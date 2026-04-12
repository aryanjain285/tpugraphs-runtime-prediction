import json
from pathlib import Path


import numpy as np
import pandas as pd

from xgboost import XGBRanker

BASE = Path()
DATA = BASE / "data"
TRAIN_TILE_DIR = DATA / "npz_all/npz/tile/xla/train"
VALID_TILE_DIR = DATA / "npz_all/npz/tile/xla/valid"

train_files = sorted(TRAIN_TILE_DIR.glob("*.npz"))
valid_files = sorted(VALID_TILE_DIR.glob("*.npz"))

print("train graphs:", len(train_files))
print("valid graphs:", len(valid_files))
print("sample train file:", train_files[0] if train_files else "none")

MAX_TRAIN_FILES = 1200
MAX_VALID_FILES = 120

train_files_small = train_files[:MAX_TRAIN_FILES]
valid_files_small = valid_files[:MAX_VALID_FILES]

print("using train graphs:", len(train_files_small))
print("using valid graphs:", len(valid_files_small))

MODEL_TAG = "tile_xgb_ranker_v1"
ARTIFACT_DIR = BASE / 'artifacts'
MODEL_ARTIFACT_DIR = ARTIFACT_DIR / MODEL_TAG

MODEL_PATH = MODEL_ARTIFACT_DIR / f"{MODEL_TAG}.json"
META_PATH = MODEL_ARTIFACT_DIR / f"{MODEL_TAG}_meta.json"
VALID_SUBMISSION_PREVIEW_PATH = MODEL_ARTIFACT_DIR / f"{MODEL_TAG}_valid_submission_preview.csv"

LOAD_MODEL_IF_AVAILABLE = True
FORCE_RETRAIN = False

print("model artifact dir:", MODEL_ARTIFACT_DIR)
print("model path:", MODEL_PATH)


MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def summarize_graph(d, opcode_bins=128):
    node_feat = d["node_feat"].astype(np.float32)      # (n, 140)
    node_opcode = d["node_opcode"].astype(np.int32)    # (n,)
    edge_index = d["edge_index"].astype(np.int64)      # (m, 2)

    n_nodes = int(node_feat.shape[0])
    n_edges = int(edge_index.shape[0])

    # Basic graph stats
    graph_stats = np.array([
        n_nodes,
        n_edges,
        n_edges / max(n_nodes, 1),
    ], dtype=np.float32)

    # Node feature summaries
    feat_mean = node_feat.mean(axis=0)
    feat_std = node_feat.std(axis=0)
    feat_min = node_feat.min(axis=0)
    feat_max = node_feat.max(axis=0)

    # Degree stats
    if n_edges > 0:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        in_deg = np.bincount(dst, minlength=n_nodes).astype(np.float32)
        out_deg = np.bincount(src, minlength=n_nodes).astype(np.float32)

        degree_stats = np.array([
            in_deg.mean(), in_deg.std(), in_deg.max(),
            out_deg.mean(), out_deg.std(), out_deg.max(),
        ], dtype=np.float32)
    else:
        degree_stats = np.zeros(6, dtype=np.float32)

    # Opcode histogram
    clipped = np.clip(node_opcode, 0, opcode_bins - 1)
    opcode_hist = np.bincount(clipped, minlength=opcode_bins).astype(np.float32)
    opcode_hist /= max(len(node_opcode), 1)

    return np.concatenate([
        graph_stats,
        degree_stats,
        feat_mean,
        feat_std,
        feat_min,
        feat_max,
        opcode_hist,
    ]).astype(np.float32)

def transform_runtime(runtime, normalizer):
    runtime = runtime.astype(np.float32)
    normalizer = normalizer.astype(np.float32)
    ratio = runtime / np.maximum(normalizer, 1e-8)   # lower is better
    order = np.argsort(ratio)                        # best first
    rel = np.empty(len(ratio), dtype=np.int32)
    rel[order] = np.arange(len(ratio) - 1, -1, -1, dtype=np.int32)
    return rel, ratio

def build_grouped_rows(file_list, opcode_bins=128):
    X_rows = []
    y_rank_rows = []
    y_ratio_rows = []
    graph_ids = []
    group_sizes = []

    for i, fp in enumerate(file_list):
        d = dict(np.load(fp, allow_pickle=False))

        graph_vec = summarize_graph(d, opcode_bins=opcode_bins)
        config_feat = d["config_feat"].astype(np.float32)  # (c, 24)

        y_rank, y_ratio = transform_runtime(
            d["config_runtime"],
            d["config_runtime_normalizers"]
        )

        num_configs = int(config_feat.shape[0])
        group_sizes.append(num_configs)

        graph_id = fp.stem
        for j in range(num_configs):
            row = np.concatenate([graph_vec, config_feat[j]]).astype(np.float32)
            X_rows.append(row)
            y_rank_rows.append(y_rank[j])
            y_ratio_rows.append(y_ratio[j])
            graph_ids.append(graph_id)

        if (i + 1) % 50 == 0:
            print(f"processed {i+1} / {len(file_list)} graphs")

    X = np.stack(X_rows)
    y_rank = np.array(y_rank_rows, dtype=np.float32)
    y_ratio = np.array(y_ratio_rows, dtype=np.float32)
    graph_ids = np.array(graph_ids)
    group_sizes = np.array(group_sizes, dtype=np.int32)

    return X, y_rank, y_ratio, graph_ids, group_sizes

X_train, y_train_rank, y_train_ratio, g_train, group_train = build_grouped_rows(train_files_small)
X_valid, y_valid_rank, y_valid_ratio, g_valid, group_valid = build_grouped_rows(valid_files_small)

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("train groups:", len(group_train), "rows:", group_train.sum())
print("valid groups:", len(group_valid), "rows:", group_valid.sum())

def build_model():
    return XGBRanker(
        objective="rank:ndcg",
        tree_method="hist",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="ndcg@5",
        ndcg_exp_gain=False,
    )

should_load = LOAD_MODEL_IF_AVAILABLE and MODEL_PATH.exists() and not FORCE_RETRAIN

if should_load:
    model = build_model()
    model.load_model(MODEL_PATH)
    print(f"loaded existing model from: {MODEL_PATH}")
else:
    model = build_model()
    model.fit(
        X_train,
        y_train_rank,
        group=group_train.tolist(),
        eval_set=[(X_valid, y_valid_rank)],
        eval_group=[group_valid.tolist()],
        verbose=50,
    )

    model.save_model(MODEL_PATH)

    metadata = {
        "model_tag": MODEL_TAG,
        "model_path": str(MODEL_PATH),
        "feature_dim": int(X_train.shape[1]),
        "train_graphs": int(len(train_files_small)),
        "valid_graphs": int(len(valid_files_small)),
        "train_rows": int(X_train.shape[0]),
        "valid_rows": int(X_valid.shape[0]),
        "group_train_sum": int(group_train.sum()),
        "group_valid_sum": int(group_valid.sum()),
        "xgb_params": model.get_xgb_params(),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved model to: {MODEL_PATH}")
    print(f"saved metadata to: {META_PATH}")

pred_valid = model.predict(X_valid)
print(pred_valid[:5])

# Optional sanity check: load the saved model again and verify predictions match.
reloaded_model = build_model()
reloaded_model.load_model(MODEL_PATH)
pred_valid_reloaded = reloaded_model.predict(X_valid)

print("reloaded predictions match:", np.allclose(pred_valid, pred_valid_reloaded))

def iter_group_slices(graph_ids, group_sizes):
    start = 0
    for size in group_sizes:
        stop = start + int(size)
        yield str(graph_ids[start]), start, stop
        start = stop


def tile_topk_error(y_true_ratio_group, y_pred_score_group, k):
    k = min(k, len(y_true_ratio_group))
    topk_idx = np.argsort(-y_pred_score_group)[:k]
    best_true = float(np.min(y_true_ratio_group))
    best_in_topk = float(np.min(y_true_ratio_group[topk_idx]))
    return (best_in_topk - best_true) / best_true


def tile_topk_score(y_true_ratio_group, y_pred_score_group, k):
    return 1.0 - tile_topk_error(y_true_ratio_group, y_pred_score_group, k)


def make_tile_submission_df(graph_ids, group_sizes, y_pred_score, top_k=5, prefix="tile:xla:"):
    rows = []

    for graph_id, start, stop in iter_group_slices(graph_ids, group_sizes):
        local_scores = y_pred_score[start:stop]
        top_idx = np.argsort(-local_scores)[:min(top_k, stop - start)]
        topconfigs = ";".join(map(str, top_idx.tolist()))
        rows.append({
            "ID": f"{prefix}{graph_id}",
            "TopConfigs": topconfigs,
        })

    return pd.DataFrame(rows)


def evaluate_tile_competition_metrics(graph_ids, group_sizes, y_true_ratio, y_pred_score):
    top1_hits = 0
    top5_hits = 0
    slowdown_at_1 = []
    slowdown_best_of_top5 = []

    top1_errors = []
    top5_errors = []
    top10_errors = []

    top1_scores = []
    top5_scores = []
    top10_scores = []

    for _, start, stop in iter_group_slices(graph_ids, group_sizes):
        true_ratio = y_true_ratio[start:stop]
        pred_score = y_pred_score[start:stop]

        true_best_idx = int(np.argmin(true_ratio))
        pred_order = np.argsort(-pred_score)

        pred_best_idx = int(pred_order[0])
        top5_pred_idx = pred_order[:min(5, len(pred_order))]

        true_best_runtime = float(np.min(true_ratio))
        chosen_runtime_top1 = float(true_ratio[pred_best_idx])
        chosen_runtime_top5 = float(np.min(true_ratio[top5_pred_idx]))

        top1_hits += int(pred_best_idx == true_best_idx)
        top5_hits += int(true_best_idx in set(top5_pred_idx.tolist()))

        slowdown_at_1.append(chosen_runtime_top1 / true_best_runtime)
        slowdown_best_of_top5.append(chosen_runtime_top5 / true_best_runtime)

        err1 = tile_topk_error(true_ratio, pred_score, k=1)
        err5 = tile_topk_error(true_ratio, pred_score, k=5)
        err10 = tile_topk_error(true_ratio, pred_score, k=10)

        top1_errors.append(err1)
        top5_errors.append(err5)
        top10_errors.append(err10)

        top1_scores.append(1.0 - err1)
        top5_scores.append(1.0 - err5)
        top10_scores.append(1.0 - err10)

    n_graphs = len(group_sizes)

    return {
        # Official tile-side error form from the TPU Graphs evaluation script
        "tile_error@1": float(np.mean(top1_errors)),
        "tile_error@5": float(np.mean(top5_errors)),
        "tile_error@10": float(np.mean(top10_errors)),

        # Competition-style score form (1 - error)
        "tile_score@1": float(np.mean(top1_scores)),
        "tile_score@5": float(np.mean(top5_scores)),
        "tile_score@10": float(np.mean(top10_scores)),

        # Same information in the more intuitive runtime-ratio view
        "mean_slowdown@1": float(np.mean(slowdown_at_1)),
        "mean_best_of_top5_slowdown": float(np.mean(slowdown_best_of_top5)),

        # Helpful debug metrics
        "top1_exact_hit_rate": top1_hits / n_graphs,
        "top5_contains_best_rate": top5_hits / n_graphs,

        # Optional percentage view for readability
        "tile_error_pct@1": float(100.0 * np.mean(top1_errors)),
        "tile_error_pct@5": float(100.0 * np.mean(top5_errors)),
        "tile_error_pct@10": float(100.0 * np.mean(top10_errors)),
    }


metrics = evaluate_tile_competition_metrics(
    g_valid,
    group_valid,
    y_valid_ratio,
    pred_valid,
)
pd.Series(metrics)

def random_baseline(graph_ids, group_sizes, y_true_ratio, seed=42):
    rng = np.random.default_rng(seed)

    top1_hits = 0
    top5_hits = 0
    slowdown_at_1 = []
    slowdown_best_of_top5 = []

    top1_errors = []
    top5_errors = []
    top10_errors = []

    top1_scores = []
    top5_scores = []
    top10_scores = []

    for _, start, stop in iter_group_slices(graph_ids, group_sizes):
        true_ratio = y_true_ratio[start:stop]
        n = len(true_ratio)

        rand_perm = rng.permutation(n)
        rand_scores = np.empty(n, dtype=np.float32)
        rand_scores[rand_perm] = np.arange(n, 0, -1, dtype=np.float32)

        true_best_idx = int(np.argmin(true_ratio))
        rand_top1 = int(rand_perm[0])
        rand_top5 = rand_perm[:min(5, n)]

        true_best_runtime = float(np.min(true_ratio))
        chosen_runtime_top1 = float(true_ratio[rand_top1])
        chosen_runtime_top5 = float(np.min(true_ratio[rand_top5]))

        top1_hits += int(rand_top1 == true_best_idx)
        top5_hits += int(true_best_idx in set(rand_top5.tolist()))

        slowdown_at_1.append(chosen_runtime_top1 / true_best_runtime)
        slowdown_best_of_top5.append(chosen_runtime_top5 / true_best_runtime)

        err1 = tile_topk_error(true_ratio, rand_scores, k=1)
        err5 = tile_topk_error(true_ratio, rand_scores, k=5)
        err10 = tile_topk_error(true_ratio, rand_scores, k=10)

        top1_errors.append(err1)
        top5_errors.append(err5)
        top10_errors.append(err10)

        top1_scores.append(1.0 - err1)
        top5_scores.append(1.0 - err5)
        top10_scores.append(1.0 - err10)

    n_graphs = len(group_sizes)

    return {
        "tile_error@1": float(np.mean(top1_errors)),
        "tile_error@5": float(np.mean(top5_errors)),
        "tile_error@10": float(np.mean(top10_errors)),
        "tile_score@1": float(np.mean(top1_scores)),
        "tile_score@5": float(np.mean(top5_scores)),
        "tile_score@10": float(np.mean(top10_scores)),
        "mean_slowdown@1": float(np.mean(slowdown_at_1)),
        "mean_best_of_top5_slowdown": float(np.mean(slowdown_best_of_top5)),
        "top1_exact_hit_rate": top1_hits / n_graphs,
        "top5_contains_best_rate": top5_hits / n_graphs,
        "tile_error_pct@1": float(100.0 * np.mean(top1_errors)),
        "tile_error_pct@5": float(100.0 * np.mean(top5_errors)),
        "tile_error_pct@10": float(100.0 * np.mean(top10_errors)),
    }


rand_metrics = random_baseline(
    g_valid,
    group_valid,
    y_valid_ratio,
    seed=42,
)
pd.Series(rand_metrics)

valid_submission_preview = make_tile_submission_df(
    g_valid,
    group_valid,
    pred_valid,
    top_k=5,
    prefix="tile:xla:",
)

valid_submission_preview.to_csv(VALID_SUBMISSION_PREVIEW_PATH, index=False)

comparison = pd.DataFrame([
    {"model": "XGBRanker", **metrics},
    {"model": "Random", **rand_metrics},
])

print("saved validation submission preview to:", VALID_SUBMISSION_PREVIEW_PATH)



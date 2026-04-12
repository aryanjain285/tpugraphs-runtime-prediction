from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class SubmissionGraphBundle:
    graph: Any
    config_groups: list[list[int]]

    @property
    def original_num_configs(self) -> int:
        return sum(len(group) for group in self.config_groups)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    if np.isnan(value):
        return default
    return value


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return re.sub(r"_+", "_", name).strip("_")


def parse_gnn_components(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part for part in text.split("|") if part]


def canonical_submission_key(experiment_name: str, dataset_name: str) -> str:
    dataset_token = f"_{dataset_name}"
    if dataset_token in experiment_name:
        return experiment_name.replace(dataset_token, "")
    return experiment_name


def prepare_module_runtime(module, base: Path | None = None):
    if base is not None:
        module.CONFIG["base"] = str(base)
    if hasattr(module, "set_runtime_device"):
        gpu_id = 0 if torch.cuda.is_available() else None
        module.set_runtime_device(gpu_id)
    return module


def dedupe_layout_configs_with_groups(module, node_config_feat: np.ndarray, strategy: str):
    if strategy == "none" or node_config_feat.shape[0] == 0:
        groups = [[int(i)] for i in range(int(node_config_feat.shape[0]))]
        dup = np.ones(node_config_feat.shape[0], dtype=np.float32)
        return node_config_feat, dup, groups

    group_map: dict[bytes, list[int]] = {}
    for i in range(node_config_feat.shape[0]):
        signature = module.config_signature_bytes(node_config_feat[i])
        if signature not in group_map:
            group_map[signature] = []
        group_map[signature].append(i)

    kept_cfg = []
    duplicate_count = []
    config_groups: list[list[int]] = []
    for indices in group_map.values():
        kept_cfg.append(node_config_feat[indices[0]])
        duplicate_count.append(float(len(indices)))
        config_groups.append([int(idx) for idx in indices])

    kept_cfg = np.stack(kept_cfg, axis=0) if kept_cfg else node_config_feat[:0]
    dup = np.asarray(duplicate_count, dtype=np.float32)
    return kept_cfg, dup, config_groups


def preprocess_test_graph(module, cfg, npz_path: Path) -> SubmissionGraphBundle:
    data = dict(np.load(npz_path))
    node_feat_raw = module.sanitize_dense_features(
        data["node_feat"],
        clip_value=cfg.feature_clip_value,
    )
    node_numeric_feat = module.standardize_node_numeric_feat_np(
        node_feat_raw[:, module.NODE_NUMERIC_INDICES],
        mean=cfg.node_numeric_mean,
        std=cfg.node_numeric_std,
    )
    node_shape_type = module.extract_shape_type_id_np(node_feat_raw)
    node_layout_feat = module.encode_layout_values_np(node_feat_raw[:, module.NODE_LAYOUT_START:module.NODE_LAYOUT_END])

    node_opcode = data["node_opcode"].astype(np.int64)
    edge_index = data["edge_index"].astype(np.int64)
    node_config_ids = data["node_config_ids"].astype(np.int64)

    if cfg.blobify_enabled:
        (
            node_numeric_feat,
            node_shape_type,
            node_layout_feat,
            node_opcode,
            node_config_ids,
            edge_index,
        ) = module.blobify_graph_numpy(
            node_numeric_feat=node_numeric_feat,
            node_shape_type=node_shape_type,
            node_layout_feat=node_layout_feat,
            node_opcode=node_opcode,
            node_config_ids=node_config_ids,
            edge_index=edge_index,
            node_feat_raw=node_feat_raw,
            keep_hops=cfg.blobify_keep_hops,
            frontier_split=cfg.blobify_frontier_split,
            summary_opcode_id=cfg.summary_opcode_id,
        )
    else:
        node_numeric_feat = module.append_blob_extra_zeros_np(node_numeric_feat)

    node_config_feat = module.prepare_node_config_storage_np(
        data["node_config_feat"],
        pack_node_config_feat=cfg.pack_node_config_feat,
    )
    node_config_feat, duplicate_count, config_groups = dedupe_layout_configs_with_groups(
        module=module,
        node_config_feat=node_config_feat,
        strategy=cfg.dedup_strategy,
    )

    sage_edge_index, sage_deg = module.build_sage_edges_numpy(int(node_numeric_feat.shape[0]), edge_index)
    graph_storage_dtype = module.get_graph_storage_numpy_dtype(cfg.graph_storage_dtype)
    if graph_storage_dtype == np.float16:
        finfo = np.finfo(np.float16)
        node_numeric_feat = np.clip(node_numeric_feat, finfo.min, finfo.max)

    graph = module.GraphExample(
        graph_id=npz_path.stem,
        num_configs=int(node_config_feat.shape[0]),
        node_numeric_feat=torch.as_tensor(node_numeric_feat.astype(graph_storage_dtype, copy=False)),
        node_shape_type=torch.as_tensor(node_shape_type.astype(np.int64, copy=False), dtype=torch.long),
        node_layout_feat=torch.as_tensor(node_layout_feat.astype(np.int64, copy=False), dtype=torch.long),
        node_opcode=torch.as_tensor(node_opcode.astype(np.int64, copy=False), dtype=torch.long),
        node_config_ids=torch.as_tensor(node_config_ids.astype(np.int64, copy=False), dtype=torch.long),
        node_config_feat=torch.as_tensor(np.ascontiguousarray(node_config_feat)),
        sage_edge_index=torch.as_tensor(sage_edge_index.astype(np.int64, copy=False), dtype=torch.long),
        sage_deg=torch.as_tensor(sage_deg.astype(np.float32, copy=False)),
        runtimes=None,
        duplicate_count=torch.as_tensor(duplicate_count.astype(np.float32, copy=False)),
    )
    return SubmissionGraphBundle(graph=graph, config_groups=config_groups)


def prepare_submission_environment(module, dataset_name: str):
    cfg, _enabled_gnns, _run_xgb, _run_ensemble = module.build_runtime_config(dataset_name)
    module.configure_torch_runtime(cfg)
    module.seed_everything(cfg.seed)

    train_dir = module.find_layout_split_dir(cfg, "train")
    valid_dir = module.find_layout_split_dir(cfg, "valid")
    test_dir = module.find_layout_split_dir(cfg, "test")

    print(f"[paths] {dataset_name}")
    print(f"        train_dir : {train_dir}")
    print(f"        valid_dir : {valid_dir}")
    print(f"        test_dir  : {test_dir}")

    train_files = module.list_npz_files(train_dir, cfg.max_train_graphs)
    valid_files = module.list_npz_files(valid_dir, cfg.max_valid_graphs)
    test_files = module.list_npz_files(test_dir, None)

    print(f"[counts] {dataset_name}: train={len(train_files)} valid={len(valid_files)} test={len(test_files)}")
    if not train_files:
        raise RuntimeError(f"No train NPZ files found in {train_dir}")
    if not valid_files:
        print(f"[warn] {dataset_name}: no valid NPZ files found in {valid_dir}")
    if not test_files:
        print(f"[warn] {dataset_name}: no test NPZ files found in {test_dir}")

    cfg.node_numeric_mean, cfg.node_numeric_std = module.fit_node_numeric_scaler(
        files=train_files,
        feature_clip_value=cfg.feature_clip_value,
    )
    max_opcode_seen = module.scan_max_opcode(train_files + valid_files)
    num_opcodes = max(129, max_opcode_seen + 2)
    cfg.summary_opcode_id = num_opcodes - 1

    test_bundles = [preprocess_test_graph(module, cfg, path) for path in test_files]
    return cfg, num_opcodes, test_bundles


def build_test_xgb_pack(module, graph_bundles: list[SubmissionGraphBundle]) -> dict[str, Any]:
    features = []
    rows = []
    for bundle in graph_bundles:
        graph = bundle.graph
        feat = module.summarize_xgb_config_features_np(graph.node_config_feat.cpu().numpy())
        dup = np.ones(graph.num_configs, dtype=np.float32) if graph.duplicate_count is None else graph.duplicate_count.cpu().numpy()
        feat = np.concatenate([feat, dup[:, None]], axis=1).astype(np.float32)
        features.append(feat)
        rows.append(pd.DataFrame({
            "graph_id": graph.graph_id,
            "row_id": np.arange(graph.num_configs, dtype=np.int32),
        }))

    if not features:
        raise RuntimeError("No test features were built for XGB inference.")

    return {
        "X": np.vstack(features),
        "meta": pd.concat(rows, ignore_index=True),
    }


def load_gnn_artifact(module, cfg, experiment_name: str, num_opcodes: int):
    experiments = {exp.name: exp for exp in module.build_gnn_experiments()}
    if experiment_name not in experiments:
        raise KeyError(f"Unknown GNN experiment: {experiment_name}")

    exp_dir = cfg.out_dir / experiment_name
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing GNN checkpoint: {model_path}")

    exp = module.maybe_load_experiment_config(exp_dir, experiments[experiment_name])
    model = module.LayoutGraphAwareModel(num_opcodes=num_opcodes, exp=exp).to(module.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=module.DEVICE))
    model.eval()
    return {
        "name": experiment_name,
        "type": "gnn",
        "model": model,
        "config": exp,
        "model_dir": exp_dir,
    }


def load_xgb_artifact(module, cfg):
    if module.xgb is None:
        raise ImportError("xgboost is not available in this environment.")

    fallback_exp = module.build_xgb_experiment()
    exp_dir = cfg.out_dir / fallback_exp.name
    model_path = exp_dir / "xgb_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing XGB checkpoint: {model_path}")

    exp = module.maybe_load_experiment_config(exp_dir, fallback_exp)
    model = module.xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        n_estimators=exp.n_estimators,
        learning_rate=exp.learning_rate,
        max_depth=exp.max_depth,
        min_child_weight=exp.min_child_weight,
        subsample=exp.subsample,
        colsample_bytree=exp.colsample_bytree,
        reg_lambda=exp.reg_lambda,
        random_state=exp.random_state,
        tree_method=exp.tree_method,
        n_jobs=getattr(exp, "n_jobs", 1),
    )
    model.load_model(model_path)
    return {
        "name": fallback_exp.name,
        "type": "xgb",
        "model": model,
        "config": exp,
        "model_dir": exp_dir,
    }


def predict_gnn(module, artifact: dict[str, Any], graph_bundles: list[SubmissionGraphBundle]) -> dict[str, np.ndarray]:
    graphs = [bundle.graph for bundle in graph_bundles]
    return module.predict_graphs(
        exp=artifact["config"],
        model=artifact["model"],
        graph_items=graphs,
    )


def predict_xgb(module, artifact: dict[str, Any], graph_bundles: list[SubmissionGraphBundle]) -> dict[str, np.ndarray]:
    pack = build_test_xgb_pack(module, graph_bundles)
    return module.predict_xgb_by_graph(artifact["model"], pack)


def load_experiment_summary(cfg, experiment_name: str) -> dict[str, Any]:
    summary_path = cfg.out_dir / experiment_name / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_predictions_for_row(
    module,
    cfg,
    row: pd.Series,
    num_opcodes: int,
    graph_bundles: list[SubmissionGraphBundle],
    prediction_cache: dict[str, dict[str, np.ndarray]],
    artifact_cache: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    experiment_name = str(row["experiment"])
    experiment_type = str(row["type"])

    if experiment_name in prediction_cache:
        return prediction_cache[experiment_name]

    if experiment_type == "gnn":
        artifact = artifact_cache.get(experiment_name)
        if artifact is None:
            artifact = load_gnn_artifact(module, cfg, experiment_name, num_opcodes)
            artifact_cache[experiment_name] = artifact
        preds = predict_gnn(module, artifact, graph_bundles)
        prediction_cache[experiment_name] = preds
        return preds

    if experiment_type == "xgb":
        artifact = artifact_cache.get(experiment_name)
        if artifact is None:
            artifact = load_xgb_artifact(module, cfg)
            artifact_cache[experiment_name] = artifact
        preds = predict_xgb(module, artifact, graph_bundles)
        prediction_cache[experiment_name] = preds
        return preds

    if experiment_type != "ensemble":
        raise ValueError(f"Unsupported experiment type: {experiment_type}")

    summary_payload = load_experiment_summary(cfg, experiment_name)
    gnn_components = parse_gnn_components(row.get("gnn_components"))
    if not gnn_components:
        gnn_components = parse_gnn_components(summary_payload.get("gnn_components"))
    if not gnn_components:
        raise ValueError(f"Could not resolve gnn_components for ensemble {experiment_name}")

    xgb_component = row.get("xgb_component")
    if pd.isna(xgb_component) if hasattr(pd, "isna") else False:
        xgb_component = None
    if not xgb_component:
        xgb_component = summary_payload.get("xgb_component")
    if not xgb_component:
        raise ValueError(f"Could not resolve xgb_component for ensemble {experiment_name}")

    alpha_best_gnn = safe_float(row.get("alpha_best_gnn"), default=None)
    if alpha_best_gnn is None:
        alpha_best_gnn = safe_float(summary_payload.get("alpha_best_gnn"), default=None)
    if alpha_best_gnn is None:
        raise ValueError(f"Could not resolve alpha_best_gnn for ensemble {experiment_name}")

    gnn_pooling = row.get("gnn_pooling")
    if pd.isna(gnn_pooling) if hasattr(pd, "isna") else False:
        gnn_pooling = None
    if not gnn_pooling:
        gnn_pooling = summary_payload.get("gnn_pooling")

    gnn_results = []
    for component in gnn_components:
        if component not in prediction_cache:
            gnn_row = pd.Series({"experiment": component, "type": "gnn"})
            prediction_cache[component] = get_predictions_for_row(
                module=module,
                cfg=cfg,
                row=gnn_row,
                num_opcodes=num_opcodes,
                graph_bundles=graph_bundles,
                prediction_cache=prediction_cache,
                artifact_cache=artifact_cache,
            )
        gnn_results.append({"name": component, "preds": prediction_cache[component]})

    if xgb_component not in prediction_cache:
        xgb_row = pd.Series({"experiment": xgb_component, "type": "xgb"})
        prediction_cache[xgb_component] = get_predictions_for_row(
            module=module,
            cfg=cfg,
            row=xgb_row,
            num_opcodes=num_opcodes,
            graph_bundles=graph_bundles,
            prediction_cache=prediction_cache,
            artifact_cache=artifact_cache,
        )

    if len(gnn_results) == 1 and not gnn_pooling:
        gnn_preds = gnn_results[0]["preds"]
    else:
        gnn_preds = module.average_rank_predictions(gnn_results)

    preds = module.ensemble_predictions(
        preds_a=gnn_preds,
        preds_b=prediction_cache[xgb_component],
        alpha=float(alpha_best_gnn),
    )
    prediction_cache[experiment_name] = preds
    return preds


def predictions_to_submission_df(cfg, graph_bundles: list[SubmissionGraphBundle], preds_by_graph: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for bundle in graph_bundles:
        graph = bundle.graph
        pred = np.asarray(preds_by_graph[graph.graph_id], dtype=np.float32)
        if pred.shape[0] != len(bundle.config_groups):
            raise ValueError(
                f"Prediction length mismatch for graph {graph.graph_id}: "
                f"got {pred.shape[0]}, expected {len(bundle.config_groups)} deduped configs"
            )

        group_order = np.argsort(pred, kind="mergesort")
        full_order: list[int] = []
        for dedup_idx in group_order:
            full_order.extend(bundle.config_groups[int(dedup_idx)])

        if len(full_order) != bundle.original_num_configs:
            raise ValueError(
                f"Expanded prediction length mismatch for graph {graph.graph_id}: "
                f"got {len(full_order)}, expected {bundle.original_num_configs}"
            )

        rows.append({
            "ID": f"layout:{cfg.source}:{cfg.search}:{graph.graph_id}",
            "TopConfigs": ";".join(str(int(idx)) for idx in full_order),
        })
    return pd.DataFrame(rows)


def inspect_dataset_jobs(module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in module.CONFIG.get("dataset_presets", {}):
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "ok": False,
            "error": None,
            "summary_exists": False,
            "summary_path": None,
            "out_dir": None,
        }
        try:
            cfg, _enabled, _run_xgb, _run_ensemble = module.build_runtime_config(dataset_name)
            summary_path = cfg.out_dir / "ablation_summary.csv"
            row.update({
                "ok": True,
                "summary_exists": summary_path.exists(),
                "summary_path": summary_path,
                "out_dir": cfg.out_dir,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate updated Kaggle layout submission CSVs using the current preprocessing pipeline.")
    parser.add_argument("--base", type=Path, default=None, help="Override CONFIG['base'] for both layout modules.")
    parser.add_argument("--default-script", type=Path, default=Path(__file__).with_name("layout_default_only.py"))
    parser.add_argument("--random-script", type=Path, default=Path(__file__).with_name("layout_random_only.py"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write submission CSVs. Defaults to <base>/artifacts/kaggle_submissions")
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset names to process. Defaults to auto-discovery from ablation_summary.csv")
    parser.add_argument("--experiments", type=str, default="", help="Optional comma-separated experiment names to keep.")
    parser.add_argument("--best-only", action="store_true", help="Only write the top-ranked experiment from each dataset's ablation_summary.csv")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    modules: list[tuple[Any, str]] = []
    if args.default_script.exists():
        modules.append((prepare_module_runtime(load_module(args.default_script, "layout_default_only_submission"), args.base), "default"))
    if args.random_script.exists():
        modules.append((prepare_module_runtime(load_module(args.random_script, "layout_random_only_submission"), args.base), "random"))
    if not modules:
        raise FileNotFoundError("Could not find any layout module scripts to import.")

    base_path = args.base
    if base_path is None:
        base_path = Path(modules[0][0].CONFIG["base"])
    output_dir = args.output_dir or (Path(base_path) / "artifacts" / "kaggle_submissions")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = {part.strip() for part in args.datasets.split(",") if part.strip()}
    requested_experiments = {part.strip() for part in args.experiments.split(",") if part.strip()}

    merged_frames: dict[str, list[pd.DataFrame]] = {}
    manifest_rows: list[dict[str, Any]] = []
    any_dataset_checked = False

    print(f"base: {base_path}")
    print(f"output_dir: {output_dir}")
    print(f"requested_datasets: {sorted(requested_datasets) if requested_datasets else '[auto]'}")
    print(f"requested_experiments: {sorted(requested_experiments) if requested_experiments else '[all]'}")
    print(f"best_only: {args.best_only}")

    for module, module_label in modules:
        inspection_rows = inspect_dataset_jobs(module)
        if requested_datasets:
            inspection_rows = [row for row in inspection_rows if row["dataset"] in requested_datasets]

        print(f"\n=== {module_label} module ===")
        if not inspection_rows:
            print("[skip] no datasets matched this module.")
            continue

        dataset_names: list[str] = []
        for info in inspection_rows:
            any_dataset_checked = True
            dataset_name = str(info["dataset"])
            if not info["ok"]:
                print(f"[skip] {dataset_name}: build_runtime_config failed: {info['error']}")
                continue
            summary_path = info["summary_path"]
            print(f"[check] {dataset_name}")
            print(f"        out_dir   : {info['out_dir']}")
            print(f"        summary   : {summary_path}")
            print(f"        summary ok: {info['summary_exists']}")
            if info["summary_exists"]:
                dataset_names.append(dataset_name)
            else:
                print(f"[skip] {dataset_name}: missing ablation_summary.csv")

        if not dataset_names:
            print("[skip] no runnable datasets found for this module.")
            continue

        for dataset_name in dataset_names:
            try:
                cfg, num_opcodes, test_bundles = prepare_submission_environment(module, dataset_name)
            except Exception as exc:
                print(f"[skip] {dataset_name}: environment prep failed: {type(exc).__name__}: {exc}")
                continue

            summary_path = cfg.out_dir / "ablation_summary.csv"
            try:
                summary_df = pd.read_csv(summary_path)
            except Exception as exc:
                print(f"[skip] {dataset_name}: could not read summary: {summary_path} ({type(exc).__name__}: {exc})")
                continue

            raw_summary_len = len(summary_df)
            if requested_experiments:
                summary_df = summary_df[summary_df["experiment"].isin(requested_experiments)].copy()
            if args.best_only and len(summary_df):
                summary_df = summary_df.head(1).copy()
            if summary_df.empty:
                available = []
                if raw_summary_len and "experiment" in summary_df.columns:
                    available = summary_df["experiment"].astype(str).tolist()
                elif raw_summary_len:
                    try:
                        available = pd.read_csv(summary_path)["experiment"].astype(str).tolist()
                    except Exception:
                        available = []
                print(f"[skip] {dataset_name}: no experiments matched.")
                if requested_experiments:
                    print(f"        requested : {sorted(requested_experiments)}")
                if available:
                    print(f"        available : {available}")
                continue

            print(f"[dataset] {dataset_name}: {len(test_bundles)} test graphs, {len(summary_df)} experiments")
            if not test_bundles:
                print(f"[skip] {dataset_name}: found 0 test graphs, so no submission CSV can be written.")
                continue

            prediction_cache: dict[str, dict[str, np.ndarray]] = {}
            artifact_cache: dict[str, dict[str, Any]] = {}

            for _, row in summary_df.iterrows():
                experiment_name = str(row["experiment"])
                print(f"  -> {experiment_name}")
                try:
                    preds = get_predictions_for_row(
                        module=module,
                        cfg=cfg,
                        row=row,
                        num_opcodes=num_opcodes,
                        graph_bundles=test_bundles,
                        prediction_cache=prediction_cache,
                        artifact_cache=artifact_cache,
                    )
                    sub_df = predictions_to_submission_df(cfg, test_bundles, preds)
                except Exception as exc:
                    print(f"     failed: {type(exc).__name__}: {exc}")
                    continue

                dataset_file = output_dir / f"submission__{sanitize_filename(dataset_name)}__{sanitize_filename(experiment_name)}.csv"
                sub_df.to_csv(dataset_file, index=False)

                merged_key = canonical_submission_key(experiment_name, dataset_name)
                merged_frames.setdefault(merged_key, []).append(sub_df)

                manifest_rows.append({
                    "dataset": dataset_name,
                    "experiment": experiment_name,
                    "type": row.get("type", ""),
                    "output_csv": str(dataset_file),
                    "num_rows": int(len(sub_df)),
                    "merged_key": merged_key,
                })
                print(f"     saved: {dataset_file}")

    merged_outputs = []
    for merged_key, frames in sorted(merged_frames.items()):
        merged_df = pd.concat(frames, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["ID"], keep="first")
        merged_df = merged_df.sort_values("ID").reset_index(drop=True)
        merged_path = output_dir / f"submission__{sanitize_filename(merged_key)}.csv"
        merged_df.to_csv(merged_path, index=False)
        merged_outputs.append({"merged_key": merged_key, "output_csv": str(merged_path), "num_rows": int(len(merged_df))})
        print(f"[merged] {merged_key}: {merged_path}")

    if not any_dataset_checked:
        print("\n[warning] No datasets were even checked. Your --datasets filter may not match any dataset preset.")
    elif not manifest_rows:
        print("\n[warning] No submission CSVs were written.")
        print("          Common causes:")
        print("          - missing ablation_summary.csv")
        print("          - 0 test graphs found")
        print("          - requested experiments did not match")
        print("          - missing model checkpoints like best_model.pt or xgb_model.json")

    manifest = {
        "base": str(base_path),
        "output_dir": str(output_dir),
        "files": manifest_rows,
        "merged_files": merged_outputs,
    }
    manifest_path = output_dir / "submission_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

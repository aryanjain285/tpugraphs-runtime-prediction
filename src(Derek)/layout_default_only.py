
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
import multiprocessing as mp
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import kendalltau
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xgboost as xgb
except Exception:
    xgb = None


SEED = 42
CONFIG_FEAT_DIM = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAPE_TYPE_START = 2
SHAPE_TYPE_END = 21
NUM_SHAPE_TYPES = SHAPE_TYPE_END - SHAPE_TYPE_START
NODE_LAYOUT_START = 134
NODE_LAYOUT_END = 140
NODE_LAYOUT_SLOTS = NODE_LAYOUT_END - NODE_LAYOUT_START
LAYOUT_VALUE_MIN = -1
LAYOUT_VALUE_MAX = 5
LAYOUT_VALUE_VOCAB_SIZE = LAYOUT_VALUE_MAX - LAYOUT_VALUE_MIN + 1
NODE_NUMERIC_INDICES = np.asarray([0] + list(range(SHAPE_TYPE_END, NODE_LAYOUT_START)), dtype=np.int64)
NODE_NUMERIC_DIM = int(NODE_NUMERIC_INDICES.shape[0])
CONFIG_LAYOUT_GROUP_SIZE = 6
CONFIG_LAYOUT_GROUPS = CONFIG_FEAT_DIM // CONFIG_LAYOUT_GROUP_SIZE
BASE7 = LAYOUT_VALUE_VOCAB_SIZE
CONFIG_LAYOUT_PACKED_DIM = CONFIG_LAYOUT_GROUPS
CONFIG_LAYOUT_BASE_POWERS_NP = (BASE7 ** np.arange(CONFIG_LAYOUT_GROUP_SIZE, dtype=np.int64)).astype(np.int64)
BLOB_EXTRA_DIM = 7
NODE_TOTAL_NUMERIC_DIM = NODE_NUMERIC_DIM + BLOB_EXTRA_DIM
BLOB_SUMMARY_SHAPE_TYPE_ID = 0
BLOB_RUNTIME_WEIGHT_SOURCE_INDICES = {
    "shape_product": 28,
    "window_size_product": 44,
    "window_stride_product": 52,
    "window_dilation_product": 76,
    "base_dilation_product": 84,
    "feature_group_count": 107,
    "batch_group_count": 108,
    "slice_limit_product": 120,
    "dynamic_slice_product": 124,
    "padding_low_product": 128,
    "padding_high_product": 132,
}



# ============================================================
# Main config block
#
# This version does not use on-disk graph cache.
# Train graphs are preprocessed from .npz by CPU worker processes while the GPU
# trains on the previous graph. The training subset is resampled every epoch.
#
# Main ideas:
# - problems_to_run: which of the 4 layout problems to train.
# - problem_models: per-problem data caps and enabled experiment names.
# - gnn_experiments: per-problem model / batching definitions.
#
# Batching policy used here:
# - default sets: chunk scope + shifted_contiguous windows
# - random sets:  chunk scope + contiguous windows
#
# Training subset policy:
# - train split: default uses contiguous window with random start; random uses pure random config subsample each epoch
# - valid split: deterministic evenly-spaced coverage
#
# CPU pipeline:
# - no graph cache
# - num_preprocess_workers worker processes
# - bounded prefetch queue feeding the GPU consumer
# ============================================================

CONFIG = {
    "base": "/scratch/users/ntu/dere0006/TPU",
    "problems_to_run": [
        "xla_default",
        "nlp_default",
    ],
    "dataset_presets": {
        "xla_random": {"source": "xla", "search": "random"},
        "xla_default": {"source": "xla", "search": "default"},
        "nlp_random": {"source": "nlp", "search": "random"},
        "nlp_default": {"source": "nlp", "search": "default"},
    },

    # Naming
    "out_dir_template": "layout_{source}_{search}_stream_v1",
    "combined_best_name": "layout_default_problem_best_models.csv",

    # Data / preprocessing
    "dedup_strategy": "median",   # none / mean / median / min
    "feature_clip_value": 1e6,
    "max_train_graphs": None,
    "max_valid_graphs": None,
    "max_train_configs_per_graph": None,
    "max_valid_configs_per_graph": None,

    # Runtime / loading
    "seed": SEED,
    "preload_valid_in_memory": True,
    "num_preprocess_workers": 12,
    "prefetch_graphs": 12,
    "log_prefetch_status": True,
    "prefetch_status_every_n_graphs": 10,
    "requested_num_cpus": 16,
    "prefer_cached_experiments": True,
    "train_missing_gnn_experiments": True,
    "train_missing_xgb_experiment": True,
    "prefer_saved_valid_preds": True,
    "recompute_static_per_chunk_train": False,
    "empty_cache_between_graphs": False,
    "allow_tf32": True,
    "amp": True,

    # Which experiments run is controlled only inside problem_models.
    # The fields below only affect how already-enabled GNNs are combined in ensembles.
    "ensemble_always_include_gnns": [],
    "ensemble_auto_select_beneficial_gnns": True,
    "ensemble_min_tau_ratio_to_best": 0.50,
    "ensemble_top_k_gnns": 2,

    # Memory controls
    # - graph_storage_dtype: store graph float features as float16 or float32
    # - autocast_static_encode: let cached static node embeddings live in AMP dtype
    "graph_storage_dtype": "float16",
    "autocast_static_encode": True,
    "pack_node_config_feat": True,
    "type_embed_dim": 4,
    "layout_value_embed_dim": 4,
    "blobify_enabled": False,
    "blobify_keep_hops": 2,
    "blobify_frontier_split": True,


    "problem_models": {
        "xla_default": {
            "enabled_gnn_experiments": ["gnn_xla_default"],
            "run_xgb_experiment": True,
            "run_ensemble_search": True,
            "max_train_configs_per_graph": 2048,
            "max_valid_configs_per_graph": 4096,
            "train_subsample_mode": "contiguous_random_start",
        },
        "xla_random": {
            "enabled_gnn_experiments": ["gnn_xla_random"],
            "run_xgb_experiment": True,
            "run_ensemble_search": True,
            "max_train_configs_per_graph": 2048,
            "max_valid_configs_per_graph": 4096,
            "train_subsample_mode": "random",
        },
        "nlp_default": {
            "enabled_gnn_experiments": ["gnn_nlp_default"],
            "run_xgb_experiment": True,
            "run_ensemble_search": True,
            "max_train_configs_per_graph": 2048,
            "max_valid_configs_per_graph": 4096,
            "train_subsample_mode": "contiguous_random_start",
        },
        "nlp_random": {
            "enabled_gnn_experiments": ["gnn_nlp_random"],
            "run_xgb_experiment": True,
            "run_ensemble_search": True,
            "max_train_configs_per_graph": 2048,
            "max_valid_configs_per_graph": 4096,
            "train_subsample_mode": "random",
        },
    },

    # Shared model/runtime defaults
    "gnn_defaults": {
        "hidden_dim": 96,
        "op_embed_dim": 32,
        "type_embed_dim": 4,
        "layout_value_embed_dim": 4,
        "use_graph_global_pool": True,
        "num_gnn_layers": 4,
        "dropout": 0.05,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "epochs": 60,
        "early_stopping": 20,
        "train_config_batch": 32,
        "eval_config_batch": 32,
        "pairwise_samples": 1024,
        "chunk_batching": "contiguous",
        "shift_stride": None,   # default -> batch_size // 2
        "grad_clip": 1.0,
        "default_max_train_configs_per_graph": 1024,
        "default_max_valid_configs_per_graph": 1024,
    },

    "gnn_experiments": [
        {
            "name": "gnn_xla_default",
            "pairwise_weight": 1.0,
            "pairwise_mode": "random",
            "chunk_batching": "shifted_contiguous",
        },
        {
            "name": "gnn_xla_random",
            "pairwise_weight": 1.0,
            "pairwise_mode": "random",
            "chunk_batching": "contiguous",
        },
        {
            "name": "gnn_nlp_default",
            "pairwise_weight": 1.0,
            "pairwise_mode": "random",
            "chunk_batching": "shifted_contiguous",
        },
        {
            "name": "gnn_nlp_random",
            "pairwise_weight": 1.0,
            "pairwise_mode": "random",
            "chunk_batching": "contiguous",
        },
    ],

    "xgb_experiment": {
        "name": "xgb_pairwise",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "tree_method": "hist",
        "n_jobs": 8,
    },
}

# ============================================================


@dataclass
class GraphExample:
    graph_id: str
    num_configs: int
    node_numeric_feat: torch.Tensor
    node_shape_type: torch.Tensor
    node_layout_feat: torch.Tensor
    node_opcode: torch.Tensor
    node_config_ids: torch.Tensor
    node_config_feat: torch.Tensor
    sage_edge_index: torch.Tensor
    sage_deg: torch.Tensor
    runtimes: torch.Tensor | None
    duplicate_count: torch.Tensor | None


@dataclass
class DeviceGraphView:
    node_numeric_feat: torch.Tensor
    node_shape_type: torch.Tensor
    node_layout_feat: torch.Tensor
    node_opcode: torch.Tensor
    node_config_ids: torch.Tensor
    sage_edge_index: torch.Tensor
    sage_deg: torch.Tensor


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    except Exception:
        pass


def get_autocast_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def get_graph_storage_numpy_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError(f"Unsupported graph_storage_dtype: {dtype_name!r}")


def autocast_context(enabled: bool):
    return torch.amp.autocast(
        device_type=DEVICE.type,
        dtype=get_autocast_dtype(),
        enabled=(enabled and DEVICE.type == "cuda"),
    )


def get_mp_context():
    # Use spawn on cluster jobs too. Forking after CUDA has been initialized is fragile,
    # and this script launches CPU preprocess workers while using CUDA in the parent process.
    return mp.get_context("spawn")


def get_preprocess_variant_name(blobify_enabled: bool) -> str:
    return "blobify_opcode" if blobify_enabled else "plain"


def append_variant_to_filename(filename: str, variant_name: str) -> str:
    path = Path(filename)
    suffix = path.suffix or ".csv"
    return f"{path.stem}_{variant_name}{suffix}"


def build_runtime_config(dataset_name: str):
    dataset_presets = CONFIG["dataset_presets"]
    if dataset_name not in dataset_presets:
        raise ValueError(f"Unknown dataset preset: {dataset_name!r}. Available: {sorted(dataset_presets)}")

    dataset_cfg = dict(dataset_presets[dataset_name])
    problem_cfg = dict(CONFIG.get("problem_models", {}).get(dataset_name, {}))
    required_problem_keys = [
        "enabled_gnn_experiments",
        "run_xgb_experiment",
        "run_ensemble_search",
    ]
    missing_problem_keys = [key for key in required_problem_keys if key not in problem_cfg]
    if missing_problem_keys:
        raise KeyError(
            f"problem_models[{dataset_name!r}] is missing required keys: {missing_problem_keys}. "
            "Per-problem config is now the single source of truth for what runs."
        )
    defaults = dict(CONFIG["gnn_defaults"])

    source = dataset_cfg["source"]
    search = dataset_cfg["search"]
    preprocess_variant = get_preprocess_variant_name(bool(CONFIG["blobify_enabled"]))
    out_dir_name = CONFIG["out_dir_template"].format(source=source, search=search) + f"_{preprocess_variant}"

    max_train_cfg = problem_cfg.get("max_train_configs_per_graph", CONFIG["max_train_configs_per_graph"])
    if max_train_cfg is None:
        max_train_cfg = defaults["default_max_train_configs_per_graph"]
    max_valid_cfg = problem_cfg.get("max_valid_configs_per_graph", CONFIG["max_valid_configs_per_graph"])
    if max_valid_cfg is None:
        max_valid_cfg = defaults["default_max_valid_configs_per_graph"]

    cfg = SimpleNamespace(
        base=Path(CONFIG["base"]),
        dataset=dataset_name,
        source=source,
        search=search,
        seed=int(CONFIG["seed"]),
        dedup_strategy=problem_cfg.get("dedup_strategy", CONFIG["dedup_strategy"]),
        out_dir_name=out_dir_name,
        feature_clip_value=float(CONFIG["feature_clip_value"]),
        max_train_graphs=CONFIG["max_train_graphs"],
        max_valid_graphs=CONFIG["max_valid_graphs"],
        max_train_configs_per_graph=max_train_cfg,
        max_valid_configs_per_graph=max_valid_cfg,
        train_subsample_mode=str(problem_cfg.get("train_subsample_mode", "random")),
        preload_valid_in_memory=bool(CONFIG["preload_valid_in_memory"]),
        num_preprocess_workers=int(CONFIG["num_preprocess_workers"]),
        prefetch_graphs=int(CONFIG["prefetch_graphs"]),
        log_prefetch_status=bool(CONFIG.get("log_prefetch_status", False)),
        prefetch_status_every_n_graphs=max(1, int(CONFIG.get("prefetch_status_every_n_graphs", 10))),
        recompute_static_per_chunk_train=bool(CONFIG["recompute_static_per_chunk_train"]),
        empty_cache_between_graphs=bool(CONFIG["empty_cache_between_graphs"]),
        allow_tf32=bool(CONFIG["allow_tf32"]),
        prefer_cached_experiments=bool(CONFIG["prefer_cached_experiments"]),
        train_missing_gnn_experiments=bool(CONFIG["train_missing_gnn_experiments"]),
        train_missing_xgb_experiment=bool(CONFIG["train_missing_xgb_experiment"]),
        prefer_saved_valid_preds=bool(CONFIG["prefer_saved_valid_preds"]),
        ensemble_gnn_experiments=list(problem_cfg.get("ensemble_gnn_experiments", problem_cfg["enabled_gnn_experiments"])),
        ensemble_always_include_gnns=list(problem_cfg.get("ensemble_always_include_gnns", CONFIG["ensemble_always_include_gnns"])),
        ensemble_auto_select_beneficial_gnns=bool(problem_cfg.get("ensemble_auto_select_beneficial_gnns", CONFIG["ensemble_auto_select_beneficial_gnns"])),
        ensemble_min_tau_ratio_to_best=float(problem_cfg.get("ensemble_min_tau_ratio_to_best", CONFIG["ensemble_min_tau_ratio_to_best"])),
        ensemble_top_k_gnns=int(problem_cfg.get("ensemble_top_k_gnns", CONFIG["ensemble_top_k_gnns"])),
        graph_storage_dtype=str(CONFIG["graph_storage_dtype"]),
        autocast_static_encode=bool(CONFIG["autocast_static_encode"]),
        pack_node_config_feat=bool(CONFIG["pack_node_config_feat"]),
        type_embed_dim=int(CONFIG["type_embed_dim"]),
        layout_value_embed_dim=int(CONFIG["layout_value_embed_dim"]),
        blobify_enabled=bool(CONFIG["blobify_enabled"]),
        blobify_keep_hops=int(CONFIG["blobify_keep_hops"]),
        blobify_frontier_split=bool(CONFIG["blobify_frontier_split"]),
        node_numeric_mean=None,
        node_numeric_std=None,
        summary_opcode_id=None,
    )

    cfg.data_dir = cfg.base / "data"
    cfg.out_dir = cfg.base / "artifacts" / cfg.out_dir_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    enabled = list(problem_cfg["enabled_gnn_experiments"])
    run_xgb = bool(problem_cfg["run_xgb_experiment"])
    run_ensemble = bool(problem_cfg["run_ensemble_search"])
    return cfg, enabled, run_xgb, run_ensemble


def build_gnn_experiments(enabled_names: list[str] | None = None):
    shared_defaults = dict(CONFIG["gnn_defaults"])
    shared_defaults.pop("default_max_train_configs_per_graph", None)
    shared_defaults.pop("default_max_valid_configs_per_graph", None)

    experiments = []
    for spec in CONFIG["gnn_experiments"]:
        merged = dict(shared_defaults)
        merged.update(spec)
        merged["chunk_batching"] = merged.get("chunk_batching", "contiguous")
        merged["amp"] = bool(CONFIG["amp"])
        experiments.append(SimpleNamespace(**merged))

    if enabled_names is None:
        return experiments

    enabled_set = set(enabled_names)
    selected = [exp for exp in experiments if exp.name in enabled_set]
    missing = sorted(enabled_set - {exp.name for exp in experiments})
    if missing:
        raise ValueError(f"Unknown GNN experiment names: {missing}")
    if not selected:
        raise ValueError("No GNN experiments selected.")
    return selected


def build_xgb_experiment():
    return SimpleNamespace(**CONFIG["xgb_experiment"])


def sanitize_dense_features(x: np.ndarray, clip_value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32, copy=False)


def encode_layout_values_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=LAYOUT_VALUE_MIN, posinf=LAYOUT_VALUE_MAX, neginf=LAYOUT_VALUE_MIN)
    x = np.rint(x).astype(np.int16, copy=False)
    x = np.clip(x, LAYOUT_VALUE_MIN, LAYOUT_VALUE_MAX)
    return (x - LAYOUT_VALUE_MIN).astype(np.int64, copy=False)


def encode_layout_values_torch(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=float(LAYOUT_VALUE_MIN), posinf=float(LAYOUT_VALUE_MAX), neginf=float(LAYOUT_VALUE_MIN))
    x = torch.round(x).clamp(min=float(LAYOUT_VALUE_MIN), max=float(LAYOUT_VALUE_MAX)).long()
    return x - LAYOUT_VALUE_MIN


def pack_layout_values_base7_np(layout_ids: np.ndarray) -> np.ndarray:
    arr = np.asarray(layout_ids)
    if arr.shape[-1] != CONFIG_FEAT_DIM:
        raise ValueError(f"Expected trailing dim {CONFIG_FEAT_DIM}, got {arr.shape}")
    flat = arr.astype(np.int64, copy=False).reshape(-1, CONFIG_LAYOUT_GROUPS, CONFIG_LAYOUT_GROUP_SIZE)
    packed = (flat * CONFIG_LAYOUT_BASE_POWERS_NP.reshape(1, 1, CONFIG_LAYOUT_GROUP_SIZE)).sum(axis=-1)
    return packed.astype(np.int32, copy=False).reshape(*arr.shape[:-1], CONFIG_LAYOUT_PACKED_DIM)


def unpack_node_config_storage_np(x: np.ndarray, output_mode: str = "layout_ids") -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape[-1] == CONFIG_LAYOUT_PACKED_DIM and np.issubdtype(arr.dtype, np.integer):
        flat = arr.astype(np.int64, copy=False).reshape(-1, CONFIG_LAYOUT_PACKED_DIM)
        out = np.empty((flat.shape[0], CONFIG_LAYOUT_GROUPS, CONFIG_LAYOUT_GROUP_SIZE), dtype=np.int64)
        tmp = flat.copy()
        for slot in range(CONFIG_LAYOUT_GROUP_SIZE):
            out[:, :, slot] = tmp % BASE7
            tmp //= BASE7
        out = out.reshape(*arr.shape[:-1], CONFIG_FEAT_DIM)
    elif arr.shape[-1] == CONFIG_FEAT_DIM:
        if np.issubdtype(arr.dtype, np.integer):
            out = arr.astype(np.int64, copy=False)
        else:
            out = encode_layout_values_np(arr).astype(np.int64, copy=False)
    else:
        raise ValueError(f"Unsupported node_config_feat storage shape: {arr.shape}")

    if output_mode == "layout_ids":
        return out.astype(np.int64, copy=False)
    if output_mode == "signed_float":
        return (out.astype(np.float32, copy=False) + float(LAYOUT_VALUE_MIN)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown output_mode: {output_mode}")


def unpack_node_config_storage_torch(x: torch.Tensor, output_mode: str = "layout_ids") -> torch.Tensor:
    if x.shape[-1] == CONFIG_LAYOUT_PACKED_DIM and not x.dtype.is_floating_point:
        orig_shape = tuple(x.shape[:-1])
        flat = x.long().reshape(-1, CONFIG_LAYOUT_PACKED_DIM)
        values = []
        tmp = flat
        for _slot in range(CONFIG_LAYOUT_GROUP_SIZE):
            values.append(torch.remainder(tmp, BASE7))
            tmp = torch.div(tmp, BASE7, rounding_mode="floor")
        out = torch.stack(values, dim=-1).reshape(*orig_shape, CONFIG_FEAT_DIM)
    elif x.shape[-1] == CONFIG_FEAT_DIM:
        out = encode_layout_values_torch(x) if x.dtype.is_floating_point else x.long()
    else:
        raise ValueError(f"Unsupported node_config_feat storage shape: {tuple(x.shape)}")

    if output_mode == "layout_ids":
        return out.long()
    if output_mode == "signed_float":
        return out.float() + float(LAYOUT_VALUE_MIN)
    raise ValueError(f"Unknown output_mode: {output_mode}")


def prepare_node_config_storage_np(node_config_feat_raw: np.ndarray, pack_node_config_feat: bool) -> np.ndarray:
    layout_ids = encode_layout_values_np(node_config_feat_raw)
    if pack_node_config_feat:
        return pack_layout_values_base7_np(layout_ids)
    return layout_ids.astype(np.uint8, copy=False)


def extract_shape_type_id_np(node_feat: np.ndarray) -> np.ndarray:
    shape_slice = np.asarray(node_feat[:, SHAPE_TYPE_START:SHAPE_TYPE_END], dtype=np.float32)
    if shape_slice.shape[1] == 0:
        return np.zeros(node_feat.shape[0], dtype=np.int64)
    return np.argmax(shape_slice, axis=1).astype(np.int64, copy=False)


def standardize_node_numeric_feat_np(node_feat_numeric: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (node_feat_numeric.astype(np.float32, copy=False) - mean.astype(np.float32, copy=False)) / std.astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def fit_node_numeric_scaler(files: list[Path], feature_clip_value: float) -> tuple[np.ndarray, np.ndarray]:
    sum_x = np.zeros(NODE_NUMERIC_DIM, dtype=np.float64)
    sum_x2 = np.zeros(NODE_NUMERIC_DIM, dtype=np.float64)
    count = 0

    for path in tqdm(files, total=len(files), desc="fit node scaler", leave=False):
        data = dict(np.load(path))
        node_feat = sanitize_dense_features(
            data["node_feat"],
            clip_value=feature_clip_value,
        )
        numeric = node_feat[:, NODE_NUMERIC_INDICES].astype(np.float64, copy=False)
        sum_x += numeric.sum(axis=0)
        sum_x2 += np.square(numeric).sum(axis=0)
        count += int(numeric.shape[0])

    if count <= 0:
        raise RuntimeError("Cannot fit node feature scaler: no nodes found in training data.")

    mean = sum_x / float(count)
    var = np.maximum(sum_x2 / float(count) - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def config_signature_bytes(cfg: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(np.asarray(cfg))
    return __import__("hashlib").blake2b(arr.tobytes(), digest_size=16).digest()


def dedupe_layout_configs(node_config_feat: np.ndarray, runtimes: np.ndarray | None, strategy: str):
    if strategy == "none" or node_config_feat.shape[0] == 0:
        dup = np.ones(node_config_feat.shape[0], dtype=np.float32)
        return node_config_feat, runtimes, dup

    groups: dict[bytes, list[int]] = defaultdict(list)
    for i in range(node_config_feat.shape[0]):
        groups[config_signature_bytes(node_config_feat[i])].append(i)

    kept_cfg = []
    kept_runtime = [] if runtimes is not None else None
    kept_dup = []

    for idxs in groups.values():
        kept_cfg.append(node_config_feat[idxs[0]])
        kept_dup.append(float(len(idxs)))
        if runtimes is not None:
            vals = runtimes[idxs]
            if strategy == "mean":
                kept_runtime.append(float(np.mean(vals)))
            elif strategy == "median":
                kept_runtime.append(float(np.median(vals)))
            elif strategy == "min":
                kept_runtime.append(float(np.min(vals)))
            else:
                raise ValueError(f"Unknown dedup strategy: {strategy}")

    kept_cfg = np.stack(kept_cfg, axis=0)
    kept_dup = np.asarray(kept_dup, dtype=np.float32)
    if kept_runtime is not None:
        kept_runtime = np.asarray(kept_runtime, dtype=np.float32)
    return kept_cfg, kept_runtime, kept_dup


def subsample_configs(
    node_config_feat: np.ndarray,
    runtimes: np.ndarray | None,
    duplicate_count: np.ndarray | None,
    max_configs: int | None,
    split: str,
    seed: int,
    train_subsample_mode: str = "random",
):
    count = node_config_feat.shape[0]
    if max_configs is None or count <= max_configs:
        return node_config_feat, runtimes, duplicate_count

    idx = np.arange(count)
    rng = np.random.default_rng(seed + count)

    if split == "train":
        if train_subsample_mode == "contiguous_random_start":
            max_start = count - max_configs
            start = 0 if max_start <= 0 else int(rng.integers(0, max_start + 1))
            idx = np.arange(start, start + max_configs, dtype=np.int64)
        elif train_subsample_mode == "random":
            idx = np.sort(rng.choice(idx, size=max_configs, replace=False))
        else:
            raise ValueError(f"Unknown train_subsample_mode: {train_subsample_mode}")
    else:
        idx = np.linspace(0, count - 1, num=max_configs, dtype=np.int64)
        idx = np.unique(idx)
        if idx.shape[0] < max_configs:
            missing = np.setdiff1d(np.arange(count), idx, assume_unique=False)
            need = max_configs - idx.shape[0]
            if need > 0:
                idx = np.sort(np.concatenate([idx, missing[:need]]))

    node_config_feat = node_config_feat[idx]
    if runtimes is not None:
        runtimes = runtimes[idx]
    if duplicate_count is not None:
        duplicate_count = duplicate_count[idx]
    return node_config_feat, runtimes, duplicate_count


def append_blob_extra_zeros_np(node_numeric_feat: np.ndarray) -> np.ndarray:
    out = np.zeros((node_numeric_feat.shape[0], NODE_TOTAL_NUMERIC_DIM), dtype=np.float32)
    out[:, :NODE_NUMERIC_DIM] = node_numeric_feat.astype(np.float32, copy=False)
    return out


def infer_blob_node_weights_np(node_feat_raw: np.ndarray) -> np.ndarray:
    node_feat_raw = np.asarray(node_feat_raw, dtype=np.float32)
    def col(name: str) -> np.ndarray:
        return np.abs(node_feat_raw[:, BLOB_RUNTIME_WEIGHT_SOURCE_INDICES[name]])
    shape_product = col("shape_product")
    window_product = col("window_size_product")
    stride_product = col("window_stride_product")
    dilation_product = col("window_dilation_product") + col("base_dilation_product")
    slice_like_product = col("slice_limit_product") + col("dynamic_slice_product")
    padding_product = col("padding_low_product") + col("padding_high_product")
    group_count = col("feature_group_count") + col("batch_group_count")
    weight = np.ones(node_feat_raw.shape[0], dtype=np.float32)
    weight += 0.15 * np.log1p(shape_product)
    weight += 0.10 * np.log1p(window_product)
    weight += 0.05 * np.log1p(stride_product)
    weight += 0.05 * np.log1p(dilation_product)
    weight += 0.03 * np.log1p(slice_like_product + padding_product)
    weight += 0.08 * np.log1p(group_count)
    weight += 0.25 * (window_product > 0).astype(np.float32)
    return np.nan_to_num(weight, nan=1.0, posinf=1e3, neginf=1.0).astype(np.float32, copy=False)


def compute_keep_distances_undirected_np(num_nodes: int, edge_index: np.ndarray, source_nodes: np.ndarray, max_hops: int) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
    e = np.asarray(edge_index, dtype=np.int64)
    if e.shape[0] == 2:
        src, dst = e[0], e[1]
    else:
        src, dst = e[:, 0], e[:, 1]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.int8)
    undirected = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    undirected.sum_duplicates()
    undirected.data[:] = 1

    dist = np.full(num_nodes, max_hops + 100, dtype=np.int16)
    source_nodes = np.unique(np.asarray(source_nodes, dtype=np.int64))
    source_nodes = source_nodes[(source_nodes >= 0) & (source_nodes < num_nodes)]
    if source_nodes.size == 0:
        return dist, undirected, src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False)

    frontier = np.zeros(num_nodes, dtype=bool)
    frontier[source_nodes] = True
    seen = frontier.copy()
    dist[source_nodes] = 0

    for step in range(1, max_hops + 1):
        nbr = np.asarray((undirected @ frontier.astype(np.int8)) > 0).ravel()
        frontier = nbr & (~seen)
        dist[frontier] = step
        seen |= frontier

    return dist, undirected, src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False)


def build_blob_summary_numeric_np(
    member_nodes: np.ndarray,
    node_numeric_feat_aug: np.ndarray,
    node_weights: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    kept_touch_nodes: np.ndarray,
    bfs_depth: np.ndarray,
    bfs_weight: np.ndarray,
) -> np.ndarray:
    summary = np.zeros(NODE_TOTAL_NUMERIC_DIM, dtype=np.float32)
    if member_nodes.size == 0:
        return summary

    summary[:NODE_NUMERIC_DIM] = node_numeric_feat_aug[member_nodes, :NODE_NUMERIC_DIM].mean(axis=0).astype(np.float32, copy=False)
    member_mask = np.zeros(node_numeric_feat_aug.shape[0], dtype=bool)
    member_mask[member_nodes] = True
    internal_edges = member_mask[src] & member_mask[dst]
    num_internal_edges = int(np.count_nonzero(internal_edges))
    total_weight = float(node_weights[member_nodes].sum())
    max_weight = float(node_weights[member_nodes].max(initial=0.0))
    max_depth = int(bfs_depth[member_nodes].max(initial=0))
    max_path_weight = float(bfs_weight[member_nodes].max(initial=0.0))
    extras = np.asarray([
        math.log1p(int(member_nodes.size)),
        math.log1p(num_internal_edges),
        math.log1p(int(np.unique(kept_touch_nodes).size)),
        math.log1p(max_depth),
        math.log1p(max_path_weight),
        math.log1p(total_weight),
        math.log1p(max_weight),
    ], dtype=np.float32)
    summary[NODE_NUMERIC_DIM:] = extras
    return summary


def blobify_graph_numpy(
    node_numeric_feat: np.ndarray,
    node_shape_type: np.ndarray,
    node_layout_feat: np.ndarray,
    node_opcode: np.ndarray,
    node_config_ids: np.ndarray,
    edge_index: np.ndarray,
    node_feat_raw: np.ndarray,
    keep_hops: int,
    frontier_split: bool,
    summary_opcode_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = int(node_numeric_feat.shape[0])
    if num_nodes == 0:
        return (
            append_blob_extra_zeros_np(node_numeric_feat),
            node_shape_type,
            node_layout_feat,
            node_opcode,
            node_config_ids,
            edge_index.astype(np.int64, copy=False),
        )

    cfg_sources = np.unique(np.asarray(node_config_ids, dtype=np.int64))
    dist, undirected, src, dst = compute_keep_distances_undirected_np(
        num_nodes=num_nodes,
        edge_index=edge_index,
        source_nodes=cfg_sources,
        max_hops=keep_hops,
    )
    keep_mask = dist <= keep_hops
    removed_mask = ~keep_mask

    node_numeric_feat_aug = append_blob_extra_zeros_np(node_numeric_feat)
    node_weights = infer_blob_node_weights_np(node_feat_raw)

    if not np.any(removed_mask):
        return (
            node_numeric_feat_aug,
            node_shape_type.astype(np.int64, copy=False),
            node_layout_feat.astype(np.int64, copy=False),
            node_opcode.astype(np.int64, copy=False),
            node_config_ids.astype(np.int64, copy=False),
            edge_index.astype(np.int64, copy=False),
        )

    removed_indices = np.flatnonzero(removed_mask)
    removed_sub = undirected[removed_mask][:, removed_mask]
    num_removed_cc, removed_cc_labels = connected_components(removed_sub, directed=False, return_labels=True)
    removed_cc_global = np.full(num_nodes, -1, dtype=np.int32)
    removed_cc_global[removed_indices] = removed_cc_labels.astype(np.int32, copy=False)

    d_frontier_mask = dist == keep_hops
    d_frontier_indices = np.flatnonzero(d_frontier_mask)
    frontier_cc_global = np.full(num_nodes, -1, dtype=np.int32)
    if d_frontier_indices.size > 0:
        frontier_sub = undirected[d_frontier_mask][:, d_frontier_mask]
        _num_frontier_cc, frontier_cc_labels = connected_components(frontier_sub, directed=False, return_labels=True)
        frontier_cc_global[d_frontier_indices] = frontier_cc_labels.astype(np.int32, copy=False)

    indptr = undirected.indptr
    indices = undirected.indices

    zone_assignment = np.full(num_nodes, -1, dtype=np.int32)
    bfs_depth = np.zeros(num_nodes, dtype=np.int16)
    bfs_weight = np.zeros(num_nodes, dtype=np.float32)

    for comp_id in range(int(num_removed_cc)):
        comp_nodes = removed_indices[removed_cc_labels == comp_id]
        if comp_nodes.size == 0:
            continue
        queue = deque()
        fallback_zone = 0
        for node in comp_nodes:
            touched = set()
            for nbr in indices[indptr[node]:indptr[node + 1]]:
                z = int(frontier_cc_global[nbr])
                if z >= 0:
                    touched.add(z)
            if touched:
                zone = min(touched)
                if zone_assignment[node] == -1:
                    zone_assignment[node] = zone
                    bfs_depth[node] = 1
                    bfs_weight[node] = float(node_weights[node])
                    queue.append(int(node))
                fallback_zone = zone
        if not queue:
            seed = int(comp_nodes[0])
            zone_assignment[seed] = fallback_zone
            bfs_depth[seed] = 1
            bfs_weight[seed] = float(node_weights[seed])
            queue.append(seed)

        while queue:
            u = queue.popleft()
            for v in indices[indptr[u]:indptr[u + 1]]:
                if removed_cc_global[v] != comp_id:
                    continue
                if zone_assignment[v] == -1:
                    zone_assignment[v] = zone_assignment[u]
                    bfs_depth[v] = bfs_depth[u] + 1
                    bfs_weight[v] = bfs_weight[u] + float(node_weights[v])
                    queue.append(int(v))

        unassigned = comp_nodes[zone_assignment[comp_nodes] == -1]
        if unassigned.size > 0:
            zone_assignment[unassigned] = fallback_zone
            bfs_depth[unassigned] = 1
            bfs_weight[unassigned] = node_weights[unassigned]

    if not frontier_split:
        zone_assignment[removed_mask] = removed_cc_global[removed_mask]

    partition_keys = np.stack([removed_cc_global[removed_mask], zone_assignment[removed_mask]], axis=1)
    key_to_part: dict[tuple[int, int], int] = {}
    part_nodes: list[list[int]] = []
    part_touch_kept: list[set[int]] = []

    removed_global = np.flatnonzero(removed_mask)
    for node, key_arr in zip(removed_global, partition_keys, strict=False):
        key = (int(key_arr[0]), int(key_arr[1]))
        if key not in key_to_part:
            key_to_part[key] = len(part_nodes)
            part_nodes.append([])
            part_touch_kept.append(set())
        pid = key_to_part[key]
        part_nodes[pid].append(int(node))
        for nbr in indices[indptr[node]:indptr[node + 1]]:
            if keep_mask[nbr]:
                part_touch_kept[pid].add(int(nbr))

    kept_old = np.flatnonzero(keep_mask)
    old_to_new = np.full(num_nodes, -1, dtype=np.int64)
    old_to_new[kept_old] = np.arange(kept_old.size, dtype=np.int64)

    summary_numeric = []
    summary_shape_type = []
    summary_layout_feat = []
    summary_opcode = []
    for pid, members in enumerate(part_nodes):
        member_arr = np.asarray(members, dtype=np.int64)
        kept_touch = np.asarray(sorted(part_touch_kept[pid]), dtype=np.int64)
        summary_numeric.append(
            build_blob_summary_numeric_np(
                member_nodes=member_arr,
                node_numeric_feat_aug=node_numeric_feat_aug,
                node_weights=node_weights,
                src=src,
                dst=dst,
                kept_touch_nodes=kept_touch,
                bfs_depth=bfs_depth,
                bfs_weight=bfs_weight,
            )
        )
        summary_shape_type.append(BLOB_SUMMARY_SHAPE_TYPE_ID)
        summary_layout_feat.append(np.zeros(NODE_LAYOUT_SLOTS, dtype=np.int64))
        summary_opcode.append(int(summary_opcode_id))

    if summary_numeric:
        summary_numeric_arr = np.stack(summary_numeric, axis=0).astype(np.float32, copy=False)
        summary_shape_arr = np.asarray(summary_shape_type, dtype=np.int64)
        summary_layout_arr = np.stack(summary_layout_feat, axis=0).astype(np.int64, copy=False)
        summary_opcode_arr = np.asarray(summary_opcode, dtype=np.int64)
    else:
        summary_numeric_arr = np.zeros((0, NODE_TOTAL_NUMERIC_DIM), dtype=np.float32)
        summary_shape_arr = np.zeros((0,), dtype=np.int64)
        summary_layout_arr = np.zeros((0, NODE_LAYOUT_SLOTS), dtype=np.int64)
        summary_opcode_arr = np.zeros((0,), dtype=np.int64)

    summary_start = kept_old.size
    for pid in range(len(part_nodes)):
        part_nodes_arr = np.asarray(part_nodes[pid], dtype=np.int64)
        if part_nodes_arr.size > 0:
            old_to_new[part_nodes_arr] = summary_start + pid

    new_numeric = np.concatenate([node_numeric_feat_aug[kept_old], summary_numeric_arr], axis=0)
    new_shape = np.concatenate([node_shape_type[kept_old], summary_shape_arr], axis=0)
    new_layout = np.concatenate([node_layout_feat[kept_old], summary_layout_arr], axis=0)
    new_opcode = np.concatenate([node_opcode[kept_old], summary_opcode_arr], axis=0)

    new_src = []
    new_dst = []
    for u, v in zip(src.tolist(), dst.tolist()):
        nu = int(old_to_new[u])
        nv = int(old_to_new[v])
        if nu < 0 or nv < 0 or nu == nv:
            continue
        new_src.append(nu)
        new_dst.append(nv)

    if new_src:
        edge_arr = np.stack([np.asarray(new_src, dtype=np.int64), np.asarray(new_dst, dtype=np.int64)], axis=0)
        edge_pairs = np.unique(edge_arr.T, axis=0)
        new_edge_index = edge_pairs.T.astype(np.int64, copy=False)
    else:
        new_edge_index = np.zeros((2, 0), dtype=np.int64)

    new_node_config_ids = old_to_new[np.asarray(node_config_ids, dtype=np.int64)].astype(np.int64, copy=False)

    return (
        new_numeric.astype(np.float32, copy=False),
        new_shape.astype(np.int64, copy=False),
        new_layout.astype(np.int64, copy=False),
        new_opcode.astype(np.int64, copy=False),
        new_node_config_ids,
        new_edge_index,
    )


def build_sage_edges_numpy(num_nodes: int, edge_index: np.ndarray):
    e = np.asarray(edge_index, dtype=np.int64)
    if e.shape[0] == 2:
        src, dst = e[0], e[1]
    else:
        src, dst = e[:, 0], e[:, 1]

    row = np.concatenate([src, dst, np.arange(num_nodes)])
    col = np.concatenate([dst, src, np.arange(num_nodes)])
    edge = np.stack([row, col], axis=0).astype(np.int64)
    deg = np.bincount(col, minlength=num_nodes).astype(np.float32)
    deg = np.maximum(deg, 1.0)
    return edge, deg


def graph_payload_to_example(payload: dict) -> GraphExample:
    return GraphExample(
        graph_id=payload["graph_id"],
        num_configs=int(payload["num_configs"]),
        node_numeric_feat=torch.as_tensor(payload["node_numeric_feat"]),
        node_shape_type=torch.as_tensor(payload["node_shape_type"], dtype=torch.long),
        node_layout_feat=torch.as_tensor(payload["node_layout_feat"], dtype=torch.long),
        node_opcode=torch.as_tensor(payload["node_opcode"], dtype=torch.long),
        node_config_ids=torch.as_tensor(payload["node_config_ids"], dtype=torch.long),
        node_config_feat=torch.as_tensor(payload["node_config_feat"]),
        sage_edge_index=torch.as_tensor(payload["sage_edge_index"], dtype=torch.long),
        sage_deg=torch.as_tensor(payload["sage_deg"]),
        runtimes=None if payload["runtimes"] is None else torch.as_tensor(payload["runtimes"], dtype=torch.float32),
        duplicate_count=None if payload["duplicate_count"] is None else torch.as_tensor(payload["duplicate_count"], dtype=torch.float32),
    )


def preprocess_graph_payload(
    npz_path: str,
    split: str,
    max_configs_per_graph: int | None,
    dedup_strategy: str,
    feature_clip_value: float,
    node_numeric_mean: np.ndarray,
    node_numeric_std: np.ndarray,
    pack_node_config_feat: bool,
    blobify_enabled: bool,
    blobify_keep_hops: int,
    blobify_frontier_split: bool,
    summary_opcode_id: int,
    seed: int,
    train_subsample_mode: str,
    graph_storage_dtype: str,
):
    path = Path(npz_path)
    data = dict(np.load(path))
    node_feat_raw = sanitize_dense_features(
        data["node_feat"],
        clip_value=feature_clip_value,
    )
    node_numeric_feat = standardize_node_numeric_feat_np(
        node_feat_raw[:, NODE_NUMERIC_INDICES],
        mean=node_numeric_mean,
        std=node_numeric_std,
    )
    node_shape_type = extract_shape_type_id_np(node_feat_raw)
    node_layout_feat = encode_layout_values_np(node_feat_raw[:, NODE_LAYOUT_START:NODE_LAYOUT_END])

    node_opcode = data["node_opcode"].astype(np.int64)
    edge_index = data["edge_index"].astype(np.int64)
    node_config_ids = data["node_config_ids"].astype(np.int64)
    if blobify_enabled:
        (
            node_numeric_feat,
            node_shape_type,
            node_layout_feat,
            node_opcode,
            node_config_ids,
            edge_index,
        ) = blobify_graph_numpy(
            node_numeric_feat=node_numeric_feat,
            node_shape_type=node_shape_type,
            node_layout_feat=node_layout_feat,
            node_opcode=node_opcode,
            node_config_ids=node_config_ids,
            edge_index=edge_index,
            node_feat_raw=node_feat_raw,
            keep_hops=blobify_keep_hops,
            frontier_split=blobify_frontier_split,
            summary_opcode_id=summary_opcode_id,
        )
    else:
        node_numeric_feat = append_blob_extra_zeros_np(node_numeric_feat)

    node_config_feat = prepare_node_config_storage_np(
        data["node_config_feat"],
        pack_node_config_feat=pack_node_config_feat,
    )

    runtimes = data.get("config_runtime")
    if runtimes is not None:
        runtimes = np.nan_to_num(runtimes.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    node_config_feat, runtimes, duplicate_count = dedupe_layout_configs(
        node_config_feat=node_config_feat,
        runtimes=runtimes,
        strategy=dedup_strategy,
    )
    node_config_feat, runtimes, duplicate_count = subsample_configs(
        node_config_feat=node_config_feat,
        runtimes=runtimes,
        duplicate_count=duplicate_count,
        max_configs=max_configs_per_graph,
        split=split,
        seed=seed,
        train_subsample_mode=train_subsample_mode,
    )

    sage_edge_index, sage_deg = build_sage_edges_numpy(int(node_numeric_feat.shape[0]), edge_index)
    graph_storage_dtype = get_graph_storage_numpy_dtype(graph_storage_dtype)

    if graph_storage_dtype == np.float16:
        finfo = np.finfo(np.float16)
        node_numeric_feat = np.clip(node_numeric_feat, finfo.min, finfo.max)

    return {
        "graph_id": path.stem,
        "num_configs": int(node_config_feat.shape[0]),
        "node_numeric_feat": node_numeric_feat.astype(graph_storage_dtype, copy=False),
        "node_shape_type": node_shape_type.astype(np.int64, copy=False),
        "node_layout_feat": node_layout_feat.astype(np.int64, copy=False),
        "node_opcode": node_opcode.astype(np.int64, copy=False),
        "node_config_ids": node_config_ids.astype(np.int64, copy=False),
        "node_config_feat": np.ascontiguousarray(node_config_feat),
        "sage_edge_index": sage_edge_index.astype(np.int64, copy=False),
        "sage_deg": sage_deg.astype(np.float32, copy=False),
        "runtimes": None if runtimes is None else runtimes.astype(np.float32, copy=False),
        "duplicate_count": None if duplicate_count is None else duplicate_count.astype(np.float32, copy=False),
    }


def make_worker_args(cfg, npz_path: Path, split: str, max_configs_per_graph: int | None, seed: int):
    return (
        str(npz_path),
        split,
        max_configs_per_graph,
        cfg.dedup_strategy,
        cfg.feature_clip_value,
        cfg.node_numeric_mean,
        cfg.node_numeric_std,
        cfg.pack_node_config_feat,
        cfg.blobify_enabled,
        cfg.blobify_keep_hops,
        cfg.blobify_frontier_split,
        cfg.summary_opcode_id,
        seed,
        cfg.train_subsample_mode,
        cfg.graph_storage_dtype,
    )


def _maybe_log_prefetch_status(cfg, split: str, consumed: int, total: int, window: int, pending: dict[int, object], next_future_ready: bool):
    if not getattr(cfg, "log_prefetch_status", False):
        return
    interval = max(1, int(getattr(cfg, "prefetch_status_every_n_graphs", 10)))
    ready_buffered = sum(1 for fut in pending.values() if fut.done())
    in_flight = len(pending) - ready_buffered
    should_log = (
        consumed <= 1
        or consumed >= total
        or (consumed % interval) == 0
        or ready_buffered == 0
        or ready_buffered >= window
        or not next_future_ready
    )
    if not should_log:
        return
    current_plus_buffered = ready_buffered + 1
    print(
        f"[prefetch {split}] consumed={consumed}/{total} current_plus_buffered={current_plus_buffered} "
        f"buffered_ready={ready_buffered}/{window} in_flight={in_flight} next_ready={int(next_future_ready)}"
    )


def preprocess_graphs_parallel(
    cfg,
    files: list[Path],
    split: str,
    max_configs_per_graph: int | None,
    base_seed: int,
    executor: ProcessPoolExecutor | None = None,
    show_progress: bool = True,
):
    own_executor = executor is None
    if own_executor:
        executor = ProcessPoolExecutor(max_workers=cfg.num_preprocess_workers, mp_context=get_mp_context())

    try:
        total = len(files)
        pbar = tqdm(total=total, desc=f"preprocess {split}", leave=False) if show_progress else None
        next_submit = 0
        next_yield = 0
        window = max(1, cfg.prefetch_graphs)
        pending: dict[int, object] = {}

        while next_submit < total and len(pending) < window:
            seed = base_seed + next_submit
            pending[next_submit] = executor.submit(
                preprocess_graph_payload,
                *make_worker_args(cfg, files[next_submit], split, max_configs_per_graph, seed),
            )
            next_submit += 1

        while next_yield < total:
            future = pending.pop(next_yield)
            next_future_ready = future.done()
            _maybe_log_prefetch_status(
                cfg=cfg,
                split=split,
                consumed=next_yield + 1,
                total=total,
                window=window,
                pending=pending,
                next_future_ready=next_future_ready,
            )
            payload = future.result()
            if pbar is not None:
                pbar.update(1)
            yield graph_payload_to_example(payload)
            next_yield += 1

            while next_submit < total and len(pending) < window:
                seed = base_seed + next_submit
                pending[next_submit] = executor.submit(
                    preprocess_graph_payload,
                    *make_worker_args(cfg, files[next_submit], split, max_configs_per_graph, seed),
                )
                next_submit += 1
        if pbar is not None:
            pbar.close()
    finally:
        if own_executor:
            executor.shutdown(wait=True)


def find_layout_split_dir(cfg, split: str) -> Path:
    candidates = [
        cfg.data_dir / "npz_all" / "npz" / "layout" / cfg.source / cfg.search / split,
        cfg.data_dir / "npz" / "layout" / cfg.source / cfg.search / split,
        cfg.data_dir / "layout" / cfg.source / cfg.search / split,
        cfg.base / "data" / "npz_all" / "npz" / "layout" / cfg.source / cfg.search / split,
        cfg.base / "data" / "npz" / "layout" / cfg.source / cfg.search / split,
        cfg.base / "data" / "layout" / cfg.source / cfg.search / split,
    ]
    for path in candidates:
        if path.exists():
            return path

    pattern_parts = ["layout", cfg.source, cfg.search, split]
    for root in [cfg.data_dir, cfg.base / "data", cfg.base]:
        if not root.exists():
            continue
        hits = []
        for path in root.rglob("*.npz"):
            parts = [part.lower() for part in path.parts]
            if all(token in parts for token in pattern_parts):
                hits.append(path.parent)
        if hits:
            return sorted(set(hits))[0]

    raise FileNotFoundError(
        f"Could not find layout split dir for split={split!r}, source={cfg.source!r}, search={cfg.search!r}."
    )


def list_npz_files(split_dir: Path, max_graphs: int | None = None):
    files = sorted(split_dir.glob("*.npz"))
    return files if max_graphs is None else files[:max_graphs]


def scan_max_opcode(files: list[Path]) -> int:
    max_opcode_seen = -1
    for path in tqdm(files, total=len(files), desc="scan opcodes", leave=False):
        data = np.load(path)
        node_opcode = np.asarray(data["node_opcode"], dtype=np.int64)
        if node_opcode.size:
            max_opcode_seen = max(max_opcode_seen, int(node_opcode.max()))
    return max_opcode_seen


def build_device_graph_view(graph: GraphExample) -> DeviceGraphView:
    return DeviceGraphView(
        node_numeric_feat=graph.node_numeric_feat.to(DEVICE, non_blocking=True),
        node_shape_type=graph.node_shape_type.to(DEVICE, non_blocking=True),
        node_layout_feat=graph.node_layout_feat.to(DEVICE, non_blocking=True),
        node_opcode=graph.node_opcode.to(DEVICE, non_blocking=True),
        node_config_ids=graph.node_config_ids.to(DEVICE, non_blocking=True),
        sage_edge_index=graph.sage_edge_index.to(DEVICE, non_blocking=True),
        sage_deg=graph.sage_deg.to(DEVICE, non_blocking=True),
    )


def topk_slowdown(runtime: np.ndarray, pred_score: np.ndarray, k: int):
    runtime = np.asarray(runtime, dtype=np.float32)
    pred_score = np.asarray(pred_score, dtype=np.float32)
    order = np.argsort(pred_score)[: min(k, len(pred_score))]
    return float(runtime[order].min() / runtime.min())


def kendall_tau_for_graph(runtime: np.ndarray, pred_score: np.ndarray):
    runtime = np.asarray(runtime, dtype=np.float64)
    pred_score = np.asarray(pred_score, dtype=np.float64)
    tau = kendalltau(runtime, pred_score, nan_policy="omit").correlation
    return float(0.0 if tau is None or np.isnan(tau) else tau)


def evaluate_predictions(
    graph_items: list[GraphExample],
    preds_by_graph: dict[str, np.ndarray],
    ks=(1, 5, 10, 100),
):
    rows = []
    for graph in graph_items:
        if graph.runtimes is None:
            continue
        pred = preds_by_graph[graph.graph_id]
        runtime = graph.runtimes.cpu().numpy()
        row = {"graph_id": graph.graph_id}
        for k in ks:
            row[f"top{k}_slowdown"] = topk_slowdown(runtime, pred, k)
        row["kendall_tau"] = kendall_tau_for_graph(runtime, pred)
        rows.append(row)
    per_graph = pd.DataFrame(rows)
    summary = per_graph.mean(numeric_only=True).to_dict() if len(per_graph) else {}
    return per_graph, summary


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0, activation=nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BatchedSAGEBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor):
        src, dst = edge_index[0], edge_index[1]
        msg = h[:, src, :]
        agg = h.new_zeros(h.shape)
        agg.index_add_(1, dst, msg)
        agg = agg / deg.view(1, -1, 1).clamp(min=1.0)
        z = self.lin_self(h) + self.lin_neigh(agg)
        z = F.gelu(z)
        z = self.dropout(z)
        return self.norm(h + z)


class LayoutGraphAwareModel(nn.Module):
    def __init__(self, num_opcodes: int, exp, node_numeric_dim: int = NODE_TOTAL_NUMERIC_DIM):
        super().__init__()
        self.exp = exp
        self.hidden_dim = exp.hidden_dim
        self.use_graph_global_pool = bool(getattr(exp, "use_graph_global_pool", False))

        self.op_emb = nn.Embedding(num_opcodes + 1, exp.op_embed_dim)
        self.shape_type_emb = nn.Embedding(NUM_SHAPE_TYPES, exp.type_embed_dim)
        self.layout_value_emb = nn.Embedding(LAYOUT_VALUE_VOCAB_SIZE, exp.layout_value_embed_dim)

        static_in_dim = node_numeric_dim + exp.op_embed_dim + exp.type_embed_dim + NODE_LAYOUT_SLOTS * exp.layout_value_embed_dim
        cfg_in_dim = CONFIG_FEAT_DIM * exp.layout_value_embed_dim

        self.static_proj = nn.Linear(static_in_dim, exp.hidden_dim)
        self.cfg_proj = MLP([cfg_in_dim, exp.hidden_dim, exp.hidden_dim], dropout=exp.dropout)
        self.input_norm = nn.LayerNorm(exp.hidden_dim)
        self.gnn_layers = nn.ModuleList([BatchedSAGEBlock(exp.hidden_dim, dropout=exp.dropout) for _ in range(exp.num_gnn_layers)])
        self.cfg_attn = nn.Linear(exp.hidden_dim, 1)
        head_in_dim = exp.hidden_dim * (4 if self.use_graph_global_pool else 2)
        self.head = MLP([head_in_dim, exp.hidden_dim * 2, 1], dropout=exp.dropout)

    def encode_static_nodes(self, view: DeviceGraphView):
        op_e = self.op_emb(view.node_opcode)
        shape_e = self.shape_type_emb(view.node_shape_type)
        layout_e = self.layout_value_emb(view.node_layout_feat).reshape(view.node_layout_feat.shape[0], -1)
        static_input = torch.cat([view.node_numeric_feat.float(), op_e, shape_e, layout_e], dim=-1)
        return F.gelu(self.static_proj(static_input))

    def _inject_cfg_hidden(self, view: DeviceGraphView, static_h: torch.Tensor, cfg_batch: torch.Tensor):
        batch_size = cfg_batch.shape[0]
        cfg_layout_ids = unpack_node_config_storage_torch(cfg_batch, output_mode="layout_ids")
        if cfg_layout_ids.ndim != 3:
            raise ValueError(
                f"Expected cfg_batch to decode to [batch, num_config_nodes, {CONFIG_FEAT_DIM}], "
                f"got shape {tuple(cfg_layout_ids.shape)}"
            )
        num_config_nodes = cfg_layout_ids.shape[1]
        if int(view.node_config_ids.numel()) != int(num_config_nodes):
            raise ValueError(
                f"Mismatch between decoded config-node features ({num_config_nodes}) and "
                f"node_config_ids ({int(view.node_config_ids.numel())})"
            )
        cfg_layout_e = self.layout_value_emb(cfg_layout_ids).reshape(batch_size, num_config_nodes, -1)
        cfg_node_h = self.cfg_proj(cfg_layout_e)
        h = static_h.unsqueeze(0).expand(batch_size, -1, -1).clone()
        h[:, view.node_config_ids, :] = h[:, view.node_config_ids, :] + cfg_node_h
        return self.input_norm(h)

    def score_configs_batch(self, view: DeviceGraphView, static_h: torch.Tensor, cfg_batch: torch.Tensor):
        cfg_batch = cfg_batch.to(DEVICE, non_blocking=True)
        h = self._inject_cfg_hidden(view, static_h, cfg_batch)

        for layer in self.gnn_layers:
            h = layer(h, view.sage_edge_index, view.sage_deg)

        hc = h[:, view.node_config_ids, :]
        attn = torch.softmax(self.cfg_attn(hc).squeeze(-1), dim=1)
        cfg_attn_pool = torch.sum(attn.unsqueeze(-1) * hc, dim=1)
        cfg_mean_pool = hc.mean(dim=1)
        if self.use_graph_global_pool:
            graph_mean = h.mean(dim=1)
            graph_max = h.max(dim=1).values
            head_input = torch.cat([cfg_attn_pool, cfg_mean_pool, graph_mean, graph_max], dim=-1)
        else:
            head_input = torch.cat([cfg_attn_pool, cfg_mean_pool], dim=-1)
        return self.head(head_input).squeeze(-1)

    def score_graph(self, graph: GraphExample, cfg_batch_size: int = 8):
        self.eval()
        view = build_device_graph_view(graph)
        with torch.inference_mode():
            with autocast_context(getattr(self.exp, "amp", False)):
                static_h = self.encode_static_nodes(view)
            scores = []
            cfg = graph.node_config_feat
            for i in range(0, graph.num_configs, cfg_batch_size):
                batch = cfg[i:i + cfg_batch_size]
                with autocast_context(getattr(self.exp, "amp", False)):
                    scores.append(self.score_configs_batch(view, static_h, batch).float().cpu())
            return torch.cat(scores, dim=0).numpy()


def runtime_targets(runtimes: torch.Tensor) -> torch.Tensor:
    runtimes = runtimes.float()
    min_rt = torch.clamp(runtimes.min(), min=1.0)
    return torch.log(runtimes / min_rt)


def sample_pair_indices(target: torch.Tensor, num_pairs: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(target.shape[0])
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=target.device)
        return empty, empty

    if mode == "random":
        max_pairs = n * (n - 1) // 2
        if num_pairs is None or max_pairs <= num_pairs:
            return torch.triu_indices(n, n, 1, device=target.device)
        ii = torch.randint(0, n, (num_pairs,), device=target.device)
        jj = torch.randint(0, n, (num_pairs,), device=target.device)
        mask = ii != jj
        return ii[mask], jj[mask]

    order = torch.argsort(target)
    pairs = []
    if mode in {"adjacent", "topk_mix"}:
        for stride in (1, 2):
            if n > stride:
                pairs.append(torch.stack([order[:-stride], order[stride:]], dim=1))
    if mode == "topk_mix":
        top = order[: max(2, n // 4)]
        rest = order[max(2, n // 8):]
        if len(rest) > 0:
            draw = min(len(top) * 3, len(rest))
            rest_sel = rest[torch.randperm(len(rest), device=target.device)[:draw]]
            top_rep = top.repeat_interleave(max(1, math.ceil(draw / max(len(top), 1))))[:draw]
            pairs.append(torch.stack([top_rep, rest_sel], dim=1))
    if not pairs:
        return sample_pair_indices(target, num_pairs=num_pairs, mode="random")

    all_pairs = torch.cat(pairs, dim=0)
    if all_pairs.shape[0] > num_pairs:
        idx = torch.randperm(all_pairs.shape[0], device=target.device)[:num_pairs]
        all_pairs = all_pairs[idx]
    ii, jj = all_pairs[:, 0], all_pairs[:, 1]
    mask = ii != jj
    return ii[mask], jj[mask]


def sampled_pairwise_loss(pred: torch.Tensor, target: torch.Tensor, num_pairs: int, mode: str):
    ii, jj = sample_pair_indices(target, num_pairs=num_pairs, mode=mode)
    if ii.numel() == 0:
        return pred.sum() * 0.0
    better_i = (target[ii] < target[jj]).float()
    logits = pred[jj] - pred[ii]
    return F.binary_cross_entropy_with_logits(logits, better_i)


def build_config_batch_indices(
    num_configs: int,
    batch_size: int,
    batching: str = "contiguous",
    epoch_num: int = 1,
    shift_stride: int | None = None,
) -> list[torch.Tensor]:
    if num_configs <= 0:
        return []

    batch_size = max(1, int(batch_size))
    if batching == "contiguous":
        return [
            torch.arange(start, min(start + batch_size, num_configs), dtype=torch.long)
            for start in range(0, num_configs, batch_size)
        ]
    if batching == "shifted_contiguous":
        shift_stride = max(1, int(shift_stride or max(1, batch_size // 2)))
        shift = ((epoch_num - 1) * shift_stride) % max(num_configs, 1)
        order = torch.roll(torch.arange(num_configs, dtype=torch.long), shifts=-shift)
        return [order[start:start + batch_size] for start in range(0, num_configs, batch_size)]
    if batching == "strided":
        num_batches = max(1, math.ceil(num_configs / batch_size))
        return [
            idx
            for idx in (torch.arange(offset, num_configs, num_batches, dtype=torch.long) for offset in range(num_batches))
            if idx.numel() > 0
        ]
    raise ValueError(f"Unknown chunk batching mode: {batching}")


def train_one_epoch(
    cfg,
    exp,
    model: LayoutGraphAwareModel,
    train_files: list[Path],
    optimizer,
    scaler,
    epoch_num: int,
    executor: ProcessPoolExecutor,
):
    model.train()
    losses = []
    skipped = 0

    graph_iter = preprocess_graphs_parallel(
        cfg=cfg,
        files=train_files,
        split="train",
        max_configs_per_graph=cfg.max_train_configs_per_graph,
        base_seed=cfg.seed + epoch_num * 1_000_003,
        executor=executor,
        show_progress=False,
    )

    for graph in tqdm(graph_iter, total=len(train_files), desc=f"train {exp.name}", leave=False):
        if graph.runtimes is None or graph.node_config_feat.shape[0] == 0:
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        view = build_device_graph_view(graph)

        graph_loss_value = 0.0
        valid_chunks = 0

        static_h = None
        if not cfg.recompute_static_per_chunk_train:
            with autocast_context(bool(cfg.autocast_static_encode and exp.amp)):
                static_h = model.encode_static_nodes(view)
        batch_indices = build_config_batch_indices(
            num_configs=graph.node_config_feat.shape[0],
            batch_size=exp.train_config_batch,
            batching=getattr(exp, "chunk_batching", "contiguous"),
            epoch_num=epoch_num,
            shift_stride=getattr(exp, "shift_stride", None),
        )
        num_chunks = len(batch_indices)

        for chunk_idx, cfg_idx in enumerate(batch_indices):
            batch = graph.node_config_feat[cfg_idx]

            with autocast_context(exp.amp):
                static_h_chunk = model.encode_static_nodes(view) if cfg.recompute_static_per_chunk_train else static_h
                pred = model.score_configs_batch(view, static_h_chunk, batch)
                if not torch.isfinite(pred).all():
                    continue
                target = runtime_targets(graph.runtimes[cfg_idx].to(DEVICE, non_blocking=True))
                pair_loss = sampled_pairwise_loss(pred, target, num_pairs=exp.pairwise_samples, mode=exp.pairwise_mode)
                chunk_loss = exp.pairwise_weight * pair_loss
                loss = chunk_loss / max(num_chunks, 1)

            if not torch.isfinite(loss):
                continue

            later_chunks_exist = (chunk_idx + 1) < num_chunks
            retain_graph = bool((not cfg.recompute_static_per_chunk_train) and later_chunks_exist)

            if exp.amp and DEVICE.type == "cuda":
                scaler.scale(loss).backward(retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)

            graph_loss_value += float(chunk_loss.detach().cpu())
            valid_chunks += 1

        if valid_chunks == 0:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            if DEVICE.type == "cuda" and cfg.empty_cache_between_graphs:
                torch.cuda.empty_cache()
            continue

        if exp.amp and DEVICE.type == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.grad_clip)
            optimizer.step()

        losses.append(graph_loss_value / valid_chunks)
        if DEVICE.type == "cuda" and cfg.empty_cache_between_graphs:
            torch.cuda.empty_cache()

    if skipped:
        print(f"[{exp.name}] skipped graphs this epoch: {skipped}")
    return float(np.mean(losses)) if losses else np.nan


@torch.inference_mode()
def predict_graphs(exp, model: LayoutGraphAwareModel, graph_items: list[GraphExample]):
    preds = {}
    for graph in tqdm(graph_items, total=len(graph_items), desc=f"predict {exp.name}", leave=False):
        preds[graph.graph_id] = model.score_graph(graph, cfg_batch_size=exp.eval_config_batch)
    return preds


def save_preds_npz(path: Path, preds_by_graph: dict[str, np.ndarray]):
    np.savez_compressed(path, **{gid: np.asarray(pred, dtype=np.float32) for gid, pred in preds_by_graph.items()})


def load_preds_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_experiment_config(exp_dir: Path, fallback_exp):
    cfg_path = exp_dir / "experiment_config.json"
    if not cfg_path.exists():
        return fallback_exp
    try:
        return SimpleNamespace(**load_json(cfg_path))
    except Exception as exc:
        print(f"[cache] failed to read {cfg_path.name} for {exp_dir.name}: {exc}; using fallback config")
        return fallback_exp


def train_gnn_experiment(
    cfg,
    exp,
    train_files,
    valid_graphs,
    num_opcodes: int,
):
    print(f"\n=== GNN experiment: {exp.name} ===")
    seed_everything(cfg.seed)

    model = LayoutGraphAwareModel(num_opcodes=num_opcodes, exp=exp).to(DEVICE)
    print(f"params[{exp.name}]: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(exp.amp and DEVICE.type == "cuda"))

    history = []
    best_state = None
    best_tau = -1e18
    best_top1 = 1e18
    epochs_without_improve = 0

    with ProcessPoolExecutor(max_workers=cfg.num_preprocess_workers, mp_context=get_mp_context()) as executor:
        for epoch in range(1, exp.epochs + 1):
            seed_everything(cfg.seed + epoch)
            t0 = time.time()

            train_loss = train_one_epoch(
                cfg=cfg,
                exp=exp,
                model=model,
                train_files=train_files,
                optimizer=optimizer,
                scaler=scaler,
                epoch_num=epoch,
                executor=executor,
            )
            valid_preds = predict_graphs(
                exp=exp,
                model=model,
                graph_items=valid_graphs,
            )
            per_graph_df, summary = evaluate_predictions(
                graph_items=valid_graphs,
                preds_by_graph=valid_preds,
                ks=(1, 5, 10, 100),
            )
            tau = float(summary.get("kendall_tau", -np.inf))
            top1 = float(summary.get("top1_slowdown", np.inf))
            row = {
                "experiment": exp.name,
                "epoch": epoch,
                "train_loss": train_loss,
                "kendall_tau": tau,
                "top1_slowdown": top1,
                "top5_slowdown": summary.get("top5_slowdown", np.nan),
                "top10_slowdown": summary.get("top10_slowdown", np.nan),
                "top100_slowdown": summary.get("top100_slowdown", np.nan),
                "seconds": time.time() - t0,
            }
            history.append(row)
            print(row)

            improved = (tau > best_tau) or (tau == best_tau and top1 < best_top1)
            if improved:
                best_tau = tau
                best_top1 = top1
                epochs_without_improve = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "summary": summary,
                    "per_graph_df": per_graph_df.copy(),
                    "preds": {k: v.copy() for k, v in valid_preds.items()},
                    "history": pd.DataFrame(history),
                }
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= exp.early_stopping:
                    print(f"[{exp.name}] early stopping after epoch {epoch}")
                    break

    if best_state is None:
        raise RuntimeError(f"Training failed for experiment {exp.name}")

    model.load_state_dict(best_state["model"])
    exp_dir = cfg.out_dir / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state["model"], exp_dir / "best_model.pt")
    best_state["history"].to_csv(exp_dir / "history.csv", index=False)
    best_state["per_graph_df"].to_csv(exp_dir / "per_graph_metrics.csv", index=False)
    save_preds_npz(exp_dir / "valid_preds.npz", best_state["preds"])
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(best_state["summary"], f, indent=2)
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(exp), f, indent=2)

    return {
        "name": exp.name,
        "type": "gnn",
        "summary": best_state["summary"],
        "per_graph": best_state["per_graph_df"],
        "preds": best_state["preds"],
        "history": best_state["history"],
        "model_dir": exp_dir,
        "config": exp,
    }


def load_existing_gnn_experiment(cfg, exp, valid_graphs, num_opcodes: int):
    exp_dir = cfg.out_dir / exp.name
    model_path = exp_dir / "best_model.pt"
    if not model_path.exists():
        return None

    loaded_exp = maybe_load_experiment_config(exp_dir, exp)
    print(f"[cache] loading GNN checkpoint for {exp.name} from {exp_dir}")
    model = LayoutGraphAwareModel(num_opcodes=num_opcodes, exp=loaded_exp).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    preds_path = exp_dir / "valid_preds.npz"
    summary_path = exp_dir / "summary.json"
    history_path = exp_dir / "history.csv"
    if cfg.prefer_saved_valid_preds and preds_path.exists():
        preds = load_preds_npz(preds_path)
    else:
        preds = predict_graphs(
            exp=loaded_exp,
            model=model,
            graph_items=valid_graphs,
        )
        save_preds_npz(preds_path, preds)

    per_graph_df, summary_eval = evaluate_predictions(
        graph_items=valid_graphs,
        preds_by_graph=preds,
        ks=(1, 5, 10, 100),
    )
    summary = load_json(summary_path) if summary_path.exists() else summary_eval
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    return {
        "name": exp.name,
        "type": "gnn",
        "summary": summary,
        "summary_eval": summary_eval,
        "per_graph": per_graph_df,
        "preds": preds,
        "history": history,
        "model_dir": exp_dir,
        "config": loaded_exp,
        "loaded_from_cache": True,
    }


def resolve_gnn_experiment(
    cfg,
    exp,
    train_files,
    valid_graphs,
    num_opcodes: int,
):
    if cfg.prefer_cached_experiments:
        loaded = load_existing_gnn_experiment(cfg, exp, valid_graphs, num_opcodes)
        if loaded is not None:
            return loaded
    if not cfg.train_missing_gnn_experiments:
        raise FileNotFoundError(
            f"Missing cached GNN experiment '{exp.name}'. Either train it once or enable train_missing_gnn_experiments."
        )
    return train_gnn_experiment(
        cfg=cfg,
        exp=exp,
        train_files=train_files,
        valid_graphs=valid_graphs,
        num_opcodes=num_opcodes,
    )


def summarize_xgb_config_features_np(node_config_feat: np.ndarray) -> np.ndarray:
    cfg = unpack_node_config_storage_np(node_config_feat, output_mode="signed_float")
    cfg = np.asarray(cfg, dtype=np.float32)
    if cfg.ndim != 3:
        raise ValueError(f"Expected decoded node_config_feat with shape (c, nc, 18), got {cfg.shape}")

    mask = (cfg >= 0).astype(np.float32)
    cfg_masked = np.where(mask > 0, cfg, 0.0)
    valid_count = mask.sum(axis=1)

    per_dim_mean = cfg_masked.sum(axis=1) / np.maximum(valid_count, 1.0)
    centered = np.where(mask > 0, cfg - per_dim_mean[:, None, :], 0.0)
    per_dim_std = np.sqrt((centered * centered).sum(axis=1) / np.maximum(valid_count, 1.0))

    overall_valid = mask.mean(axis=(1, 2), keepdims=False)[:, None]
    overall_mean = cfg_masked.sum(axis=(1, 2), keepdims=False)[:, None] / np.maximum(mask.sum(axis=(1, 2), keepdims=False)[:, None], 1.0)
    overall_std = np.sqrt(
        (np.where(mask > 0, cfg - overall_mean[:, None, :], 0.0) ** 2).sum(axis=(1, 2), keepdims=False)[:, None]
        / np.maximum(mask.sum(axis=(1, 2), keepdims=False)[:, None], 1.0)
    )

    out = np.concatenate([overall_valid, overall_mean, overall_std, per_dim_mean, per_dim_std], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def iter_graph_source(
    cfg,
    graph_source,
    split: str,
    max_configs_per_graph: int | None,
    base_seed: int,
    show_progress: bool = False,
):
    if not graph_source:
        return
    first = graph_source[0]
    if isinstance(first, GraphExample):
        for graph in graph_source:
            yield graph
        return

    yield from preprocess_graphs_parallel(
        cfg=cfg,
        files=list(graph_source),
        split=split,
        max_configs_per_graph=max_configs_per_graph,
        base_seed=base_seed,
        executor=None,
        show_progress=show_progress,
    )


def build_layout_xgb_table(cfg, graph_source, split: str, max_configs_per_graph: int | None, base_seed: int):
    features = []
    rows = []
    group_sizes = []

    for graph in tqdm(
        iter_graph_source(
            cfg=cfg,
            graph_source=graph_source,
            split=split,
            max_configs_per_graph=max_configs_per_graph,
            base_seed=base_seed,
            show_progress=False,
        ),
        total=len(graph_source) if hasattr(graph_source, "__len__") else None,
        desc=f"build xgb {split}",
        leave=False,
    ):
        if graph.runtimes is None:
            continue
        rt = graph.runtimes.cpu().numpy()
        feat = summarize_xgb_config_features_np(graph.node_config_feat.cpu().numpy())
        dup = np.ones(graph.num_configs, dtype=np.float32) if graph.duplicate_count is None else graph.duplicate_count.cpu().numpy()
        feat = np.concatenate([feat, dup[:, None]], axis=1).astype(np.float32)

        order = np.argsort(rt)
        rel = np.empty_like(order, dtype=np.float32)
        rel[order] = np.arange(len(order) - 1, -1, -1, dtype=np.float32)

        features.append(feat)
        rows.append(pd.DataFrame({
            "graph_id": graph.graph_id,
            "runtime": rt,
            "label": rel,
            "duplicate_weight": dup,
            "row_id": np.arange(graph.num_configs, dtype=np.int32),
        }))
        group_sizes.append(graph.num_configs)

    if not features:
        raise RuntimeError(f"No rows were built for split={split}")

    return {
        "X": np.vstack(features),
        "meta": pd.concat(rows, ignore_index=True),
        "group_sizes": group_sizes,
    }


def predict_xgb_by_graph(model, xgb_pack: dict):
    pred = -np.asarray(model.predict(xgb_pack["X"]), dtype=np.float32)
    preds_by_graph = {}
    for gid, grp in xgb_pack["meta"].assign(pred=pred).groupby("graph_id", sort=False):
        preds_by_graph[gid] = grp["pred"].to_numpy(dtype=np.float32)
    return preds_by_graph


def load_existing_xgb_experiment(cfg, exp, valid_graphs):
    if xgb is None:
        raise ImportError("xgboost is not installed in this environment.")

    exp_dir = cfg.out_dir / exp.name
    model_path = exp_dir / "xgb_model.json"
    if not model_path.exists():
        return None

    loaded_exp = maybe_load_experiment_config(exp_dir, exp)
    print(f"[cache] loading XGB checkpoint for {exp.name} from {exp_dir}")
    model = xgb.XGBRanker(
        objective="rank:pairwise",
        tree_method=loaded_exp.tree_method,
        n_estimators=loaded_exp.n_estimators,
        learning_rate=loaded_exp.learning_rate,
        max_depth=loaded_exp.max_depth,
        min_child_weight=loaded_exp.min_child_weight,
        subsample=loaded_exp.subsample,
        colsample_bytree=loaded_exp.colsample_bytree,
        reg_lambda=loaded_exp.reg_lambda,
        random_state=loaded_exp.random_state,
        n_jobs=getattr(loaded_exp, "n_jobs", getattr(cfg, "xgb_n_jobs", 1)),
    )
    model.load_model(model_path)

    preds_path = exp_dir / "valid_preds.npz"
    summary_path = exp_dir / "summary.json"
    if cfg.prefer_saved_valid_preds and preds_path.exists():
        preds = load_preds_npz(preds_path)
    else:
        valid_pack = build_layout_xgb_table(
            cfg=cfg,
            graph_source=valid_graphs,
            split="valid",
            max_configs_per_graph=cfg.max_valid_configs_per_graph,
            base_seed=cfg.seed + 313_131,
        )
        preds = predict_xgb_by_graph(model, valid_pack)
        save_preds_npz(preds_path, preds)

    per_graph_df, summary_eval = evaluate_predictions(
        graph_items=valid_graphs,
        preds_by_graph=preds,
        ks=(1, 5, 10, 100),
    )
    summary = load_json(summary_path) if summary_path.exists() else summary_eval
    return {
        "name": exp.name,
        "type": "xgb",
        "summary": summary,
        "summary_eval": summary_eval,
        "per_graph": per_graph_df,
        "preds": preds,
        "model_dir": exp_dir,
        "config": loaded_exp,
        "loaded_from_cache": True,
    }


def resolve_xgb_experiment(cfg, exp, train_files, valid_graphs):
    if cfg.prefer_cached_experiments:
        loaded = load_existing_xgb_experiment(cfg, exp, valid_graphs)
        if loaded is not None:
            return loaded
    if not cfg.train_missing_xgb_experiment:
        raise FileNotFoundError(
            f"Missing cached XGB experiment '{exp.name}'. Either train it once or enable train_missing_xgb_experiment."
        )
    return train_xgb_experiment(cfg, exp, train_files=train_files, valid_graphs=valid_graphs)


def train_xgb_experiment(cfg, exp, train_files, valid_graphs):
    if xgb is None:
        raise ImportError("xgboost is not installed in this environment.")

    print(f"\n=== XGB experiment: {exp.name} ===")
    train_pack = build_layout_xgb_table(
        cfg=cfg,
        graph_source=train_files,
        split="train",
        max_configs_per_graph=cfg.max_train_configs_per_graph,
        base_seed=cfg.seed + 919_191,
    )
    valid_pack = build_layout_xgb_table(
        cfg=cfg,
        graph_source=valid_graphs,
        split="valid",
        max_configs_per_graph=cfg.max_valid_configs_per_graph,
        base_seed=cfg.seed + 313_131,
    )

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        tree_method=exp.tree_method,
        n_estimators=exp.n_estimators,
        learning_rate=exp.learning_rate,
        max_depth=exp.max_depth,
        min_child_weight=exp.min_child_weight,
        subsample=exp.subsample,
        colsample_bytree=exp.colsample_bytree,
        reg_lambda=exp.reg_lambda,
        random_state=exp.random_state,
        n_jobs=getattr(exp, "n_jobs", getattr(cfg, "xgb_n_jobs", 1)),
    )
    model.fit(
        train_pack["X"],
        train_pack["meta"]["label"].to_numpy(np.float32),
        group=train_pack["group_sizes"],
        verbose=False,
    )

    valid_preds = predict_xgb_by_graph(model, valid_pack)
    per_graph_df, summary = evaluate_predictions(
        graph_items=valid_graphs,
        preds_by_graph=valid_preds,
        ks=(1, 5, 10, 100),
    )

    exp_dir = cfg.out_dir / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    per_graph_df.to_csv(exp_dir / "per_graph_metrics.csv", index=False)
    save_preds_npz(exp_dir / "valid_preds.npz", valid_preds)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(exp_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(exp), f, indent=2)
    model.save_model(exp_dir / "xgb_model.json")

    return {
        "name": exp.name,
        "type": "xgb",
        "summary": summary,
        "per_graph": per_graph_df,
        "preds": valid_preds,
        "model_dir": exp_dir,
        "config": exp,
    }


def fractional_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    rank = np.empty_like(order, dtype=np.float32)
    rank[order] = np.arange(len(scores), dtype=np.float32)
    if len(scores) > 1:
        rank /= float(len(scores) - 1)
    return rank


def average_rank_predictions(results: list[dict], weights: list[float] | None = None) -> dict[str, np.ndarray]:
    if not results:
        return {}
    if weights is None:
        weights = [1.0] * len(results)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    common = set(results[0]["preds"].keys())
    for result in results[1:]:
        common &= set(result["preds"].keys())

    out = {}
    for gid in common:
        agg = None
        for result, weight in zip(results, weights):
            rank = fractional_rank(result["preds"][gid])
            agg = rank * float(weight) if agg is None else agg + rank * float(weight)
        out[gid] = agg / total
    return out


def ensemble_predictions(preds_a: dict[str, np.ndarray], preds_b: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    out = {}
    for gid in preds_a.keys() & preds_b.keys():
        ra = fractional_rank(preds_a[gid])
        rb = fractional_rank(preds_b[gid])
        out[gid] = alpha * ra + (1.0 - alpha) * rb
    return out


def search_best_ensemble(
    valid_graphs,
    preds_a: dict[str, np.ndarray],
    preds_b: dict[str, np.ndarray],
    name_a: str,
    name_b: str,
):
    rows = []
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        preds = ensemble_predictions(preds_a, preds_b, float(alpha))
        per_graph_df, summary = evaluate_predictions(
            graph_items=valid_graphs,
            preds_by_graph=preds,
            ks=(1, 5, 10, 100),
        )
        row = {
            "ensemble_of": f"{name_a}+{name_b}",
            "alpha_a": float(alpha),
            "alpha_b": float(1.0 - alpha),
            **summary,
        }
        rows.append(row)

        best_tau = float(best["summary"].get("kendall_tau", -np.inf)) if best is not None else -np.inf
        best_top1 = float(best["summary"].get("top1_slowdown", np.inf)) if best is not None else np.inf
        row_tau = float(row.get("kendall_tau", -np.inf))
        row_top1 = float(row.get("top1_slowdown", np.inf))
        if best is None or row_tau > best_tau or (row_tau == best_tau and row_top1 < best_top1):
            best = {
                "summary": summary,
                "per_graph": per_graph_df,
                "preds": preds,
                "alpha": float(alpha),
            }

    return best, pd.DataFrame(rows)


def save_ensemble_artifact(exp_dir: Path, best: dict, sweep: pd.DataFrame, extra_summary: dict | None = None):
    exp_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(exp_dir / "weight_sweep.csv", index=False)
    best["per_graph"].to_csv(exp_dir / "per_graph_metrics.csv", index=False)
    save_preds_npz(exp_dir / "valid_preds.npz", best["preds"])
    payload = dict(best["summary"])
    if extra_summary:
        payload.update(extra_summary)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def select_gnn_candidates_for_ensemble(cfg, gnn_results: list[dict]) -> list[dict]:
    if not gnn_results:
        return []
    by_name = {result["name"]: result for result in gnn_results}
    ranked = sorted(
        gnn_results,
        key=lambda result: (
            float(result["summary"].get("kendall_tau", -np.inf)),
            -float(result["summary"].get("top1_slowdown", np.inf)),
        ),
        reverse=True,
    )
    best_tau = float(ranked[0]["summary"].get("kendall_tau", -np.inf))

    selected_names = []
    for name in cfg.ensemble_always_include_gnns:
        if name in by_name and name not in selected_names:
            selected_names.append(name)

    explicit_pool = [name for name in cfg.ensemble_gnn_experiments if name in by_name]
    if cfg.ensemble_auto_select_beneficial_gnns:
        for result in ranked:
            tau = float(result["summary"].get("kendall_tau", -np.inf))
            beneficial = True if best_tau <= 0 else tau >= best_tau * cfg.ensemble_min_tau_ratio_to_best
            if beneficial and result["name"] not in selected_names:
                selected_names.append(result["name"])
            if len(selected_names) >= cfg.ensemble_top_k_gnns:
                break
    else:
        for name in explicit_pool:
            if name not in selected_names:
                selected_names.append(name)
            if len(selected_names) >= cfg.ensemble_top_k_gnns:
                break

    if ranked[0]["name"] not in selected_names:
        selected_names.insert(0, ranked[0]["name"])

    selected = []
    seen = set()
    for name in selected_names:
        if name in by_name and name not in seen:
            selected.append(by_name[name])
            seen.add(name)
    return selected


def prepare_graph_sets(cfg):
    train_dir = find_layout_split_dir(cfg, "train")
    valid_dir = find_layout_split_dir(cfg, "valid")

    train_files = list_npz_files(train_dir, cfg.max_train_graphs)
    valid_files = list_npz_files(valid_dir, cfg.max_valid_graphs)

    cfg.node_numeric_mean, cfg.node_numeric_std = fit_node_numeric_scaler(
        files=train_files,
        feature_clip_value=cfg.feature_clip_value,
    )

    max_opcode_seen = scan_max_opcode(train_files + valid_files)
    num_opcodes = max(129, max_opcode_seen + 2)
    cfg.summary_opcode_id = num_opcodes - 1

    valid_graphs = []
    if cfg.preload_valid_in_memory:
        valid_graphs = list(
            preprocess_graphs_parallel(
                cfg=cfg,
                files=valid_files,
                split="valid",
                max_configs_per_graph=cfg.max_valid_configs_per_graph,
                base_seed=cfg.seed + 123_457,
                executor=None,
                show_progress=True,
            )
        )
    print("device:", DEVICE)
    print("train graphs:", len(train_files), "valid graphs:", len(valid_files))
    print("num_opcodes:", num_opcodes)
    print("summary_opcode_id:", cfg.summary_opcode_id)
    print("node_numeric_dim:", NODE_TOTAL_NUMERIC_DIM)
    return train_files, valid_graphs, num_opcodes


def run_suite(
    cfg,
    enabled_gnn_experiments: list[str] | None = None,
    run_xgb_experiment: bool = True,
    run_ensemble_search: bool = True,
):
    configure_torch_runtime(cfg)
    seed_everything(cfg.seed)
    train_files, valid_graphs, num_opcodes = prepare_graph_sets(cfg)

    gnn_results = []
    summary_rows = []
    for exp in build_gnn_experiments(enabled_gnn_experiments):
        local_cfg = copy.deepcopy(cfg)
        result = resolve_gnn_experiment(
            cfg=local_cfg,
            exp=exp,
            train_files=train_files,
            valid_graphs=valid_graphs,
            num_opcodes=num_opcodes,
        )
        gnn_results.append(result)
        summary_rows.append({
            "experiment": result["name"],
            "type": result["type"],
            "loaded_from_cache": bool(result.get("loaded_from_cache", False)),
            **result["summary"],
        })

    xgb_result = None
    if run_xgb_experiment:
        xgb_result = resolve_xgb_experiment(
            cfg,
            build_xgb_experiment(),
            train_files=train_files,
            valid_graphs=valid_graphs,
        )
        summary_rows.append({
            "experiment": xgb_result["name"],
            "type": xgb_result["type"],
            "loaded_from_cache": bool(xgb_result.get("loaded_from_cache", False)),
            **xgb_result["summary"],
        })

    if run_ensemble_search:
        if xgb_result is None:
            raise ValueError("Ensemble search requires run_xgb_experiment=True.")
        if not gnn_results:
            raise ValueError("Ensemble search requires at least one GNN experiment.")

        selected_gnns = select_gnn_candidates_for_ensemble(cfg, gnn_results)
        if not selected_gnns:
            selected_gnns = [max(
                gnn_results,
                key=lambda result: (
                    float(result["summary"].get("kendall_tau", -np.inf)),
                    -float(result["summary"].get("top1_slowdown", np.inf)),
                ),
            )]

        print("\n=== Ensemble GNN candidates ===")
        for result in selected_gnns:
            print(result["name"], result["summary"])

        ensemble_rows = []
        for gnn_result in selected_gnns:
            ensemble_best, ensemble_sweep = search_best_ensemble(
                valid_graphs=valid_graphs,
                preds_a=gnn_result["preds"],
                preds_b=xgb_result["preds"],
                name_a=gnn_result["name"],
                name_b=xgb_result["name"],
            )
            ensemble_name = f"ensemble_{gnn_result['name']}_plus_{xgb_result['name']}"
            save_ensemble_artifact(
                cfg.out_dir / ensemble_name,
                ensemble_best,
                ensemble_sweep,
                extra_summary={
                    "alpha_best_gnn": ensemble_best["alpha"],
                    "gnn_components": [gnn_result["name"]],
                    "xgb_component": xgb_result["name"],
                },
            )
            row = {
                "experiment": ensemble_name,
                "type": "ensemble",
                "loaded_from_cache": False,
                "alpha_best_gnn": ensemble_best["alpha"],
                "gnn_components": "|".join([gnn_result["name"]]),
                "xgb_component": xgb_result["name"],
                **ensemble_best["summary"],
            }
            summary_rows.append(row)
            ensemble_rows.append(row)

        if len(selected_gnns) >= 2:
            pooled_preds = average_rank_predictions(selected_gnns)
            pooled_name = "rankavg_" + "_".join(result["name"] for result in selected_gnns)
            pooled_best, pooled_sweep = search_best_ensemble(
                valid_graphs=valid_graphs,
                preds_a=pooled_preds,
                preds_b=xgb_result["preds"],
                name_a=pooled_name,
                name_b=xgb_result["name"],
            )
            ensemble_name = f"ensemble_{pooled_name}_plus_{xgb_result['name']}"
            save_ensemble_artifact(
                cfg.out_dir / ensemble_name,
                pooled_best,
                pooled_sweep,
                extra_summary={
                    "alpha_best_gnn": pooled_best["alpha"],
                    "gnn_components": [result["name"] for result in selected_gnns],
                    "xgb_component": xgb_result["name"],
                    "gnn_pooling": "uniform_rank_average",
                },
            )
            row = {
                "experiment": ensemble_name,
                "type": "ensemble",
                "loaded_from_cache": False,
                "alpha_best_gnn": pooled_best["alpha"],
                "gnn_components": "|".join(result["name"] for result in selected_gnns),
                "xgb_component": xgb_result["name"],
                "gnn_pooling": "uniform_rank_average",
                **pooled_best["summary"],
            }
            summary_rows.append(row)
            ensemble_rows.append(row)

        if ensemble_rows:
            pd.DataFrame(ensemble_rows).sort_values(["kendall_tau", "top1_slowdown"], ascending=[False, True]).to_csv(
                cfg.out_dir / "ensemble_candidates.csv",
                index=False,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["kendall_tau", "top1_slowdown"], ascending=[False, True])
    summary_df.to_csv(cfg.out_dir / "ablation_summary.csv", index=False)
    print("\n=== Final ranking ===")
    print(summary_df.to_string(index=False))
    return summary_df


def print_run_plan(cfg, enabled_gnn_experiments: list[str] | None, run_xgb_experiment: bool, run_ensemble_search: bool):
    print("\n=== Run settings ===")
    print(f"base={cfg.base}")
    print(f"dataset={cfg.dataset}  source={cfg.source}  search={cfg.search}")
    print(f"max_train_graphs={cfg.max_train_graphs}  max_valid_graphs={cfg.max_valid_graphs}")
    print(f"max_train_configs_per_graph={cfg.max_train_configs_per_graph}  max_valid_configs_per_graph={cfg.max_valid_configs_per_graph}")
    print(f"dedup_strategy={cfg.dedup_strategy}  prefer_cached_experiments={cfg.prefer_cached_experiments}")
    print(f"num_preprocess_workers={cfg.num_preprocess_workers}  prefetch_graphs={cfg.prefetch_graphs}")
    print(f"log_prefetch_status={cfg.log_prefetch_status}  prefetch_status_every_n_graphs={cfg.prefetch_status_every_n_graphs}")
    print(f"graph_storage_dtype={cfg.graph_storage_dtype}  autocast_static_encode={cfg.autocast_static_encode}")
    print(f"blobify_enabled={cfg.blobify_enabled}  blobify_keep_hops={cfg.blobify_keep_hops}  blobify_frontier_split={cfg.blobify_frontier_split}")
    print(f"pack_node_config_feat={cfg.pack_node_config_feat}")
    print(f"type_embed_dim={cfg.type_embed_dim}  layout_value_embed_dim={cfg.layout_value_embed_dim}")
    print(f"enabled_gnn_experiments={enabled_gnn_experiments}")
    print(f"train_subsample_mode={cfg.train_subsample_mode}")
    for exp in build_gnn_experiments(enabled_gnn_experiments):
        print(
            f"  - {exp.name}: batching={getattr(exp, 'chunk_batching', 'contiguous')} "
            f"hidden={exp.hidden_dim} layers={exp.num_gnn_layers} train_batch={exp.train_config_batch} "
            f"pairs={exp.pairwise_samples} epochs={exp.epochs}"
        )
    print(f"run_xgb_experiment={run_xgb_experiment}  run_ensemble_search={run_ensemble_search}")
    print(
        f"prefer_cached_experiments={cfg.prefer_cached_experiments}  "
        f"train_missing_gnn={cfg.train_missing_gnn_experiments}  "
        f"train_missing_xgb={cfg.train_missing_xgb_experiment}"
    )


def set_runtime_device(gpu_id: int | None):
    global DEVICE
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        DEVICE = torch.device(f"cuda:{gpu_id}")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_single_job_resources() -> tuple[int, int, int]:
    requested_cpus = int(CONFIG.get("requested_num_cpus", CONFIG.get("num_preprocess_workers", 1)))
    requested_workers = int(CONFIG.get("num_preprocess_workers", 1))
    requested_prefetch = int(CONFIG.get("prefetch_graphs", 1))
    cpu_budget = max(2, requested_cpus)
    preprocess_workers = max(1, min(requested_workers, cpu_budget - 1))
    xgb_n_jobs = max(1, cpu_budget)
    prefetch_graphs = max(1, min(requested_prefetch, cpu_budget))
    return preprocess_workers, prefetch_graphs, xgb_n_jobs


def write_combined_best_rows(all_best_rows: list[dict]):
    if not all_best_rows:
        return
    combined_df = pd.DataFrame(all_best_rows)
    combined_base_name = CONFIG.get("combined_best_name", "layout_all_problem_best_models.csv")
    combined_name = append_variant_to_filename(
        combined_base_name,
        get_preprocess_variant_name(bool(CONFIG.get("blobify_enabled", False))),
    )
    combined_path = Path(CONFIG["base"]) / "artifacts" / combined_name
    combined_df.to_csv(combined_path, index=False)
    if len(combined_df) > 1:
        print("\n=== Best model per problem ===")
        print(combined_df.to_string(index=False))
    print(f"saved: {combined_path}")


def main():
    problems_to_run = list(CONFIG.get("problems_to_run") or [CONFIG.get("dataset")])
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    set_runtime_device(0 if available_gpus >= 1 else None)
    preprocess_workers, prefetch_graphs, xgb_n_jobs = compute_single_job_resources()

    all_best_rows = []
    for dataset_name in problems_to_run:
        cfg, enabled_gnn_experiments, run_xgb_experiment, run_ensemble_search = build_runtime_config(dataset_name)
        cfg.num_preprocess_workers = preprocess_workers
        cfg.prefetch_graphs = prefetch_graphs
        cfg.xgb_n_jobs = xgb_n_jobs
        print_run_plan(cfg, enabled_gnn_experiments, run_xgb_experiment, run_ensemble_search)
        summary_df = run_suite(
            cfg,
            enabled_gnn_experiments=enabled_gnn_experiments,
            run_xgb_experiment=run_xgb_experiment,
            run_ensemble_search=run_ensemble_search,
        )
        if len(summary_df):
            best_row = summary_df.iloc[0].to_dict()
            best_row["dataset"] = dataset_name
            best_row["out_dir"] = str(cfg.out_dir)
            best_row["gpu_id"] = 0 if available_gpus >= 1 else None
            all_best_rows.append(best_row)

    write_combined_best_rows(all_best_rows)


if __name__ == "__main__":
    main()

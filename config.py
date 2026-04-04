"""
config.py — Central configuration for hyperparameters, paths, and constants.

All tunable settings are collected here so experiments are reproducible
and easy to modify without touching training logic.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get("TPUGRAPHS_DATA", "data/npz_all/npz")
OUTPUT_DIR = os.environ.get("TPUGRAPHS_OUTPUT", "outputs")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submission")

# Number of distinct HLO opcodes in the dataset (0-indexed, max ~120)
NUM_OPCODES = 128

# Node feature dimensionality in the .npz files
NODE_FEAT_DIM = 140

# Config feature dimensionality
TILE_CONFIG_DIM = 24
LAYOUT_CONFIG_DIM = 18


@dataclass
class TileConfig:
    """Hyperparameters for the tile:xla sub-collection."""

    # Model
    hidden_dim: int = 128
    num_gnn_layers: int = 4
    opcode_embed_dim: int = 64
    dropout: float = 0.1
    gnn_type: str = "sage"          # "sage" or "gatv2"
    heads: int = 4                  # only used if gnn_type == "gatv2"

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    batch_size: int = 4             # number of graphs per batch
    max_configs: int = 256          # sample this many configs per graph
    swa_start_epoch: int = 60       # start SWA from this epoch
    swa_lr: float = 5e-4

    # Loss
    loss_type: str = "listmle"      # "listmle", "pairwise", or "combined"
    aux_mse_weight: float = 0.1     # weight of auxiliary MSE loss in combined

    # Evaluation
    top_k: int = 5                  # top-K slowdown metric

    # Paths
    data_dir: str = os.path.join(DATA_ROOT, "tile", "xla")
    save_dir: str = os.path.join(OUTPUT_DIR, "tile_xla")


@dataclass
class LayoutConfig:
    """Hyperparameters for layout sub-collections."""

    # Model
    hidden_dim: int = 128
    num_gnn_layers: int = 4
    opcode_embed_dim: int = 64
    dropout: float = 0.1
    gnn_type: str = "sage"
    heads: int = 4

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    max_configs: int = 512          # sample configs per graph per epoch
    swa_start_epoch: int = 45
    swa_lr: float = 5e-4
    num_pairs: int = 64             # pairs sampled per graph for pairwise loss

    # Graph segmentation
    max_segment_size: int = 5000    # partition graphs larger than this

    # Loss
    loss_type: str = "pairwise"     # "pairwise", "listmle", or "combined"
    margin: float = 0.1             # margin for pairwise loss
    aux_mse_weight: float = 0.1

    # Paths — set dynamically based on source/search
    source: str = "xla"             # "xla" or "nlp"
    search: str = "default"         # "default" or "random"

    @property
    def data_dir(self) -> str:
        return os.path.join(DATA_ROOT, "layout", self.source, self.search)

    @property
    def save_dir(self) -> str:
        return os.path.join(OUTPUT_DIR, f"layout_{self.source}_{self.search}")

    @property
    def collection_name(self) -> str:
        return f"layout:{self.source}:{self.search}"


# ──────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────

ALL_LAYOUT_COLLECTIONS = [
    ("xla", "default"),
    ("xla", "random"),
    ("nlp", "default"),
    ("nlp", "random"),
]


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    tile_cfg = TileConfig()
    os.makedirs(tile_cfg.save_dir, exist_ok=True)
    for source, search in ALL_LAYOUT_COLLECTIONS:
        layout_cfg = LayoutConfig(source=source, search=search)
        os.makedirs(layout_cfg.save_dir, exist_ok=True)

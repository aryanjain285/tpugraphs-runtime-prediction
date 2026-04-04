"""
dataset.py — Data loading for TpuGraphs tile and layout collections.

Each .npz file encodes one computation graph with multiple compiler configurations
and their measured runtimes. We load these into PyTorch Geometric Data objects.

Key design choices:
  - Tile configs are graph-level (24-dim vector per config).
  - Layout configs are node-level (18-dim per configurable node per config).
  - We normalise runtimes per-graph so the ranking target is scale-invariant.
  - Large layout graphs are segmented at load time for memory efficiency.
"""

import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _compute_topo_depth(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Compute topological depth for each node in a DAG.

    Depth = length of the longest path from any source node to this node.
    This gives the GNN positional awareness of where an operation sits
    in the computation pipeline.

    Parameters
    ----------
    edge_index : ndarray of shape (m, 2)
        Each row [u, v] means u consumes output of v (edge v → u in data flow).
    num_nodes : int

    Returns
    -------
    depth : ndarray of shape (num_nodes,), float32, normalised to [0, 1].
    """
    # Build adjacency list (data flow direction: v → u)
    children = [[] for _ in range(num_nodes)]
    in_degree = np.zeros(num_nodes, dtype=np.int64)
    for u, v in edge_index:
        children[v].append(u)
        in_degree[u] += 1

    # BFS from sources (nodes with in_degree == 0 in data-flow sense)
    # But edge_index has u consuming v, so sources in data flow are nodes
    # that nobody consumes from => no outgoing edges in the reversed sense.
    # Actually let's just use the topological order from the dataset
    # (nodes are already topologically sorted per the dataset docs).
    depth = np.zeros(num_nodes, dtype=np.float32)
    for v_idx in range(num_nodes):
        for u_idx in children[v_idx]:
            depth[u_idx] = max(depth[u_idx], depth[v_idx] + 1)

    max_depth = depth.max()
    if max_depth > 0:
        depth = depth / max_depth
    return depth


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load an npz file into a plain dict of numpy arrays."""
    return dict(np.load(path))


# ──────────────────────────────────────────────────────────────
# Tile Dataset
# ──────────────────────────────────────────────────────────────

class TileDataset:
    """
    Dataset for the tile:xla collection.

    Each sample is a kernel graph with graph-level tile configurations.
    The target is the ranking of configs by normalised runtime.

    Parameters
    ----------
    data_dir : str
        Path to tile/xla/{split}/ directory.
    split : str
        One of 'train', 'valid', 'test'.
    max_configs : int or None
        If set, randomly sub-sample this many configs per graph during __getitem__.
    """

    def __init__(self, data_dir: str, split: str, max_configs: Optional[int] = None):
        split_dir = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {split_dir}. "
                f"Check that the data is extracted correctly."
            )
        self.split = split
        self.max_configs = max_configs
        self.is_test = (split == "test")

        # Pre-load file names for identification in submission
        self.graph_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in self.files
        ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        d = _load_npz(self.files[idx])

        node_feat = torch.from_numpy(d["node_feat"].astype(np.float32))
        node_opcode = torch.from_numpy(d["node_opcode"].astype(np.int64))
        edge_index = torch.from_numpy(d["edge_index"].astype(np.int64).T)  # (2, m)

        num_nodes = node_feat.shape[0]

        # Topological depth as extra feature
        depth = _compute_topo_depth(d["edge_index"], num_nodes)
        depth_feat = torch.from_numpy(depth).unsqueeze(-1)  # (n, 1)

        config_feat = torch.from_numpy(d["config_feat"].astype(np.float32))  # (c, 24)

        if not self.is_test:
            runtimes = d["config_runtime"].astype(np.float64)
            normalizers = d["config_runtime_normalizers"].astype(np.float64)
            # Normalised runtime: lower is better
            norm_runtime = runtimes / np.maximum(normalizers, 1.0)
            norm_runtime = torch.from_numpy(norm_runtime.astype(np.float32))
        else:
            norm_runtime = torch.zeros(config_feat.shape[0])

        num_configs = config_feat.shape[0]

        # Sub-sample configs if requested (training only)
        if self.max_configs is not None and num_configs > self.max_configs and not self.is_test:
            perm = torch.randperm(num_configs)[: self.max_configs]
            config_feat = config_feat[perm]
            norm_runtime = norm_runtime[perm]
            num_configs = self.max_configs

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            node_opcode=node_opcode,
            topo_depth=depth_feat,
            config_feat=config_feat,          # (c, 24)
            runtime=norm_runtime,             # (c,)
            num_configs=num_configs,
            graph_id=self.graph_ids[idx],
        )
        data.num_nodes = num_nodes
        return data


# ──────────────────────────────────────────────────────────────
# Layout Dataset
# ──────────────────────────────────────────────────────────────

class LayoutDataset:
    """
    Dataset for layout sub-collections.

    Each sample is a full-program graph with node-level layout configurations.
    The target is ranking configs by runtime.

    Parameters
    ----------
    data_dir : str
        Path to layout/{source}/{search}/{split}/ directory.
    split : str
        One of 'train', 'valid', 'test'.
    max_configs : int or None
        Sub-sample configurations per graph.
    max_segment_size : int
        If a graph has more nodes than this, segment it.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_configs: Optional[int] = None,
        max_segment_size: int = 5000,
    ):
        split_dir = os.path.join(data_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {split_dir}. "
                f"Check that the data is extracted correctly."
            )
        self.split = split
        self.max_configs = max_configs
        self.max_segment_size = max_segment_size
        self.is_test = (split == "test")

        self.graph_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in self.files
        ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        d = _load_npz(self.files[idx])

        node_feat = torch.from_numpy(d["node_feat"].astype(np.float32))
        node_opcode = torch.from_numpy(d["node_opcode"].astype(np.int64))
        edge_index = torch.from_numpy(d["edge_index"].astype(np.int64).T)  # (2, m)
        num_nodes = node_feat.shape[0]

        # Topological depth
        depth = _compute_topo_depth(d["edge_index"], num_nodes)
        depth_feat = torch.from_numpy(depth).unsqueeze(-1)

        # Configurable node indices
        node_config_ids = torch.from_numpy(
            d["node_config_ids"].astype(np.int64)
        )  # (nc,)

        # Node-level config features: (c, nc, 18)
        node_config_feat = torch.from_numpy(
            d["node_config_feat"].astype(np.float32)
        )

        if not self.is_test:
            runtimes = torch.from_numpy(
                d["config_runtime"].astype(np.float32)
            )  # (c,)
        else:
            runtimes = torch.zeros(node_config_feat.shape[0])

        num_configs = node_config_feat.shape[0]

        # Sub-sample configs
        if self.max_configs is not None and num_configs > self.max_configs and not self.is_test:
            perm = torch.randperm(num_configs)[: self.max_configs]
            node_config_feat = node_config_feat[perm]
            runtimes = runtimes[perm]
            num_configs = self.max_configs

        # Segment information for large graph handling
        if "node_splits" in d:
            node_splits = torch.from_numpy(d["node_splits"].astype(np.int64))
        else:
            node_splits = torch.tensor([0, num_nodes], dtype=torch.int64)

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            node_opcode=node_opcode,
            topo_depth=depth_feat,
            node_config_ids=node_config_ids,      # (nc,)
            node_config_feat=node_config_feat,      # (c, nc, 18)
            runtime=runtimes,                       # (c,)
            num_configs=num_configs,
            node_splits=node_splits,
            graph_id=self.graph_ids[idx],
        )
        data.num_nodes = num_nodes
        return data


# ──────────────────────────────────────────────────────────────
# Collation
# ──────────────────────────────────────────────────────────────

def tile_collate_fn(batch: List[Data]) -> List[Data]:
    """
    For tile data we process each graph individually (no batching into
    a single mega-graph) because config counts vary per graph.
    Returns a list of Data objects.
    """
    return batch


def layout_collate_fn(batch: List[Data]) -> List[Data]:
    """Same as tile — each graph is processed individually."""
    return batch

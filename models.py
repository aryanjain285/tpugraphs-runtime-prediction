"""
models.py — GNN architectures for TpuGraphs runtime prediction.

Two model families:
  1. TileModel  — for tile:xla (graph-level config features)
  2. LayoutModel — for layout collections (node-level config features)

Both share a common GNN backbone (GraphSAGE or GATv2 with residual connections
and LayerNorm) but differ in how configuration features are injected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional

from config import NUM_OPCODES, NODE_FEAT_DIM, TILE_CONFIG_DIM, LAYOUT_CONFIG_DIM


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

class CrossConfigAttention(nn.Module):
    """
    Cross-config attention from 1st place solution.
    Lets configs compare against each other during forward pass,
    rather than scoring each config independently.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (num_configs, num_nodes, hidden_dim) or (num_configs, hidden_dim)
        scores = (x / self.temperature).softmax(dim=0)
        return x * scores


class SqueezeExcitation(nn.Module):
    """
    Channel-wise attention (Squeeze-and-Excitation) from 1st place solution.
    Captures correlations between channels, suppresses less useful ones.
    """
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (..., dim)
        scale = self.fc(x)
        return x * scale

class OpcodeEmbedding(nn.Module):
    """Learnable embedding for HLO opcodes."""

    def __init__(self, num_opcodes: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_opcodes, embed_dim, padding_idx=0)

    def forward(self, opcodes: torch.Tensor) -> torch.Tensor:
        return self.embed(opcodes.clamp(0, self.embed.num_embeddings - 1))


class ResidualGNNBlock(nn.Module):
    """
    One GNN message-passing layer with residual connection and LayerNorm.

    Architecture:  h' = LayerNorm(h + Dropout(GNN(h, edge_index)))

    This prevents gradient degradation in deeper networks and handles
    the heterogeneous node feature distributions across different opcodes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gnn_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn_type = gnn_type

        if gnn_type == "sage":
            self.conv = SAGEConv(in_dim, out_dim, aggr="mean")
        elif gnn_type == "gatv2":
            assert out_dim % heads == 0, "out_dim must be divisible by heads"
            self.conv = GATv2Conv(
                in_dim, out_dim // heads, heads=heads, concat=True
            )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # If dimensions change, project the residual
        self.residual_proj = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.conv(x, edge_index)
        out = self.dropout(F.gelu(out))
        return self.norm(residual + out)


class GNNBackbone(nn.Module):
    """
    Shared GNN backbone: project features → stack of ResidualGNNBlocks.

    Input features are the concatenation of:
      - Projected node features (140-dim → hidden_dim)
      - Opcode embedding (opcode_embed_dim)
      - Topological depth (1-dim)

    These are linearly projected to hidden_dim, then processed by
    num_layers ResidualGNNBlocks.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        opcode_embed_dim: int = 64,
        gnn_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.opcode_embed = OpcodeEmbedding(NUM_OPCODES, opcode_embed_dim)

        # Input projection: node_feat(140) + opcode(embed_dim) + depth(1) → hidden_dim
        input_dim = NODE_FEAT_DIM + opcode_embed_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            ResidualGNNBlock(
                hidden_dim, hidden_dim,
                gnn_type=gnn_type, heads=heads, dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_feat: torch.Tensor,
        node_opcode: torch.Tensor,
        topo_depth: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feat : (n, 140)
        node_opcode : (n,) int
        topo_depth : (n, 1)
        edge_index : (2, m)

        Returns
        -------
        node_embeddings : (n, hidden_dim)
        """
        opcode_h = self.opcode_embed(node_opcode)           # (n, embed_dim)
        h = torch.cat([node_feat, opcode_h, topo_depth], dim=-1)  # (n, 140+embed+1)
        h = self.input_proj(h)                               # (n, hidden_dim)

        for layer in self.layers:
            h = layer(h, edge_index)

        return h


# ──────────────────────────────────────────────────────────────
# Tile Model
# ──────────────────────────────────────────────────────────────

class TileModel(nn.Module):
    """
    Runtime prediction model for tile:xla collection.

    Pipeline:
      1. GNN backbone produces node embeddings.
      2. Global mean pooling → graph embedding.
      3. For each config: concat(graph_embed, config_feat) → MLP → scalar score.

    The model outputs one score per configuration. Training optimises these
    scores so that faster configs get higher scores (for ranking losses) or
    so that scores approximate log-normalised runtimes (for MSE loss).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        opcode_embed_dim: int = 64,
        gnn_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = GNNBackbone(
            hidden_dim, num_gnn_layers, opcode_embed_dim,
            gnn_type, heads, dropout,
        )

        # Config feature projection
        self.config_proj = nn.Sequential(
            nn.Linear(TILE_CONFIG_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Score head: graph_embed + config_embed → scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data object with fields x, edge_index, node_opcode,
               topo_depth, config_feat.

        Returns
        -------
        scores : (num_configs,) predicted scores (higher = predicted faster).
        """
        device = data.x.device
        node_h = self.backbone(
            data.x, data.node_opcode, data.topo_depth, data.edge_index
        )  # (n, hidden_dim)

        # Global mean pooling (single graph, no batch vector needed)
        graph_embed = node_h.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        # Expand for each config
        config_feat = data.config_feat.to(device)  # (c, 24)
        config_h = self.config_proj(config_feat)    # (c, hidden_dim)

        graph_embed_expanded = graph_embed.expand(config_h.shape[0], -1)  # (c, hidden_dim)

        combined = torch.cat([graph_embed_expanded, config_h], dim=-1)  # (c, 2*hidden_dim)
        scores = self.score_head(combined).squeeze(-1)  # (c,)

        return scores


# ──────────────────────────────────────────────────────────────
# Layout Model
# ──────────────────────────────────────────────────────────────

class LayoutModel(nn.Module):
    """
    Improved layout model with cross-config attention and SE blocks.
    
    Key improvements over baseline:
    1. Vectorized config processing (no per-config loop)
    2. Cross-config attention lets configs compare against each other
    3. Squeeze-and-Excitation for channel-wise attention
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        opcode_embed_dim: int = 64,
        gnn_type: str = "sage",
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = GNNBackbone(
            hidden_dim, num_gnn_layers, opcode_embed_dim,
            gnn_type, heads, dropout,
        )

        # Project config features at configurable nodes
        self.config_node_proj = nn.Sequential(
            nn.Linear(hidden_dim + LAYOUT_CONFIG_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Squeeze-and-Excitation on config embeddings
        self.se = SqueezeExcitation(hidden_dim, reduction=8)
        
        # Cross-config attention
        self.cross_config_attn = CrossConfigAttention()

        # Score head: config_aware_embed + global_embed → scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _segment_forward(
        self,
        node_feat: torch.Tensor,
        node_opcode: torch.Tensor,
        topo_depth: torch.Tensor,
        edge_index: torch.Tensor,
        node_splits: torch.Tensor,
        max_segment_size: int = 5000,
    ) -> torch.Tensor:
        """
        Run the GNN backbone with graph segmentation for large graphs.

        If the graph is small enough, just run it directly.
        Otherwise, split into segments based on node_splits, process each,
        and concatenate the results.
        """
        num_nodes = node_feat.shape[0]

        if num_nodes <= max_segment_size:
            return self.backbone(node_feat, node_opcode, topo_depth, edge_index)

        # Build segments from node_splits
        split_pts = node_splits.cpu().numpy().flatten().astype(int).tolist()
        if len(split_pts) == 0 or split_pts[0] != 0:
            split_pts = [0] + split_pts
        if split_pts[-1] != num_nodes:
            split_pts.append(num_nodes)
        # Deduplicate and sort
        split_pts = sorted(set(split_pts))

        # Merge small consecutive splits to reach ~max_segment_size
        segments = []
        seg_start = split_pts[0]
        for i in range(1, len(split_pts)):
            seg_size = split_pts[i] - seg_start
            if seg_size >= max_segment_size or i == len(split_pts) - 1:
                segments.append((seg_start, split_pts[i]))
                if i < len(split_pts) - 1:
                    seg_start = split_pts[i]

        if len(segments) == 0:
            segments = [(0, num_nodes)]

        # Process each segment
        all_node_h = torch.zeros(num_nodes, self.backbone.input_proj[0].out_features,
                                 device=node_feat.device)

        for seg_start, seg_end in segments:
            # Get node indices for this segment
            seg_mask = torch.zeros(num_nodes, dtype=torch.bool, device=node_feat.device)
            seg_mask[seg_start:seg_end] = True

            # Remap edge indices to segment-local
            src, dst = edge_index
            edge_mask = seg_mask[src] & seg_mask[dst]
            seg_edges = edge_index[:, edge_mask]
            seg_edges = seg_edges - seg_start  # remap to local indices

            seg_feat = node_feat[seg_start:seg_end]
            seg_opcode = node_opcode[seg_start:seg_end]
            seg_depth = topo_depth[seg_start:seg_end]

            if seg_edges.shape[1] > 0 and seg_feat.shape[0] > 0:
                seg_h = self.backbone(seg_feat, seg_opcode, seg_depth, seg_edges)
                all_node_h[seg_start:seg_end] = seg_h

        return all_node_h

    def forward(
        self, data: Data, max_segment_size: int = 5000, config_chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Forward with cross-config attention, processed in chunks to avoid OOM.
        """
        device = data.x.device

        # Get node embeddings (with segmentation for large graphs)
        node_h = self._segment_forward(
            data.x, data.node_opcode, data.topo_depth,
            data.edge_index, data.node_splits, max_segment_size,
        )  # (n, hidden_dim)

        # Global graph embedding (config-agnostic)
        global_embed = node_h.mean(dim=0)  # (hidden_dim,)

        # Config-specific embeddings
        config_ids = data.node_config_ids.to(device)     # (nc,)
        config_feat = data.node_config_feat.to(device)   # (c, nc, 18)
        num_configs = config_feat.shape[0]

        # Gather embeddings at configurable nodes
        config_node_h = node_h[config_ids]  # (nc, hidden_dim)

        # Process configs in chunks to avoid OOM
        all_scores = []
        for start in range(0, num_configs, config_chunk_size):
            end = min(start + config_chunk_size, num_configs)
            chunk_feat = config_feat[start:end]  # (chunk, nc, 18)
            chunk_size = chunk_feat.shape[0]

            # Expand node embeddings for this chunk
            chunk_node_h = config_node_h.unsqueeze(0).expand(chunk_size, -1, -1)

            # Concatenate and project
            combined = torch.cat([chunk_node_h, chunk_feat], dim=-1)
            proj = self.config_node_proj(combined)
            
            # Squeeze-excitation
            proj = self.se(proj)
            
            # Cross-config attention within chunk
            proj = self.cross_config_attn(proj)

            # Mean over configurable nodes
            config_embeds = proj.mean(dim=1)  # (chunk, hidden_dim)

            # Score
            global_expanded = global_embed.unsqueeze(0).expand(chunk_size, -1)
            final = torch.cat([global_expanded, config_embeds], dim=-1)
            chunk_scores = self.score_head(final).squeeze(-1)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=0)  # (c,)
        return scores


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def build_tile_model(cfg) -> TileModel:
    """Construct a TileModel from a TileConfig."""
    return TileModel(
        hidden_dim=cfg.hidden_dim,
        num_gnn_layers=cfg.num_gnn_layers,
        opcode_embed_dim=cfg.opcode_embed_dim,
        gnn_type=cfg.gnn_type,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )


def build_layout_model(cfg) -> LayoutModel:
    """Construct a LayoutModel from a LayoutConfig."""
    return LayoutModel(
        hidden_dim=cfg.hidden_dim,
        num_gnn_layers=cfg.num_gnn_layers,
        opcode_embed_dim=cfg.opcode_embed_dim,
        gnn_type=cfg.gnn_type,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )

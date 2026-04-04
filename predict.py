"""
predict.py — Generate ranked predictions on the test set for all 5 collections.

Loads trained models (SWA weights if available, otherwise best checkpoint)
and produces per-collection CSV files with config rankings.

The CSV format follows Kaggle's requirement:
  - Column 1: graph_id (e.g., "tile:xla:graph_name")
  - Column 2: top-K config indices, semicolon-separated

Usage:
    python predict.py
    python predict.py --model_variant best   # use best_model.pt instead of swa
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TileConfig, LayoutConfig, ALL_LAYOUT_COLLECTIONS,
    OUTPUT_DIR, SUBMISSION_DIR, ensure_dirs,
)
from dataset import TileDataset, LayoutDataset
from models import build_tile_model, build_layout_model
from utils import get_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Generate test predictions")
    parser.add_argument("--model_variant", type=str, default="swa",
                        choices=["swa", "best"],
                        help="Which checkpoint to load: swa_model.pt or best_model.pt")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top configs to include in submission")
    return parser.parse_args()


def load_model(model, save_dir: str, variant: str, device: torch.device):
    """Load model weights from checkpoint."""
    ckpt_name = f"{variant}_model.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        # Fallback
        alt = "best_model.pt" if variant == "swa" else "swa_model.pt"
        alt_path = os.path.join(save_dir, alt)
        if os.path.exists(alt_path):
            print(f"  {ckpt_name} not found, using {alt}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {save_dir}. "
                f"Train the model first."
            )

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_tile(cfg: TileConfig, variant: str, top_k: int, device: torch.device):
    """
    Generate predictions for tile:xla test set.

    Returns a list of (graph_id, ranking_str) tuples.
    """
    print(f"\n{'='*50}")
    print(f"Predicting tile:xla")
    print(f"{'='*50}")

    model = build_tile_model(cfg).to(device)
    model = load_model(model, cfg.save_dir, variant, device)

    test_ds = TileDataset(cfg.data_dir, "test")
    print(f"  Test graphs: {len(test_ds)}")

    results = []
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="tile:xla"):
            data = test_ds[idx]
            graph_id = data.graph_id
            data = data.to(device)

            scores = model(data).cpu().numpy()

            # Rank configs by score (highest = fastest predicted)
            ranked_indices = np.argsort(-scores)[:top_k]
            ranking_str = ";".join(str(i) for i in ranked_indices)

            results.append((f"tile:xla:{graph_id}", ranking_str))

    return results


def predict_layout(
    cfg: LayoutConfig, variant: str, top_k: int, device: torch.device
):
    """
    Generate predictions for a layout sub-collection test set.

    Returns a list of (graph_id, ranking_str) tuples.
    """
    collection = cfg.collection_name
    print(f"\n{'='*50}")
    print(f"Predicting {collection}")
    print(f"{'='*50}")

    model = build_layout_model(cfg).to(device)
    model = load_model(model, cfg.save_dir, variant, device)

    test_ds = LayoutDataset(
        cfg.data_dir, "test",
        max_segment_size=cfg.max_segment_size,
    )
    print(f"  Test graphs: {len(test_ds)}")

    results = []
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc=collection):
            data = test_ds[idx]
            graph_id = data.graph_id
            data = data.to(device)

            try:
                scores = model(
                    data, max_segment_size=cfg.max_segment_size
                ).cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM on {graph_id}, using random ranking")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scores = np.random.randn(data.num_configs)
                else:
                    raise

            ranked_indices = np.argsort(-scores)[:top_k]
            ranking_str = ";".join(str(i) for i in ranked_indices)

            results.append((f"{collection}:{graph_id}", ranking_str))

    return results


def main():
    args = parse_args()
    seed_everything(42)
    ensure_dirs()

    device = get_device()
    print(f"Device: {device}")

    all_results = []

    # Tile
    tile_cfg = TileConfig()
    try:
        tile_results = predict_tile(tile_cfg, args.model_variant, args.top_k, device)
        all_results.extend(tile_results)
        print(f"  Generated {len(tile_results)} tile predictions")
    except FileNotFoundError as e:
        print(f"  Skipping tile:xla — {e}")

    # Layout collections
    for source, search in ALL_LAYOUT_COLLECTIONS:
        layout_cfg = LayoutConfig(source=source, search=search)
        try:
            layout_results = predict_layout(
                layout_cfg, args.model_variant, args.top_k, device
            )
            all_results.extend(layout_results)
            print(f"  Generated {len(layout_results)} predictions for {layout_cfg.collection_name}")
        except FileNotFoundError as e:
            print(f"  Skipping {layout_cfg.collection_name} — {e}")

    # Save individual collection CSVs
    if all_results:
        df = pd.DataFrame(all_results, columns=["ID", "TopConfigs"])
        out_path = os.path.join(SUBMISSION_DIR, "all_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(all_results)} predictions to {out_path}")
    else:
        print("\nNo predictions generated. Train models first.")


if __name__ == "__main__":
    main()

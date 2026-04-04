"""
train_tile.py — Training pipeline for the tile:xla sub-collection.

Trains a TileModel using ListMLE ranking loss with optional auxiliary MSE.
Applies Stochastic Weight Averaging in the final epochs.

Usage:
    python train_tile.py
    python train_tile.py --epochs 100 --hidden_dim 256
"""

import os
import argparse
import torch
import numpy as np


from config import TileConfig, ensure_dirs
from dataset import TileDataset
from models import build_tile_model
from losses import build_loss
from utils import (
    setup_logger, evaluate_tile, SWAAccumulator,
    seed_everything, get_device, Timer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train tile:xla model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--gnn_type", type=str, default=None, choices=["sage", "gatv2"])
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["listmle", "pairwise", "combined"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(
    model, dataset, optimizer, criterion, device, logger, epoch
):
    """Train for one epoch over all graphs in the dataset."""
    model.train()
    total_loss = 0.0
    num_graphs = len(dataset)

    indices = np.random.permutation(num_graphs)

    for step, idx in enumerate(indices):
        data = dataset[int(idx)]
        data = data.to(device)

        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, data.runtime)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping graph {idx}: loss is {loss.item()}")
            continue

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{num_graphs}, loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / max(num_graphs, 1)
    return avg_loss


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Build config, overriding with CLI args
    cfg = TileConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.gnn_type is not None:
        cfg.gnn_type = args.gnn_type
    if args.loss_type is not None:
        cfg.loss_type = args.loss_type
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    ensure_dirs()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = setup_logger("tile", os.path.join(cfg.save_dir, "train.log"))
    logger.info(f"Configuration: {cfg}")

    device = get_device()
    logger.info(f"Device: {device}")

    # Load datasets
    logger.info("Loading tile:xla datasets...")
    train_ds = TileDataset(cfg.data_dir, "train", max_configs=cfg.max_configs)
    valid_ds = TileDataset(cfg.data_dir, "valid", max_configs=None)
    logger.info(f"  Train: {len(train_ds)} graphs")
    logger.info(f"  Valid: {len(valid_ds)} graphs")

    # Build model
    model = build_tile_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.swa_lr
    )

    # Loss
    criterion = build_loss(
        cfg.loss_type,
        aux_weight=cfg.aux_mse_weight,
    )

    # SWA accumulator
    swa = None
    best_metric = 0.0

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        with Timer():
            avg_loss = train_one_epoch(
                model, train_ds, optimizer, criterion, device, logger, epoch
            )
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} — loss: {avg_loss:.5f} — "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Start SWA
        if epoch >= cfg.swa_start_epoch:
            if swa is None:
                swa = SWAAccumulator(model)
                logger.info(f"Started SWA at epoch {epoch}")
            swa.update(model)

        # Validate every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == cfg.epochs:
            metrics = evaluate_tile(model, valid_ds, device, k=cfg.top_k)
            logger.info(
                f"  Valid — slowdown@1: {metrics['slowdown_1']:.5f}, "
                f"slowdown@5: {metrics['slowdown_5']:.5f}, "
                f"tau: {metrics['kendall_tau']:.5f}, "
                f"opa: {metrics['opa']:.5f}"
            )

            if metrics["slowdown_5"] > best_metric:
                best_metric = metrics["slowdown_5"]
                torch.save(model.state_dict(),
                           os.path.join(cfg.save_dir, "best_model.pt"))
                logger.info(f"  Saved best model (slowdown@5 = {best_metric:.5f})")

    # Apply SWA weights and save
    if swa is not None:
        swa.apply(model)
        logger.info("Applied SWA weights")

        metrics = evaluate_tile(model, valid_ds, device, k=cfg.top_k)
        logger.info(
            f"SWA Valid — slowdown@1: {metrics['slowdown_1']:.5f}, "
            f"slowdown@5: {metrics['slowdown_5']:.5f}, "
            f"tau: {metrics['kendall_tau']:.5f}, "
            f"opa: {metrics['opa']:.5f}"
        )

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "swa_model.pt"))
    logger.info(f"Training complete. Models saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()

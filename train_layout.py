"""
train_layout.py — Training pipeline for layout sub-collections.

Handles all four layout collections:
  layout:xla:default, layout:xla:random,
  layout:nlp:default, layout:nlp:random

Uses pairwise ranking loss by default (pairs of configs per graph).

Usage:
    python train_layout.py --source xla --search default
    python train_layout.py --source nlp --search random --epochs 80
"""

import os
import argparse
import torch
import numpy as np


from config import LayoutConfig, ensure_dirs
from dataset import LayoutDataset
from models import build_layout_model
from losses import build_loss
from utils import (
    setup_logger, evaluate_layout, SWAAccumulator,
    seed_everything, get_device, Timer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train layout model")
    parser.add_argument("--source", type=str, required=True, choices=["xla", "nlp"])
    parser.add_argument("--search", type=str, required=True, choices=["default", "random"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--gnn_type", type=str, default=None, choices=["sage", "gatv2"])
    parser.add_argument("--loss_type", type=str, default=None,
                        choices=["pairwise", "listmle", "combined"])
    parser.add_argument("--max_configs", type=int, default=None)
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(
    model, dataset, optimizer, criterion, device, logger, epoch, cfg
):
    """Train for one epoch over all layout graphs."""
    model.train()
    total_loss = 0.0
    num_graphs = len(dataset)
    skipped = 0

    indices = np.random.permutation(num_graphs)

    for step, idx in enumerate(indices):
        data = dataset[int(idx)]

        # Skip graphs with too few configs for ranking
        if data.num_configs < 2:
            skipped += 1
            continue

        data = data.to(device)

        optimizer.zero_grad()

        try:
            scores = model(data, max_segment_size=cfg.max_segment_size)
            loss = criterion(scores, data.runtime)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM on graph {idx} ({data.num_nodes} nodes), skipping")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                skipped += 1
                continue
            raise

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Skipping graph {idx}: loss is {loss.item()}")
            skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{num_graphs}, loss: {loss.item():.4f}", flush=True)

    processed = num_graphs - skipped
    avg_loss = total_loss / max(processed, 1)
    if skipped > 0:
        logger.info(f"  Skipped {skipped}/{num_graphs} graphs")
    return avg_loss


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Build config
    cfg = LayoutConfig(source=args.source, search=args.search)
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
    if args.max_configs is not None:
        cfg.max_configs = args.max_configs
    if args.num_pairs is not None:
        cfg.num_pairs = args.num_pairs
    
    # Adjust SWA start to 90% of epochs if epochs were changed
    if args.epochs is not None:
        cfg.swa_start_epoch = max(1, int(cfg.epochs * 0.9))

    ensure_dirs()
    os.makedirs(cfg.save_dir, exist_ok=True)

    logger = setup_logger(
        cfg.collection_name,
        os.path.join(cfg.save_dir, "train.log"),
    )
    logger.info(f"Configuration: {cfg}")
    logger.info(f"Collection: {cfg.collection_name}")

    device = get_device()
    logger.info(f"Device: {device}")

    # Load datasets
    logger.info(f"Loading {cfg.collection_name} datasets...")
    train_ds = LayoutDataset(
        cfg.data_dir, "train",
        max_configs=cfg.max_configs,
        max_segment_size=cfg.max_segment_size,
    )
    valid_ds = LayoutDataset(
        cfg.data_dir, "valid",
        max_configs=None,
        max_segment_size=cfg.max_segment_size,
    )
    logger.info(f"  Train: {len(train_ds)} graphs")
    logger.info(f"  Valid: {len(valid_ds)} graphs")

    # Build model
    model = build_layout_model(cfg).to(device)
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
        margin=cfg.margin,
        num_pairs=cfg.num_pairs,
    )

    # Training
    swa = None
    best_metric = 0.0

    logger.info("=" * 60)
    logger.info(f"Starting training for {cfg.collection_name}")
    logger.info("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        with Timer():
            avg_loss = train_one_epoch(
                model, train_ds, optimizer, criterion, device, logger, epoch, cfg
            )
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{cfg.epochs} — loss: {avg_loss:.5f} — "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )

        # SWA
        if epoch >= cfg.swa_start_epoch:
            if swa is None:
                swa = SWAAccumulator(model)
                logger.info(f"Started SWA at epoch {epoch}")
            swa.update(model)

        # Validate every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == cfg.epochs:
            metrics = evaluate_layout(
                model, valid_ds, device,
                max_segment_size=cfg.max_segment_size,
            )
            logger.info(
                f"  Valid — tau: {metrics['kendall_tau']:.5f}, "
                f"opa: {metrics['opa']:.5f}"
            )

            if metrics["opa"] > best_metric:
                best_metric = metrics["opa"]
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_dir, "best_model.pt"),
                )
                logger.info(f"  Saved best model (opa = {best_metric:.5f})")

    # Apply SWA
    if swa is not None:
        swa.apply(model)
        logger.info("Applied SWA weights")

        metrics = evaluate_layout(
            model, valid_ds, device,
            max_segment_size=cfg.max_segment_size,
        )
        logger.info(
            f"SWA Valid — tau: {metrics['kendall_tau']:.5f}, "
            f"opa: {metrics['opa']:.5f}"
        )

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "swa_model.pt"))
    logger.info(f"Training complete. Models saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()

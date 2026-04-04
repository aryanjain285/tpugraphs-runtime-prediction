"""
utils.py — Metrics, SWA helpers, and logging utilities.
"""

import os
import copy
import time
import logging
import numpy as np
import torch
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that prints to console and optionally to file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s", datefmt="%H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────

def topk_slowdown(scores: np.ndarray, runtimes: np.ndarray, k: int = 5) -> float:
    """
    Top-K Slowdown metric for tile collection.

    Measures how close the best runtime among the top-K predicted configs
    is to the actual best runtime.

    Parameters
    ----------
    scores : (c,) predicted scores (higher = predicted faster).
    runtimes : (c,) normalised runtimes (lower = faster).
    k : int, number of top predictions to consider.

    Returns
    -------
    slowdown : float in [0, 1] ideally close to 1.0.
               Defined as best_runtime / min(runtimes among top-K predicted).
    """
    top_k_indices = np.argsort(-scores)[:k]  # highest scores first
    best_predicted_runtime = runtimes[top_k_indices].min()
    best_actual_runtime = runtimes.min()

    if best_predicted_runtime <= 0:
        return 0.0
    return float(best_actual_runtime / best_predicted_runtime)


def opa_score(scores: np.ndarray, runtimes: np.ndarray) -> float:
    """
    OPA (Ordered Pair Accuracy) — fraction of correctly ordered pairs.

    For all pairs (i, j) where runtime_i < runtime_j, check whether
    score_i > score_j.

    Parameters
    ----------
    scores : (c,) predicted scores.
    runtimes : (c,) ground-truth runtimes.

    Returns
    -------
    opa : float in [0, 1].
    """
    n = len(scores)
    if n < 2:
        return 1.0

    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if runtimes[i] != runtimes[j]:
                total += 1
                if (runtimes[i] < runtimes[j]) == (scores[i] > scores[j]):
                    correct += 1
    return correct / max(total, 1)


def kendall_tau(scores: np.ndarray, runtimes: np.ndarray) -> float:
    """
    Kendall's Tau correlation between predicted ranking and true ranking.

    Parameters
    ----------
    scores : (c,) predicted scores (higher = predicted faster).
    runtimes : (c,) ground-truth runtimes (lower = faster).

    Returns
    -------
    tau : float in [-1, 1].
    """
    if len(scores) < 2:
        return 1.0
    # Negate scores so that lower = better aligns with runtimes
    tau, _ = kendalltau(-scores, runtimes)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def evaluate_tile(
    model, dataset, device: torch.device, k: int = 5
) -> Dict[str, float]:
    """
    Evaluate tile model on a dataset split.

    Returns
    -------
    metrics : dict with keys 'slowdown_1', 'slowdown_5', 'slowdown_10',
              'kendall_tau', 'opa'.
    """
    model.eval()
    slowdowns = {1: [], 5: [], 10: []}
    taus = []
    opas = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            data = data.to(device)
            scores = model(data).cpu().numpy()
            runtimes = data.runtime.cpu().numpy()

            for kk in [1, 5, 10]:
                sd = topk_slowdown(scores, runtimes, k=kk)
                slowdowns[kk].append(sd)

            taus.append(kendall_tau(scores, runtimes))
            opas.append(opa_score(scores, runtimes))

    return {
        "slowdown_1": np.mean(slowdowns[1]),
        "slowdown_5": np.mean(slowdowns[5]),
        "slowdown_10": np.mean(slowdowns[10]),
        "kendall_tau": np.mean(taus),
        "opa": np.mean(opas),
    }


def evaluate_layout(
    model, dataset, device: torch.device, max_segment_size: int = 5000
) -> Dict[str, float]:
    """
    Evaluate layout model on a dataset split.

    Returns
    -------
    metrics : dict with keys 'kendall_tau', 'opa'.
    """
    model.eval()
    taus = []
    opas = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            data = data.to(device)
            scores = model(data, max_segment_size=max_segment_size).cpu().numpy()
            runtimes = data.runtime.cpu().numpy()

            taus.append(kendall_tau(scores, runtimes))
            opas.append(opa_score(scores, runtimes))

    return {
        "kendall_tau": np.mean(taus),
        "opa": np.mean(opas),
    }


# ──────────────────────────────────────────────────────────────
# Stochastic Weight Averaging (SWA)
# ──────────────────────────────────────────────────────────────

class SWAAccumulator:
    """
    Manual SWA implementation — averages model weights across epochs.

    Usage:
        swa = SWAAccumulator(model)
        for epoch in range(start, end):
            train_one_epoch(model)
            swa.update(model)
        swa.apply(model)  # overwrite model weights with SWA average
    """

    def __init__(self, model: torch.nn.Module):
        self.avg_state = copy.deepcopy(model.state_dict())
        self.n_averaged = 0

    def update(self, model: torch.nn.Module):
        """Incorporate current model weights into the running average."""
        state = model.state_dict()
        self.n_averaged += 1
        for key in self.avg_state:
            if state[key].is_floating_point():
                self.avg_state[key] = (
                    self.avg_state[key] * (self.n_averaged - 1) + state[key]
                ) / self.n_averaged

    def apply(self, model: torch.nn.Module):
        """Overwrite model parameters with the averaged weights."""
        model.load_state_dict(self.avg_state)


# ──────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Timer:
    """Simple context-manager timer."""

    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.1f}s")

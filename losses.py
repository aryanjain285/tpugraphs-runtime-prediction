"""
losses.py — Ranking loss functions for runtime prediction.

Standard regression losses (MSE) optimise pointwise accuracy, but our
evaluation metric cares about the *ordering* of configurations, not
their absolute predicted runtimes.  We implement two families of
ranking-aware losses:

  1. ListMLE   — a listwise loss that maximises the probability of the
                 correct permutation under a Plackett-Luce model.
  2. Pairwise  — a margin-based loss that penalises mis-ordered pairs.

We also provide a combined loss that adds an auxiliary MSE term for
gradient stability during early training.

References
----------
- Xia et al., "Listwise Approach to Learning to Rank," ICML 2008.
- Burges et al., "Learning to Rank with Non-smooth Cost Functions," NeurIPS 2007.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def listmle_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    ListMLE: Listwise ranking loss based on the Plackett-Luce model.

    Given predicted scores and ground-truth runtimes, this loss maximises
    the log-likelihood of sampling items in order of increasing runtime
    (i.e., the fastest config should have the highest score).

    Parameters
    ----------
    scores : (c,)  predicted scores (higher = predicted faster).
    targets : (c,) ground-truth normalised runtimes (lower = faster).

    Returns
    -------
    loss : scalar tensor.
    """
    # Sort targets in ascending order (fastest first)
    # The model should assign highest scores to the fastest configs
    sorted_indices = targets.argsort()  # ascending runtime
    sorted_scores = scores[sorted_indices]

    # ListMLE: log P(π*) = Σ_i [s_π(i) - log Σ_{j≥i} exp(s_π(j))]
    # Compute from the top of the ranking downward
    # We negate because we want highest score first but targets are ascending runtime
    # So we reverse: process from fastest (should have highest score) to slowest
    n = sorted_scores.shape[0]

    # For numerical stability, compute log-sum-exp from bottom up
    max_score = sorted_scores.max()
    shifted = sorted_scores - max_score

    # Cumulative log-sum-exp from the end
    # cumsumexp[i] = log(sum_{j=i}^{n-1} exp(shifted[j]))
    # We compute this using a reverse cumsum trick
    exp_shifted = torch.exp(shifted)
    reverse_cumsum = torch.flip(
        torch.cumsum(torch.flip(exp_shifted, [0]), dim=0), [0]
    )
    log_cumsum = torch.log(reverse_cumsum + 1e-10) + max_score

    loss = (log_cumsum - sorted_scores).mean()
    return loss


def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    num_pairs: int = 64,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss.

    Randomly sample pairs of configs where one is faster than the other,
    and penalise the model if the faster config doesn't score higher
    by at least `margin`.

    Parameters
    ----------
    scores : (c,) predicted scores (higher = predicted faster).
    targets : (c,) ground-truth runtimes (lower = faster).
    num_pairs : int, number of pairs to sample.
    margin : float, minimum score gap between correctly-ordered pairs.

    Returns
    -------
    loss : scalar tensor.
    """
    c = scores.shape[0]
    if c < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # Sample random pairs
    actual_pairs = min(num_pairs, c * (c - 1) // 2)
    idx_i = torch.randint(0, c, (actual_pairs,), device=scores.device)
    idx_j = torch.randint(0, c, (actual_pairs,), device=scores.device)

    # Ensure i != j
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % c

    # Determine which is faster (lower runtime)
    rt_i = targets[idx_i]
    rt_j = targets[idx_j]

    # target_sign: +1 if i is faster (should score higher), -1 otherwise
    target_sign = torch.sign(rt_j - rt_i)

    # Filter out ties (target_sign == 0)
    non_tie = target_sign != 0
    if non_tie.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    score_diff = scores[idx_i[non_tie]] - scores[idx_j[non_tie]]
    target_sign = target_sign[non_tie]

    # Margin ranking loss: max(0, -target_sign * score_diff + margin)
    loss = F.relu(-target_sign * score_diff + margin).mean()
    return loss


def mse_on_log_runtime(
    scores: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Auxiliary MSE loss on log-transformed normalised runtimes.

    This provides smooth gradients during early training when the ranking
    losses alone may have sparse/noisy gradients.

    We negate the scores to align with targets (lower runtime = higher score
    but lower target).

    Parameters
    ----------
    scores : (c,) predicted scores (higher = predicted faster).
    targets : (c,) ground-truth normalised runtimes (lower = faster).

    Returns
    -------
    loss : scalar tensor.
    """
    log_targets = torch.log(targets.clamp(min=1e-8))
    # Normalise targets to zero-mean unit-variance for stable MSE
    mean_t = log_targets.mean()
    std_t = log_targets.std().clamp(min=1e-6)
    norm_targets = (log_targets - mean_t) / std_t

    # Negate scores because higher score should map to lower runtime
    neg_scores = -scores
    mean_s = neg_scores.mean()
    std_s = neg_scores.std().clamp(min=1e-6)
    norm_scores = (neg_scores - mean_s) / std_s

    return F.mse_loss(norm_scores, norm_targets)


class CombinedRankingLoss(nn.Module):
    """
    Combined loss = primary_ranking_loss + aux_weight * MSE_on_log_runtime.

    Parameters
    ----------
    primary : str, one of 'listmle' or 'pairwise'.
    aux_weight : float, weight of the auxiliary MSE component.
    margin : float, margin for pairwise loss.
    num_pairs : int, pairs per graph for pairwise loss.
    """

    def __init__(
        self,
        primary: str = "listmle",
        aux_weight: float = 0.1,
        margin: float = 0.1,
        num_pairs: int = 64,
    ):
        super().__init__()
        self.primary = primary
        self.aux_weight = aux_weight
        self.margin = margin
        self.num_pairs = num_pairs

    def forward(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.primary == "listmle":
            rank_loss = listmle_loss(scores, targets)
        elif self.primary == "pairwise":
            rank_loss = pairwise_ranking_loss(
                scores, targets, self.num_pairs, self.margin
            )
        else:
            raise ValueError(f"Unknown primary loss: {self.primary}")

        if self.aux_weight > 0:
            aux_loss = mse_on_log_runtime(scores, targets)
            return rank_loss + self.aux_weight * aux_loss
        return rank_loss


def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory for loss functions."""
    if loss_type == "listmle":
        return CombinedRankingLoss(primary="listmle", aux_weight=0.0, **kwargs)
    elif loss_type == "pairwise":
        return CombinedRankingLoss(primary="pairwise", aux_weight=0.0, **kwargs)
    elif loss_type == "combined":
        return CombinedRankingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

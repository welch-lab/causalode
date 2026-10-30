"""Reconstruction and causal (DEG) losses."""
from __future__ import annotations

import torch

from .model import libsize_log1p


def reconstruction_loss(u_hat, s_hat, u_obs, s_obs) -> torch.Tensor:
    """Mean squared error of library-normalized log1p U and S."""
    return 0.5 * ((libsize_log1p(u_hat) - libsize_log1p(u_obs)).square().mean()
                  + (libsize_log1p(s_hat) - libsize_log1p(s_obs)).square().mean())


def strength_from_effect(q_mean: torch.Tensor) -> torch.Tensor:
    """Perturbation strength in [0, 1] from a group's mean effect q.

    s = |q| / max(1, 1 - q) = 1 - 1/fold: equal to q for knockdown (q in [0, 1]) and
    1 - 1/m for an m-fold activation (q = 1 - m).
    """
    return (q_mean.abs() / torch.clamp(1.0 - q_mean, min=1.0)).clamp(0.0, 1.0)


def noisy_or_response(edge_prob: torch.Tensor, strength: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """P(gene responds) = 1 - prod_j (1 - s_j p_gj) over the perturbed regulators j.

    edge_prob: genes x regulators; strength: (regulators,), 0 for regulators not perturbed.
    """
    active = strength > 0.0
    if not active.any():
        return torch.zeros(edge_prob.shape[0], device=edge_prob.device)
    selected = edge_prob[:, active].clamp(0.0, 1.0 - eps) * strength[active].unsqueeze(0)
    return (1.0 - torch.exp(torch.log1p(-selected).sum(dim=1))).clamp(0.0, 1.0)


def balanced_bce(prediction: torch.Tensor, labels: torch.Tensor, lambda_neg: float = 1.0, eps: float = 1e-6):
    """Class-balanced BCE between predicted responses and DEG labels; None if one class is absent."""
    positive = labels > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return None
    pos = -torch.log(prediction[positive].clamp_min(eps)).mean()
    neg = -torch.log1p(-prediction[negative].clamp_max(1.0 - eps)).mean()
    return 0.5 * (pos + float(lambda_neg) * neg)

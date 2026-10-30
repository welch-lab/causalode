"""Coupled RNA / regulator-activity ODE and the edge-probability link."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVITY_MODES = ("lagged", "instantaneous")


def _inverse_softplus(value: float) -> float:
    x = np.float32(max(value, 1e-6))
    return float(np.float32(x if x > 20.0 else np.log(np.expm1(x) + 1e-8)))


def libsize_log1p(x: torch.Tensor, scale: float = 1e4, eps: float = 1e-12) -> torch.Tensor:
    """log1p of library-size-normalized counts (per row)."""
    return torch.log1p(x / x.sum(dim=1, keepdim=True).clamp_min(eps) * scale)


class CausalODE(nn.Module):
    """Unspliced/spliced RNA of every gene driven by the activity of the perturbed regulators.

    For gene g and regulator j:
        dU_g/dt = alpha_g * m_g * exp(sum_j w_gj (o_gj - 1/2)) - beta_g U_g
        dS_g/dt = beta_g U_g - gamma_g S_g
        dr_j/dt = k_j (S_j / S_j^ref - r_j),   r_j(0) = 1
        o_gj    = r_j^h_gj / (1 + r_j^h_gj)
    m_g = clamp(1 - q_g, 0, max_fold) is the measured perturbation of the cell on gene g
    (q > 0 knockdown, q < 0 activation, q = 0 not targeted). Integration starts from the
    control state; U and S are updated exactly over each step with transcription held fixed,
    and r by an exact half step before and after.
    """

    def __init__(self, n_genes: int, regulator_idx, control_u0, control_s0, *, max_fold: float = 1.0,
                 elapsed: float = 300.0, time_scale: float = 25.0, ode_steps: int = 8, init_w: float = 0.05,
                 hill_init: float = 1.0, hill_range=(0.1, 4.0), kinetic_init: float = 1.0,
                 activity: str = "lagged", eps: float = 1e-8, activity_clip=(1e-6, 1e6)):
        super().__init__()
        if activity not in ACTIVITY_MODES:
            raise ValueError(f"activity must be one of {ACTIVITY_MODES}")
        if max_fold < 1.0:
            raise ValueError("max_fold must be >= 1 (1 = knockdown only)")
        if ode_steps < 1 or time_scale <= 0:
            raise ValueError("ode_steps and time_scale must be positive")
        u0 = np.asarray(control_u0, dtype=np.float32)
        s0 = np.asarray(control_s0, dtype=np.float32)
        if u0.shape != (n_genes,) or s0.shape != (n_genes,):
            raise ValueError("control states need one entry per gene")
        idx = np.asarray(regulator_idx, dtype=np.int64)
        self.n_genes, self.n_regulators = int(n_genes), len(idx)
        self.max_fold, self.elapsed, self.time_scale, self.ode_steps = float(max_fold), float(elapsed), float(time_scale), int(ode_steps)
        self.activity, self.eps = activity, float(eps)
        self.amin, self.amax = activity_clip
        self.hill_min, self.hill_max = hill_range
        self.register_buffer("regulator_idx", torch.tensor(idx))
        self.register_buffer("control_u0", torch.tensor(u0).clamp_min(0.0))
        self.register_buffer("control_s0", torch.tensor(s0).clamp_min(0.0))
        self.register_buffer("control_s_ref", torch.tensor(s0).clamp_min(self.eps))

        k0 = _inverse_softplus(kinetic_init)
        self.log_alpha = nn.Parameter(torch.full((n_genes,), k0))
        self.log_beta = nn.Parameter(torch.full((n_genes,), k0))
        self.log_gamma = nn.Parameter(torch.full((n_genes,), k0))
        self.log_activity_decay = nn.Parameter(torch.full((self.n_regulators,), float(np.log(np.expm1(1.0)))))
        self.w = nn.Parameter(torch.full((n_genes, self.n_regulators), float(init_w)))
        p = min(max((hill_init - self.hill_min) / max(self.hill_max - self.hill_min, 1e-6), 1e-4), 1 - 1e-4)
        self.hill_u = nn.Parameter(torch.full((n_genes, self.n_regulators), float(np.log(p / (1.0 - p)))))

    def rate(self, name: str) -> torch.Tensor:
        """Positive kinetic rate: 'alpha' (transcription), 'beta' (splicing), 'gamma' (mRNA decay), 'activity_decay'."""
        return F.softplus(getattr(self, f"log_{name}")) + self.eps

    def hill(self) -> torch.Tensor:
        return self.hill_min + (self.hill_max - self.hill_min) * torch.sigmoid(self.hill_u)

    def _activity_half_step(self, r, s_reg, half_delta, decay_rate, s_ref):
        target = s_reg / s_ref.clamp_min(self.eps)
        if self.activity == "instantaneous":
            return target.clamp(self.amin, self.amax)
        decay = torch.exp(-(decay_rate * half_delta).clamp(0.0, 80.0))
        return (r * decay + target * (1.0 - decay)).clamp(self.amin, self.amax)

    def _rna_step(self, u, s, alpha, beta, gamma, delta):
        exp_b = torch.exp(-(beta * delta).clamp(0.0, 80.0))
        exp_g = torch.exp(-(gamma * delta).clamp(0.0, 80.0))
        u_ss = alpha / beta.clamp_min(self.eps)
        s_ss = alpha / gamma.clamp_min(self.eps)
        u_off = u - u_ss
        gap = gamma - beta
        close = gap.abs() < 1e-4
        cross = beta / torch.where(close, torch.ones_like(gap), gap) * u_off * (exp_b - exp_g)
        cross = torch.where(close.expand_as(cross), beta * delta * u_off * exp_b, cross)
        return (u_ss + u_off * exp_b).clamp_min(0.0), (s_ss + (s - s_ss) * exp_g + cross).clamp_min(0.0)

    def forward(self, effect: torch.Tensor, elapsed: torch.Tensor | float | None = None):
        """Integrate cells with per-cell perturbation effects q (n x genes) for `elapsed` time.

        Returns raw U, S (n x genes) and a dict with the regulator activity r (n x regulators).
        """
        n = effect.shape[0]
        if elapsed is None:
            elapsed = self.elapsed
        if not torch.is_tensor(elapsed):
            elapsed = torch.full((n, 1), float(elapsed), dtype=effect.dtype, device=effect.device)
        alpha0 = self.rate("alpha").unsqueeze(0)
        beta = self.rate("beta").unsqueeze(0)
        gamma = self.rate("gamma").unsqueeze(0)
        multiplier = (1.0 - effect).clamp(0.0, self.max_fold)
        hill = self.hill().unsqueeze(0)
        reg = self.regulator_idx
        u = self.control_u0.unsqueeze(0).expand(n, -1)
        s = self.control_s0.unsqueeze(0).expand(n, -1)
        r = torch.ones((n, self.n_regulators), dtype=effect.dtype, device=effect.device)
        delta = (elapsed / self.time_scale) / float(self.ode_steps)
        decay_rate = self.rate("activity_decay").unsqueeze(0)
        s_ref = self.control_s_ref[reg].unsqueeze(0)
        for _ in range(self.ode_steps):
            r = self._activity_half_step(r, s[:, reg], delta / 2.0, decay_rate, s_ref)
            powered = torch.exp((hill * torch.log(r).unsqueeze(1)).clamp(-30.0, 30.0))
            occupancy = powered / (1.0 + powered)
            preact = torch.sum(self.w.unsqueeze(0) * (occupancy - 0.5), dim=2).clamp(-8.0, 8.0)
            u, s = self._rna_step(u, s, alpha0 * torch.exp(preact) * multiplier, beta, gamma, delta)
            r = self._activity_half_step(r, s[:, reg], delta / 2.0, decay_rate, s_ref)
        return u, s, {"activity": r}


class EdgeLink(nn.Module):
    """Edge probability p_gj = sigmoid(a |w_gj| + b) with learned slope a > 0 and bias b."""

    def __init__(self, init_slope: float = 8.0, init_bias: float = -2.0):
        super().__init__()
        self.log_slope = nn.Parameter(torch.log(torch.tensor(float(init_slope))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.exp(self.log_slope).clamp(1e-3, 1e3) * torch.abs(w) + self.bias)

"""Training loop, checkpoint selection and outputs."""
from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PerturbData
from .losses import balanced_bce, noisy_or_response, reconstruction_loss, strength_from_effect
from .model import CausalODE, EdgeLink


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    lr: float = 4e-3
    graph_lr_multiplier: float = 0.01
    grad_clip: float = 2.0
    lambda_neg: float = 1.0
    causal_weight: float = 1.0
    recon_weight: float = 1.0
    selector: str = "relaxed"
    relaxed_start_epoch: int = 10
    relaxed_threshold: float = 0.03
    relaxed_sustained: int = 3
    early_stop: bool = True
    seed: int = 1
    device: str = "auto"
    elapsed: float = 300.0
    time_scale: float = 25.0
    ode_steps: int = 8
    init_w: float = 0.05
    hill_init: float = 1.0
    kinetic_init: float = 1.0
    activity: str = "lagged"
    save_epoch_graphs: bool = False


def relaxed_stop(val_causal, epochs, threshold: float = 0.03, sustained: int = 3, start_epoch: int = 10):
    """First epoch at which the validation causal loss plateaus.

    The loss is smoothed over 3 epochs; an epoch is flat when the smoothed loss improved by less than
    `threshold` (relative) over the previous 5 epochs; the selected epoch is the first of `sustained`
    consecutive flat epochs (eligible from `start_epoch`). Returns (epoch, detection epoch, fired); if the
    rule never fires, the eligible epoch with the lowest validation causal loss.
    """
    loss = pd.Series(np.asarray(val_causal, float))
    smooth = loss.rolling(3, min_periods=3).mean().to_numpy()
    epochs = np.asarray(epochs, int)
    flat = np.zeros(len(loss), dtype=bool)
    for i in range(5, len(loss)):
        if epochs[i] < start_epoch or not np.isfinite(smooth[i - 5:i + 1]).all():
            continue
        flat[i] = (smooth[i - 5] - smooth[i]) / max(abs(smooth[i - 5]), 1e-12) < threshold
    for start in range(len(loss) - sustained + 1):
        if flat[start:start + sustained].all():
            return int(epochs[start]), int(epochs[start + sustained - 1]), True
    eligible = epochs >= start_epoch
    if not eligible.any():
        return int(epochs[-1]), int(epochs[-1]), False
    return int(epochs[eligible][np.argmin(loss.to_numpy()[eligible])]), int(epochs[-1]), False


def full_score(edge_prob: np.ndarray, regulator_idx, n_genes: int) -> np.ndarray:
    """(genes x genes) score, row = regulator, column = target; rows of unperturbed genes are 0."""
    score = np.zeros((n_genes, n_genes), dtype=np.float32)
    score[np.asarray(regulator_idx), :] = np.asarray(edge_prob).T
    np.fill_diagonal(score, 0.0)
    return score


@dataclass
class TrainResult:
    edge_prob: np.ndarray
    genes: np.ndarray
    regulator_idx: np.ndarray
    selected_epoch: int
    criterion_fired: bool
    history: pd.DataFrame
    model: CausalODE
    link: EdgeLink
    config: TrainConfig

    @property
    def regulators(self) -> np.ndarray:
        return self.genes[self.regulator_idx]

    def score_matrix(self) -> np.ndarray:
        return full_score(self.edge_prob, self.regulator_idx, len(self.genes))

    def edge_table(self) -> pd.DataFrame:
        """Regulator -> target edges ranked by probability (self edges excluded)."""
        reg = np.repeat(self.regulators[None, :], len(self.genes), axis=0)
        tgt = np.repeat(self.genes[:, None], len(self.regulator_idx), axis=1)
        t = pd.DataFrame({"regulator": reg.ravel(), "target": tgt.ravel(), "probability": self.edge_prob.ravel()})
        return t[t.regulator != t.target].sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)


def _dense(matrix, rows) -> np.ndarray:
    value = matrix[rows]
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value, dtype=np.float32)


def _resolve_device(name: str) -> torch.device:
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def build_model(data: PerturbData, config: TrainConfig) -> CausalODE:
    return CausalODE(len(data.genes), data.regulator_idx, data.control_u0, data.control_s0, max_fold=data.max_fold,
                     elapsed=config.elapsed, time_scale=config.time_scale, ode_steps=config.ode_steps,
                     init_w=config.init_w, hill_init=config.hill_init, kinetic_init=config.kinetic_init,
                     activity=config.activity)


def fit(data: PerturbData, config: TrainConfig | None = None, out_dir=None, verbose: bool = True) -> TrainResult:
    """Train CausalODE and return the graph of the selected epoch (written to `out_dir` if given)."""
    config = config or TrainConfig()
    if config.selector not in ("relaxed", "final"):
        raise ValueError("selector must be 'relaxed' or 'final'")
    has_val = len(data.val_idx) > 0
    if config.selector == "relaxed" and not has_val:
        raise ValueError("the relaxed selector needs validation groups; use selector='final' with a fixed epoch count")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)

    model = build_model(data, config).to(device)
    link = EdgeLink().to(device)
    optimizer = torch.optim.Adam([
        {"params": [model.w, *link.parameters()], "lr": config.lr * config.graph_lr_multiplier},
        {"params": [p for name, p in model.named_parameters() if name != "w"], "lr": config.lr},
    ])
    reg_idx = model.regulator_idx
    group_deg = {g: torch.tensor(v, device=device) for g, v in data.group_deg.items()}
    group_strength = {g: strength_from_effect(torch.tensor(v, device=device)[reg_idx]) for g, v in data.group_effect.items()}

    def batch_tensors(rows):
        as_t = lambda m: torch.tensor(_dense(m, rows), device=device)  # noqa: E731
        elapsed = torch.full((len(rows), 1), config.elapsed, dtype=torch.float32, device=device)
        return as_t(data.unspliced), as_t(data.spliced), as_t(data.effect), elapsed

    def one_pass(indices, train, epoch):
        model.train(train)
        link.train(train)
        rng = np.random.default_rng(config.seed * 100000 + epoch + (0 if train else 50000))
        order = rng.permutation(indices) if train else np.asarray(indices)
        totals = dict.fromkeys(("loss", "recon", "causal", "mean_prob"), 0.0)
        n_total = 0
        with torch.enable_grad() if train else torch.no_grad():
            for start in range(0, len(order), config.batch_size):
                batch = order[start:start + config.batch_size]
                u_obs, s_obs, effect, elapsed = batch_tensors(batch)
                u_hat, s_hat, _ = model(effect, elapsed)
                recon = reconstruction_loss(u_hat, s_hat, u_obs, s_obs)
                prob = link(model.w)
                causal_total, causal_cells = prob.sum() * 0.0, 0
                for group, count in Counter(data.labels[batch]).items():
                    if group not in group_deg:
                        continue
                    value = balanced_bce(noisy_or_response(prob, group_strength[group]), group_deg[group],
                                         lambda_neg=config.lambda_neg)
                    if value is not None:
                        causal_total = causal_total + int(count) * value
                        causal_cells += int(count)
                causal = causal_total / max(causal_cells, 1)
                loss = config.recon_weight * recon + config.causal_weight * causal
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {epoch}")
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([*model.parameters(), *link.parameters()], config.grad_clip)
                    optimizer.step()
                n_total += len(batch)
                for name, value in (("loss", loss), ("recon", recon), ("causal", causal), ("mean_prob", prob.mean())):
                    totals[name] += len(batch) * float(value.detach().cpu())
        return {k: v / max(n_total, 1) for k, v in totals.items()}

    def snapshot():
        with torch.no_grad():
            prob = link(model.w).cpu().numpy().astype(np.float32)
        return copy.deepcopy(model.state_dict()), copy.deepcopy(link.state_dict()), prob

    history, recent, epoch_graphs = [], deque(maxlen=config.relaxed_sustained), []
    best, fired_at, fired = None, None, False
    t_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in one_pass(data.train_idx, True, epoch).items()}}
        if has_val:
            row.update({f"val_{k}": v for k, v in one_pass(data.val_idx, False, epoch).items()})
        history.append(row)
        snap = snapshot()
        recent.append((epoch, snap))
        if config.save_epoch_graphs:
            epoch_graphs.append(snap[2])
        if verbose:
            val = f" val_causal={row['val_causal']:.5f}" if has_val else ""
            print(f"epoch {epoch:3d} loss={row['train_loss']:.5f} recon={row['train_recon']:.5f} "
                  f"causal={row['train_causal']:.5f}{val}", flush=True)
        if config.selector != "relaxed":
            continue
        if epoch >= config.relaxed_start_epoch and (best is None or row["val_causal"] < best[1]):
            best = (epoch, row["val_causal"], snap)
        if not fired:
            sel, _, fired = relaxed_stop([h["val_causal"] for h in history], [h["epoch"] for h in history],
                                         config.relaxed_threshold, config.relaxed_sustained, config.relaxed_start_epoch)
            if fired:
                fired_at = next((e, s) for e, s in recent if e == sel)
                if config.early_stop:
                    break
    if config.selector == "final":
        selected_epoch, state = history[-1]["epoch"], recent[-1][1]
    elif fired:
        selected_epoch, state = fired_at
    else:
        selected_epoch, state = best[0], best[2]
    model.load_state_dict(state[0])
    link.load_state_dict(state[1])
    result = TrainResult(edge_prob=state[2], genes=data.genes, regulator_idx=data.regulator_idx,
                         selected_epoch=int(selected_epoch), criterion_fired=bool(fired),
                         history=pd.DataFrame(history), model=model.cpu().eval(), link=link.cpu().eval(), config=config)
    if out_dir is not None:
        save_result(result, out_dir, data.info, train_seconds=time.perf_counter() - t_start,
                    epoch_graphs=np.stack(epoch_graphs) if epoch_graphs else None)
    return result


def save_result(result: TrainResult, out_dir, data_info=None, train_seconds=None, epoch_graphs=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "genes.txt").write_text("\n".join(result.genes) + "\n")
    (out / "regulators.txt").write_text("\n".join(result.regulators) + "\n")
    np.save(out / "edge_prob_target_by_regulator.npy", result.edge_prob)
    np.save(out / "score.npy", result.score_matrix())
    result.edge_table().to_csv(out / "edges.csv.gz", index=False)
    result.history.to_csv(out / "history.csv", index=False)
    if epoch_graphs is not None:
        np.save(out / "edge_prob_by_epoch.npy", epoch_graphs)
    summary = {"selected_epoch": result.selected_epoch, "criterion_fired": result.criterion_fired,
               "epochs_run": int(len(result.history)), "train_seconds": train_seconds,
               "config": asdict(result.config), "data": data_info or {}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    torch.save({"model": result.model.state_dict(), "link": result.link.state_dict(), "config": asdict(result.config),
                "genes": list(result.genes), "regulator_idx": np.asarray(result.regulator_idx).tolist(),
                "max_fold": result.model.max_fold, "selected_epoch": result.selected_epoch}, out / "model.pt")


def load_model(path):
    """(CausalODE, EdgeLink, genes, regulator_idx) from a saved model.pt."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    state = ckpt["model"]
    n = len(ckpt["genes"])
    model = CausalODE(n, ckpt["regulator_idx"], state["control_u0"].numpy(), state["control_s0"].numpy(),
                      max_fold=ckpt["max_fold"], elapsed=cfg.elapsed, time_scale=cfg.time_scale,
                      ode_steps=cfg.ode_steps, activity=cfg.activity)
    model.load_state_dict(state)
    link = EdgeLink()
    link.load_state_dict(ckpt["link"])
    return model.eval(), link.eval(), np.asarray(ckpt["genes"]), np.asarray(ckpt["regulator_idx"])

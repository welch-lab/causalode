"""Causal reach violation (CRV) score of a gene regulatory graph on held-out perturbations.

A graph is converted to density-matched edge probabilities, and for each held-out perturbation we
compute the expected fraction of its DEGs that are NOT reached from the perturbed genes within L steps
(noisy-OR over paths through perturbed genes). The same quantity for random stand-in perturbed genes
gives the baseline; R = 1 - CRV / CRV_standin (0 = no better than random regulators, 1 = every DEG reached).

Graph convention: S[i, j] = score of gene i regulating gene j (row = regulator).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DENSITIES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def ks_for(n: int, densities=DENSITIES) -> tuple:
    """Expected targets per regulator k = d * n for a density grid on an n-gene panel."""
    densities = tuple(float(d) for d in densities)
    if not densities or min(densities) <= 0 or max(densities) >= 1:
        raise ValueError("densities must lie in (0, 1)")
    return tuple(d * n for d in densities)


def scorable_mask(n: int, rows=None) -> np.ndarray:
    off = ~np.eye(n, dtype=bool)
    if rows is not None:
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray(rows)] = True
        off &= keep[:, None]
    return off


def probabilities(S, rows=None, densities=DENSITIES, ks=None) -> np.ndarray:
    """(k, n, n) density-matched edge probabilities from the ranks of |S|.

    Off-diagonal entries of the scorable rows (the perturbed genes, `rows`) are ranked; rank quantile
    q in (0, 1) becomes p = q^(1/d - 1) with d = k / (n - 1), so every regulator has k expected targets
    and only the ranking matters. Tied scores share the mean of their block. Other entries are 0.
    """
    S = np.abs(np.nan_to_num(np.asarray(S, dtype=float)))
    n = S.shape[0]
    ks = ks_for(n, densities) if ks is None else tuple(ks)
    off = scorable_mask(n, rows)
    v = S[off]
    N = len(v)
    order = np.argsort(-v, kind="stable")
    vs = v[order]
    grp = np.r_[0, np.cumsum(vs[1:] != vs[:-1])]
    cnt = np.bincount(grp)
    q = (N - np.arange(N)) / (N + 1)
    out = np.zeros((len(ks), n, n))
    tmp = np.empty(N)
    for a, k in enumerate(ks):
        d = k / (n - 1)
        tmp[order] = (np.bincount(grp, q ** (1.0 / d - 1.0)) / cnt)[grp]
        out[a][off] = tmp
    return out


def standin_draws(rng, pool, targets, B: int) -> np.ndarray:
    """(B, |T|) random stand-in target sets drawn from the pool of perturbed genes minus the targets."""
    cand = np.setdiff1d(pool, targets)
    return np.stack([rng.choice(cand, len(targets), replace=False) for _ in range(B)])


def miss_probability(P, sources, cols, steps: int = 1, rows=None) -> np.ndarray:
    """(k, B, |cols|) probability that no path of length <= steps from each source set reaches each column.

    P: (k, n, n); sources: (B, s) gene indices; rows: genes paths may pass through (default: rows of P
    with any nonzero entry, i.e. the scorable regulators). steps = 1 is the direct-edge noisy-OR.
    """
    sources = np.atleast_2d(sources)
    if steps == 1:
        return np.prod(1.0 - P[:, sources, :], axis=2)[:, :, cols]
    R = np.flatnonzero(P[-1].any(axis=1)) if rows is None else np.asarray(rows)
    src = (R[None, :, None] == sources[:, None, :]).any(-1)
    out = []
    for Pk in P:
        Prr, Prc = Pk[np.ix_(R, R)], Pk[np.ix_(R, cols)]
        r = src.astype(float)
        for _ in range(steps - 1):
            r = 1.0 - np.exp(np.log1p(-r[..., None] * Prr[None]).sum(1))
            r[src] = 1.0
        out.append(np.exp(np.log1p(-r[..., None] * Prc[None]).sum(1)))
    return np.stack(out)


def experiment_crv(P, targets, degs, steps: int = 1) -> np.ndarray:
    """(k,) expected fraction of the DEGs not reached from the targets within `steps`."""
    return miss_probability(P, np.asarray(targets)[None], np.asarray(degs), steps)[:, 0].mean(-1)


def standin_crv(P, targets, degs, draws, steps: int = 1) -> np.ndarray:
    """(k, B) CRV of each stand-in set; DEGs that are stand-in genes are dropped for that draw."""
    degs = np.asarray(degs)
    miss = miss_probability(P, draws, degs, steps)
    keep = ~(degs[None, :] == draws[:, :, None]).any(axis=1)
    with np.errstate(invalid="ignore"):
        val = (miss * keep[None]).sum(-1) / keep.sum(-1)[None]
    val[:, keep.sum(-1) == 0] = np.nan
    return val


def pooled_r(table: pd.DataFrame) -> float:
    """R = mean over k of 1 - sum_e CRV_e / sum_e CRV_standin_e (every experiment weighted equally)."""
    g = table.groupby("k")[["crv_real", "crv_standin"]].sum()
    return float((1 - g.crv_real / g.crv_standin).mean())


def crv_table(scores: dict, experiments, pool, steps=(1,), n_standins: int = 200, seed: int = 2032,
              densities=DENSITIES) -> pd.DataFrame:
    """Per (method, experiment, k, steps) CRV of the real targets and of the stand-ins.

    scores: {method: (n, n) score matrix}; experiments: list of (name, target indices, DEG indices), DEGs
    excluding the targets; pool: indices of all perturbed genes (scorable regulators and stand-in pool).
    Stand-in draws are shared by all methods and step counts.
    """
    pool = np.asarray(sorted(set(np.asarray(pool).tolist())))
    rng = np.random.default_rng(seed)
    exps = [(name, np.asarray(sorted(t)), np.asarray(sorted(d), dtype=int)) for name, t, d in experiments]
    exps = [(name, t, d, standin_draws(rng, pool, t, n_standins)) for name, t, d in exps if len(d)]
    rows = []
    for method, S in scores.items():
        n = np.asarray(S).shape[0]
        ks = ks_for(n, densities)
        P = probabilities(S, rows=pool, densities=densities)
        for L in steps:
            for name, t, d, draws in exps:
                obs = experiment_crv(P, t, d, L)
                sw = np.nanmean(standin_crv(P, t, d, draws, L), axis=1)
                rows += [{"method": method, "experiment": name, "steps": L, "k": k, "density": k / n,
                          "crv_real": o, "crv_standin": w, "R": 1 - o / w} for k, o, w in zip(ks, obs, sw)]
    return pd.DataFrame(rows)


def crv_score(scores: dict, experiments, pool, steps=(1,), **kwargs) -> pd.DataFrame:
    """Pooled R per method and step count (see crv_table for the arguments)."""
    t = crv_table(scores, experiments, pool, steps, **kwargs)
    return pd.DataFrame([{"method": m, "steps": L, "R": pooled_r(sub)} for (m, L), sub in t.groupby(["method", "steps"])])


def experiments_from_deg(deg: pd.DataFrame, groups, genes, sep: str = ",") -> list:
    """(name, target indices, DEG indices) for each group from a DEG table (perturbation, gene, is_sig[, is_target]).

    Targets are parsed from the group name; DEGs exclude the targeted genes.
    """
    gi = {g: i for i, g in enumerate(np.asarray(genes, dtype=str))}
    sig = deg[deg["is_sig"].astype(str).str.lower().isin(("true", "1", "t", "1.0"))]
    if "is_target" in sig:
        sig = sig[~sig["is_target"].astype(str).str.lower().isin(("true", "1", "t", "1.0"))]
    by_group = sig.groupby(sig["perturbation"].astype(str))["gene"].apply(list).to_dict()
    out = []
    for g in groups:
        t = sorted(gi[x] for x in str(g).split(sep) if x)
        d = sorted(gi[x] for x in by_group.get(str(g), []) if x in gi and gi[x] not in t)
        out.append((str(g), t, d))
    return out

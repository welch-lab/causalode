"""Differentially expressed genes per perturbation group (labels for the causal loss and the CRV score)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def wilcoxon_degs(adata, perturbation_key: str = "perturbation", control_label: str = "NT", layer: str = "spliced",
                  library_key: str | None = None, sep: str = ",", alpha: float = 0.05, min_cells: int = 3) -> pd.DataFrame:
    """Two-sided tie-corrected Wilcoxon rank-sum test of each group against the control cells.

    Expression = log1p(counts / library x median library); `library_key` gives each cell's library over
    all genes (default: the layer summed over adata's genes). BH correction within each group over genes.
    Returns columns perturbation, gene, pval, padj, is_sig, is_target.
    """
    X = sparse.csr_matrix(adata.layers[layer], dtype=np.float64)
    lib = adata.obs[library_key].to_numpy(float) if library_key else np.asarray(X.sum(axis=1)).ravel()
    X = sparse.diags(np.median(lib) / np.maximum(lib, 1e-12)) @ X
    X.data = np.log1p(X.data)
    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    genes = np.asarray(adata.var_names, dtype=str)
    ctrl = X[labels == control_label].toarray()
    frames = []
    for group in sorted(set(labels) - {control_label}):
        cells = labels == group
        if cells.sum() < min_cells:
            continue
        grp = X[cells].toarray()
        with np.errstate(invalid="ignore", divide="ignore"):
            p = mannwhitneyu(grp, ctrl, alternative="two-sided", method="asymptotic", use_continuity=False, axis=0).pvalue
        p = np.where(np.isfinite(p), p, 1.0)
        padj = benjamini_hochberg(p)
        targets = set(group.split(sep))
        frames.append(pd.DataFrame({"perturbation": group, "gene": genes, "pval": p, "padj": padj,
                                    "is_sig": padj < alpha, "is_target": np.isin(genes, list(targets))}))
    return pd.concat(frames, ignore_index=True)

"""Measured per-cell perturbation effects q on the targeted genes."""
from __future__ import annotations

import numpy as np
from scipy import sparse

MODALITIES = ("knockdown", "activation")


def measured_effects(spliced, library, control, target_lists, gene_index, modality: str,
                     train_cell=None, max_fold: float = 100.0):
    """Sparse (cells x genes) effect q for every targeted gene of every cell.

    ratio = (cell spliced CP10k of the target) / (mean control CP10k of the target)
      knockdown:  q = clip(1 - ratio, 0, 1);  zero control mean -> q = 1
      activation: q = min(1 - ratio, 0);      zero control mean -> q = 1 - m_fb, with
                  m_fb = max(mean, median) of min(1 - q, max_fold) over the measured
                  training-cell entries of the other targets
    Returns (effect csr, info dict).
    """
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    spliced = sparse.csr_matrix(spliced)
    library = np.asarray(library, dtype=float)
    if (library <= 0).any():
        raise ValueError("library sizes must be positive")
    control = np.asarray(control)
    ctrl_cp10k = (np.asarray(spliced[control].toarray(), dtype=np.float32) / library[control, None] * 1e4).mean(axis=0)
    rows, cols, values, zero_control = [], [], [], set()
    for row, targets in enumerate(target_lists):
        for gene in targets:
            col = gene_index[gene]
            if ctrl_cp10k[col] <= 0.0:
                zero_control.add(gene)
                value = 1.0 if modality == "knockdown" else np.nan
            else:
                ratio = float(spliced[row, col]) / library[row] * 1e4 / float(ctrl_cp10k[col])
                value = float(np.clip(1.0 - ratio, 0.0, 1.0)) if modality == "knockdown" else min(1.0 - ratio, 0.0)
            rows.append(row)
            cols.append(col)
            values.append(value)
    values = np.asarray(values, dtype=float)
    rows = np.asarray(rows, dtype=int)
    fallback = None
    if np.isnan(values).any():
        train_cell = np.ones(spliced.shape[0], bool) if train_cell is None else np.asarray(train_cell, bool)
        measured = ~np.isnan(values) & train_cell[rows]
        if not measured.any():
            raise ValueError("no measured activation entries to set the zero-control fallback")
        m = np.minimum(1.0 - values[measured], max_fold)
        fallback = float(max(m.mean(), np.median(m)))
        values[np.isnan(values)] = 1.0 - fallback
    effect = sparse.csr_matrix((values.astype(np.float32), (rows, np.asarray(cols, dtype=int))), shape=spliced.shape)
    return effect, {"zero_control_targets": sorted(zero_control), "zero_control_fold": fallback}


def group_mean_effect(effect_rows, max_fold: float) -> np.ndarray:
    """Group mean of per-cell effects, each activation first capped at the ODE bound 1 - q <= max_fold."""
    effect_rows = sparse.csr_matrix(effect_rows, copy=True)
    effect_rows.data = np.maximum(effect_rows.data, 1.0 - max_fold)
    return np.asarray(effect_rows.mean(axis=0)).ravel().astype(np.float32)

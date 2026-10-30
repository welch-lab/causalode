"""Training inputs from an AnnData object of a Perturb-seq screen."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .effects import MODALITIES, group_mean_effect, measured_effects


@dataclass
class PerturbData:
    """Everything the trainer needs. Arrays are indexed by cell (rows) and by `genes` (columns)."""

    genes: np.ndarray
    regulator_idx: np.ndarray
    spliced: sparse.csr_matrix
    unspliced: sparse.csr_matrix
    effect: sparse.csr_matrix
    labels: np.ndarray
    control_label: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    control_u0: np.ndarray
    control_s0: np.ndarray
    group_effect: dict
    group_deg: dict
    modality: str
    max_fold: float
    train_groups: list
    val_groups: list
    test_groups: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def regulators(self) -> np.ndarray:
        return self.genes[self.regulator_idx]


def split_targets(label: str, sep: str = ",") -> list:
    return [g for g in str(label).split(sep) if g and g != "nan"]


def choose_validation_groups(target_lists: dict, fraction: float = 0.1, seed: int = 2028) -> list:
    """round(fraction x groups) groups drawn at random, from the combinations (>= 2 targets) when there are enough."""
    groups = sorted(target_lists)
    n_val = int(round(fraction * len(groups)))
    combos = [g for g in groups if len(target_lists[g]) >= 2]
    pool = combos if len(combos) >= n_val else groups
    return sorted(np.random.default_rng(seed).choice(pool, n_val, replace=False).tolist()) if n_val else []


def deg_matrix(deg: pd.DataFrame, groups, genes) -> dict:
    """(genes,) 0/1 DEG label per group. With an `is_sig` column every (group, gene) row must be present;
    without it, the listed (perturbation, gene) rows are the DEGs."""
    genes = pd.Index(np.asarray(genes, dtype=str))
    out = {}
    has_sig = "is_sig" in deg.columns
    by_group = dict(tuple(deg.groupby(deg["perturbation"].astype(str))))
    for g in groups:
        part = by_group.get(g)
        if has_sig:
            if part is None:
                raise ValueError(f"no DEG rows for group {g}")
            sig = part.set_index(part["gene"].astype(str))["is_sig"].reindex(genes)
            if sig.isna().any():
                raise ValueError(f"DEG table lacks {int(sig.isna().sum())} genes for group {g}")
            values = sig.astype(str).str.lower().isin(("true", "1", "t", "1.0")).to_numpy()
        else:
            values = genes.isin([] if part is None else part["gene"].astype(str))
        out[g] = values.astype(np.float32)
    return out


def prepare_data(adata, deg: pd.DataFrame, *, modality: str, perturbation_key: str = "perturbation",
                 control_label: str = "NT", targets_key: str | None = None, sep: str = ",",
                 spliced_layer: str = "spliced", unspliced_layer: str = "unspliced", library_key: str | None = None,
                 val_groups=None, val_fraction: float = 0.1, test_groups=(), seed: int = 2028,
                 max_fold: float | None = None) -> PerturbData:
    """Build training inputs.

    adata: cells x genes with raw spliced / unspliced count layers; its genes are the modelled panel.
    deg: DEG calls per perturbation group (columns perturbation, gene[, is_sig]); the targeted genes
        themselves count as DEGs when significant.
    modality: 'knockdown' (CRISPRi / Cas13) or 'activation' (CRISPRa).
    library_key: obs column with each cell's spliced library size over all genes (recommended when adata is
        a gene panel); default = spliced counts summed over adata's genes.
    val_groups / val_fraction: perturbation groups used for checkpoint selection (default: 10% of the groups,
        combinations first). test_groups: groups removed from training and validation entirely.
    max_fold: upper bound of the transcription multiplier 1 - q (default 1 for knockdown, 100 for activation).
    """
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    max_fold = (1.0 if modality == "knockdown" else 100.0) if max_fold is None else float(max_fold)
    genes = np.asarray(adata.var_names, dtype=str)
    gene_index = {g: i for i, g in enumerate(genes)}
    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    control = labels == control_label
    if not control.any():
        raise ValueError(f"no control cells labelled {control_label!r}")
    source = adata.obs[targets_key].astype(str).to_numpy() if targets_key else labels
    target_lists = [[] if c else split_targets(t, sep) for c, t in zip(control, source)]
    missing = sorted({g for ts in target_lists for g in ts} - set(genes))
    if missing:
        raise ValueError(f"targeted genes missing from adata.var_names: {missing[:10]}")

    group_targets = {}
    for lab, ts in zip(labels, target_lists):
        if lab != control_label:
            group_targets.setdefault(lab, ts)
    test_groups = sorted(set(test_groups))
    candidates = {g: t for g, t in group_targets.items() if g not in test_groups}
    val_groups = choose_validation_groups(candidates, val_fraction, seed) if val_groups is None else sorted(set(val_groups))
    unknown = (set(val_groups) | set(test_groups)) - set(group_targets)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)[:10]}")
    if set(val_groups) & set(test_groups):
        raise ValueError("validation and test groups overlap")
    split = np.where(np.isin(labels, test_groups), "test", np.where(np.isin(labels, val_groups), "val", "train"))
    train_idx, val_idx = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    train_groups = sorted(set(labels[train_idx]) - {control_label})

    spliced = sparse.csr_matrix(adata.layers[spliced_layer], dtype=np.float32)
    unspliced = sparse.csr_matrix(adata.layers[unspliced_layer], dtype=np.float32)
    library = (adata.obs[library_key].to_numpy(float) if library_key
               else np.asarray(spliced.sum(axis=1)).ravel().astype(float))
    effect, effect_info = measured_effects(spliced, library, np.flatnonzero(control), target_lists, gene_index,
                                           modality, train_cell=split == "train", max_fold=max_fold)
    regulator_idx = np.flatnonzero(np.isin(genes, sorted({g for ts in target_lists for g in ts})))
    fit_groups = train_groups + val_groups
    group_effect = {g: group_mean_effect(effect[labels == g], max_fold) for g in fit_groups}
    info = {"modality": modality, "max_fold": max_fold, "n_cells": int(len(labels)), "n_genes": int(len(genes)),
            "n_regulators": int(len(regulator_idx)), "n_control_cells": int(control.sum()),
            "n_train_cells": int(len(train_idx)), "n_val_cells": int(len(val_idx)),
            "n_train_groups": len(train_groups), "n_val_groups": len(val_groups), "n_test_groups": len(test_groups),
            **effect_info}
    return PerturbData(
        genes=genes, regulator_idx=regulator_idx, spliced=spliced, unspliced=unspliced, effect=effect,
        labels=labels, control_label=control_label, train_idx=train_idx, val_idx=val_idx,
        control_u0=np.asarray(unspliced[control].mean(axis=0)).ravel().astype(np.float32),
        control_s0=np.asarray(spliced[control].mean(axis=0)).ravel().astype(np.float32),
        group_effect=group_effect, group_deg=deg_matrix(deg, fit_groups, genes), modality=modality,
        max_fold=max_fold, train_groups=train_groups, val_groups=val_groups, test_groups=test_groups, info=info)

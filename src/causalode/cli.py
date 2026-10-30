"""Command line: `causalode train` and `causalode crv`."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _train(a):
    import anndata as ad

    from .data import prepare_data
    from .deg import wilcoxon_degs
    from .train import TrainConfig, fit

    adata = ad.read_h5ad(a.h5ad)
    if a.genes:
        adata = adata[:, Path(a.genes).read_text().split()].copy()
    deg = pd.read_csv(a.deg) if a.deg else wilcoxon_degs(adata, a.perturbation_key, a.control, library_key=a.library_key)
    val = Path(a.val_groups).read_text().split() if a.val_groups else None
    test = Path(a.test_groups).read_text().split() if a.test_groups else ()
    data = prepare_data(adata, deg, modality=a.modality, perturbation_key=a.perturbation_key, control_label=a.control,
                        targets_key=a.targets_key, library_key=a.library_key, val_groups=val, test_groups=test,
                        val_fraction=a.val_fraction)
    cfg = TrainConfig(epochs=a.epochs, seed=a.seed, device=a.device, selector=a.selector,
                      causal_weight=a.causal_weight, early_stop=not a.no_early_stop)
    r = fit(data, cfg, out_dir=a.out)
    print(f"selected epoch {r.selected_epoch} (plateau rule fired: {r.criterion_fired}); outputs in {a.out}")


def _crv(a):
    from .crv import crv_score, crv_table, experiments_from_deg

    genes = Path(a.genes).read_text().split()
    deg = pd.read_csv(a.deg)
    groups = Path(a.groups).read_text().split()
    gi = {g: i for i, g in enumerate(genes)}
    pool = sorted({gi[t] for g in (Path(a.pool).read_text().split() if a.pool else groups) for t in g.split(",")})
    scores = {Path(p).stem if len(a.score) > 1 else "graph": np.load(p) for p in a.score}
    exps = experiments_from_deg(deg, groups, genes)
    steps = tuple(int(s) for s in a.steps.split(","))
    print(crv_score(scores, exps, pool, steps, n_standins=a.n_standins, seed=a.seed).to_string(index=False))
    if a.out:
        crv_table(scores, exps, pool, steps, n_standins=a.n_standins, seed=a.seed).to_csv(a.out, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="causalode")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="infer a graph from a Perturb-seq AnnData")
    t.add_argument("--h5ad", required=True, help="cells x genes with raw 'spliced' and 'unspliced' layers")
    t.add_argument("--modality", choices=("knockdown", "activation"), required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--deg", help="CSV with perturbation, gene, is_sig (default: Wilcoxon on spliced)")
    t.add_argument("--genes", help="text file of genes to model (default: all genes of the h5ad)")
    t.add_argument("--perturbation-key", default="perturbation")
    t.add_argument("--targets-key", help="obs column with comma-separated targets (default: parse the group label)")
    t.add_argument("--control", default="NT")
    t.add_argument("--library-key", help="obs column with each cell's spliced library over all genes")
    t.add_argument("--val-groups", help="text file of validation groups (default: 10%% of the groups)")
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--test-groups", help="text file of groups to leave out entirely")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--selector", choices=("relaxed", "final"), default="relaxed")
    t.add_argument("--causal-weight", type=float, default=1.0)
    t.add_argument("--no-early-stop", action="store_true", help="train all epochs after the selected one")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--device", default="auto")
    t.set_defaults(func=_train)
    c = sub.add_parser("crv", help="CRV score of graphs on held-out perturbation groups")
    c.add_argument("--score", nargs="+", required=True, help=".npy genes x genes matrices, row = regulator")
    c.add_argument("--genes", required=True, help="gene order of the score matrices")
    c.add_argument("--deg", required=True, help="CSV with perturbation, gene, is_sig[, is_target]")
    c.add_argument("--groups", required=True, help="text file of held-out groups to score")
    c.add_argument("--pool", help="text file of all perturbation groups (stand-in pool; default: --groups)")
    c.add_argument("--steps", default="1", help="comma-separated maximum path lengths, e.g. 1,2,3")
    c.add_argument("--n-standins", type=int, default=200)
    c.add_argument("--seed", type=int, default=2032)
    c.add_argument("--out", help="CSV of per-experiment values")
    c.set_defaults(func=_crv)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

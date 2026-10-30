"""Small simulated Perturb-seq screen with a known graph, for examples and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .model import CausalODE


def toy_dataset(n_genes: int = 200, n_regulators: int = 10, targets_per_regulator: int = 5, n_pairs: int = 15,
                cells_per_group: int = 60, n_control: int = 400, background: float = 20000.0, modality: str = "knockdown",
                seed: int = 0):
    """AnnData with raw 'spliced' / 'unspliced' counts and the true graph (regulator x gene, signed) in uns['truth'].

    The first `n_regulators` genes are the perturbed regulators; each regulates `targets_per_regulator` other genes.
    obs['library_spliced'] is each cell's spliced library over the whole transcriptome (panel + background genes);
    pass library_key='library_spliced' to wilcoxon_degs and prepare_data.
    """
    import anndata as ad

    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    reg = np.arange(n_regulators)
    truth = np.zeros((n_genes, n_genes))
    for j in reg:
        tg = rng.choice(np.arange(n_regulators, n_genes), targets_per_regulator, replace=False)
        truth[j, tg] = rng.choice([-1.0, 1.0], targets_per_regulator) * rng.uniform(1.5, 3.0, targets_per_regulator)
    model = CausalODE(n_genes, reg, np.ones(n_genes), np.ones(n_genes), max_fold=10.0, ode_steps=16)
    with torch.no_grad():
        model.log_alpha.copy_(torch.log(torch.expm1(torch.tensor(rng.uniform(2.0, 6.0, n_genes), dtype=torch.float32))))
        model.log_gamma.copy_(torch.log(torch.expm1(torch.tensor(rng.uniform(0.3, 1.0, n_genes), dtype=torch.float32))))
        model.w.copy_(torch.tensor(truth[reg].T, dtype=torch.float32))
        model.hill_u.fill_(1.0)
        zero = torch.zeros((1, n_genes))
        for _ in range(20):
            u, s, _ = model(zero, 2000.0)
            model.control_u0.copy_(u[0])
            model.control_s0.copy_(s[0])
            model.control_s_ref.copy_(s[0].clamp_min(model.eps))

    singles = [(j,) for j in reg]
    pairs = set()
    while len(pairs) < n_pairs:
        pairs.add(tuple(sorted(rng.choice(reg, 2, replace=False).tolist())))
    groups = singles + sorted(pairs)
    labels, effects = ["NT"] * n_control, [np.zeros(n_genes)] * n_control
    for g in groups:
        for _ in range(cells_per_group):
            q = np.zeros(n_genes)
            for j in g:
                q[j] = rng.beta(8, 2) if modality == "knockdown" else 1.0 - rng.uniform(3.0, 8.0)
            labels.append(",".join(genes[list(g)]))
            effects.append(q)
    with torch.no_grad():
        u, s, _ = model(torch.tensor(np.stack(effects), dtype=torch.float32))
    depth = rng.lognormal(0.0, 0.2, (len(labels), 1)) * 2.0
    U = rng.poisson(u.numpy() * depth).astype(np.float32)
    S = rng.poisson(s.numpy() * depth * 3.0).astype(np.float32)
    library = S.sum(1) + rng.poisson(depth[:, 0] * 3.0 * background)
    obs = pd.DataFrame({"perturbation": labels, "library_spliced": library.astype(float)},
                       index=[f"c{i}" for i in range(len(labels))])
    adata = ad.AnnData(X=S.copy(), obs=obs,
                       var=pd.DataFrame(index=genes), layers={"spliced": S, "unspliced": U})
    adata.uns["truth"] = truth
    return adata

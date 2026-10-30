# CausalODE

CausalODE infers a causal gene regulatory network from a Perturb-seq screen (CRISPRi, Cas13 or CRISPRa).
It fits a mechanistic ODE of unspliced and spliced RNA, in which every perturbed regulator acts through a
latent activity coupled to its own RNA, and supervises the graph with the genes each perturbation changes.
The package also provides the **causal reach violation (CRV) score**, which evaluates any inferred graph on
held-out perturbations without a reference network.

## Installation

```bash
git clone https://github.com/welch-lab/causalode.git
cd causalode
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, SciPy, pandas and anndata. A GPU is used automatically when available.

## Quick start

```python
import causalode

adata = ...  # cells x genes, raw counts in layers "spliced" and "unspliced"
deg = causalode.wilcoxon_degs(adata, perturbation_key="perturbation", control_label="NT",
                              library_key="library_spliced")
data = causalode.prepare_data(adata, deg, modality="knockdown",      # or "activation"
                              perturbation_key="perturbation", control_label="NT",
                              library_key="library_spliced")
result = causalode.fit(data, out_dir="outputs/my_screen")

result.edge_table().head()        # regulator, target, probability
S = result.score_matrix()         # genes x genes, row = regulator
```

`examples/quickstart.py` runs the full workflow, including CRV scoring, on a simulated screen with a known graph.

## Input

| Field | Requirement |
|---|---|
| `adata.layers["spliced"]`, `["unspliced"]` | raw counts, cells × genes; the genes of `adata` are the genes modelled |
| `adata.obs[perturbation_key]` | perturbation group of each cell, e.g. `"GATA1"` or `"GATA1,KLF1"`; control cells carry `control_label` |
| `targets_key` (optional) | obs column with comma-separated targets, if the group label does not list them |
| `library_key` (recommended) | obs column with each cell's spliced library size over **all** genes, when `adata` is a gene panel |
| `deg` | DEG calls per group: columns `perturbation`, `gene`, `is_sig` (or only the significant rows). A targeted gene counts as a DEG of its own perturbation when it is significant. `wilcoxon_degs` computes them. |

Every targeted gene must be in `adata.var_names`; the targeted genes are the candidate regulators.

## Model

For gene *g* and perturbed regulator *j*:

```
dU_g/dt = α_g · m_g · exp(Σ_j w_gj (o_gj − ½)) − β_g U_g
dS_g/dt = β_g U_g − γ_g S_g
dr_j/dt = k_j (S_j / S_j^ref − r_j),        r_j(0) = 1
o_gj    = r_j^h_gj / (1 + r_j^h_gj)
```

* **Perturbation.** `m_g = clamp(1 − q_g, 0, max_fold)`, with q measured in every cell from the target's
  spliced RNA relative to control: knockdown `q = clip(1 − ratio, 0, 1)`; activation `q = min(1 − ratio, 0)`
  (`max_fold` = 1 for knockdown, 100 for activation). Activation targets with no control expression get the
  typical measured fold of the other targets.
* **Cells** are integrated from the control state, so no time or pseudotime is needed.
* **Edge probability** `p_gj = sigmoid(a |w_gj| + b)`.
* **Loss** = reconstruction of library-normalized log U and S + a causal loss: for each perturbation group, the
  probability that gene *g* responds is the noisy-OR `1 − Π_j (1 − s_j p_gj)` over its perturbed regulators,
  with strength `s = |q̄| / max(1, 1 − q̄)` (= q̄ for knockdown, 1 − 1/fold for activation), fit to the DEG labels
  with a class-balanced cross-entropy. The causal loss acts on the graph only, not on the kinetic rates.
* **Checkpoint.** About 10% of the perturbation groups (combinations first) are held out for validation. Training
  stops at the first epoch where the validation causal loss plateaus (3-epoch smoothing, < 3% relative
  improvement over 5 epochs, three consecutive times, from epoch 10).

## Training options

`causalode.TrainConfig` defaults are the settings used in the paper.

| Option | Default | Meaning |
|---|---|---|
| `epochs` | 120 | maximum epochs |
| `batch_size` | 128 | cells per batch |
| `lr`, `graph_lr_multiplier` | 4e-3, 0.01 | learning rate; the graph (w and the edge link) uses lr × 0.01 |
| `grad_clip` | 2.0 | gradient-norm clip |
| `causal_weight`, `recon_weight` | 1, 1 | loss weights (`causal_weight=0`: reconstruction-only ablation) |
| `selector` | `"relaxed"` | `"relaxed"` = plateau rule on validation groups; `"final"` = last epoch (fixed-epoch training, no validation) |
| `early_stop` | True | stop once the plateau epoch is found (the selected graph is the same) |
| `activity` | `"lagged"` | `"instantaneous"` sets r = S/S_ref (no activity layer) |
| `elapsed`, `time_scale`, `ode_steps` | 300, 25, 8 | integration time and steps |
| `device` | `"auto"` | `"cuda"` if available |

## Outputs (`out_dir`)

| File | Content |
|---|---|
| `edges.csv.gz` | regulator → target edges ranked by probability |
| `edge_prob_target_by_regulator.npy` | genes × regulators edge probabilities |
| `score.npy` | genes × genes matrix, row = regulator (0 for unperturbed genes) |
| `genes.txt`, `regulators.txt` | row / column names |
| `history.csv` | losses per epoch |
| `summary.json` | selected epoch, configuration, data summary |
| `model.pt` | model and edge link at the selected epoch (`causalode.load_model`) |

```python
model, link, genes, regulator_idx = causalode.load_model("outputs/my_screen/model.pt")
U, S, aux = model(effect)          # effect: cells x genes tensor of q; aux["activity"]: regulator activity
```

## CRV score

CRV asks whether the regulators a graph assigns to a held-out perturbation explain the genes that perturbation
changed. The graph's scores are ranked and converted to edge probabilities with the same expected number of
targets k per regulator for every method, so only the ranking matters. For each held-out perturbation, CRV is
the expected fraction of its DEGs **not** reached from the perturbed genes within L steps (noisy-OR over paths
through perturbed genes). The same quantity with random perturbed genes in their place (200 stand-in draws)
is the baseline:

```
R = mean over k of [ 1 − Σ_e CRV_e / Σ_e CRV_e(stand-in) ]
```

R = 0: no better than random regulators; R = 1: every DEG is reached. k runs over a fixed density grid
(0.1%–10% of genes per regulator).

```python
experiments = causalode.experiments_from_deg(deg, held_out_groups, genes)        # DEGs exclude the targets
pool = [genes.index(g) for grp in all_groups for g in grp.split(",")]              # all perturbed genes
causalode.crv_score({"CausalODE": S, "other method": S2}, experiments, pool, steps=(1, 2, 3))
```

`crv_table` returns the per-experiment, per-k values. For a fair comparison, train on groups that exclude the
held-out perturbations (`prepare_data(..., test_groups=held_out_groups)`), and use the same experiments,
pool and seed for every method.

## Command line

```bash
causalode train --h5ad screen.h5ad --modality knockdown --library-key library_spliced \
    --test-groups held_out.txt --out outputs/screen
causalode crv --score outputs/screen/score.npy other_method.npy --genes outputs/screen/genes.txt \
    --deg deg.csv --groups held_out.txt --pool all_groups.txt --steps 1,2,3
```

## Tests

```bash
python -m unittest discover -s tests      # or: pytest
```

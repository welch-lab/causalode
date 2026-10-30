"""Infer a graph from a simulated knockdown screen and score it with CRV on held-out combinations."""
import numpy as np

import causalode
from causalode import toy

adata = toy.toy_dataset(modality="knockdown", seed=0)
deg = causalode.wilcoxon_degs(adata, library_key="library_spliced")

pairs = sorted({p for p in adata.obs["perturbation"] if "," in p})
held_out = pairs[:5]
data = causalode.prepare_data(adata, deg, modality="knockdown", library_key="library_spliced", test_groups=held_out)

result = causalode.fit(data, causalode.TrainConfig(epochs=60), out_dir="outputs/toy_knockdown")
print(f"selected epoch {result.selected_epoch}")
print(result.edge_table().head(10))

experiments = causalode.experiments_from_deg(deg, held_out, data.genes)
scores = {"CausalODE": result.score_matrix(), "true graph": adata.uns["truth"],
          "random": np.random.default_rng(0).random(result.score_matrix().shape)}
print(causalode.crv_score(scores, experiments, pool=data.regulator_idx, steps=(1, 2, 3)))

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from causalode import TrainConfig, fit, load_model, prepare_data, relaxed_stop, toy, wilcoxon_degs
from causalode.effects import measured_effects


class EffectsTest(unittest.TestCase):
    def test_knockdown_and_activation_rules(self):
        S = sparse.csr_matrix(np.array([[10, 0], [10, 0], [5, 0], [30, 2]], dtype=float))
        lib = np.full(4, 100.0)
        ctrl = np.array([0, 1])
        gi = {"A": 0, "B": 1}
        kd, info = measured_effects(S, lib, ctrl, [[], [], ["A"], ["A", "B"]], gi, "knockdown")
        self.assertAlmostEqual(kd[2, 0], 0.5)
        self.assertAlmostEqual(kd[3, 0], 0.0)
        self.assertAlmostEqual(kd[3, 1], 1.0)
        self.assertEqual(info["zero_control_targets"], ["B"])
        act, info = measured_effects(S, lib, ctrl, [[], [], ["A"], ["A", "B"]], gi, "activation")
        self.assertAlmostEqual(act[2, 0], 0.0)
        self.assertAlmostEqual(act[3, 0], -2.0)
        self.assertAlmostEqual(act[3, 1], 1.0 - info["zero_control_fold"])

    def test_relaxed_stop(self):
        loss = np.r_[np.linspace(1.0, 0.5, 15), np.full(15, 0.5)]
        epoch, _, fired = relaxed_stop(loss, np.arange(1, 31))
        self.assertTrue(fired)
        self.assertTrue(10 <= epoch <= 22)


class PipelineTest(unittest.TestCase):
    def test_toy_end_to_end(self):
        a = toy.toy_dataset(n_genes=80, n_pairs=8, cells_per_group=30, n_control=150, seed=1)
        deg = wilcoxon_degs(a, library_key="library_spliced")
        self.assertEqual(set(deg.columns), {"perturbation", "gene", "pval", "padj", "is_sig", "is_target"})
        data = prepare_data(a, deg, modality="knockdown", library_key="library_spliced")
        self.assertEqual(len(data.regulator_idx), 10)
        self.assertGreater(len(data.val_groups), 0)
        with tempfile.TemporaryDirectory() as tmp:
            r = fit(data, TrainConfig(epochs=12, device="cpu"), out_dir=tmp, verbose=False)
            self.assertEqual(r.edge_prob.shape, (80, 10))
            for f in ("model.pt", "score.npy", "edges.csv.gz", "history.csv", "summary.json"):
                self.assertTrue((Path(tmp) / f).is_file(), f)
            model, link, genes, reg = load_model(Path(tmp) / "model.pt")
            self.assertTrue(np.allclose(link(model.w).detach().numpy(), r.edge_prob, atol=1e-6))
        edges = r.edge_table()
        self.assertFalse((edges.regulator == edges.target).any())

    def test_final_selector_without_validation(self):
        a = toy.toy_dataset(n_genes=50, n_pairs=4, cells_per_group=20, n_control=80, seed=2)
        deg = wilcoxon_degs(a, library_key="library_spliced")
        data = prepare_data(a, deg, modality="knockdown", library_key="library_spliced", val_groups=[])
        with self.assertRaises(ValueError):
            fit(data, TrainConfig(epochs=2, device="cpu"), verbose=False)
        r = fit(data, TrainConfig(epochs=2, device="cpu", selector="final"), verbose=False)
        self.assertEqual(r.selected_epoch, 2)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from causalode import crv


class CRVTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 60
        self.pool = np.arange(12)
        self.truth = np.zeros((self.n, self.n))
        for j in self.pool:
            self.truth[j, rng.choice(np.arange(12, self.n), 4, replace=False)] = 1.0
        self.exps = []
        for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
            degs = np.flatnonzero(self.truth[[a, b]].any(0))
            self.exps.append((f"G{a},G{b}", [a, b], degs.tolist()))

    def test_probabilities_density_and_ties(self):
        S = np.random.default_rng(1).random((self.n, self.n))
        P = crv.probabilities(S, rows=self.pool, densities=(0.05, 0.1))
        ks = crv.ks_for(self.n, (0.05, 0.1))
        expected = P.sum(axis=2)[:, self.pool].mean(axis=1)
        self.assertTrue(np.allclose(expected, np.array(ks) * (self.n - 1) / self.n, rtol=0.05))
        self.assertTrue((P[:, 12:, :] == 0).all())
        T = np.zeros((self.n, self.n))
        T[0, 1] = T[0, 2] = 1.0
        Pt = crv.probabilities(T, rows=self.pool)
        self.assertEqual(Pt[0, 0, 1], Pt[0, 0, 2])

    def test_one_step_matches_direct_formula(self):
        P = crv.probabilities(np.random.default_rng(2).random((self.n, self.n)), rows=self.pool)
        t, d = np.array([0, 1]), np.array([20, 30, 40])
        direct = np.prod(1.0 - P[:, t, :], axis=1)[:, d].mean(axis=1)
        self.assertTrue(np.allclose(crv.experiment_crv(P, t, d, steps=1), direct))
        multi = crv.miss_probability(P, t[None], d, steps=1)
        self.assertTrue(np.allclose(multi[:, 0].mean(-1), direct))

    def test_more_steps_reach_more(self):
        P = crv.probabilities(np.random.default_rng(3).random((self.n, self.n)), rows=self.pool)
        t, d = np.array([0]), np.array([20, 30, 40])
        c1, c2, c3 = (crv.experiment_crv(P, t, d, steps=L) for L in (1, 2, 3))
        self.assertTrue((c2 <= c1 + 1e-12).all() and (c3 <= c2 + 1e-12).all())

    def test_true_graph_beats_random(self):
        rnd = np.random.default_rng(4).random((self.n, self.n))
        res = crv.crv_score({"truth": self.truth, "random": rnd}, self.exps, self.pool, steps=(1, 2), n_standins=50)
        r = res.set_index(["method", "steps"]).R
        self.assertGreater(r[("truth", 1)], 0.15)
        self.assertLess(abs(r[("random", 1)]), 0.15)
        self.assertGreater(r[("truth", 2)], r[("random", 2)])

    def test_experiments_from_deg_excludes_targets(self):
        import pandas as pd
        genes = [f"G{i}" for i in range(5)]
        deg = pd.DataFrame({"perturbation": ["G0,G1"] * 5, "gene": genes, "is_sig": [True, True, True, False, True]})
        (name, t, d), = crv.experiments_from_deg(deg, ["G0,G1"], genes)
        self.assertEqual(t, [0, 1])
        self.assertEqual(d, [2, 4])


if __name__ == "__main__":
    unittest.main()

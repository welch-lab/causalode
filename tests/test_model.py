import unittest

import numpy as np
import torch

from causalode.losses import balanced_bce, noisy_or_response, strength_from_effect
from causalode.model import CausalODE, EdgeLink


def small_model(max_fold=1.0, activity="lagged"):
    g = 6
    return CausalODE(g, [0, 1], np.full(g, 2.0), np.full(g, 5.0), max_fold=max_fold, activity=activity)


class ModelTest(unittest.TestCase):
    def test_shapes_and_finite(self):
        m = small_model()
        u, s, aux = m(torch.zeros((4, 6)))
        self.assertEqual(tuple(u.shape), (4, 6))
        self.assertEqual(tuple(aux["activity"].shape), (4, 2))
        self.assertTrue(torch.isfinite(u).all() and torch.isfinite(s).all())

    def test_knockdown_lowers_and_activation_raises_target(self):
        m = small_model(max_fold=10.0)
        q = torch.zeros((3, 6))
        q[1, 0] = 0.9
        q[2, 0] = -4.0
        _, s, aux = m(q)
        self.assertLess(s[1, 0], s[0, 0])
        self.assertGreater(s[2, 0], s[0, 0])
        self.assertLess(aux["activity"][1, 0], aux["activity"][0, 0])
        self.assertGreater(aux["activity"][2, 0], aux["activity"][0, 0])

    def test_max_fold_caps_activation(self):
        m = small_model(max_fold=1.0)
        q = torch.zeros((2, 6))
        q[1, 0] = -4.0
        _, s, _ = m(q)
        self.assertTrue(torch.allclose(s[0], s[1]))

    def test_instantaneous_activity(self):
        _, s, aux = small_model(activity="instantaneous")(torch.zeros((1, 6)))
        self.assertTrue(torch.allclose(aux["activity"], s[:, :2] / 5.0))

    def test_gradients_reach_graph_and_rates(self):
        m = small_model()
        q = torch.zeros((2, 6))
        q[1, 1] = 0.8
        _, s, _ = m(q)
        s.sum().backward()
        self.assertGreater(float(m.w.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.log_gamma.grad.abs().sum()), 0.0)


class LossTest(unittest.TestCase):
    def test_strength(self):
        q = torch.tensor([0.0, 0.5, 1.0, -1.0, -9.0])
        self.assertTrue(torch.allclose(strength_from_effect(q), torch.tensor([0.0, 0.5, 1.0, 0.5, 0.9])))

    def test_noisy_or(self):
        prob = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        out = noisy_or_response(prob, torch.tensor([1.0, 0.5]))
        self.assertTrue(torch.allclose(out, torch.tensor([1 - 0.5 * 0.75, 0.5]), atol=1e-5))
        self.assertTrue(torch.equal(noisy_or_response(prob, torch.zeros(2)), torch.zeros(2)))

    def test_balanced_bce(self):
        self.assertIsNone(balanced_bce(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 1.0])))
        good = balanced_bce(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0]))
        bad = balanced_bce(torch.tensor([0.1, 0.9]), torch.tensor([1.0, 0.0]))
        self.assertLess(good, bad)

    def test_edge_link_monotone_in_abs_w(self):
        p = EdgeLink()(torch.tensor([0.0, 0.5, -0.5, 1.0]))
        self.assertAlmostEqual(float(p[1]), float(p[2]), places=6)
        self.assertLess(float(p[0]), float(p[1]))
        self.assertLess(float(p[1]), float(p[3]))


if __name__ == "__main__":
    unittest.main()

"""
Tiny network (20 inputs, 3 hidden neurons) so the tests run in milliseconds.
Input pattern: pixels 0-4 spike every timestep, the other 15 pixels never spike.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lif_model import LIFLayer  # noqa: E402
from stdp import STDPTrainer    # noqa: E402

NEURON = {"beta": 0.9, "threshold": 50.0, "reset": "subtract", "w_norm": 2.0}
LEARN = {"threshold": 1.0, "lr": 0.05, "trace_decay": 0.8, "x_target": 2.0,
         "w_max": 1.0, "mu": 1.0, "theta_plus": 0.5}
T = 25


def pattern():
    s = torch.zeros(T, 20)
    s[:, :5] = 1.0
    return s


def make():
    layer = LIFLayer(20, 3, NEURON, seed=0)
    return layer, STDPTrainer(layer, LEARN)


class TestSTDP(unittest.TestCase):
    def test_at_most_one_spike_per_timestep(self):
        layer, trainer = make()
        self.assertLessEqual(trainer.train_sample(pattern()), T)

    def test_theta_grows_by_theta_plus_per_spike(self):
        layer, trainer = make()
        n = trainer.train_sample(pattern())
        self.assertGreater(n, 0)
        self.assertAlmostEqual(layer.theta.sum().item(), n * LEARN["theta_plus"], places=5)

    def test_weights_stay_non_negative_and_normalised(self):
        layer, trainer = make()
        for _ in range(10):
            trainer.train_sample(pattern())
        self.assertGreaterEqual(layer.W.min().item(), 0.0)
        for s in layer.W.sum(0).tolist():
            self.assertAlmostEqual(s, NEURON["w_norm"], places=4)

    def test_winner_learns_the_input_pattern(self):
        layer, trainer = make()
        for _ in range(30):
            trainer.train_sample(pattern())
        winner = int(layer.theta.argmax())
        active = layer.W[:5, winner].mean().item()
        silent = layer.W[5:, winner].mean().item()
        self.assertGreater(active, 3 * silent)

    def test_no_input_means_no_learning(self):
        layer, trainer = make()
        before = layer.W.clone()
        n = trainer.train_sample(torch.zeros(T, 20))
        self.assertEqual(n, 0)
        self.assertTrue(torch.allclose(layer.W, before))
        self.assertEqual(layer.theta.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

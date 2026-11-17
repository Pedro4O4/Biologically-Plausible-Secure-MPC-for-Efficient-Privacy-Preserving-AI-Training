"""
Run:  .venv/bin/python -m unittest discover -s tests -v

One neuron, one input, weight 0.6, beta 0.5, threshold 1.0, input spikes 1,1,1,0.
Worked by hand:
  t0: u = 0.5*0    + 0.6 = 0.60  -> no spike
  t1: u = 0.5*0.60 + 0.6 = 0.90  -> no spike
  t2: u = 0.5*0.90 + 0.6 = 1.05  -> SPIKE, reset: u = 1.05 - 1.0 = 0.05
  t3: u = 0.5*0.05 + 0    = 0.025 -> no spike
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lif_model import LIFLayer  # noqa: E402

CFG = {"beta": 0.5, "threshold": 1.0, "reset": "subtract", "w_norm": 1.0}


def one_neuron(cfg=CFG):
    layer = LIFLayer(1, 1, cfg, seed=0)
    layer.W = torch.tensor([[0.6]])
    return layer


class TestLIF(unittest.TestCase):
    def test_matches_hand_calculation(self):
        out = one_neuron().run(torch.tensor([[1.0], [1.0], [1.0], [0.0]]))
        self.assertEqual(out["spikes"].flatten().tolist(), [0, 0, 1, 0])
        for got, want in zip(out["membrane"].flatten().tolist(), [0.6, 0.9, 1.05, 0.025]):
            self.assertAlmostEqual(got, want, places=5)

    def test_reset_to_zero(self):
        layer = one_neuron({**CFG, "reset": "zero"})
        out = layer.run(torch.tensor([[1.0], [1.0], [1.0], [1.0]]))
        # after the spike at t2 the membrane is 0, so t3 is just 0.6 again
        self.assertEqual(out["spikes"].flatten().tolist(), [0, 0, 1, 0])
        self.assertAlmostEqual(out["membrane"][3, 0].item(), 0.6, places=5)

    def test_always_runs_exactly_T_steps(self):
        layer = LIFLayer(784, 100, {**CFG, "w_norm": 78.4}, seed=0)
        for T in (5, 25):
            quiet = layer.run(torch.zeros(T, 784))      # an empty image
            busy = layer.run(torch.ones(T, 784))        # a full image
            self.assertEqual(tuple(quiet["spikes"].shape), (T, 100))
            self.assertEqual(tuple(busy["spikes"].shape), (T, 100))

    def test_no_input_no_spikes(self):
        out = LIFLayer(784, 100, {**CFG, "w_norm": 78.4}, seed=0).run(torch.zeros(25, 784))
        self.assertEqual(out["spikes"].sum().item(), 0)

    def test_weights_normalised(self):
        layer = LIFLayer(784, 100, {**CFG, "w_norm": 78.4}, seed=0)
        for s in layer.W.sum(0).tolist():
            self.assertAlmostEqual(s, 78.4, places=2)


if __name__ == "__main__":
    unittest.main()

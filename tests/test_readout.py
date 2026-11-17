import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lif_model import LIFLayer                         # noqa: E402
from readout import LinearReadout, classification_metrics  # noqa: E402

NEURON = {"beta": 0.9, "threshold": 50.0, "reset": "subtract", "w_norm": 78.4}


class TestBatchedForward(unittest.TestCase):
    def test_batched_counts_match_single_runs(self):
        torch.manual_seed(0)
        layer = LIFLayer(784, 100, NEURON, seed=0)
        layer.theta = torch.rand(100) * 5          # pretend some learning happened
        T, B = 25, 6
        spikes = (torch.rand(T, B, 784) < 0.12).float()
        batched = layer.spike_counts(spikes, threshold=15.0)
        for b in range(B):
            single = layer.run(spikes[:, b, :], threshold=15.0, wta=True)["spikes"].sum(0)
            self.assertTrue(torch.equal(batched[b], single))


class TestReadout(unittest.TestCase):
    def test_learns_separable_features(self):
        g = torch.Generator().manual_seed(0)
        y = torch.randint(0, 3, (300,), generator=g)
        X = torch.zeros(300, 6)
        X[torch.arange(300), y] = 1.0                       # neuron k fires for class k
        X += 0.05 * torch.rand(300, 6, generator=g)
        ro = LinearReadout(6, 3)
        ro.fit(X, y, X, y, epochs=100, lr=0.1)
        self.assertGreater((ro.predict(X) == y).float().mean().item(), 0.98)

    def test_metrics_match_hand_calculation(self):
        # true:  0 0 0 1 1 2     predicted: 0 0 1 1 1 0
        y_true, y_pred = torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 0, 1, 1, 1, 0])
        m = classification_metrics(y_true, y_pred, num_classes=3)
        self.assertAlmostEqual(m["accuracy"], 4 / 6, places=6)
        self.assertEqual(m["confusion"], [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
        self.assertEqual([round(v, 4) for v in m["per_class"]], [0.6667, 1.0, 0.0])       # recall
        # precision: class0 2/3, class1 2/3, class2 0/0 -> 0
        self.assertAlmostEqual(m["macro_precision"], (2 / 3 + 2 / 3 + 0) / 3, places=6)


if __name__ == "__main__":
    unittest.main()

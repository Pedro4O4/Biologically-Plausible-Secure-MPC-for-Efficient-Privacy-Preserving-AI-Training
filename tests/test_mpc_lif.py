"""
Phase 6: the secure LIF layer must agree with the plaintext Phase 2 layer.
The integration test needs MP-SPDZ (replicated-ring-party.x) and is skipped without it.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import mpc_lif    # noqa: E402
import mpc_real   # noqa: E402
from lif_model import LIFLayer  # noqa: E402

NEURON = {"beta": 0.9, "threshold": 2.0, "reset": "subtract", "w_norm": 4.0}


def tiny_layer(n_in=8, n_hid=4, **over):
    return LIFLayer(n_in, n_hid, {**NEURON, **over}, seed=0)


class TestParsingAndComparison(unittest.TestCase):
    def test_parse_full_reads_raw_fixed_point_integers(self):
        text = "SPK 0: 0 1\nMEM 0: 65536 -32768\nCUR 0: 0 131072\nSPK 1: 1 0\nMEM 1: 1 2\nCUR 1: 3 4\nTime = 1 seconds\n"
        spk, mem, cur = mpc_lif.parse_full(text, T=2, H=2)
        self.assertEqual(spk.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(mem[0].tolist(), [1.0, -0.5])
        self.assertEqual(cur[0].tolist(), [0.0, 2.0])

    def test_parse_full_rejects_incomplete_output(self):
        with self.assertRaises(RuntimeError):
            mpc_lif.parse_full("SPK 0: 0 1\n", T=2, H=2)

    def test_compare_counts_mismatches_and_errors(self):
        plain = {"spikes": torch.tensor([[0., 1.], [1., 0.]]), "membrane": torch.zeros(2, 2), "current": torch.zeros(2, 2)}
        secure = {"spikes": torch.tensor([[0., 1.], [0., 0.]]), "membrane": torch.full((2, 2), 0.25), "current": torch.zeros(2, 2)}
        c = mpc_lif.compare(plain, secure)
        self.assertEqual(c["spike_mismatches"], 1)
        self.assertEqual(c["first_mismatch"], [1, 0])
        self.assertAlmostEqual(c["max_membrane_error"], 0.25)
        self.assertEqual(c["cells"], 4)

    def test_schedule_name_matches_what_mp_spdz_creates(self):
        self.assertEqual(mpc_lif.schedule_name("full", 25, 784, 100, 0.9, 50.0), "secure_lif-full-25-784-100-0.9-50.0")


class TestGuards(unittest.TestCase):
    def test_only_subtract_reset_is_supported(self):
        layer = tiny_layer(reset="zero")
        with self.assertRaises(RuntimeError):
            mpc_lif.run_secure_lif(layer, torch.zeros(3, 8))

    def test_values_outside_the_fixed_point_range_are_refused(self):
        layer = tiny_layer()
        layer.W = layer.W * 1e5
        with self.assertRaises(RuntimeError):
            mpc_lif.run_secure_lif(layer, torch.ones(3, 8))


@unittest.skipUnless(mpc_real.available()[0], "MP-SPDZ is not built")
class TestSecureEqualsPlaintext(unittest.TestCase):
    def test_tiny_layer_end_to_end(self):
        layer = tiny_layer()
        g = torch.Generator().manual_seed(3)
        spikes = (torch.rand(3, 8, generator=g) < 0.7).float()
        with mpc_real.RUN_LOCK:
            r = mpc_lif.run_secure_lif(layer, spikes)
        plain = layer.run(spikes, layer.beta, layer.threshold)
        near = int(((plain["membrane"] - layer.threshold).abs() < 0.01).sum())     # only these cells may flip
        self.assertLessEqual(r["check"]["spike_mismatches"], near)
        self.assertLess(r["check"]["max_membrane_error"], 0.01)
        self.assertLess(r["check"]["max_current_error"], 0.001)
        self.assertEqual(torch.tensor(r["plain"]["spikes"]).float().tolist(), plain["spikes"].tolist())
        parts = {p["mode"]: p for p in r["parts"]}
        self.assertEqual(parts["reset"]["per_step"]["global_bytes"], 0)       # secret bit x public WHOLE number: free
        self.assertGreater(parts["thresh"]["per_step"]["rounds"], 0)          # a comparison needs communication
        self.assertGreater(r["total"]["rounds"], parts["thresh"]["per_step"]["rounds"])

    def test_fractional_threshold_makes_the_reset_cost_communication(self):
        layer = tiny_layer(threshold=1.5)
        spikes = torch.ones(3, 8)
        with mpc_real.RUN_LOCK:
            r = mpc_lif.run_secure_lif(layer, spikes)
        parts = {p["mode"]: p for p in r["parts"]}
        self.assertGreater(parts["reset"]["per_step"]["global_bytes"], 0)     # a fractional constant needs a truncation


if __name__ == "__main__":
    unittest.main()

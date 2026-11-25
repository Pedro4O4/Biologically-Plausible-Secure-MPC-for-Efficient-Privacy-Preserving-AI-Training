"""
Phase 7: the secure STDP image must follow the plaintext STDPTrainer, and cost the same whatever the input.
The integration tests need MP-SPDZ (replicated-ring-party.x) and are skipped without it.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import mpc_real   # noqa: E402
import mpc_stdp   # noqa: E402
from lif_model import LIFLayer  # noqa: E402
from stdp import STDPTrainer    # noqa: E402

NEURON = {"beta": 0.9, "threshold": 3.0, "reset": "subtract", "w_norm": 4.0}
LEARN = {"threshold": 3.0, "lr": 0.05, "trace_decay": 0.8, "x_target": 2.0, "w_max": 1.0, "mu": 1.0, "theta_plus": 0.5}
T, N, H = 8, 16, 6


def layer(seed=1):
    return LIFLayer(N, H, NEURON, seed=seed)


def spikes(seed=5, p=0.6):
    return (torch.rand(T, N, generator=torch.Generator().manual_seed(seed)) < p).float()


class TestPlainReference(unittest.TestCase):
    def test_step_by_step_copy_equals_the_real_trainer(self):
        for seed in range(6):
            a, b = layer(seed), layer(seed)
            sp = spikes(seed + 10)
            out = mpc_stdp.plain_image(a, LEARN, sp)
            STDPTrainer(b, LEARN).train_sample(sp)
            self.assertTrue(torch.allclose(out["W"], b.W, atol=1e-6))
            self.assertTrue(torch.allclose(out["theta"], b.theta))

    def test_the_given_model_is_not_modified(self):
        a = layer()
        w, th = a.W.clone(), a.theta.clone()
        mpc_stdp.plain_image(a, LEARN, spikes())
        self.assertTrue(torch.equal(a.W, w) and torch.equal(a.theta, th))


class TestParsingAndGuards(unittest.TestCase):
    def test_parse_full(self):
        text = "NWIN 1\nWIN 0: 0 0 1\nWIN 1: 0 0 0\nTHETA: 0 32768 65536\nW 0: 65536 0 32768\nW 1: 1 2 3\n"
        winners, th, W = mpc_stdp.parse_full(text, T=2, N=2, H=3)
        self.assertEqual(winners, [2, -1])
        self.assertEqual(th.tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(W[0].tolist(), [1.0, 0.0, 0.5])

    def test_parse_rejects_two_winners_in_one_step(self):
        with self.assertRaises(RuntimeError):
            mpc_stdp.parse_full("NWIN 1\nWIN 0: 1 1 0\nTHETA: 0 0 0\nW 0: 0 0 0\n", T=1, N=1, H=3)

    def test_compare_counts_wrong_winners(self):
        plain = {"winners": [-1, 2, -1], "W": torch.zeros(2, 3), "theta": torch.zeros(3)}
        c = mpc_stdp.compare(plain, [-1, 1, -1], torch.zeros(3), torch.full((2, 3), 0.001))
        self.assertEqual((c["winner_mismatches"], c["first_mismatch"]), (1, 1))
        self.assertAlmostEqual(c["max_weight_error"], 0.001, places=6)

    def test_only_a_soft_bound_exponent_of_one_is_supported(self):
        with self.assertRaises(RuntimeError):
            mpc_stdp.run_secure_stdp(layer(), {**LEARN, "mu": 2.0}, spikes())

    def test_values_outside_the_fixed_point_range_are_refused(self):
        a = layer()
        a.theta = a.theta + 40000.0
        with self.assertRaises(RuntimeError):
            mpc_stdp.run_secure_stdp(a, LEARN, spikes())


@unittest.skipUnless(mpc_real.available()[0], "MP-SPDZ is not built")
class TestSecureFollowsPlaintext(unittest.TestCase):
    def test_tiny_image_end_to_end(self):
        a, sp = layer(), spikes()
        with mpc_real.RUN_LOCK:
            r = mpc_stdp.run_secure_stdp(a, LEARN, sp)
        plain = mpc_stdp.plain_image(a, LEARN, sp)
        gaps = [float((s["u"] - (LEARN["threshold"] + a.theta)).abs().min()) for s in plain["steps"]]
        if min(gaps) > 0.01:                                  # a spike decision can only flip within the rounding error
            self.assertEqual(r["secure"]["winners"], plain["winners"])
            self.assertEqual(r["check"]["winner_mismatches"], 0)
        self.assertLess(r["check"]["max_weight_error"], 1e-3)
        self.assertEqual(r["plain"]["winners"], plain["winners"])
        parts = {p["mode"]: p for p in r["parts"]}
        self.assertEqual(parts["theta"]["per_unit"]["global_bytes"], 0)      # secret bit x public integer: local
        self.assertGreater(parts["wta"]["per_unit"]["rounds"], parts["thresh"]["per_unit"]["rounds"])
        self.assertLess(parts["current"]["per_unit"]["global_bytes"], r["current_trunc"]["global_bytes"])
        self.assertGreater(r["total_nearest"]["rounds"], r["total"]["rounds"])           # exact truncation costs more
        self.assertGreater(r["naive_clamp"]["global_bytes"], parts["clamp"]["per_unit"]["global_bytes"])

    def test_cost_does_not_depend_on_the_input(self):
        a = layer()
        costs = []
        for sp in (spikes(5, 0.6), torch.zeros(T, N), torch.ones(T, N)):
            with mpc_real.RUN_LOCK:
                r = mpc_stdp.run_secure_stdp(a, LEARN, sp)
            costs.append((r["total"]["rounds"], r["total"]["global_bytes"], r["check"]["winning_steps_plain"]))
        self.assertEqual(len({c[:2] for c in costs}), 1, costs)             # same rounds and bytes
        self.assertGreater(len({c[2] for c in costs}), 1, costs)            # although the number of winning steps differs


if __name__ == "__main__":
    unittest.main()

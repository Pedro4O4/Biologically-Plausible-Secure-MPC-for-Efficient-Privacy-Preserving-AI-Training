"""
The secret-sharing simulation must give the right answers, count its own cost correctly,
and every single party's view must be independent of the secret.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import mpc_sim as m  # noqa: E402


def signed(v, k):
    v %= 1 << k
    return v - (1 << k) if v >= 1 << (k - 1) else v


class TestCorrectness(unittest.TestCase):
    def test_add_matches_ring_arithmetic(self):
        rng = random.Random(1)
        for k in (8, 16, 32, 64):
            for _ in range(60):
                a, b = (rng.randrange(-(1 << (k - 1)), 1 << (k - 1)) for _ in range(2))
                r = m.run("add", a, b, k, seed=rng.randrange(10**6))
                self.assertEqual(int(r["result"]), signed(a + b, k))
                self.assertTrue(r["ok"])

    def test_mul_matches_ring_arithmetic_including_wraparound(self):
        rng = random.Random(2)
        for k in (8, 16, 32, 64):
            for _ in range(60):
                a, b = (rng.randrange(-(1 << (k - 1)), 1 << (k - 1)) for _ in range(2))
                r = m.run("mul", a, b, k, seed=rng.randrange(10**6))
                self.assertEqual(int(r["result"]), signed(a * b, k))
        self.assertTrue(m.run("mul", 100, 100, 8)["wrapped"])     # 10000 does not fit in 8 bits
        self.assertFalse(m.run("mul", 7, 12, 8)["wrapped"])

    def test_compare_all_edges_and_random_8_bit(self):
        k = 8
        lo, hi = -128, 127
        cases = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (lo, -1), (hi, 0), (0, lo + 1), (-5, 5), (5, -5),
                 (63, -64), (-64, 63), (hi, 1), (lo, 0)]
        rng = random.Random(3)
        while len(cases) < 400:
            a, b = rng.randint(lo, hi), rng.randint(lo, hi)
            if lo <= a - b <= hi:
                cases.append((a, b))
        for a, b in cases:
            r = m.run("cmp", a, b, k, seed=a * 1000 + b)
            self.assertEqual(int(r["result"]), int(a < b), (a, b))

    def test_compare_wide_rings(self):
        rng = random.Random(4)
        for k in (16, 32, 64):
            for _ in range(40):
                a = rng.randint(-(1 << (k - 2)), (1 << (k - 2)) - 1)
                b = rng.randint(-(1 << (k - 2)), (1 << (k - 2)) - 1)
                self.assertEqual(int(m.run("cmp", a, b, k, seed=7)["result"]), int(a < b))

    def test_out_of_range_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            m.run("add", 200, 1, 8)
        with self.assertRaises(ValueError):
            m.run("cmp", 100, -100, 8)        # a - b = 200 does not fit: the sign bit would lie
        with self.assertRaises(ValueError):
            m.run("sqrt", 1, 1, 8)

    def test_same_seed_same_shares(self):
        a = m.run("mul", 5, 9, 8, seed=11)["steps"]
        b = m.run("mul", 5, 9, 8, seed=11)["steps"]
        self.assertEqual(a, b)
        self.assertNotEqual(a, m.run("mul", 5, 9, 8, seed=12)["steps"])


class TestCost(unittest.TestCase):
    def test_addition_is_free(self):
        c = m.run("add", 3, 4, 8)["cost"]["op"]
        self.assertEqual((c["rounds"], c["messages"], c["bits"]), (0, 0, 0))

    def test_multiplication_costs_one_round_and_three_ring_elements(self):
        for k in (8, 16, 64):
            c = m.run("mul", 3, 4, k)["cost"]["op"]
            self.assertEqual((c["rounds"], c["messages"], c["bits"]), (1, 3, 3 * k))

    def test_counted_cost_equals_formula(self):
        for k in (8, 16, 32, 64):
            r = m.run("cmp", 1, 2, k)
            self.assertEqual(r["cost"]["op"]["rounds"], r["formula"]["rounds"])
            self.assertEqual(r["cost"]["op"]["bits"], r["formula"]["bits"])
            self.assertEqual(r["and_bits"], r["formula"]["and_bits"])
            self.assertEqual(r["cost"]["op"]["messages"], r["formula"]["messages"])

    def test_comparison_dwarfs_multiplication(self):
        self.assertGreater(m.cost_compare(64)["bits"], 10 * m.cost_mul(64)["bits"])   # 2175 vs 192 bits
        self.assertGreater(m.cost_compare(64)["rounds"], 5 * m.cost_mul(64)["rounds"])

    def test_input_sharing_runs_in_one_shared_round(self):
        c = m.run("add", 3, 4, 8)["cost"]
        self.assertEqual(c["input"]["rounds"], 1)
        self.assertEqual(c["input"]["messages"], 4)     # two owners, two messages each
        self.assertEqual(c["reveal"]["rounds"], 1)


class TestSharingStructure(unittest.TestCase):
    def test_every_step_is_complete(self):
        for op in ("add", "mul", "cmp"):
            for s in m.run(op, 5, 9, 8)["steps"]:
                self.assertEqual(len(s["parties"]), 3)
                self.assertTrue(s["title"] and s["why"])
                for msg in s["msgs"]:
                    self.assertNotEqual(msg["src"], msg["dst"])
                self.assertEqual(s["local"], s["round"] is None)

    def test_local_steps_send_nothing(self):
        for op in ("add", "mul", "cmp"):
            for s in m.run(op, 5, 9, 8)["steps"]:
                if s["local"]:
                    self.assertEqual(s["msgs"], [])


class TestSecrecyOfOneParty(unittest.TestCase):
    def test_view_does_not_depend_on_the_secret(self):
        # Party 0's two pieces are uniform whatever the secret is: both histograms are flat.
        d = m.secrecy_demo(8, 37, 100, n=8000, seed=5)
        expect = d["expected_per_bin"]
        sigma = (expect * (1 - 1 / d["bins"])) ** 0.5
        for h in (d["hist_a"], d["hist_b"]):
            for c in h:
                self.assertLess(abs(c - expect), 6 * sigma)

    def test_two_pieces_of_a_sharing_are_independent_of_the_secret(self):
        # With the secret fixed, (x0, x1) are drawn without looking at it: only x2 depends on the secret.
        k = 8
        for secret in (0, 1, 200):
            seen = set()
            for seed in range(200):
                sim = m.MPC(k, seed)
                X = sim.share_input(0, secret, "a")
                seen.add(X.sh[0])
            self.assertGreater(len(seen), 100)     # many different values: not determined by the secret


class TestSnnPreview(unittest.TestCase):
    def test_rows_use_the_formulas(self):
        p = m.snn_preview(25, 100, 784, 64)
        rows = {r["name"]: r for r in p["rows"]}
        thr = rows["Threshold test u ≥ θ"]
        self.assertEqual(thr["count"], 2500)
        self.assertEqual(thr["bits"], 2500 * m.cost_compare(64)["bits"])
        self.assertEqual(rows["Leak and add: u = β·u + I"]["bits"], 0)
        stdp = rows["STDP update, secret winner"]
        self.assertEqual(stdp["count"], 2 * 784 * 100 * 25)


if __name__ == "__main__":
    unittest.main()

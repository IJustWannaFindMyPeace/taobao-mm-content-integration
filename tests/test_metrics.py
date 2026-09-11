import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from metrics import auc, evaluate


class MetricContracts(unittest.TestCase):
    def test_known_rankings_and_ties(self):
        self.assertEqual(auc([0, 1], [0, 1]), 1)
        self.assertEqual(auc([0, 1], [1, 0]), 0)
        self.assertEqual(auc([0, 1], [1, 1]), .5)
        self.assertAlmostEqual(auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)

    def test_matches_pairwise_oracle(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            y = np.r_[0, 1, rng.integers(0, 2, 20)]
            s = rng.integers(-3, 4, len(y))
            delta = s[y == 1, None] - s[y == 0]
            oracle = np.mean((delta > 0) + .5 * (delta == 0))
            self.assertAlmostEqual(auc(y, s), oracle)

    def test_gauc_weight_and_exclusions(self):
        r = evaluate([0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 1], ['a','a','b','b','b','c'])
        self.assertAlmostEqual(r['gauc'], 2/5)
        self.assertEqual((r['valid_users'], r['valid_rows'], r['excluded_rows']), (2, 5, 1))

    def test_batch_boundaries_and_permutation(self):
        y = np.array([0, 0, 1, 1, 0]); s = np.array([0., 1., 2., 3., 4.]); u = np.array([1, 2, 1, 2, 3])
        expected = evaluate(y, s, u)
        for size in [1, 2, 3, 4, 8]:
            arrays = [np.concatenate([a[i:i+size] for i in range(0, len(a), size)]) for a in [y,s,u]]
            self.assertEqual(evaluate(*arrays), expected)
        order = [4, 2, 0, 3, 1]
        self.assertAlmostEqual(evaluate(y[order], s[order], u[order])['gauc'], expected['gauc'])

    def test_undefined_and_extreme_logits(self):
        r = evaluate([1,1], [1000., -1000.], [1,2])
        self.assertIsNone(r['auc']); self.assertIsNone(r['gauc'])
        self.assertAlmostEqual(r['logloss'], 500)

    def test_reject_invalid_input(self):
        for y,s in [([],[]), ([2],[0]), ([1],[float('nan')]), ([1],[0,1])]:
            with self.assertRaises(ValueError): auc(y,s)


if __name__ == '__main__':
    unittest.main()

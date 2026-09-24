"""Regression checks for the independent audit's grouped rank arithmetic."""
import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts'))
sys.path.insert(0, str(root/'src'))
from audit_cross_project_completion import independent_metrics
from metrics import evaluate


class IndependentMetrics(unittest.TestCase):
    def test_ties_and_single_class_users(self):
        result = independent_metrics([0, 1, 1, 1], [2, 2, 0, 1], [0, 0, 1, 1])
        self.assertEqual(result['gauc'], 0.5)
        self.assertEqual(result['valid_users'], 1)
        self.assertEqual(result['valid_rows'], 2)
        self.assertEqual(result['excluded_rows'], 2)

    def test_exposure_weighting(self):
        # Perfect group of size 2 and reversed group of size 4: GAUC = 2/6.
        result = independent_metrics([0, 1, 0, 0, 1, 1], [0, 1, 3, 4, 1, 2], [0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(result['gauc'], 1/3)

    def test_random_tied_unsorted_groups_match_reference(self):
        rng = np.random.default_rng(901)
        for _ in range(20):
            labels = rng.integers(0, 2, 120)
            scores = rng.integers(-3, 4, 120)
            users = rng.integers(0, 20, 120)
            independent = independent_metrics(labels, scores, users)
            reference = evaluate(labels, scores, users)
            for key, value in independent.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, reference[key], places=12)
                else:
                    self.assertEqual(value, reference[key])


if __name__ == '__main__':
    unittest.main()

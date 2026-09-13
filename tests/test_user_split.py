import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from user_split import user_masks, mix64


class UserSplitContracts(unittest.TestCase):
    def test_nested_sizes_preserve_roles(self):
        ids = np.arange(-100000, 100000, dtype=np.int64)
        for role in [0, 1]:
            a, b, c, d = [user_masks(ids, size)[role] for size in [1, 5, 10, 50]]
            self.assertFalse(np.any(a & ~b))
            self.assertFalse(np.any(b & ~c))
            self.assertFalse(np.any(c & ~d))

    def test_known_mix_value(self):
        self.assertEqual(int(mix64([0], 0)[0]), 0xE220A8397B1DCDAF)

    def test_cohorts_disjoint_and_batch_independent(self):
        ids = np.arange(-100000, 100000, dtype=np.int64)
        train, validation = user_masks(ids)
        self.assertFalse(np.any(train & validation))
        self.assertGreater(train.sum(), 0)
        self.assertGreater(validation.sum(), 0)
        for index in [0, 1]:
            pieces = np.concatenate([user_masks(x)[index] for x in np.array_split(ids, 7)])
            np.testing.assert_array_equal(pieces, (train, validation)[index])
        np.testing.assert_array_equal(user_masks(ids[::-1])[0][::-1], train)


if __name__ == '__main__': unittest.main()

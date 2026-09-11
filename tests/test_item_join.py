import sys
import unittest
from pathlib import Path
import pyarrow as pa
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from item_join import attach_items


class ItemJoinContracts(unittest.TestCase):
    def test_repeated_exposures_preserve_order(self):
        items = pa.table({'205': [-5, 7], '206': [9, 8], '213': [1, 2], '214': [3, 4]})
        result = attach_items(pa.table({'205': [7, -5, 7]}), items)
        self.assertEqual(result['206'].to_pylist(), [8, 9, 8])

    def test_reject_duplicate_and_missing_keys(self):
        for keys in [[1, 1], [1, 2]]:
            items = pa.table({'205': keys, '206': [9, 8], '213': [1, 2], '214': [3, 4]})
            with self.assertRaises(ValueError): attach_items(pa.table({'205': [3]}), items)


if __name__ == '__main__': unittest.main()

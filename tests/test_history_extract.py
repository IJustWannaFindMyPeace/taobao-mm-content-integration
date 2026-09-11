import sys
import tempfile
import unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extract_histories import extract


class HistoryExtractContracts(unittest.TestCase):
    def run_case(self, ids, wanted):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.parquet', Path(directory)/'out.parquet'
            pq.write_table(pa.table({'129_1':ids, '150_2_180': [[1,2] for _ in ids], '151_2_180': [[3,4] for _ in ids]}), source)
            result = extract(source, wanted, output)
            self.assertEqual(set(pq.read_table(output)['129_1'].to_pylist()), wanted)
            return result

    def test_exact_selection(self):
        self.assertEqual(self.run_case([-3,4,5], {-3,5})['selected_users'],2)

    def test_missing_user_rejected(self):
        with self.assertRaises(ValueError): self.run_case([1,2], {3})

    def test_duplicate_user_rejected(self):
        with self.assertRaises(ValueError): self.run_case([1,1], {1})


if __name__ == '__main__': unittest.main()

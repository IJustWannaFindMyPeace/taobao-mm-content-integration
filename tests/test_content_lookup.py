import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_content import positions


class ContentLookupContracts(unittest.TestCase):
    def test_signed_missing_and_upper_bound(self):
        requested=np.array([-8,2,9],dtype=np.int64)
        idx,valid=positions(np.array([9,-8,11,0,2]),requested)
        self.assertEqual(valid.tolist(),[True,True,False,False,True])
        self.assertEqual(idx[valid].tolist(),[2,0,1])


if __name__ == '__main__': unittest.main()

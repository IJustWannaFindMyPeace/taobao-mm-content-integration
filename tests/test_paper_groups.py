import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_paper_groups import balanced_groups


class BalancedGroups(unittest.TestCase):
    def test_order_independent_ties_and_balance(self):
        ids=np.arange(-10,15);values=np.zeros(len(ids),dtype=int)
        groups=balanced_groups(ids,values,8)
        order=np.random.default_rng(42).permutation(len(ids))
        np.testing.assert_array_equal(balanced_groups(ids[order],values[order],8),groups[order])
        sizes=np.bincount(groups);self.assertLessEqual(sizes.max()-sizes.min(),1)

    def test_numeric_value_is_primary(self):
        ids=np.arange(40);values=np.arange(40)[::-1]
        groups=balanced_groups(ids,values,8)
        for g in range(7):self.assertLess(values[groups==g].max(),values[groups==g+1].min())


if __name__=='__main__':unittest.main()

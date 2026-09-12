import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c0 import C0
from simtier import C3,histogram
from feature_store import FIELDS


class SimTierContracts(unittest.TestCase):
    def test_endpoints_missing_and_permutation(self):
        target=torch.tensor([[1.,0.]])
        history=torch.tensor([[[-1.,0.],[0.,1.],[1.,0.],[0.,0.]]])
        valid=torch.ones((1,4),dtype=torch.bool)
        count=histogram(target,history,valid,torch.tensor([True]),valid,bins=20)
        self.assertEqual(count.sum().item(),3)
        self.assertEqual(count[0,[0,10,19]].tolist(),[1,1,1])
        order=[2,3,0,1]
        torch.testing.assert_close(count,histogram(target,history[:,order],valid,torch.tensor([True]),valid,bins=20))
        self.assertEqual(histogram(target,history,valid,torch.tensor([False]),valid).sum().item(),0)

    def test_common_initialization_and_added_gradient(self):
        sizes={f:12 for f in FIELDS}
        torch.manual_seed(42);base=C0(sizes,width=4)
        torch.manual_seed(42);augmented=C3(sizes,width=4)
        for name,value in base.state_dict().items():torch.testing.assert_close(value,augmented.state_dict()[name])
        fields=torch.full((2,10),2,dtype=torch.long)
        hist=torch.full((2,3),3,dtype=torch.long);mask=hist!=0;counts=torch.ones((2,20))
        torch.testing.assert_close(base(fields,hist,hist,mask),augmented(fields,hist,hist,mask,counts))
        augmented(fields,hist,hist,mask,counts).sum().backward()
        self.assertGreater(augmented.histogram_projection.weight.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()

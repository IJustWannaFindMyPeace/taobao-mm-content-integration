import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c0 import C0
from feature_store import FIELDS
from side_information import SideInformation


class SideContracts(unittest.TestCase):
    def test_initialization_masks_and_gradient(self):
        sizes={f:12 for f in FIELDS}
        features=torch.full((2,10),2,dtype=torch.long)
        items=torch.tensor([[3,4,0],[0,0,0]])
        valid=items!=0
        q,h=torch.randn(2,128,requires_grad=True),torch.randn(2,3,128,requires_grad=True)
        available=torch.tensor([[True,False,False],[False,False,False]])
        target_available=torch.ones(2,dtype=torch.bool)
        for mode in ['vector','simscore']:
            torch.manual_seed(42);base=C0(sizes)
            torch.manual_seed(42);model=SideInformation(sizes,mode=mode)
            args=(features,items,items,valid,q,h,target_available,available)
            torch.testing.assert_close(base(*args[:4]),model(*args),rtol=0,atol=0)
            model(*args).sum().backward()
            self.assertGreater(model.side_projection.weight.grad.abs().sum().item(),0)
            self.assertIsNone(q.grad);self.assertIsNone(h.grad)
            with torch.no_grad():model.side_projection.weight.normal_(std=.1)
            changed=h.detach().clone();changed[~available]=100
            torch.testing.assert_close(model(*args),model(features,items,items,valid,q,changed,target_available,available))
            self.assertTrue(torch.isfinite(model(*args)).all())


if __name__=='__main__':unittest.main()

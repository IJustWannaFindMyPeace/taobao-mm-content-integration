import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from make import ContentMAKE


class MAKEContracts(unittest.TestCase):
    def test_masks_and_fixed_inputs(self):
        torch.manual_seed(42)
        model = ContentMAKE()
        q = torch.randn(2,128,requires_grad=True)
        h = torch.randn(2,5,128,requires_grad=True)
        valid = torch.tensor([[True,True,False,False,False],[False]*5])
        available = torch.ones_like(valid)
        target_available = torch.ones(2,dtype=torch.bool)
        args = (q,h,valid,target_available,available)
        reference = model(*args)
        changed = h.detach().clone(); changed[~valid] = 100
        torch.testing.assert_close(reference,model(q,changed,valid,target_available,available))
        interest, hidden = model.extract(*args)
        self.assertEqual(hidden.shape,(2,32))
        self.assertEqual(interest[1].abs().sum().item(),0)
        reference.sum().backward()
        self.assertIsNone(q.grad); self.assertIsNone(h.grad)
        self.assertGreater(model.activation[0].weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(reference).all())

    def test_missing_target_and_permutation(self):
        model = ContentMAKE()
        q,h = torch.randn(2,128),torch.randn(2,5,128)
        valid = torch.ones(2,5,dtype=torch.bool)
        available = torch.tensor([False,True])
        a=model.extract(q,h,valid,available,valid)[0]
        b=model.extract(q,h.flip(1),valid,available,valid)[0]
        torch.testing.assert_close(a,b)
        self.assertEqual(a[0].abs().sum().item(),0)


if __name__=='__main__':unittest.main()

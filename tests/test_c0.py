import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c0 import C0
from feature_store import FIELDS


class C0Contracts(unittest.TestCase):
    def test_empty_histories_and_sparse_gradients(self):
        model=C0({f:12 for f in FIELDS},width=4,short_length=2)
        fields=torch.full((2,10),2,dtype=torch.long)
        item=torch.tensor([[0,0,0],[0,3,4]])
        valid=item!=0
        logits=model(fields,item,item,valid)
        logits.sum().backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(model.embeddings['205'].weight.grad.is_sparse)
        self.assertIsNotNone(model.short_attention.query.weight.grad)
        self.assertIsNotNone(model.long_attention.query.weight.grad)

    def test_masked_values_invariant(self):
        model=C0({f:12 for f in FIELDS},width=4,short_length=2).eval()
        fields=torch.full((1,10),2,dtype=torch.long)
        valid=torch.tensor([[False,True,True]])
        with torch.no_grad():
            a=model(fields,torch.tensor([[0,3,4]]),torch.tensor([[0,3,4]]),valid)
            b=model(fields,torch.tensor([[9,3,4]]),torch.tensor([[8,3,4]]),valid)
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()

import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FIELDS
from c0 import C0
from make_fusion import MAKEFusion, C6


class FusionContracts(unittest.TestCase):
    def test_combination_preserves_make_and_trains_histogram(self):
        sizes={f:12 for f in FIELDS}
        torch.manual_seed(42);baseline=MAKEFusion(sizes,freeze_content=True)
        torch.manual_seed(42);combined=C6(sizes)
        for name,value in baseline.state_dict().items():
            torch.testing.assert_close(value,combined.state_dict()[name],rtol=0,atol=0)
        features=torch.full((2,10),2,dtype=torch.long)
        items=torch.full((2,4),3,dtype=torch.long)
        valid=torch.ones_like(items,dtype=torch.bool)
        content=(torch.randn(2,128),torch.randn(2,4,128),torch.ones(2,dtype=torch.bool),valid)
        counts=torch.ones(2,20)
        torch.testing.assert_close(baseline(features,items,items,valid,*content),
            combined(features,items,items,valid,counts,*content),rtol=0,atol=0)
        combined(features,items,items,valid,counts,*content).sum().backward()
        self.assertGreater(combined.histogram_projection.weight.grad.abs().sum().item(),0)
        self.assertGreater(combined.content_projection.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in combined.content_branch.parameters()))

    def test_common_initialization_and_gradient_paths(self):
        sizes={f:12 for f in FIELDS}
        torch.manual_seed(42);baseline=C0(sizes)
        features=torch.full((2,10),2,dtype=torch.long)
        items=torch.full((2,4),3,dtype=torch.long)
        valid=torch.ones_like(items,dtype=torch.bool)
        q,h=torch.randn(2,128),torch.randn(2,4,128)
        target_valid=torch.ones(2,dtype=torch.bool)
        args=(features,items,items,valid,q,h,target_valid,valid)
        for frozen in [False,True]:
            torch.manual_seed(42);model=MAKEFusion(sizes,freeze_content=frozen)
            for name,value in baseline.state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(baseline(*args[:4]),model(*args),rtol=0,atol=0)
            model(*args).sum().backward()
            self.assertGreater(model.content_projection.weight.grad.abs().sum().item(),0)
            # Zero fusion initialization blocks branch gradients on the first step.
            # Once the projection changes, joint training must reach the extractor.
            model.zero_grad(set_to_none=True)
            with torch.no_grad():model.content_projection.weight.fill_(0.01)
            model(*args).sum().backward()
            grad=model.content_branch.activation[0].weight.grad
            if frozen:self.assertIsNone(grad)
            else:self.assertGreater(grad.abs().sum().item(),0)
            self.assertIsNone(model.content_branch.output.weight.grad)


if __name__=='__main__':unittest.main()

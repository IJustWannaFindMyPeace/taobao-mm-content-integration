"""Behavioral checks for ID and padding contracts."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from id_probe import IDProbe, build_vocab, encode


class ProbeContracts(unittest.TestCase):
    def test_signed_ids_and_unknown(self):
        vocab = build_vocab([-17, 0, 92, -17])
        self.assertEqual(vocab, build_vocab([92, -17, 0]))
        self.assertGreaterEqual(encode(-17, vocab), 2)
        self.assertNotEqual(encode(-17, vocab), encode(92, vocab))
        self.assertEqual(encode(0, vocab), 0)
        self.assertEqual(encode(999, vocab), 1)

    def test_padding_does_not_change_interest(self):
        torch.manual_seed(42)
        model = IDProbe(10).eval()
        target = torch.tensor([2])
        with torch.no_grad():
            plain = model(target, torch.tensor([[3, 4]]), torch.tensor([[True, True]]))
            padded = model(target, torch.tensor([[8, 3, 4]]), torch.tensor([[False, True, True]]))
        torch.testing.assert_close(plain, padded)

    def test_empty_history_backward_is_finite(self):
        model = IDProbe(10)
        output = model(torch.tensor([2]), torch.tensor([[0, 0]]), torch.tensor([[False, False]]))
        output.sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(torch.count_nonzero(model.embedding.weight.grad[0]).item(), 0)


if __name__ == '__main__':
    unittest.main()

import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cache_simtier_boundaries import count_scores

class Boundaries(unittest.TestCase):
 def test_exact_edges_and_omission(self):
  scores=torch.tensor([[-1.,-.5,0.,.5,1.]])
  mask=torch.ones_like(scores,dtype=torch.bool)
  self.assertEqual(count_scores(scores,mask,4,'floor').tolist(),[[1,1,1,2]])
  self.assertEqual(count_scores(scores,mask,4,'paper_literal').tolist(),[[1,1,1,1]])
  self.assertEqual(count_scores(scores,mask,4,'right_closed').tolist(),[[2,1,1,1]])
 def test_interior_top_interval_and_mask(self):
  scores=torch.tensor([[.75,.9,-.75,0.]])
  mask=torch.tensor([[True,True,True,False]])
  self.assertEqual(count_scores(scores,mask,4,'paper_literal').sum().item(),1)
  self.assertEqual(count_scores(scores,mask,4,'right_closed').sum().item(),3)
  self.assertEqual(count_scores(scores,mask & False,4,'right_closed').sum().item(),0)
if __name__=='__main__':unittest.main()

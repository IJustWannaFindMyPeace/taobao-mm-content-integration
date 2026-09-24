import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_supplementary_design import check_pair
class PairContract(unittest.TestCase):
 def setUp(self):self.a={'store_manifest_sha256':'same','config':dict(seed=42,batch_size=256,embedding_width=16,dense_learning_rate=.001,sparse_learning_rate=.001,weight_decay=0,epochs=1,history_length=200)}
 def test_rejects_different_seed_and_data(self):
  b=copy.deepcopy(self.a);b['config']['seed']=2026
  with self.assertRaises(ValueError):check_pair(self.a,b)
  b=copy.deepcopy(self.a);b['store_manifest_sha256']='other'
  with self.assertRaises(ValueError):check_pair(self.a,b)
 def test_rejects_changed_batch(self):
  b=copy.deepcopy(self.a);b['config']['batch_size']=32
  with self.assertRaises(ValueError):check_pair(self.a,b)
 def test_allows_explicit_study_axes(self):
  b=copy.deepcopy(self.a);b['config'].update(epochs=3,history_length=1000);check_pair(self.a,b)
if __name__=='__main__':unittest.main()

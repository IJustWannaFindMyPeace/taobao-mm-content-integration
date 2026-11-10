"""Evaluate the fixed additional model; no training or model selection."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simscore_make import SimScoreMAKE
from feature_store import FeatureStore
from metrics import evaluate
from train_simscore_make import tensors
from fetch_source import sha256
p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
r=json.loads((a.run/'report.json').read_text());cfg=r['config'];assert sha256(a.store/'manifest.json')==r['store_manifest_sha256']
m=SimScoreMAKE(json.loads((a.store/'manifest.json').read_text())['vocab_sizes'],cfg['embedding_width'],cfg['short_length'],True).cuda()
m.load_state_dict(torch.load(a.run/'model.pt',map_location='cpu',weights_only=True));m.eval()
# Verify saved validation predictions before opening the test split.
v=FeatureStore(a.store,'validation');saved=np.load(a.run/'validation_predictions.npz');idx=np.arange(256)
with torch.no_grad():pred=m(*tensors(v.batch(idx,cfg['history_length'],with_content=True))).cpu().numpy()
np.testing.assert_array_equal(pred,saved['logits'][idx]);del v,saved
store=FeatureStore(a.store,'test');preds=[];start=time.time()
with torch.no_grad():
 for i in range(0,len(store),cfg['batch_size']):
  b=store.batch(np.arange(i,min(i+cfg['batch_size'],len(store))),cfg['history_length'],with_content=True)
  preds.append(m(*tensors(b)).cpu().numpy())
  if i%(256*500)==0:print('test rows',i,flush=True)
logits=np.concatenate(preds);assert np.isfinite(logits).all()
np.savez(a.run/'test_predictions.npz',logits=logits,labels=store.labels,user_rows=store.user_rows,source_rows=store.source_rows)
result={'metrics':evaluate(store.labels,logits,store.user_rows),'model_sha256':sha256(a.run/'model.pt'),'predictions_sha256':sha256(a.run/'test_predictions.npz'),'seconds':time.time()-start,'validation_probe_exact':True,'scope':'additional evaluation after previous test results were known'}
(a.run/'test_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

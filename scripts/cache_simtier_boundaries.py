"""Cache literal-paper and count-preserving right-closed SimTier variants."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from feature_store import FeatureStore
from fetch_source import sha256


def count_scores(scores,mask,bins,mode):
 scaled=(scores.clamp(-1,1)+1)*bins/2
 if mode=='floor':index=scaled.floor().long().clamp(0,bins-1);keep=mask
 elif mode=='paper_literal':
  index=scaled.ceil().long();keep=mask & (index>=0) & (index<bins)
 elif mode=='right_closed':index=(scaled.ceil().long()-1).clamp(0,bins-1);keep=mask
 else:raise ValueError(mode)
 counts=torch.zeros((len(scores),bins),device=scores.device)
 counts.scatter_add_(1,index.clamp(0,bins-1),keep.float())
 return counts


def main():
 p=argparse.ArgumentParser()
 for name in ('store','reference-cache','output-dir'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();out=a.output_dir
 if out.exists():raise ValueError('Fresh cache directory required')
 out.mkdir();reference=json.loads((a.reference_cache/'manifest.json').read_text())
 assert reference['store_manifest_sha256']==sha256(a.store/'manifest.json')
 bins=reference['bins'];length=reference['history_length'];started=time.time()
 modes=('paper_literal','right_closed');results={}
 for mode in modes:
  (out/mode).mkdir();results[mode]=dict(version='simtier_boundary_sensitivity_v1',mode=mode,bins=bins,history_length=length,store_manifest_sha256=reference['store_manifest_sha256'],implementation_sha256=sha256(Path(__file__)),splits={},test_evaluated=False)
 for split in ('train','validation'):
  store=FeatureStore(a.store,split);refpath=a.reference_cache/f'{split}.npy'
  assert sha256(refpath)==reference['splits'][split]['sha256']
  ref=np.load(refpath,mmap_mode='r')
  arrays={mode:np.lib.format.open_memmap(out/mode/f'{split}.npy',mode='w+',dtype=np.float32,shape=ref.shape) for mode in modes}
  totals={m:0 for m in modes};changed={m:0 for m in modes};eligible=0
  for start in range(0,len(store),256):
   b=store.batch(np.arange(start,min(start+256,len(store))),length,with_content=True)
   target,hist,valid,ta,ha=[torch.as_tensor(b[k],device='cuda') for k in ('target_content','history_content','history_valid','target_content_available','history_content_available')]
   tn=target.norm(dim=-1);hn=hist.norm(dim=-1)
   mask=valid & ha & ta[:,None] & (hn>0) & (tn[:,None]>0)
   scores=((hist/hn.clamp_min(1e-12)[...,None])*(target/tn.clamp_min(1e-12)[:,None])[:,None]).sum(-1).clamp(-1,1)
   floor=count_scores(scores,mask,bins,'floor').cpu().numpy()
   np.testing.assert_array_equal(floor,ref[start:start+len(floor)])
   eligible+=int(mask.sum())
   for mode in modes:
    counts=count_scores(scores,mask,bins,mode).cpu().numpy();arrays[mode][start:start+len(counts)]=counts
    totals[mode]+=int(counts.sum());changed[mode]+=int(np.any(counts!=floor,axis=1).sum())
   if start%65536==0:print(split,start,len(store),flush=True)
  for mode in modes:
   arrays[mode].flush();results[mode]['splits'][split]=dict(rows=len(store),eligible_pairs=eligible,matched_pairs=totals[mode],omitted_pairs=eligible-totals[mode],rows_changed_vs_floor=changed[mode],floor_reference_exact=True)
   assert totals[mode]<=eligible
   if mode=='right_closed':assert totals[mode]==eligible
  arrays.clear()
  for mode in modes:results[mode]['splits'][split]['sha256']=sha256(out/mode/f'{split}.npy')
 for mode in modes:
  results[mode]['cache_seconds']=time.time()-started
  (out/mode/'manifest.json').write_text(json.dumps(results[mode],indent=2))
 (out/'summary.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()

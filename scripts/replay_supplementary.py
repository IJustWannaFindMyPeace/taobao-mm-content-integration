"""Read-only exact validation replay for inventoried supplementary models."""
import argparse,json,sys,time,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 for key in ('inventory','store','output-dir'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--model-id',required=True);a=p.parse_args()
 inv=json.loads(a.inventory.read_text());entry=inv['models'][a.model_id];run=Path(entry['run']);family=entry['family'];cfg=entry['config']
 contentroot=Path(__file__).resolve().parents[1];root=contentroot if family=='content' else contentroot.parent/'taobao-mm-muse-repro'
 sys.path[:0]=[str(root/'src'),str(root/'extensions'),str(contentroot/'scripts')]
 import torch,numpy as np,psutil
 from feature_store import FeatureStore
 from metrics import evaluate
 assert sha(run/'model.pt')==entry['model_sha256'] and sha(run/'report.json')==entry['training_report_sha256']
 assert sha(run/'validation_predictions.npz')==entry['validation_predictions_sha256']
 assert sha(a.store/'manifest.json')==entry['store_manifest_sha256']
 if a.output_dir.exists():raise ValueError('Fresh output required')
 a.output_dir.mkdir(parents=True)
 r=json.loads((run/'report.json').read_text());assert r['config']==cfg
 sources={f.name:sha(f) for folder in ('src','extensions') for f in (root/folder).glob('*.py')}
 for name,digest in r['implementation_sha256'].items():
  if name in sources and name!='baseline_model.py':assert sources[name]==digest,(name,'source changed')
 store=FeatureStore(a.store,'validation');sizes=store.manifest['vocab_sizes'];width=cfg['embedding_width'];short=cfg['short_length'];bins=cfg.get('simtier_bins')
 if family=='content':
  from c0 import C0
  from simtier import C3,histogram
  from make_fusion import MAKEFusion,C6
  if cfg.get('make_fusion'):model=(C6 if bins else MAKEFusion)(sizes,width,short,cfg['freeze_content'],**({'bins':bins} if bins else {}))
  elif bins:model=C3(sizes,width,short,bins)
  else:model=C0(sizes,width,short)
 elif family=='din':
  from din import ShortInterest
  model=ShortInterest(sizes,width,short,mechanism=cfg['mechanism'])
 elif family=='twin':
  from twin import TwinAdaptation
  model=TwinAdaptation(sizes,width=width,short_length=short,k=cfg['k'],retrieval=cfg['retrieval'])
 elif family=='miss':
  from miss import MissAdaptation
  model=MissAdaptation(sizes,width=width,short_length=short,k=cfg['k'],variant=cfg['variant'])
 elif family=='g3':
  from fused_attention_search import FusedAttentionSearch
  model=FusedAttentionSearch(sizes,width=width,short_length=short,k=cfg['k'],retrieval=cfg['variant'])
 elif family=='g1':
  from learned_content_search import LearnedContentSearch
  model=LearnedContentSearch(sizes,width=width,short_length=short,k=cfg['k'])
 elif family=='muse':
  from muse_model import MUSEModel
  model=MUSEModel(sizes,width=width,short_length=short,k=cfg['k'],mode=cfg['mode'],semantic=cfg['semantic'],histogram=cfg['histogram'],long_id=cfg['long_id'])
 else:raise ValueError(family)
 model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True,mmap=True),strict=True);model.requires_grad_(False);model.eval();model.cuda()
 torch.cuda.reset_peak_memory_stats();process=psutil.Process();limits=cfg['resource_limits'];samples=[]
 def guard():
  sample=dict(rss_gib=process.memory_info().rss/2**30,ram_free_gib=psutil.virtual_memory().available/2**30,gpu_free_gib=torch.cuda.mem_get_info()[0]/2**30,allocated_gib=torch.cuda.max_memory_allocated()/2**30);samples.append(sample)
  assert sample['rss_gib']<=limits['max_process_rss_gib'] and sample['ram_free_gib']>=limits['min_system_available_gib'] and sample['gpu_free_gib']>=limits['min_device_free_gib'] and sample['allocated_gib']<=limits['max_allocated_gib'],'Resource gate exceeded'
 guard();started=time.time();logits=np.empty(len(store),dtype=np.float32)
 with torch.inference_mode():
  for start in range(0,len(store),cfg['batch_size']):
   end=min(start+cfg['batch_size'],len(store));need_content=family not in ('content','din') or bool(bins or cfg.get('make_fusion'))
   b=store.batch(np.arange(start,end),cfg['history_length'],with_content=need_content)
   args=[torch.as_tensor(b[k],device='cuda',dtype=torch.bool if k=='history_valid' else torch.long) for k in ('features','history_item','history_category','history_valid')]
   ct=[torch.as_tensor(b[k],device='cuda') for k in ('target_content','history_content','target_content_available','history_content_available')] if need_content else []
   if bins:
    if entry['histogram_mode']:
     from cache_simtier_boundaries import count_scores
     tc,hc,ta,ha=ct;tn=tc.norm(dim=-1);hn=hc.norm(dim=-1);mask=args[3]&ha&ta[:,None]&(hn>0)&(tn[:,None]>0)
     scores=((hc/hn.clamp_min(1e-12)[...,None])*(tc/tn.clamp_min(1e-12)[:,None])[:,None]).sum(-1).clamp(-1,1)
     counts=count_scores(scores,mask,bins,entry['histogram_mode'])
    else:counts=histogram(ct[0],ct[1],args[3],ct[2],ct[3],bins=bins)
    args.append(counts)
   if family not in ('content','din') or cfg.get('make_fusion'):args+=ct
   logits[start:end]=model(*args).cpu().numpy()
   if start%(100*cfg['batch_size'])==0:
    guard();(a.output_dir/'progress.json').write_text(json.dumps(dict(rows=end,total=len(store))))
 guard()
 with np.load(run/'validation_predictions.npz') as old:
  for k,v in [('source_rows',store.source_rows),('labels',store.labels),('user_rows',store.user_rows)]:np.testing.assert_array_equal(old[k],v)
  error=float(np.max(np.abs(logits-old['logits'])));np.testing.assert_array_equal(logits,old['logits'])
 metrics=evaluate(store.labels,logits,store.user_rows);assert metrics==r['validation']
 result=dict(scope='exact_supplementary_validation_replay',model_id=a.model_id,rows=len(store),max_abs_logit_error=error,metrics=metrics,inventory_sha256=sha(a.inventory),model_sha256=entry['model_sha256'],source_sha256=sources,evaluator_sha256=sha(Path(__file__)),seconds=time.time()-started,resource_samples=samples,test_evaluated=False,optimizer_created=False)
 (a.output_dir/'report.json').write_text(json.dumps(result,indent=2));print(a.model_id,error,flush=True)
if __name__=='__main__':main()

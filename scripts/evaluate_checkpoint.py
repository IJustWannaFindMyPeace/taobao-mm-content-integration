"""Read-only full validation replay, without optimizer construction or training."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--family',choices=['content','muse'],required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--hist-cache',type=Path)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output required')
    sys.path.insert(0,str(args.project.resolve()/'src'))
    import numpy as np
    import torch
    import psutil
    from feature_store import FeatureStore
    from metrics import evaluate
    r=json.loads((args.run/'report.json').read_text());cfg=r['config']
    if r['test_evaluated'] or r['store_manifest_sha256']!=digest(args.store/'manifest.json'):
        raise ValueError('Validation provenance mismatch')
    if r['scope']!='one_seed_validation_run':raise ValueError('Full trained run required')
    source_hashes={f.name:digest(f) for f in (args.project/'src').glob('*.py')}
    for name,value in r['implementation_sha256'].items():
        if name in source_hashes and name!='baseline_model.py' and source_hashes[name]!=value:
            raise ValueError('Model source changed: '+name)
    # Old MUSE references do not instantiate BaselineModel; existing baseline
    # variants have dedicated forward/gradient regression tests after extension.
    weights_hash=digest(args.run/'model.pt')
    store=FeatureStore(args.store,'validation')
    width=cfg['embedding_width'];short=cfg['short_length'];bins=cfg.get('simtier_bins')
    cache=None
    if args.family=='content':
        from c0 import C0
        from simtier import C3,histogram
        from make_fusion import MAKEFusion,C6
        from side_information import SideInformation
        if cfg.get('side_information'):
            model=SideInformation(store.manifest['vocab_sizes'],width,short,cfg['side_information'])
        elif cfg.get('make_fusion'):
            cls=C6 if bins else MAKEFusion
            model=cls(store.manifest['vocab_sizes'],width,short,cfg['freeze_content'],**({'bins':bins} if bins else {}))
        elif bins:model=C3(store.manifest['vocab_sizes'],width,short,bins)
        else:model=C0(store.manifest['vocab_sizes'],width,short)
        if bins:
            if args.hist_cache is None:raise ValueError('Validation histogram parity cache required')
            meta=json.loads((args.hist_cache/'manifest.json').read_text())
            if (meta['store_manifest_sha256']!=r['store_manifest_sha256'] or meta['bins']!=bins
                or meta['history_length']!=cfg['history_length']
                or meta['splits']['validation']['sha256']!=digest(args.hist_cache/'validation.npy')):
                raise ValueError('Histogram cache provenance mismatch')
            cache=np.load(args.hist_cache/'validation.npy',mmap_mode='r')
    else:
        from muse_model import MUSEModel
        from baseline_model import BaselineModel
        variant=cfg.get('model_variant','muse')
        cls=MUSEModel if variant=='muse' else BaselineModel
        model=cls(store.manifest['vocab_sizes'],width=width,short_length=short,k=cfg['k'],mode=cfg['mode'],
                  semantic=cfg['semantic'],histogram=cfg['histogram'],long_id=cfg['long_id'],
                  **({} if variant=='muse' else {'variant':variant}))
    model.load_state_dict(torch.load(args.run/'model.pt',map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False);model.eval();model.cuda()
    limits=cfg['resource_limits'];process=psutil.Process();peak_rss=0.;minimum_free=float('inf');minimum_ram=float('inf')
    def guard():
        nonlocal peak_rss,minimum_free,minimum_ram
        peak_rss=max(peak_rss,process.memory_info().rss/2**30)
        minimum_free=min(minimum_free,torch.cuda.mem_get_info()[0]/2**30)
        minimum_ram=min(minimum_ram,psutil.virtual_memory().available/2**30)
        if (peak_rss>limits['max_process_rss_gib'] or minimum_ram<limits['min_system_available_gib']
            or minimum_free<limits['min_device_free_gib'] or torch.cuda.max_memory_allocated()/2**30>limits['max_allocated_gib']):
            raise RuntimeError('Read-only evaluation resource gate exceeded')
    torch.cuda.reset_peak_memory_stats();guard()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    logits=np.empty(len(store),dtype=np.float32)
    started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(store),cfg['batch_size']):
            end=min(start+cfg['batch_size'],len(store))
            content=args.family=='muse' or bool(bins or cfg.get('make_fusion') or cfg.get('side_information'))
            batch=store.batch(np.arange(start,end),cfg['history_length'],with_content=content)
            base=[torch.as_tensor(batch[k],device='cuda',dtype=torch.bool if k=='history_valid' else torch.long)
                  for k in ('features','history_item','history_category','history_valid')]
            ct=[torch.as_tensor(batch[k],device='cuda') for k in
                ('target_content','history_content','target_content_available','history_content_available')] if content else []
            inputs=list(base)
            if bins:
                counts=histogram(ct[0],ct[1],base[3],ct[2],ct[3],bins=bins)
                np.testing.assert_array_equal(counts.cpu().numpy(),cache[start:end])
                inputs.append(counts)
            if args.family=='muse' or cfg.get('make_fusion') or cfg.get('side_information'):inputs.extend(ct)
            logits[start:end]=model(*inputs).cpu().numpy()
            if start%(100*cfg['batch_size'])==0:
                guard()
                (args.output_dir/'progress.json').write_text(json.dumps({'rows':end,'total':len(store)}))
    metrics=evaluate(store.labels,logits,store.user_rows)
    with np.load(args.run/'validation_predictions.npz') as old:
        for key,value in [('source_rows',store.source_rows),('labels',store.labels),('user_rows',store.user_rows)]:
            np.testing.assert_array_equal(old[key],value)
        error=float(np.max(np.abs(logits-old['logits'])))
        # Exact replay is the gate. Any numerical discrepancy requires inspection,
        # not an automatic relaxation of tolerance or silent checkpoint replacement.
        np.testing.assert_array_equal(logits,old['logits'])
    if metrics!=r['validation']:raise ValueError('Validation metric replay mismatch')
    result={'scope':'read_only_full_validation_replay','family':args.family,'config':cfg,
            'rows':len(store),'metrics':metrics,'max_abs_logit_error':error,
            'histogram_online_equals_cache':True if bins else None,
            'model_sha256':weights_hash,'training_report_sha256':digest(args.run/'report.json'),
            'original_predictions_sha256':digest(args.run/'validation_predictions.npz'),
            'store_manifest_sha256':r['store_manifest_sha256'],'source_sha256':source_hashes,
            'evaluator_sha256':digest(Path(__file__)),'seconds':time.perf_counter()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_process_rss_gib':peak_rss,
            'min_device_free_gib':minimum_free,'min_system_available_gib':minimum_ram,
            'optimizer_created':False,'test_evaluated':False}
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(store),'max_abs_logit_error':error,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()

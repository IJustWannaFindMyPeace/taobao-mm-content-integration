"""Freeze validated artifacts and exact comparisons before test inference."""
import argparse
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
from evaluate_checkpoint import digest


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('inventory','replay-summary','regressions','store','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Freeze file must be new')
    inventory=read(args.inventory);replays=read(args.replay_summary);manifest=read(args.store/'manifest.json')
    if replays['inventory_sha256']!=digest(args.inventory) or replays['test_evaluated']:
        raise ValueError('Inventory provenance mismatch')
    if len(inventory['models'])!=54 or set(replays['models'])!={e['id'] for e in inventory['models']}:
        raise ValueError('All 54 replays required')
    evaluator=Path(__file__).with_name('evaluate_checkpoint.py');eval_hash=digest(evaluator)
    models={};sources={};replay_details={}
    for entry in inventory['models']:
        evidence=replays['models'][entry['id']];replay_path=Path(evidence['replay_dir'])/'report.json'
        r=read(replay_path);training=read(Path(entry['run'])/'report.json')
        if (digest(replay_path)!=evidence['report_sha256'] or r['max_abs_logit_error']!=0
            or r['optimizer_created'] or r['test_evaluated'] or r['rows']!=manifest['splits']['validation']['rows']
            or r['model_sha256']!=evidence['model_sha256'] or r['evaluator_sha256']!=replays['evaluator_sha256']
            or digest(Path(entry['run'])/'report.json')!=entry['report_sha256']
            or entry['report_sha256']!=r['training_report_sha256'] or r['config']!=training['config']):
            raise ValueError('Replay provenance failed: '+entry['id'])
        current={f.name:digest(f) for f in (Path(entry['project'])/'src').glob('*.py')}
        if current!=r['source_sha256']:raise ValueError('Model source changed after replay')
        sources[entry['family']]=current
        models[entry['id']]={'family':entry['family'],'config':training['config'],
            'model_sha256':r['model_sha256'],'training_report_sha256':entry['report_sha256'],
            'validation_replay_sha256':evidence['report_sha256']}
    for name in ('c6-seed42','r4-l1000-seed42','content-id-cosine-l1000-seed42'):
        file=args.regressions/name/'report.json';r=read(file)
        if (r['evaluator_sha256']!=eval_hash or r['max_abs_logit_error']!=0 or r['test_evaluated']
            or r['model_sha256']!=models[name]['model_sha256'] or r['rows']!=manifest['splits']['validation']['rows']):
            raise ValueError('Updated evaluator regression failed')
        replay_details[name]=digest(file)
    # Byte integrity only; no test outcomes or model predictions are calculated here.
    arrays={}
    for name,meta in manifest['arrays'].items():
        if name.startswith(('test.','test_users.')) or name=='content_int8.npy':
            if digest(args.store/name)!=meta['sha256']:raise ValueError('Test input hash mismatch: '+name)
            arrays[name]=meta['sha256']
    pairs=[(b,'c0') for b in ('c2','c3','c5','c6','c7')]
    pairs += [('c6','c3'),('c6','c5'),('c7','c6'),('r1','r0'),('r2','r1'),('r3','r1'),
              ('r4','r2'),('r4','r3'),('r4','r5'),('r5','r1'),('r1-l1000','r1'),('r4-l1000','r4'),
              ('r4-l1000','r1-l1000'),('r4-l1000','b1-hard-enhanced'),('r4-l1000','b2-soft-enhanced'),
              ('r4-l1000','g-id-item'),('content-id-cosine-l1000','r4-l1000')]
    comparisons=[{'candidate':f'{a}-seed{s}','baseline':f'{b}-seed{s}'} for s in (42,2026,3407) for a,b in pairs]
    assert all(x['candidate'] in models and x['baseline'] in models for x in comparisons)
    import torch,numpy
    out={'scope':'frozen_54_model_test_inventory','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
         'all_54_validation_replays_passed':True,'updated_entry_validation_regressions_passed':True,
         'inventory_sha256':digest(args.inventory),'historical_replay_summary_sha256':digest(args.replay_summary),
         'evaluator_sha256':eval_hash,'regression_report_sha256':replay_details,'source_sha256':sources,
         'evaluation_support_sha256':{name:digest(Path(__file__).with_name(name)) for name in
                                     ('run_frozen_test.py','compare_test.py','fetch_source.py')},
         'store_manifest_sha256':digest(args.store/'manifest.json'),'test_arrays_sha256':arrays,
         'test_rows':manifest['splits']['test']['rows'],'models':models,'comparisons':comparisons,
         'bootstrap':{'replicates':2000,'seed':2026,'unit':'user','interval':'95_percentile','multiplicity_adjusted':False},
         'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'platform':platform.platform()},
         'test_evaluated_at_freeze':False,
         'policy':'All fixed models reported; no tuning or model selection from test outcomes; single-seed diagnostics remain validation-only.'}
    args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('Frozen 54 models, 66 comparisons and test input hashes',flush=True)


if __name__=='__main__':main()

"""Summarize audited content controls with independently selected pretraining per seed."""
import argparse,json
from pathlib import Path
import numpy as np
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--studies',nargs=3,type=Path,required=True)
    parser.add_argument('--audits',nargs=3,type=Path,required=True)
    parser.add_argument('--pretraining',nargs=3,type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise ValueError('Fresh report required')
    names=['c0','c2','c3','c5','c6','c7'];runs={};provenance={}
    for study,audit_path,pretrain in zip(args.studies,args.audits,args.pretraining):
        audit=json.loads(audit_path.read_text());state=json.loads((study/'status.json').read_text())
        if state['active'] or state['failed'] or set(state['completed'])!=set(names):raise ValueError('Incomplete study')
        pretrain_path=pretrain/'report.json';content=json.loads(pretrain_path.read_text())
        seed=content['config']['seed']
        if seed in runs or content['test_evaluated'] or len(content['epochs'])!=content['config']['epochs']:raise ValueError('Invalid pretraining')
        if sha256(pretrain_path)!=state['content_report_sha256']:raise ValueError('Pretraining report changed')
        best=max(content['epochs'],key=lambda e:(e['validation']['gauc'],-e['epoch']))
        checkpoint=pretrain/f"epoch-{best['epoch']}.pt"
        if sha256(checkpoint)!=best['checkpoint_sha256'] or state['selected_epoch']!=best['epoch'] or state['checkpoint_sha256']!=best['checkpoint_sha256']:
            raise ValueError('Selected checkpoint mismatch')
        reports={}
        for name in names:
            path=study/name/'report.json';r=json.loads(path.read_text());check=audit['models'][name]
            if sha256(path)!=check['report_sha256'] or not check['saved_rows_and_metrics_verified']:raise ValueError('Invalid audit binding')
            if r['config']['seed']!=seed or r['store_manifest_sha256']!=audit['store_manifest_sha256'] or r['store_manifest_sha256']!=content['store_manifest_sha256']:
                raise ValueError('Seed/store mismatch')
            if name in ['c5','c6'] and (r['content_checkpoint_sha256']!=best['checkpoint_sha256'] or r['config']['content_checkpoint_sha256']!=best['checkpoint_sha256']):
                raise ValueError('MAKE must use this seed pretraining')
            reports[name]=r
        runs[seed]=reports
        provenance[str(seed)]={'audit_sha256':sha256(audit_path),'pretraining_report_sha256':sha256(pretrain_path),
            'selected_epoch':best['epoch'],'checkpoint_sha256':best['checkpoint_sha256'],
            'report_sha256':{n:audit['models'][n]['report_sha256'] for n in names}}
    if set(runs)!={42,2026,3407}:raise ValueError('Expected prespecified seeds')
    seeds=[42,2026,3407];base=runs[42]
    for reports in runs.values():
        for name,r in reports.items():
            strip=lambda c:{k:v for k,v in c.items() if k not in ['seed','content_checkpoint_sha256']}
            if strip(r['config'])!=strip(base[name]['config']) or r['implementation_sha256']!=base[name]['implementation_sha256'] or r['store_manifest_sha256']!=base[name]['store_manifest_sha256']:
                raise ValueError('Changes beyond seed and selected content checkpoint')
    def stats(values):return {'values':values,'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))}
    metrics=['auc','gauc','logloss']
    models={n:{m:stats([runs[s][n]['validation'][m] for s in seeds]) for m in metrics} for n in names}
    pairs=[('c2','c0'),('c3','c0'),('c5','c0'),('c6','c3'),('c6','c5'),('c7','c6')]
    deltas={a+'-minus-'+b:{m:stats([runs[s][a]['validation'][m]-runs[s][b]['validation'][m] for s in seeds]) for m in metrics} for a,b in pairs}
    result={'scope':'three_seed_content_core_validation_summary','seeds':seeds,'models':models,'paired_seed_deltas':deltas,
        'store_manifest_sha256':base['c0']['store_manifest_sha256'],'provenance':provenance,'source_sha256':sha256(Path(__file__)),
        'test_evaluated':False,'limitations':['three seeds on one fixed validation population',
        'sample standard deviation is not a confidence interval','seed-specific validation-selected pretraining is part of the pipeline',
        'no best-seed selection; supplementary MAKE strategy controls remain separate']}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({n:v['gauc'] for n,v in models.items()},indent=2))


if __name__=='__main__':main()

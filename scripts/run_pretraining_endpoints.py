"""Train frozen P1 and P5 downstream endpoints after exact prefix verification."""
import argparse,json,subprocess,sys
from pathlib import Path
from fetch_source import sha256

def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser()
    for key in ['store','pretraining','prefix-audit','reference-c5','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();audit=read(a.prefix_audit);source=read(a.pretraining/'report.json');base=read(a.reference_c5/'report.json')
    if a.output_dir.exists():raise ValueError('Fresh directory required')
    if audit['test_evaluated'] or audit['store_manifest_sha256']!=sha256(a.store/'manifest.json') or audit['extended_report_sha256']!=sha256(a.pretraining/'report.json'):raise ValueError('Audit provenance mismatch')
    if len(audit['prefix'])!=3 or any(x['max_prediction_error']!=0 or not x['all_weights_exact'] for x in audit['prefix']):raise ValueError('Prefix not identical')
    if base['test_evaluated'] or base['store_manifest_sha256']!=audit['store_manifest_sha256'] or base['config']['seed']!=source['config']['seed']:raise ValueError('Downstream reference mismatch')
    cfg=base['config']
    if not cfg['pretrained_content'] or not cfg['freeze_content'] or cfg['batch_size']!=256 or cfg['epochs']!=1:raise ValueError('Unexpected frozen downstream reference')
    a.output_dir.mkdir();(a.output_dir/'configs').mkdir()
    state=dict(planned=['p1','p5'],completed=[],active=None,failed=None,prefix_audit_sha256=sha256(a.prefix_audit),test_evaluated=False)
    def save():
        p=a.output_dir/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(a.output_dir/'status.json')
    save()
    for epoch in [1,5]:
        name=f'p{epoch}';checkpoint=a.pretraining/f'epoch-{epoch}.pt';digest=sha256(checkpoint)
        if digest!=source['epochs'][epoch-1]['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        config=dict(cfg,experiment=f'MAKE-frozen-P{epoch}-shared-v1',content_checkpoint_sha256=digest)
        path=a.output_dir/'configs'/f'{name}.json';path.write_text(json.dumps(config,indent=2))
        state['active']=name;save()
        with (a.output_dir/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('train_c0.py')),'--store',str(a.store.resolve()),'--config',str(path.resolve()),'--output-dir',str((a.output_dir/name).resolve()),'--content-checkpoint',str(checkpoint.resolve())],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(active=None,failed=name);save();raise RuntimeError(name+' failed')
        report=read(a.output_dir/name/'report.json')
        if report['config']!=config or report['test_evaluated'] or report['reload_max_abs_error']!=0:
            state.update(active=None,failed=name);save();raise ValueError('Run integrity failed')
        state['completed'].append(name);state['active']=None;save()

if __name__=='__main__':main()

"""Extend fixed MAKE P0-joint/P1/P3/P5 curves to the remaining seeds."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from fetch_source import sha256


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();w=a.work.resolve();out=a.output_dir.resolve();root=Path(__file__).resolve().parents[1]
    store=w/'shared-feature-store-050'
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir();(out/'configs').mkdir()
    state=dict(phase='preflight',completed=[],active=None,failed=None,test_evaluated=False)
    def save():
        t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
    sources=[*sorted((root/'src').glob('*.py')),*[root/'scripts'/f for f in ('train_content.py','train_c0.py','audit_pretraining_prefix.py','run_pretraining_endpoints.py','audit_pretraining_endpoints.py','audit_paper_zero_reference.py')]]
    hashes={f.name:sha256(f) for f in sources}
    manifest=sha256(store/'manifest.json')
    references={}
    for seed in (2026,3407):
        pre=w/f'c1-050-seed{seed}-epochs3';r=read(pre/'report.json')
        assert r['store_manifest_sha256']==manifest and not r['test_evaluated'] and len(r['epochs'])==3
        assert all(hashes[k]==v for k,v in r['implementation_sha256'].items())
        cfg=dict(r['config'],epochs=5)
        (out/f'configs/pretraining-seed{seed}.json').write_text(json.dumps(cfg,indent=2))
        references[str(seed)]={'pretraining_report_sha256':sha256(pre/'report.json')}
    (out/'plan.json').write_text(json.dumps(dict(seeds=[2026,3407],pretraining_epochs=[0,1,3,5],zero='random joint',positive_epochs='frozen exact epoch',source_hashes=hashes,manifest_sha256=manifest,references=references,seed42='reuse published shared-paper-zero-reference-v1-seed42-audit',test_evaluated=False),indent=2))
    def execute(name,script,args):
        assert all(sha256(f)==hashes[f.name] for f in sources)
        state.update(phase='running',active=name);save();started=time.time()
        command=[sys.executable,str(root/'scripts'/script),*map(str,args)]
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append({'job':name,'seconds':time.time()-started});state['active']=None;save()
    try:
        for seed in (2026,3407):
            pre=out/f'pretraining-seed{seed}';prefix=out/f'prefix-seed{seed}.json'
            core=w/f'shared-core-v1-remaining-seeds/seed{seed}/c5'
            strategy=w/f'post-main-make-seeds-v1/seed{seed}'
            execute(f'pretraining-seed{seed}','train_content.py',['--store',store,'--config',out/f'configs/pretraining-seed{seed}.json','--output-dir',pre])
            execute(f'prefix-seed{seed}','audit_pretraining_prefix.py',['--store',store,'--reference',w/f'c1-050-seed{seed}-epochs3','--extended',pre,'--report',prefix])
            endpoints=out/f'endpoints-seed{seed}'
            execute(f'endpoints-seed{seed}','run_pretraining_endpoints.py',['--store',store,'--pretraining',pre,'--prefix-audit',prefix,'--reference-c5',core,'--output-dir',endpoints])
            p3=core
            r=read(core/'report.json');pr=read(pre/'report.json')
            if r['content_checkpoint_sha256']!=pr['epochs'][2]['checkpoint_sha256']:
                p3=out/f'p3-seed{seed}'
                cfg=dict(r['config'],experiment='MAKE-frozen-exact-P3',content_checkpoint_sha256=pr['epochs'][2]['checkpoint_sha256'])
                config=out/f'configs/p3-seed{seed}.json';config.write_text(json.dumps(cfg,indent=2))
                execute(f'p3-seed{seed}','train_c0.py',['--store',store,'--config',config,'--output-dir',p3,'--content-checkpoint',pre/'epoch-3.pt'])
            audit=out/f'endpoints-audit-seed{seed}'
            execute(f'endpoints-audit-seed{seed}','audit_pretraining_endpoints.py',['--store',store,'--p0',strategy/'c4-frozen','--p1',endpoints/'p1','--p3',p3,'--p5',endpoints/'p5','--pretraining',pre,'--prefix-audit',prefix,'--output-dir',audit])
            execute(f'paper-zero-seed{seed}','audit_paper_zero_reference.py',['--store',store,'--joint-zero',strategy/'c4','--p1',endpoints/'p1','--p3',p3,'--p5',endpoints/'p5','--strategy-audit',w/f'post-main-make-seeds-v1/seed{seed}-audit/summary.json','--endpoint-audit',audit/'summary.json','--output-dir',out/f'paper-zero-seed{seed}'])
        state.update(phase='completed',active=None);save()
    except Exception as e:
        state.update(phase='failed',failed={'job':state['active'],'error':repr(e)});save();raise


if __name__=='__main__':main()

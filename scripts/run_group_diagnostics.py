"""Run validation subgroup diagnostics serially without training or test access."""
import argparse,json,subprocess,sys
from pathlib import Path
from fetch_source import sha256

def main():
    p=argparse.ArgumentParser()
    for k in ['store','items-dir','models','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise ValueError('Fresh output directory required')
    a.output_dir.mkdir()
    state=dict(planned=['frequency','coverage','history_length'],completed=[],active=None,failed=None,models_sha256=sha256(a.models),test_evaluated=False)
    def save():
        temp=a.output_dir/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(a.output_dir/'status.json')
    for name,script in [('frequency','audit_frequency.py'),('coverage','audit_prediction_coverage.py'),('history_length','audit_history_length.py')]:
        state['active']=name;save()
        cmd=[sys.executable,str(Path(__file__).with_name(script)),'--store',str(a.store.resolve()),'--models',str(a.models.resolve()),'--report',str((a.output_dir/(name+'.json')).resolve())]
        if name=='frequency':cmd+=['--items-dir',str(a.items_dir.resolve())]
        with (a.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(active=None,failed=name);save();raise RuntimeError(name+' failed')
        state['completed'].append(name);state['active']=None;save()

if __name__=='__main__':main()

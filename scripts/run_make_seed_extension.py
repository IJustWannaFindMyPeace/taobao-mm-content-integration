"""Post-main-study seed extension; preserve training settings, validation only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from fetch_source import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();work=a.work.resolve();root=Path(__file__).resolve().parents[1]
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise ValueError('Fresh output required')
    protocol=root/'configs/shared-downstream-protocol-v1.json'
    plan={'scope':'post_main_study_make_seed_extension','seeds':[2026,3407],
          'protocol_sha256':sha256(protocol),'test_used_for_training_or_selection':False,
          'changes':'training seed only; retain established validation checkpoint selection rule',
          'pretraining_report_sha256':{str(s):sha256(work/f'c1-050-seed{s}-epochs3/report.json') for s in [2026,3407]},
          'jobs':['c0-epochs3','c4','c4-frozen','c5-joint'],'reused':'same-seed frozen pretrained C5 from shared core'}
    a.output_dir.mkdir(parents=True);(a.output_dir/'plan.json').write_text(json.dumps(plan,indent=2))
    state={'planned':[2026,3407],'completed':[],'active':None,'failed':None,'test_evaluated':False}
    def save():
        tmp=a.output_dir/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(a.output_dir/'status.json')
    for seed in state['planned']:
        state['active']=seed;save()
        try:
            output=a.output_dir/f'seed{seed}'
            with (a.output_dir/f'seed{seed}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(root/'scripts/run_make_study.py'),'--store',str(work/'shared-feature-store-050'),
                    '--pretraining-run',str(work/f'c1-050-seed{seed}-epochs3'),'--protocol',str(protocol),
                    '--supplement-only','--output-dir',str(output.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
                subprocess.run([sys.executable,str(root/'scripts/summarize_make_study.py'),'--study',str(output.resolve()),
                    '--original-c0',str(work/f'shared-core-v1-remaining-seeds/seed{seed}/c0'),
                    '--frozen-pretrained',str(work/f'shared-core-v1-remaining-seeds/seed{seed}/c5'),
                    '--store',str(work/'shared-feature-store-050'),'--output-dir',str((a.output_dir/f'seed{seed}-audit').resolve())],
                    check=True,stdout=log,stderr=subprocess.STDOUT)
            state['completed'].append(seed);state['active']=None;save();print('Completed and audited '+str(seed),flush=True)
        except Exception as exc:
            state['failed']={'seed':seed,'error':str(exc)};state['active']=None;save();raise


if __name__=='__main__':main()

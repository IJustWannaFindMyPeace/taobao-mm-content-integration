"""Frozen supplementary test comparison; identical user bootstrap computation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import auc,evaluate
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--pair-index',type=int,required=True)
    parser.add_argument('--baseline-report',type=Path,required=True)
    parser.add_argument('--candidate-report',type=Path,required=True)
    args=parser.parse_args()
    from supplementary_test_contract import verify_dependencies
    frozen=json.loads(args.freeze.read_text());verify_dependencies(frozen,Path(__file__).resolve().parents[1])
    if frozen.get('scope')!='frozen_supplementary_test_v1':raise ValueError('Test freeze required')
    pair=frozen['comparisons'][args.pair_index]
    for role in ('baseline','candidate'):
        report_path=getattr(args,role+'_report');r=json.loads(report_path.read_text());key=pair[role]
        if not r.get('test_evaluated') or r['rows']!=923211:raise ValueError('Test report required')
        if sha256(getattr(args,role))!=r['test_predictions_sha256']:raise ValueError('Test prediction digest mismatch')
        if key.startswith('main:'):
            reference=frozen['references'][key]
            if sha256(report_path)!=reference['report_sha256']:raise ValueError('Main test reference changed')
        elif r['model_id']!=key or r['freeze_sha256']!=sha256(args.freeze):raise ValueError('Wrong test model or freeze')
    if args.report.exists():raise ValueError('Fresh comparison report required')
    a=np.load(args.baseline);b=np.load(args.candidate)
    for key in ['labels','source_rows','user_rows']:
        np.testing.assert_array_equal(a[key],b[key])
    labels=a['labels'];users=a['user_rows']
    a_logits=a['logits'];b_logits=b['logits']
    ma=evaluate(labels,a_logits,users);mb=evaluate(labels,b_logits,users)
    if len(labels)!=923211 or ma!=json.loads(args.baseline_report.read_text())['metrics'] or mb!=json.loads(args.candidate_report.read_text())['metrics']:raise ValueError('Test metric contract mismatch')
    groups={}
    for i,u in enumerate(users):groups.setdefault(int(u),[]).append(i)
    weights=[];numerators=[]
    for indices in groups.values():
        av=auc(labels[indices],a_logits[indices]);bv=auc(labels[indices],b_logits[indices])
        if av is None:weights.append(0);numerators.append(0.)
        else:weights.append(len(indices));numerators.append(len(indices)*(bv-av))
    weights=np.asarray(weights);numerators=np.asarray(numerators)
    rng=np.random.default_rng(2026);samples=[]
    for _ in range(20):
        draw=rng.integers(0,len(weights),size=(100,len(weights)))
        denom=weights[draw].sum(1)
        if np.any(denom==0):raise ValueError('Bootstrap has no valid users')
        samples.extend((numerators[draw].sum(1)/denom).tolist())
    result={'scope':'paired_single_seed_supplementary_test','test_evaluated':True,'freeze_sha256':sha256(args.freeze),'pair_index':args.pair_index,'pair':pair,'baseline_sha256':sha256(args.baseline),'candidate_sha256':sha256(args.candidate),
        'rows':len(labels),'baseline':ma,'candidate':mb,
        'candidate_minus_baseline':{k:mb[k]-ma[k] for k in ['auc','gauc','logloss']},
        'gauc_delta_user_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),
        'bootstrap_replicates':2000,'bootstrap_seed':2026,
        'limitations':['conditional on these trained models and this cohort','does not measure training-seed or cohort-selection variability','additional test evaluation after main test was already inspected; not a new blind test']}
    temporary=args.report.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');temporary.replace(args.report)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

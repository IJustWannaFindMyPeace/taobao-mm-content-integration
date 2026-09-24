"""Paired validation comparison; user bootstrap conditional on fixed fitted models."""
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
    args=parser.parse_args()
    a=np.load(args.baseline);b=np.load(args.candidate)
    for key in ['labels','source_rows','user_rows']:
        np.testing.assert_array_equal(a[key],b[key])
    labels=a['labels'];users=a['user_rows']
    a_logits=a['logits'];b_logits=b['logits']
    ma=evaluate(labels,a_logits,users);mb=evaluate(labels,b_logits,users)
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
    result={'scope':'paired_single_seed_validation','baseline_sha256':sha256(args.baseline),'candidate_sha256':sha256(args.candidate),
        'rows':len(labels),'baseline':ma,'candidate':mb,
        'candidate_minus_baseline':{k:mb[k]-ma[k] for k in ['auc','gauc','logloss']},
        'gauc_delta_user_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),
        'bootstrap_replicates':2000,'bootstrap_seed':2026,
        'limitations':['conditional on these trained models and this cohort','does not measure training-seed or cohort-selection variability','validation comparison, not test evidence']}
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

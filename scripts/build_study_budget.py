"""Measured training-only budget scenarios; never an OOM or runtime guarantee."""
import argparse,json
from pathlib import Path
from fetch_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--content-repo',type=Path,required=True)
    parser.add_argument('--muse-repo',type=Path,required=True)
    parser.add_argument('--cohort-report',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    paths={name:args.content_repo/'reports'/file for name,file in [
        ('C0','c0-seed42-epoch1.json'),('C2','c2-seed42-epoch1.json'),('C3','c3-seed42-epoch1.json'),
        ('C5','c5-seed42.json'),('C6','c6-seed42-epoch1.json'),('C7','c7-seed42-epoch1.json')]}
    paths.update({f'R{i}':args.muse_repo/'reports'/f'r{i}-seed42-epoch1.json' for i in range(6)})
    measured={}
    for name,path in paths.items():
        r=json.loads(path.read_text())
        measured[name]={'report_filename':path.name,'report_sha256':sha256(path),
                        'seconds':r['training_seconds'],'peak_allocated_mib':r['training_peak_allocated_mib']}
    pretrain_path=args.content_repo/'reports/c1-seed42-epochs3.json'
    pretrain=json.loads(pretrain_path.read_text())
    pretrain_seconds=sum(e['training_seconds'] for e in pretrain['epochs'])
    cohort=json.loads(args.cohort_report.read_text())
    ratio=cohort['subsets']['train']['rows']/609902
    seconds=sum(r['seconds'] for r in measured.values())+pretrain_seconds
    result={'scope':'training_only_planning_scenarios','candidate_cohort':cohort['subsets'],
            'candidate_cohort_report_sha256':sha256(args.cohort_report),'train_row_ratio_to_1_percent':ratio,
            'measured_1_percent_main_runs':measured,
            'content_pretraining':{'epochs':3,'seconds':pretrain_seconds,'source_sha256':sha256(pretrain_path)},
            'planned_seeds':[42,2026,3407],
            'one_percent_three_seed_training_hours':seconds*3/3600,
            'candidate_linear_three_seed_training_hours':seconds*3*ratio/3600,
            'candidate_sensitivity_hours_0_7_to_1_4_of_linear':[seconds*3*ratio*x/3600 for x in [.7,1.4]],
            'scope_includes':'12 main configurations plus one 3-epoch content pretraining per seed',
            'scope_excludes':['validation and checkpoints','data preparation','MAKE factorial extras and epoch grids',
                              'history-length scans','DIN/SIM/TWIN/MISS and other supplementary baselines','failed runs'],
            'limitations':['sensitivity range is a planning assumption, not a confidence interval',
                           'sample ratio does not predict vocabulary growth or GPU memory',
                           'candidate feature store and full-model resource gate are not yet complete']}
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if 'hours' in k or k=='train_row_ratio_to_1_percent'},indent=2))


if __name__=='__main__':main()

"""Audit narrative data, source attribution, figure assets and selected claim directions."""
import argparse
import csv
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    content=Path(__file__).resolve().parents[1]
    roots={'content':content,'muse':content.parent/'taobao-mm-muse-repro','work':a.work.resolve()}
    evidence=read(content/'reports/storyline-evidence.json')
    primary=read(content/'reports/final-results-summary.json');extra=read(content/'reports/supplementary-test-summary.json')
    manifest=read(content/'reports/storyline-figure-manifest.json');claims=read(content/'reports/storyline-claims.json')
    assert manifest['data_sha256']==sha(content/'reports/storyline-evidence.json')
    assert manifest['renderer_sha256']==sha(content/'scripts/render_storyline_figures.py')
    for source,value in evidence['sources'].items():
        project,relative=source.split('/',1);assert sha(roots[project]/relative)==value,source
    figures=evidence['figures'];assert len(figures)==len(manifest['figures'])==12
    for fig in manifest['figures']:
        assert fig['data_rows']==len(figures[fig['id']]['rows'])
        for extension,entry in fig['files'].items():
            path=content/entry['path'];assert sha(path)==entry['sha256']
            assert path.read_bytes()==(roots['muse']/entry['path']).read_bytes()
            if extension=='png':
                data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
                width,height=struct.unpack('>II',data[16:24]);assert width>=2600 and height>=1400
            else:ET.parse(path)
    count=0
    for key,fig in figures.items():
        rows=fig['rows'];count+=len(rows)
        if fig['kind'] not in ('diagram','forest','behavior'):
            groups={}
            for r in rows:
                group=(r['label'],r.get('length'),r.get('epoch'),r.get('frequency'))
                groups.setdefault(group,set()).add(r['seed'])
            assert all(v=={42,2026,3407} for v in groups.values()),key
        for r in rows:
            if 'model_id' in r:
                if r['model_id'] in extra['models']:
                    expected=extra['models'][r['model_id']]['test_metrics']
                else:
                    model,seed=r['model_id'].rsplit('-seed',1);i=primary['training_seeds'].index(int(seed))
                    expected={m:primary['models'][model]['test'][m]['values'][i] for m in ('auc','gauc','logloss')}
                for m in ('auc','gauc','logloss'):assert r[m]==expected[m],(key,r['model_id'],m)
            if 'pair_index' in r:
                original=extra['comparisons'][r['pair_index']]
                assert r['delta_gauc']==original['candidate_minus_baseline']['gauc']
                assert [r['ci_low'],r['ci_high']]==original['gauc_delta_user_bootstrap_95_percentile']
    strategy=read(content/'reports/post-main-make-three-seed-summary.json')
    for r in figures['03-training']['rows']:
        if r['label']=='ID下游':
            original=strategy['audits'][str(r['seed'])]['c0_epochs'][r['epoch']-1]
        else:
            original=read(roots['work']/f"c1-050-seed{r['seed']}-epochs3/report.json")['epochs'][r['epoch']-1]
        assert r['gauc']==original['validation']['gauc'] and r['logloss']==original['validation']['logloss']
        assert r['online_train_logloss']==original['online_train_logloss']
    frequency=read(content/'reports/posthoc-test-frequency-v1.json')
    for r in figures['09-frequency']['rows']:
        candidate,baseline=('c6','c0') if r['label'].startswith('内容集成') else ('r4-l1000','g-id-item')
        def group(model):
            return next(x for x in frequency['models'][model+f"-seed{r['seed']}"]
                        if x['training_target_exposure_frequency']==r['frequency'])
        c,b=group(candidate),group(baseline)
        assert r['rows']==c['test_rows']==b['test_rows']
        assert r['delta_auc']==c['auc']-b['auc'] and r['delta_logloss']==c['logloss']-b['logloss']
    behavior=read(roots['muse']/'reports/post-main-g3-validation.json')['behavior_audit']['results']
    for r in figures['10-g3-diagnostic']['rows']:
        original=behavior[str(r['seed'])]
        for field,source in [('negative_slope_fraction','negative_semantic_slope_fraction'),
                             ('overlap_fraction','content_topk_overlap_fraction'),
                             ('selected_cosine','mean_selected_content_cosine_fused'),
                             ('reference_cosine','mean_selected_content_cosine_reference')]:
            assert r[field]==original[source]
    csv_rows=list(csv.DictReader((content/'reports/storyline-chart-data.csv').open(encoding='utf-8',newline='')))
    expected=[dict(figure=k,**r) for k,f in figures.items() for r in f['rows']]
    assert len(csv_rows)==len(expected)==count==186
    for actual,row in zip(csv_rows,expected):
        for k,v in row.items():assert actual[k]==str(v),(k,actual[k],v)
    for claim in claims['claims']:
        assert claim['prohibited_inference'] and claim['paper_location']
        assert all(f in figures for f in claim['figure_ids'])
    assert len(claims['claims'])==11
    # Numerical assertions are separate from the prose writing path.
    mm=primary['models']
    assert mm['c7']['test']['gauc']['mean']>mm['c6']['test']['gauc']['mean']
    assert mm['c6']['test']['auc']['mean']>mm['c7']['test']['auc']['mean']
    assert mm['c2']['test']['logloss']['mean']>mm['c0']['test']['logloss']['mean']
    pairs=extra['comparisons']
    assert all(pairs[i]['gauc_delta_user_bootstrap_95_percentile'][0]>0 for i in [0,8,16,3,11,19])
    assert all(pairs[i]['gauc_delta_user_bootstrap_95_percentile'][1]<0 for i in [60,61,62,64,65,90,91,92])
    assert pairs[63]['gauc_delta_user_bootstrap_95_percentile'][0]<0<pairs[63]['gauc_delta_user_bootstrap_95_percentile'][1]
    assert all(pairs[i]['gauc_delta_user_bootstrap_95_percentile'][0]>0 for i in [75,76,78,79,81,82,83])
    assert all(pairs[i]['gauc_delta_user_bootstrap_95_percentile'][0]<0<pairs[i]['gauc_delta_user_bootstrap_95_percentile'][1] for i in [77,80])
    shared=['reports/storyline-evidence.json','reports/storyline-chart-data.csv','reports/storyline-claims.json',
            'reports/storyline-figure-manifest.json','docs/research-narrative.md','docs/paper-comparison-and-analysis.md',
            'docs/presentation-plan.md','docs/storyline-figures.md']
    for relative in shared:assert (content/relative).read_bytes()==(roots['muse']/relative).read_bytes()
    for root in [content,roots['muse']]:
        for relative in ['README.md']+[s for s in shared if s.startswith('docs/')]:
            path=root/relative;text=path.read_text(encoding='utf-8')
            assert '\ufffd' not in text and 'C:\\Users' not in text
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if not link.startswith(('http','#')):assert (path.parent/link.split('#')[0]).exists(),(relative,link)
    result=dict(scope='narrative_data_and_presentation_readiness_audit',date='2026-09-25',all_passed=True,
                source_files_verified=len(evidence['sources']),figures_verified=12,asset_files_verified=24,
                chart_rows_verified=count,core_claims_mapped=11,shared_outputs_identical=True,
                provenance_verified=True,selected_claim_directions_checked=True,
                presentation_file_created=False,
                boundaries=['Does not establish causal explanations for paper-result differences.',
                            'Visual inspection is a separate manual step, not asserted by this numeric audit.',
                            'No new training, test-driven tuning or bootstrap rerun.',
                            'A future PPT still requires page-level visual and numerical QA.'])
    for root in [content,roots['muse']]:
        (root/'reports/narrative-readiness-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()

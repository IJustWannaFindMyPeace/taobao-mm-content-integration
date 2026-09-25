"""Export presentation-ready evidence from audited reports; no model fitting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = {'content': root, 'muse': root.parent/'taobao-mm-muse-repro', 'work': a.work.resolve()}
    provenance = {}

    def source(project, relative):
        path = roots[project]/relative
        provenance[project+'/'+relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8-sig'))

    primary = source('content', 'reports/final-results-summary.json')
    extra = source('content', 'reports/supplementary-test-summary.json')
    strategy = source('content', 'reports/post-main-make-three-seed-summary.json')
    frequency = source('content', 'reports/posthoc-test-frequency-v1.json')
    g3 = source('muse', 'reports/post-main-g3-validation.json')['behavior_audit']
    baselines = source('muse', 'reports/adapted-baseline-test-comparison.json')
    seeds = primary['training_seeds']
    figures = {}

    def rows_main(ids):
        return [dict(label=label, seed=seed, model_id=key+'-seed'+str(seed), split='test', phase='main',
                     **{m: primary['models'][key]['test'][m]['values'][i] for m in ('auc','gauc','logloss')})
                for key, label in ids for i, seed in enumerate(seeds)]

    def extra_row(key, label):
        r = extra['models'][key]
        return dict(label=label, model_id=key, seed=r['config']['seed'], split='test', phase='supplement',
                    **{m:r['test_metrics'][m] for m in ('auc','gauc','logloss')})

    def add(key, title, kind, rows, note):
        figures[key] = dict(title=title, kind=kind, rows=rows, note=note)

    add('01-content', '内容信息有用，最佳接入方式依赖指标', 'metrics',
        rows_main([('c0','ID基线'),('c2','直接向量'),('c7','SimScore'),('c3','SimTier'),('c5','MAKE'),('c6','SimTier＋MAKE')]),
        '主测试；点为三种子均值，短线为样本标准差，不是置信区间。横轴为局部范围。')
    rows = []
    for seed in seeds:
        prefix = 'shared-make-strategy-v1-seed42' if seed == 42 else f'post-main-make-seeds-v1--seed{seed}'
        for suffix, label in [('c4-frozen','随机／冻结'),('c4','随机／联合'),('c5-joint','预训练／联合')]:
            rows.append(extra_row(prefix+'--'+suffix,label))
        rows += rows_main([('c5','预训练／冻结')])[seeds.index(seed):seeds.index(seed)+1]
    order = ['随机／冻结','随机／联合','预训练／冻结','预训练／联合']
    rows.sort(key=lambda r:(order.index(r['label']),r['seed']))
    add('02-make-strategy','MAKE：初始化与接入后更新必须分开比较','metrics',rows,
        '相同下游设置；预训练另有计算成本。主测试与追加测试分别留存，非新的盲测。')
    rows = []
    for seed in seeds:
        for e in strategy['audits'][str(seed)]['c0_epochs']:
            rows.append(dict(label='ID下游',seed=seed,epoch=e['epoch'],split='validation',batch_size=256,
                             gauc=e['validation']['gauc'],logloss=e['validation']['logloss'],
                             online_train_logloss=e['online_train_logloss']))
        pretrain = source('work', f'c1-050-seed{seed}-epochs3/report.json')
        for e in pretrain['epochs']:
            rows.append(dict(label='内容预训练',seed=seed,epoch=e['epoch'],split='validation',batch_size=32,
                             gauc=e['validation']['gauc'],logloss=e['validation']['logloss'],
                             online_train_logloss=e['online_train_logloss']))
    add('03-training','训练诊断：ID泛化退步，内容分支具有不同的学习曲线','training',rows,
        '验证过程诊断；两分支结构与batch不同，不能归因为ID embedding单一因素。训练损失为在线批次均值。')
    rows=[]
    for seed in seeds:
        prefix='shared-make-strategy-v1-seed42' if seed==42 else f'post-main-make-seeds-v1--seed{seed}'
        r=extra_row(prefix+'--c4','P0联合');r['epoch']=0;rows.append(r)
        for epoch in (1,3,5):
            if epoch==3 and seed!=2026:
                r=rows_main([('c5','正轮冻结')])[seeds.index(seed)]
            else:
                key=(f'shared-pretraining-endpoints-v1-seed42--p{epoch}' if seed==42 else
                     'post-main-make-epochs-v1--p3-seed2026' if epoch==3 else
                     f'post-main-make-epochs-v1--endpoints-seed{seed}--p{epoch}')
                r=extra_row(key,'正轮冻结')
            r['epoch']=epoch;rows.append(r)
    add('04-pretraining-epochs','更多预训练并未稳定转化为更多下游收益','epochs',rows,
        '测试；P0参与联合训练，P1/P3/P5冻结接入。P0与正轮同时改变初始化及冻结策略，不是纯轮数效应。')
    add('05-retrieval','相同聚合结构下，内容检索优于本地ID与类目检索','metrics',
        rows_main([('b1-hard-enhanced','类目检索'),('b2-soft-enhanced','商品＋类目ID'),('g-id-item','商品ID'),
                   ('content-id-cosine-l1000','内容＋ID余弦'),('r4-l1000','内容余弦')]),
        '主测试；L1000→K50，完整聚合结构相同。结果限定于当前表示及检索实现。')
    add('06-aggregation','统计与语义注意力各有作用，叠加增量却不必明确','metrics',
        rows_main([('r1','普通注意力'),('r2','注意力＋统计'),('r3','语义注意力'),('r4','语义注意力＋统计'),('r5','仅长历史统计')]),
        '主测试；固定内容检索，L200→K50。“仅长历史统计”仍保留公共ID和短历史分支。')
    rows=[]
    for mechanism,label in [('r1','普通注意力'),('r4','完整MUSE')]:
        for length in (50,200,500,1000):
            for seed in seeds:
                if length in (200,1000):
                    key=mechanism if length==200 else mechanism+'-l1000'
                    r=rows_main([(key,label)])[seeds.index(seed)]
                else:
                    key=(f'shared-length-sensitivity-v1-seed42--shared-{mechanism}-l{length}' if seed==42 else
                         f'post-main-length-seeds-v1--{mechanism}-l{length}-seed{seed}')
                    r=extra_row(key,label)
                r['length']=length;rows.append(r)
    add('07-history-length','扩大候选历史有效，但最后一段增益具有种子差异','length',rows,
        '测试；所有长度固定K50。细线为各个种子，粗线为均值；不能由该测试曲线重新选择长度。')
    rows=[]
    for r in extra['comparisons']:
        source_name=r['pair']['source']
        if '/post-main-g1-validation.json/' in source_name or '/post-main-g3-validation.json/' in source_name:
            seed=int(source_name.rsplit('/',1)[1])
            rows.append(dict(label='G1两阶段投影' if 'g1-' in source_name else 'G3共享融合',seed=seed,
                             split='test',pair_index=r['pair_index'],delta_gauc=r['candidate_minus_baseline']['gauc'],
                             ci_low=r['gauc_delta_user_bootstrap_95_percentile'][0],
                             ci_high=r['gauc_delta_user_bootstrap_95_percentile'][1]))
    add('08-learned-retrieval','复杂检索没有稳定优势，必须展示失败种子','forest',rows,
        '追加测试；差值＝学习式检索−内容余弦；单项95%用户bootstrap区间，未作多重校正。')
    rows=[]
    for candidate,baseline,label in [('c6','c0','内容集成：统计＋MAKE − ID'),('r4-l1000','g-id-item','MUSE：内容检索 − 商品ID')]:
        for seed in seeds:
            for x,y in zip(frequency['models'][candidate+f'-seed{seed}'],frequency['models'][baseline+f'-seed{seed}']):
                assert x['training_target_exposure_frequency']==y['training_target_exposure_frequency']
                rows.append(dict(label=label,seed=seed,split='posthoc_test',frequency=x['training_target_exposure_frequency'],
                                 rows=x['test_rows'],delta_auc=x['auc']-y['auc'],delta_logloss=x['logloss']-y['logloss']))
    add('09-frequency','低频组收益更大：描述性证据，不是机制因果证明','frequency',rows,
        '测试事后分组；频次仅由训练目标曝光计算。点为三种子均值，无分组置信区间；频次0不等于新商品。')
    rows=[]
    for seed,r in g3['results'].items():
        rows.append(dict(label='最终检索行为',seed=int(seed),split='validation_diagnostic',
                         negative_slope_fraction=r['negative_semantic_slope_fraction'],
                         overlap_fraction=r['content_topk_overlap_fraction'],
                         selected_cosine=r['mean_selected_content_cosine_fused'],
                         reference_cosine=r['mean_selected_content_cosine_reference']))
    add('10-g3-diagnostic','G3异常种子的检索行为：内容排序方向出现反转','behavior',rows,
        '固定1024条验证位置的最终检查点诊断；与测试下降相容，但未证明训练因果或约束系数后的效果。')
    rows=[]
    for family in baselines['models']:
        for r in family['reports']:
            rows.append(dict(label=family['label'],budget=family['budget'],seed=r['seed'],model_id=r['model_id'],
                             split='test',phase=r['phase'],**{m:r['metrics'][m] for m in ('auc','gauc','logloss')}))
    add('11-baselines','适配基线提供背景，不能替代受控机制比较','metrics',rows,
        '同一测试集；结构与历史预算不同。包含额外内容检索对照，不是原表数值复刻或等计算量排名。')
    add('12-framework','两个项目的联系：内容信息 → 历史选择 → 兴趣聚合','diagram',[],
        '概念示意，不对应一条把所有模块串联起来的实测网络；原始图文编码器属于给定前提。')
    result=dict(scope='presentation_storyline_evidence',sources=provenance,figures=figures,
                seeds=seeds,test_rows=923211,policy='No new model selection, fitting or inference. Historical and test phases remain explicit.')
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    columns=['figure','label','seed','model_id','split','phase','epoch','length','frequency','rows','batch_size',
             'auc','gauc','logloss','online_train_logloss','delta_auc','delta_gauc','delta_logloss','ci_low','ci_high',
             'negative_slope_fraction','overlap_fraction','selected_cosine','reference_cosine','pair_index','budget']
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for key,figure in figures.items():
        for row in figure['rows']:writer.writerow(dict(figure=key,**row))
    for project in ('content','muse'):
        (roots[project]/'reports/storyline-evidence.json').write_text(text,encoding='utf-8',newline='\n')
        (roots[project]/'reports/storyline-chart-data.csv').write_text(buf.getvalue(),encoding='utf-8',newline='\n')
    print(f'Exported {len(figures)} figure specifications, {sum(len(f["rows"]) for f in figures.values())} rows.')


if __name__=='__main__':main()

"""Verify frozen test artifacts and publish complete descriptive summaries."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from evaluate_checkpoint import digest

NAMES={
 'c0':'ID 基线','c2':'逐商品内容融合','c3':'SimTier 相似度统计','c5':'MAKE 预训练冻结接入',
 'c6':'SimTier＋MAKE','c7':'SimScore 逐行为相似度',
 'r0':'近期历史＋普通注意力（L200）','r1':'内容检索＋普通注意力（L200）',
 'r2':'内容检索＋注意力与统计（L200）','r3':'内容检索＋语义注意力（L200）',
 'r4':'内容检索＋语义注意力与统计（L200）','r5':'内容检索＋仅长历史统计（L200）',
 'r1-l1000':'内容检索＋普通注意力（L1000）','r4-l1000':'内容检索＋完整聚合（L1000）',
 'b1-hard-enhanced':'类目检索＋完整聚合（L1000）',
 'b2-soft-enhanced':'商品与类目 ID 检索＋完整聚合（L1000）',
 'g-id-item':'商品 ID 检索＋完整聚合（L1000）',
 'content-id-cosine-l1000':'内容＋ID 等权检索＋完整聚合（L1000）'}
SEEDS=(42,2026,3407)


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('test-output','freeze','inventory','content-project','muse-project'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();d=read(a.test_output/'summary.json');f=read(a.freeze);inventory=read(a.inventory)
    state=read(a.test_output/'status.json')
    if state['phase']!='complete' or state['failed'] or d['freeze_sha256']!=digest(a.freeze):raise ValueError('Incomplete or changed freeze')
    if set(d['runs'])!=set(f['models']) or len(d['runs'])!=54:raise ValueError('Model set mismatch')
    expected_pairs={x['candidate']+'-vs-'+x['baseline'] for x in f['comparisons']}
    if set(d['comparisons'])!=expected_pairs or len(expected_pairs)!=66:raise ValueError('Comparison set mismatch')
    if digest(a.inventory)!=f['inventory_sha256']:raise ValueError('Inventory mismatch')
    validation={};resources={}
    for e in inventory['models']:
        name=e['id'];entry=d['runs'][name];run=a.test_output/name;r=read(run/'report.json')
        if (digest(run/'report.json')!=entry['report_sha256'] or digest(run/'test_predictions.npz')!=entry['predictions_sha256']
            or r['metrics']!=entry['metrics'] or r['rows']!=f['test_rows'] or r['freeze_sha256']!=digest(a.freeze)
            or r['model_sha256']!=f['models'][name]['model_sha256'] or r['config']!=f['models'][name]['config']
            or r['source_sha256']!=f['source_sha256'][e['family']] or r['evaluator_sha256']!=f['evaluator_sha256']
            or not r['test_evaluated'] or r['optimizer_created']):raise ValueError('Test artifact mismatch: '+name)
        training=Path(e['run'])/'report.json'
        if digest(training)!=e['report_sha256']:raise ValueError('Training evidence changed')
        validation[name]=read(training)['validation'];resources[name]={k:r[k] for k in ('seconds','peak_allocated_gib','peak_process_rss_gib','min_device_free_gib')}
    for pair in f['comparisons']:
        name=pair['candidate']+'-vs-'+pair['baseline'];comp=d['comparisons'][name]
        if read(a.test_output/'comparisons'/(name+'.json'))!=comp:raise ValueError('Comparison artifact changed')
        for side in ('baseline','candidate'):
            if comp[side]!=d['runs'][pair[side]]['metrics'] or comp[side+'_sha256']!=d['runs'][pair[side]]['predictions_sha256']:
                raise ValueError('Comparison provenance mismatch')
    models={}
    for name,label in NAMES.items():
        models[name]={'name':label}
        for split,source in [('validation',validation),('test',{k:v['metrics'] for k,v in d['runs'].items()})]:
            models[name][split]={}
            for metric in ('auc','gauc','logloss'):
                values=[source[f'{name}-seed{s}'][metric] for s in SEEDS]
                models[name][split][metric]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'values':values}
    pair_rows=[];pair_summaries=[]
    for pair in f['comparisons'][:22]:
        candidate=pair['candidate'].rsplit('-seed',1)[0];baseline=pair['baseline'].rsplit('-seed',1)[0]
        signs=[];values=[]
        for seed in SEEDS:
            comp=d['comparisons'][f'{candidate}-seed{seed}-vs-{baseline}-seed{seed}'];lo,hi=comp['gauc_delta_user_bootstrap_95_percentile']
            signs.append('正' if lo>0 else '负' if hi<0 else '跨零');values.append(comp['candidate_minus_baseline']['gauc'])
            pair_rows.append({'candidate':NAMES[candidate],'baseline':NAMES[baseline],'seed':seed,
                **{f'delta_{k}':v for k,v in comp['candidate_minus_baseline'].items()},'gauc_ci_low':lo,'gauc_ci_high':hi})
        pair_summaries.append({'candidate':candidate,'baseline':baseline,'mean_gauc_delta':statistics.mean(values),'interval_signs':signs})
    publication={'scope':'audited_frozen_test_publication','freeze_sha256':digest(a.freeze),
        'test_summary_sha256':digest(a.test_output/'summary.json'),'publication_script_sha256':digest(Path(__file__)),
        'models':models,'comparisons':pair_summaries,'resources':resources,
        'all_54_reports_and_prediction_hashes_verified':True,'all_66_comparison_artifacts_verified':True,
        'test_rows':f['test_rows'],'training_seeds':list(SEEDS),'test_evaluated':True}
    for repo in (a.content_project,a.muse_project):
        (repo/'reports/final-heldout-test-summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
        (repo/'reports/final-results-summary.json').write_text(json.dumps(publication,indent=2,ensure_ascii=False),encoding='utf-8')
        with (repo/'reports/final-test-paired-comparisons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(pair_rows[0]));writer.writeheader();writer.writerows(pair_rows)
        with (repo/'reports/final-test-model-metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['method','seed','auc','gauc','logloss'])
            for name,label in NAMES.items():
                for seed in SEEDS:
                    v=d['runs'][f'{name}-seed{seed}']['metrics'];writer.writerow([label,seed,v['auc'],v['gauc'],v['logloss']])
        selected=list(NAMES)[:6] if repo==a.content_project else list(NAMES)[6:]
        lines=['# 最终保留测试集结果','',
            '54 个模型和 66 项预先冻结的配对比较全部完成，完整产物摘要复核通过。测试结果未用于调整模型或选择训练检查点。本文表格只展示本项目方法；两仓库的完整机器可读证据一致。','',
            '固定 5% 用户队列，测试 923,211 条曝光、82,542 位用户、126,722 条点击。GAUC 有效用户 45,985 位，对应 700,297 条曝光（75.8545%）；其余用户未同时含两类标签，不参与 GAUC，但仍参与全局 AUC 和 LogLoss。','',
            '## 三种子指标','',
            'GAUC 列为均值 ± 样本标准差；AUC、LogLoss 为均值。标准差仅反映三个训练种子的波动，不是置信区间。','',
            '|方法|AUC ↑|GAUC ↑（均值 ± 标准差）|LogLoss ↓|','|---|---:|---:|---:|']
        for name in selected:
            v=models[name]['test'];lines.append(f"|{NAMES[name]}|{v['auc']['mean']:.6f}|{v['gauc']['mean']:.6f} ± {v['gauc']['sample_std']:.6f}|{v['logloss']['mean']:.6f}|")
        lines+=['','## 受控比较','',
            '差值均为前者减后者的绝对 GAUC 差，不是相对百分比。“正／负／跨零”按 seed=42、2026、3407 排列，表示各自 95% 用户配对区间与零的关系。每项 2000 次重采样，66 项未作多重校正，不能作为严格的确认性显著性清单。','',
            '|比较|平均 GAUC 差|三个种子的区间方向|','|---|---:|---|']
        for x in pair_summaries:
            if x['candidate'] in selected:
                lines.append(f"|{NAMES[x['candidate']]} − {NAMES[x['baseline']]}|{x['mean_gauc_delta']:+.6f}|{'／'.join(x['interval_signs'])}|")
        if repo==a.content_project:
            lines+=['','## 与验证结论的差异','',
                '五种内容融合方式相对 ID 基线的 GAUC 区间在三个种子均为正；但逐商品融合的平均 LogLoss 比 ID 基线更差，收益不能解释为所有指标都改善。','',
                '统计＋MAKE 相对统计的测试区间仅一个种子为正，另外两个跨零；相对 MAKE 的两个种子为正、一个跨零。验证阶段两项比较均为三种子正向区间，测试证据明显减弱。因此不能继续写成“两个模块在所有种子都稳定互补”。','',
                '逐行为相似度的平均测试 GAUC 高于完整组合，两个种子的配对区间为正、一个跨零；完整组合的平均 AUC 更高。模型排序依赖指标，不能将论文方案包装成必然逐级胜出的路线。','',
                'MAKE 初始化/冻结、预训练轮数、低频分组仍是单列的验证诊断，未作为这些测试比较的一部分。测试结果不能替代它们缺少的独立验证。']
        else:
            lines+=['','## 与验证结论的关系','',
                '统计特征、单独加入语义注意力、两种聚合下增加历史长度，以及 L1000 的内容检索相对类目/ID 检索，均保持三个种子 GAUC 正向区间。','',
                '已有统计后再加语义注意力，三个区间仍跨零。完整聚合相对仅长历史统计，两个种子跨零、一个为负；不能据此宣称 ID 与语义融合必然胜过统计分支。该“仅统计”模型仍保留公共 ID 和短历史分支。','',
                'L200 内容检索相对近期历史仅一个种子的区间为正，另外两个跨零；目标相关检索不是在每一种历史条件下都能体现稳定收益。','',
                '内容＋商品 ID 等权检索在三个种子的 GAUC 区间均为负，AUC、LogLoss 也均变差。这仅验证一种固定融合规则，不代表所有可学习检索、TWIN 或 MISS 无效。']
        lines+=['','## 证据与边界','',
            '三种子共享一个固定用户队列；未覆盖抽样队列变化。5% 用户队列不是全量 TAOBAO-MM，L1000 不是工业十万历史。公开固定 SCL 向量的下游适配不等于 SCL 编码器重训、原始工业网络和线上收益复刻。','',
            '[完整三种子逐模型指标](../reports/final-test-model-metrics.csv) · [全部 66 项配对比较](../reports/final-test-paired-comparisons.csv) · [统计汇总与验证指标](../reports/final-results-summary.json) · [完整测试审计结果](../reports/final-heldout-test-summary.json) · [冻结清单](../reports/final-test-freeze-v1.json)','']
        (repo/'docs/final-test-results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Verified and published 54 model results and 66 comparisons.',flush=True)


if __name__=='__main__':main()

"""Verify and publish SimTier boundary sensitivity artifacts."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];runs=a.runs;s=read(runs/'status.json');plan=read(runs/'plan.json')
 assert s['phase']=='completed' and not s['failed'] and len(s['completed'])==25
 cache=read(runs/'cache/summary.json');models={};comparisons={}
 for key,base in plan['reference'].items():
  path=Path(base['run']);assert sha(path/'report.json')==base['report_sha256']
  assert sha(path/'validation_predictions.npz')==base['predictions_sha256']
  for mode in plan['modes']:
   job=key+'-'+mode;folder=runs/job;r=read(folder/'report.json');c=read(runs/f'compare-{job}.json')
   assert r['config']==plan['configs'][key] and r['steps']==11924 and r['epochs'][0]['rows']==3052475
   assert r['parameters_updated'] and r['reload_max_abs_error']==0 and not r['test_evaluated']
   assert r['store_manifest_sha256']==plan['manifest_sha256']
   assert r['implementation_sha256']==read(path/'report.json')['implementation_sha256']
   assert r['histogram_cache_sha256']==sha(runs/'cache'/mode/'manifest.json')
   assert c['candidate']==r['validation'] and c['baseline']==read(path/'report.json')['validation']
   assert c['candidate_sha256']==sha(folder/'validation_predictions.npz') and c['baseline_sha256']==base['predictions_sha256']
   models[job]=dict(report=r,report_sha256=sha(folder/'report.json'));comparisons[job]=c
 for mode in plan['modes']:
  for split in ('train','validation'):
   x=cache[mode]['splits'][split];assert x['floor_reference_exact']
   assert x['sha256']==sha(runs/'cache'/mode/f'{split}.npy')
   if mode=='right_closed':assert x['omitted_pairs']==0
 result=dict(scope='post_main_validation_sensitivity',models=models,comparisons=comparisons,cache_audit=cache,total_seconds=sum(x['seconds'] for x in s['completed']),test_evaluated=False)
 (root/'reports/post-main-simtier-boundaries-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 text='''# SimTier 分桶敏感性：三种子结果

原伪代码字面实现确实遗漏部分有效计数，且仅 SimTier 模型三个种子的 GAUC 均下降。保留全部计数的右闭区间解释没有一致优势：不同模型与种子的差值方向不同。结果不支持静默替换原分桶并改写冻结主表。

[协议](post-main-simtier-boundaries-protocol.md)区分原 floor、paper_literal（原伪代码字面实现）和 right_closed（保留计数的右闭区间解释）。模型、初始化、数据、训练设置及 C6 的 MAKE 检查点均保持不变，只替换计数缓存。

## 缓存证据

|口径|划分|有效配对|遗漏配对|相对 floor 改变的曝光数|
|---|---|---:|---:|---:|
'''
 for mode in plan['modes']:
  for split,x in cache[mode]['splits'].items():text+=f"|{mode}|{split}|{x['eligible_pairs']:,}|{x['omitted_pairs']:,}|{x['rows_changed_vs_floor']:,}|\n"
 text+='''
原 floor 缓存已逐元素复核一致。paper_literal 在验证集遗漏约 0.098% 的有效目标—历史配对；该比例不是曝光比例。其变化还包含索引平移与边界归属，不能把指标变化全部归因于遗漏。right_closed 保留全部配对，但约 2.33% 验证曝光的直方图不同于 floor。

## 相对原 floor 的 GAUC 差

C3 表示仅 SimTier；C6 表示 SimTier＋MAKE。下表为“新口径 − 原 floor”。

|模型|种子|新口径|GAUC 差|95% 用户配对区间|
|---|---:|---|---:|---|
'''
 for key in sorted(plan['reference']):
  for mode in plan['modes']:
   c=comparisons[key+'-'+mode];lo,hi=c['gauc_delta_user_bootstrap_95_percentile'];model,seed=key.split('-seed')
   text+=f"|{model.upper()}|{seed}|{mode}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
 text+='''
C3 的 paper_literal 三个区间均负向；C6 两个负向、一个跨零。right_closed 在 C3 中一个负向、两个跨零，在 C6 中一个正向、两个跨零。跨零不代表统计等效；局部正向也不能证明整体优越。AUC 与 LogLoss 并不总与 GAUC 同向，完整指标保存在机器可读报告中。

## 解释与限制

这项检查补足的是实现忠实度与边界敏感性，不是提出新的推荐算法。论文公式存在歧义时，应同时公开字面实现及本地解释，而不是按最终指标选一种后称其为唯一原实现。

12 次完整训练及 12 项配对统计均通过配置、源文件、数据清单、缓存、预测指纹与重载核验；总耗时约 90 分钟，含缓存。沿用原数据规模和 batch=256，没有改动冻结主模型或重跑测试集。

所有区间使用 2,000 次用户级配对 bootstrap，条件于已训练模型及当前队列，无多重校正。这些是验证集补充证据，不能作为新的盲测结论。

完整指标与核验记录见[机器可读报告](../reports/post-main-simtier-boundaries-validation.json)。
'''
 (root/'docs/post-main-simtier-boundaries-results.md').write_text(text,encoding='utf-8')
 print('Verified and published 12 models and 12 comparisons')
if __name__=='__main__':main()

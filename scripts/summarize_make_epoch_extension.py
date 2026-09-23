"""Publish audited three-seed fixed-epoch MAKE evidence without reselecting models."""
import argparse,json,hashlib,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];base=a.work/'post-main-make-epochs-v1'
 status=json.loads((base/'status.json').read_text());assert status['phase']=='completed' and not status['failed']
 reports={};fingerprints={}
 for seed in (42,2026,3407):
  path=root/'reports/shared-paper-zero-reference-v1-seed42-audit.json' if seed==42 else base/f'paper-zero-seed{seed}/summary.json'
  r=json.loads(path.read_text());assert not r['test_evaluated']
  assert all(v['all_rows_and_recomputed_metrics_equal'] and v['config']['seed']==seed for v in r['runs'].values())
  reports[str(seed)]=r;fingerprints[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
 means={p:{m:statistics.mean(r['runs'][p]['metrics'][m] for r in reports.values()) for m in ('auc','gauc','logloss')} for p in ('p0_joint','p1','p3','p5')}
 result=dict(scope='three_seed_fixed_epoch_validation_extension',reports=reports,source_sha256=fingerprints,three_seed_means=means,test_evaluated=False)
 (root/'reports/post-main-make-epochs-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 text='''# MAKE 固定轮数：三种子结果

多轮预训练的收益并非单调，且存在种子差异。相对随机联合零轮，三轮接入在两个种子上有正向 GAUC 配对区间、一个跨零；五轮在一个种子上正向、一个跨零、一个负向。不能把单种子的改善解释为稳定的“训练越久越好”。

本实验遵循[固定协议](post-main-make-epochs-protocol.md)：零轮随机联合，正轮预训练后冻结。正轮与零轮同时改变初始化及冻结策略，总训练预算不同；这是本地策略比较，不是原论文数值复刻。

|种子|零轮联合 GAUC|1轮 GAUC|3轮 GAUC|5轮 GAUC|
|---|---:|---:|---:|---:|
'''
 for seed,r in reports.items():text+='|'+seed+'|'+ '|'.join(f"{r['runs'][p]['metrics']['gauc']:.6f}" for p in ('p0_joint','p1','p3','p5'))+'|\n'
 text+='\n|种子|预训练轮数|相对零轮 GAUC 差|95% 用户配对区间|\n|---|---:|---:|---|\n'
 for seed,r in reports.items():
  for p,c in r['comparisons'].items():
   lo,hi=c['gauc_delta_user_bootstrap_95_percentile'];text+=f"|{seed}|{p.split('-')[0][1:]}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
 text+='\n|三种子均值|AUC|GAUC|LogLoss|\n|---|---:|---:|---:|\n'
 for p,m in means.items():text+=f"|{p}|{m['auc']:.6f}|{m['gauc']:.6f}|{m['logloss']:.6f}|\n"
 text+='''
## 核验与故事线

两个追加种子的五轮轨迹，其前三轮与原三轮训练的权重、验证预测逐元素一致。2026 原主模型选第 2 轮，本次另训准确第 3 轮接入模型；3407 精确匹配后复用原第 3 轮。完整预测行、标签、用户及重算指标均通过审计。

训练轮数差异是 MAKE 的动机，预训练是应对方式，而轮数实验表明需要警惕收益饱和与退化。这里没有使用结果重新选主实验轮次，也没有为了得到单调曲线追加调参。

追加训练、验证和审计累计约 2 小时 41 分钟，不含种子 42 和复用模型的历史成本。保留 batch=32 的既有内容预训练协议及 batch=256 的共同下游协议。

全部结果是同一队列的验证集补充证据；每项区间使用 2,000 次用户级配对 bootstrap，条件于已训练模型，无多重校正。原单种子结果已被查看，本扩展不是未见结果的预注册研究。三种子均值没有独立置信区间，不能直接用于统计显著性判断。

完整审计及指标见[机器可读报告](../reports/post-main-make-epochs-validation.json)。
'''
 (root/'docs/post-main-make-epochs-results.md').write_text(text,encoding='utf-8')
 print(json.dumps(means,indent=2))
if __name__=='__main__':main()

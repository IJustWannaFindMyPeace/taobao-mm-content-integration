"""Publish all predefined group metrics with signed, descriptive contrasts."""
import argparse,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--muse-project',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.report.read_text());root=Path(__file__).resolve().parents[1]
    text='''# 8/9 组事后测试诊断

本报告对齐论文的分组数量，不声称还原其未公开分组。复用冻结模型的完整测试预测，未训练或选择模型。按实体数量平衡，商品频次来自训练目标曝光，用户长度来自上限 1000 的输入历史；并列按实体 ID 哈希拆分。分组全集来自测试输入，未使用标签或分数确定分组。

以下差值为三个种子的平均带符号差值，无子组置信区间。ΔAUC/ΔGAUC 为候选减基线；LogLoss 改善为基线减候选，正值表示改善。完整每种子指标和 GAUC 有效用户覆盖保存在机器可读报告中。商品组内 GAUC 只用该组曝光，不能当成完整用户 GAUC；用户长度组保留用户全部曝光。
'''
    for name,title,baseline,candidate in [
        ('cikm_item_8','CIKM：商品频次 8 组，SimTier＋MAKE 对 ID 基线','c0','c6'),
        ('muse_item_9','MUSE：商品频次 9 组，完整聚合对普通聚合','r1-l1000','r4-l1000'),
        ('muse_user_9','MUSE：用户历史长度 9 组，完整聚合对普通聚合','r1-l1000','r4-l1000')]:
        scheme=d['schemes'][name];text+=f'\n## {title}\n\n|组|实体数|曝光数|值域|ΔAUC|ΔGAUC|LogLoss 改善|\n|---|---:|---:|---|---:|---:|---:|\n'
        assert sum(x['rows'] for x in scheme['groups'])==d['test_rows']
        for i,g in enumerate(scheme['groups']):
            delta={}
            for metric in ('auc','gauc','logloss'):
                values=[]
                for seed in (42,2026,3407):
                    b=scheme['models'][f'{baseline}-seed{seed}']['metrics'][i][metric]
                    c=scheme['models'][f'{candidate}-seed{seed}']['metrics'][i][metric]
                    if b is not None and c is not None:values.append((b-c) if metric=='logloss' else (c-b))
                delta[metric]=f'{statistics.mean(values):+.6f}' if len(values)==3 else 'N/A'
            text+=f"|{g['group']}|{g['entities']}|{g['rows']}|{g['value_min']}–{g['value_max']}|{delta['auc']}|{delta['gauc']}|{delta['logloss']}|\n"
    text+='''
## 阅读边界

同频率商品可能落在不同组，零频组不能等同新商品或完全没有历史交互。长度为 1000 的多个组体现输入截断，不能解释为不同的真实终身历史长度。分组差异可能还包含曝光量、标签分布、内容覆盖和用户组成差异，不能单独作因果归因。

本结果是对原固定分组诊断的补充，不替换原报告，也不把“分组数量对齐”提升为“原论文分组完全复现”。全量测试覆盖、每种子预测指纹与组成员指纹见[完整报告](../reports/posthoc-paper-groups-v1.json)。[分组协议](paper-groups-protocol.md)记录规则。
'''
    for repo in (root,a.muse_project):
        (repo/'reports/posthoc-paper-groups-v1.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
        (repo/'docs/posthoc-paper-groups-v1.md').write_text(text,encoding='utf-8')
    (a.muse_project/'docs/paper-groups-protocol.md').write_text((root/'docs/paper-groups-protocol.md').read_text(encoding='utf-8'),encoding='utf-8')


if __name__=='__main__':main()

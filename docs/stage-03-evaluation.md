# 阶段 3 · 统一评估链路

本文记录该阶段的验证范围；当前项目状态见 [README](../README.md)。

## 指标契约

[metrics.py](../src/metrics.py) 接收完整集合的标签、logit 和用户 ID。输入必须非空、长度一致、标签为 0/1 且分数有限。

- AUC：按分数排序并计算平均秩，相同分数的正负样本对计 0.5；单类别集合返回 null。排序复杂度 O(N log N)，不构造正负样本笛卡尔积。
- GAUC：在指定评估集合内聚合同一用户的全部记录，仅保留同时具有正负标签的用户，以各有效用户的曝光条数加权；全部用户无效时返回 null。同步报告排除用户、排除行数和有效行覆盖率。
- LogLoss：直接通过 logaddexp(0, logit) − label × logit 计算，避免先转概率造成极端分数溢出。

AUC 可直接使用 logit 排序；不需要将 logit 误当成概率。跨 batch 汇总原始预测后统一计算指标，不能平均各 batch 的 AUC 或 GAUC。

## 真实预测检查

[evaluate_probe.py](../scripts/evaluate_probe.py) 核对输入 SHA256、连续行号和真实标签，拒绝缺失、重复或错序记录，再为已保存的训练分片预测补充用户 ID。

```powershell
python scripts/evaluate_probe.py --input ../taobao-mm-probe/probe.parquet --predictions ../id-probe/train_predictions.pt --training-report reports/id-probe.json --report reports/metrics-probe.json
python -m unittest discover -s tests -v
```

路径按实际产物目录调整。当前评估入口将全部预测保存在 CPU 内存，仅适用于受控子集；全量评估仍需外部排序或分区聚合实现。

|训练分片诊断|结果|
|---|---|
|预测记录|1,890|
|AUC|0.671258|
|GAUC|0.640680|
|LogLoss|0.382571|
|有效用户 / 全部用户|78 / 156|
|GAUC 有效曝光 / 全部曝光|1,403 / 1,890（74.23%）|

这些数值仅来自用于训练的同一分片，不能作为验证集、测试集或论文收益成绩。完整用户聚合仅指当前 probe 内的完整记录，不意味着用户在官方数据中的全部曝光已经收齐。机器记录见 [metrics-probe.json](../reports/metrics-probe.json)。

## 验证

本阶段 6 项指标测试与 3 项 ID/mask 测试通过。覆盖完美/逆序/并列排序、随机样本与独立成对比较参考结果、用户加权与排除、分批拼接与顺序变化、单类别、极端 logit 和非法输入。

## 正式数据协议待办

固定数据 revision 的目录统计见 [data-layout.json](../reports/data-layout.json)。这些数字来自文件清单，未扫描全量行或确认用户是否跨分片。不能根据参考 README 的分片数量假定当前版本的布局。

正式协议采用以下约束，具体用户集合与验证比例尚未冻结：

1. 保留官方 train/test 边界；测试标签不参与用户抽样或超参数选择。
2. 以固定种子对用户 ID 做稳定哈希抽样，跨所有相关分片提取被选用户的记录，保存版本、文件和行级来源。只下载一个小分片无法满足此条件。
3. 在官方训练边界内建立验证集。若没有可核验时间字段，采用用户互斥的确定性划分，并明确该验证设置包含新用户泛化条件；与官方 train/test 时间划分分别说明。
4. 可训练词表只从训练划分构建，验证/测试未知 ID 使用 OOV。内容查表覆盖与可训练 ID 词表分开。
5. 子集扫描、内容覆盖率、重复记录及预测行数检查通过后，再发布固定 manifest 并启动正式 C0。

本阶段完成评估机制与数据规模检查，尚未完成正式数据集的获取和划分。

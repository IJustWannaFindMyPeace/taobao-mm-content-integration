# TAOBAO-MM · Content Integration

**最新结果：**主实验54份模型及补充73份检查点的测试评估已完成，追加93项统计比较全部归档。先读[结果与故事线](docs/supplementary-test-results.md)和[当前覆盖范围](docs/current-reproduction-status.md)。追加测试单独标注，不混作新的盲测。


[当前复现状态与两项目故事线](docs/current-reproduction-status.md)汇总已完成证据、适配边界及剩余工作。


基于 TAOBAO-MM 固定内容向量，研究内容表示如何融入 CTR 推荐模型。对应 CIKM 2024 论文 [Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights](https://arxiv.org/html/2407.19467v1)，为非官方下游适配复现。

**研究主线：ID 信息不足 → 内容融合方式 → 内容与 ID 的优化差异 → MAKE 与统计特征的互补性。**

## 阅读入口

- [最新论文覆盖审计：已做、适配与可补缺口](docs/paper-coverage-audit-2026-09-23.md)

- [两项目最终故事线：痛点、改进、证据与局限](docs/final-storyline.md)
- [本项目最终测试结果](docs/final-test-results.md)
- [共同评估协议与冻结清单](docs/final-evaluation-protocol-v1.md)
- [方法覆盖与适配边界](docs/reproduction-scope.md)
- [后续 MUSE 项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)

## 后续补充研究

已补充 [8/9 组事后诊断](docs/posthoc-paper-groups-v1.md)。商品分组保留同频并列限制；用户组大多同为截断长度 1000，不能据此判断真实长度趋势。

第一轮冻结实验完成后，已补[测试集低频分组](docs/posthoc-test-frequency-v1.md)，明确标为事后描述性分析。MAKE 训练策略已补齐三个种子的验证证据；DIN/TWIN/MISS 的本地适配补充已完成，共 21 次验证训练和 12 项配对比较；原始工业完整模型仍不在覆盖范围。见[统一收尾说明](docs/study-completion.md)。

## 完成状态

两个项目共 54 个固定模型、三个训练种子、66 项配对比较全部完成。每组评估完整 923,211 条保留测试曝光；测试前所有模型的完整验证回放与原预测逐元素一致。测试结果未用于重新训练、调参或替换主表模型。

|本项目方法|三种子平均测试 GAUC|
|---|---:|
|ID 基线|0.573708|
|逐商品内容融合|0.577532|
|SimTier 相似度统计|0.590215|
|MAKE 预训练冻结接入|0.590055|
|SimTier＋MAKE|0.591200|
|SimScore 逐行为相似度|0.593142|

五种内容方案相对 ID 的 GAUC 区间均为三种子正向，但统计＋MAKE 相对单模块的测试互补证据弱于验证阶段。逐行为相似度的平均 GAUC 更高，组合的平均 AUC 更高，不能按单个指标宣称一种方式普遍最佳。完整 AUC、LogLoss、波动与区间见[结果报告](docs/final-test-results.md)。

## 实验协议

使用固定 5% 用户队列、训练集构建的 ID 词表、128 维固定内容向量；下游 batch=256、单轮，种子 42/2026/3407，SparseAdam 与 AdamW 学习率均为 0.001。内容预训练采用独立预算。5% 为既定研究规模，不是全量数据或硬件最大能力的证明。

设备为单张 RTX 5070 12 GB、32 GB RAM。训练策略、预训练轮数及分组分析是明确单列的验证诊断，不与三种子保留测试主表混合。论文式零轮联合训练与额外随机冻结控制组见[零轮参考分析](docs/shared-paper-zero-reference-v1.md)。

## 代码与产物

- 模型：`src/c0.py`、`src/side_information.py`、`src/simtier.py`、`src/make_fusion.py`。
- 训练：`scripts/train_c0.py`、`scripts/train_content.py`；共同参数见 `configs/shared-downstream-protocol-v1.json`。
- 只读评估：`scripts/evaluate_checkpoint.py`；冻结与队列：`scripts/freeze_test_inventory.py`、`scripts/run_frozen_test.py`。
- [逐模型测试指标](reports/final-test-model-metrics.csv)、[全部配对比较](reports/final-test-paired-comparisons.csv)、[完整统计汇总](reports/final-results-summary.json)。

数据文件、预测数组和模型权重不提交到仓库；公开配置、实现、指标和可核验摘要。数据准备与阶段命令见[数据契约](docs/dataset-contract.md)和[实施路线](docs/roadmap.md)。阶段文档保留当时的实验条件，早期 batch=32 结果不替代当前共同主表。

## 复现边界

不宣称复现 SCL 图文编码器训练、CLIP 对照、私有工业网络和线上收益。SimTier 分桶、MAKE 结构及训练策略的本地适配均单列。两项目共享研究协议，但 MAKE 不是 MUSE 必需的前置模块，完整模型分数差不是单一模块的升级收益。

[MAKE 固定轮数三种子结果](docs/post-main-make-epochs-results.md)已完成：收益不随预训练轮数单调增加，保留种子差异。[SimTier 分桶敏感性结果](docs/post-main-simtier-boundaries-results.md)已完成。

[本轮补充工作收尾核验](docs/reproduction-closeout.md)：96项产物一致性检查通过；完成范围与原论文限制分别说明。

[两论文结果阅读指南](docs/storyline-reading-guide.md)：按问题、方法、证据与局限理解完整迭代路径，无需先记实验代号。

[统一追加测试结果](docs/supplementary-test-results.md)已归档：73份验证重放、73份测试评估与93项比较全部完成。原验证报告保留时点记录，最新结论单独发布。

# TAOBAO-MM · 内容集成

研究固定内容表示如何进入点击预测模型，以及预训练与下游更新策略如何影响收益。

两个项目共同复现CIKM 2024与MUSE的下游研究问题，采用TAOBAO-MM固定5%用户队列和三个训练种子。主实验54份模型及追加73份检查点已完成统一测试，159项配对比较已归档。这是公共协议下的适配与机制验证，不是原工业系统的原样复刻。

## 从研究问题开始阅读

1. [研究叙事](docs/research-narrative.md)：问题、方法、证据与解释。
2. [论文对照与差异分析](docs/paper-comparison-and-analysis.md)：哪些结论一致，哪些不同，原因证据支持到哪里。
3. [图表与数据](docs/storyline-figures.md)：12张PNG/SVG图及三种子原始数值。
4. [PPT内容计划](docs/presentation-plan.md)：正文、附录与学习顺序。

[关联项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)共享数据与评估协议；两个项目衔接的是研究问题，不是将所有模块串接成一个模型。

[数据字段字典](docs/dataset-fields.md)：官方字段含义、模型输入顺序与本地索引约定。

[正式抽样与划分](docs/sampling-and-splits.md)：5% 用户规则、训练/验证/测试来源与评估边界。

[Vector 内容融合](docs/vector-integration.md)：方法适配、实验协议、正式结果与结论边界。

[SimScore 逐行为相似度](docs/simscore-integration.md)：显式关系特征、融合结构与对照结果。

[SimTier 相似度统计](docs/simtier-integration.md)：分桶计数、预测层接入与分桶敏感性；三种接入方式的对照见[研究主线](docs/research-narrative.md#内容接入方式对照)。

## 完整实验与审计

- [主测试结果](docs/final-test-results.md)与[追加测试结果](docs/supplementary-test-results.md)。追加测试发生在主测试已查看之后，单独标注。
- [当前覆盖范围](docs/current-reproduction-status.md)、[整体实验审计](docs/cross-project-completion-audit.md)与[叙事及展示就绪审计](docs/narrative-readiness-audit.md)。
- [全部追加配对表](docs/supplementary-test-comparisons.md)、[公共下游配置](configs/shared-downstream-protocol-v1.json)与[执行协议](docs/supplementary-test-protocol.md)。
- [方法覆盖与适配边界](docs/reproduction-scope.md)、[实施路线](docs/roadmap.md)。阶段文档保留历史条件，不替代当前正式结果。

数据文件、预测数组与权重保留在本地，不提交到仓库；公开代码、配置、指标和来源指纹。图表数据与统计图绘制说明见展示计划，工程调试记录不作为研究主线。

## 结论边界

内容与训练策略有价值，但组合、更多预训练和更复杂检索并不始终带来增量。正结果、负结果与跨零区间一并保留；AUC、GAUC与LogLoss分别解释。

没有重训SCL编码器或补齐CLIP/OpenCLIP/I2I原表示对照，没有完成全量数据、100K历史或线上试验。5%不是显卡最大规模；适配模型的结论不能直接外推到原工业网络。

## 研究表达

以受控实验明确回答方法是否有效，以结构与行为诊断解释结果；保留影响判断的适配范围和统计信息。细部机制的唯一归因属于可选扩展，不作为现有结论的前置条件。教学问答不纳入公开研究正文。见[表达审计](docs/research-expression-audit.md)。

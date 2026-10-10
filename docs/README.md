# 项目文档

建议先阅读[项目概览](../README.md)，再按方法、数据协议和测试结果查阅。历史阶段记录保留当时的配置与结论，不作为当前完成状态。

## 项目概览与方法

- [从多模态表示到长历史兴趣：研究叙事](research-narrative.md)
- [从多模态表示到长历史兴趣：结果阅读指南](storyline-reading-guide.md)
- [Vector：逐商品多模态融合](vector-integration.md)
- [SimScore：逐行为语义相似度](simscore-integration.md)
- [SimTier：目标相关的相似度统计](simtier-integration.md)
- [MAKE：推荐预训练（点击率预测任务）与接入](make-integration.md)
- [论文术语与实现命名](paper-terminology.md)
- [方法来源、实现差异与覆盖范围](reproduction-scope.md)
- [两篇论文对照与结果差异分析](paper-comparison-and-analysis.md)

## 数据与评估协议

- [TAOBAO-MM 字段字典与模型输入](dataset-fields.md)
- [TAOBAO-MM 数据定义与覆盖率核验](dataset-contract.md)
- [正式用户抽样与评估划分](sampling-and-splits.md)
- [最终评估候选清单与验证回放门槛](final-evaluation-protocol-v1.md)
- [统一追加测试评估：范围与门槛](supplementary-test-protocol.md)
- [8/9 组补充诊断协议](paper-groups-protocol.md)

## 测试结果与诊断

- [最终保留测试集结果](final-test-results.md)
- [统一追加测试：结果与故事线](supplementary-test-results.md)
- [追加测试：完整配对结果](supplementary-test-comparisons.md)
- [测试集商品频次分组：主实验后的补充分析](posthoc-test-frequency-v1.md)
- [8/9 组事后测试诊断](posthoc-paper-groups-v1.md)
- [故事线图表与原始数值](storyline-figures.md)

## 覆盖状态与实施路线

- [实验覆盖与评估状态](current-reproduction-status.md)
- [实施路线](roadmap.md)

## 审计与历史记录

<details>
<summary>展开协议演进、阶段实验与审计记录</summary>

以下记录用于追踪实验过程。带日期的审计和展示计划只说明对应版本，不代表最新运行环境或在线演示稿状态。

- [多模态表示集成项目：收尾审计](content-project-closeout-audit.md)
- [两项目整体收尾审计](cross-project-completion-audit.md)
- [研究表达审计](research-expression-audit.md)
- [追加测试前：设计与执行审计](supplementary-pretest-audit.md)
- [主实验证据清单（历史归档）](completion-checklist.md)
- [实验导读：从研究问题到证据](experiment-guide.md)
- [主实验阶段故事线（历史归档）](final-storyline.md)
- [研究叙事与展示就绪审计](narrative-readiness-audit.md)
- [论文覆盖审计：2026-09-23](paper-coverage-audit-2026-09-23.md)
- [MAKE 固定预训练轮数：三种子扩展协议](post-main-make-epochs-protocol.md)
- [MAKE 固定轮数：三种子结果](post-main-make-epochs-results.md)
- [MAKE 策略多种子补充结果](post-main-make-seed-extension.md)
- [SimTier 分桶边界敏感性：固定协议](post-main-simtier-boundaries-protocol.md)
- [SimTier 分桶敏感性：三种子结果](post-main-simtier-boundaries-results.md)
- [展示计划与材料就绪情况](presentation-plan.md)
- [本轮复现补充：收尾核验](reproduction-closeout.md)
- [两项目研究主线与证据约束](research-storyline.md)
- [共同协议核心结果：三个种子的证据](shared-core-three-seed-results.md)
- [共同协议下的分组更新](shared-group-diagnostics-v1.md)
- [共同协议下的 MAKE 训练策略诊断](shared-make-strategy-v1.md)
- [MAKE：论文式零轮参考与预训练接入](shared-paper-zero-reference-v1.md)
- [MAKE 预训练轮数诊断：固定 P=0/1/3/5](shared-pretraining-epochs-v1.md)
- [共同主实验规模：5% 用户队列](shared-scale-decision.md)
- [阶段 1 · 真实训练分片验证](stage-01-real-data.md)
- [阶段 2 · 真实数据 ID 基线工程验证](stage-02-id-probe.md)
- [阶段 3 · 统一评估链路](stage-03-evaluation.md)
- [阶段 4 · 固定用户曝光子集](stage-04-user-cohort.md)
- [阶段 5 · 嵌套扩容与商品属性关联](stage-05-scale-and-items.md)
- [阶段 6 · 历史文件传输与提取准备](stage-06-history-transfer.md)
- [阶段 7 · 专用环境与历史、多模态表示提取](stage-07-environment-and-features.md)
- [阶段 8 · 统一特征存储与数据加载](stage-08-feature-store.md)
- [阶段 9 · C0 ID 基线](stage-09-c0.md)
- [阶段 10 · SimTier 多模态融合对照](stage-10-simtier.md)
- [多模态分支与 MAKE 预训练](stage-11-content-pretraining.md)
- [MAKE 接入与训练轮数对照](stage-12-make-factorial.md)
- [SimTier 与 MAKE 的互补性对照](stage-13-combined-fusion.md)
- [Vector 与 SimScore 融合对照](stage-14-side-information.md)
- [低频诊断与下一档数据预算](stage-15-frequency-and-budget.md)
- [阶段 16：5% 数据准备与资源验证](stage-16-scale-preparation.md)
- [阶段 17：5% 多模态表示学习曲线与核心对照](stage-17-five-percent-core-study.md)
- [阶段 18：多模态融合的训练随机性验证](stage-18-training-seed-validation.md)
- [阶段 19：三种子核心汇总与 MAKE 策略归因](stage-19-core-summary-and-make-controls.md)
- [阶段 20：5% MAKE 策略与训练轮数证据](stage-20-make-strategy-evidence.md)
- [阶段 21：商品频次与多模态表示覆盖率分组诊断](stage-21-frequency-and-coverage.md)
- [阶段 22：故事线证据审计与最终测试准入](stage-22-storyline-readiness.md)
- [两项目故事线审计 · 2026-09-22](storyline-audit-2026-09-22.md)
- [两项目研究收尾：故事线与证据边界](study-completion.md)

</details>

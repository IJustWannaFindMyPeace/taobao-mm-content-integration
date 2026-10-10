# 两项目研究收尾：故事线与证据边界

> 文档性质：阶段记录／历史协议。本文中的规模、配置、验证结果及待办状态属于记录时点；当前结果以[主测试](final-test-results.md)、[补充测试](supplementary-test-results.md)和[覆盖说明](current-reproduction-status.md)为准。

> 历史阶段记录。后续统一追加测试已完成，当前证据与完成范围见[追加测试结果](supplementary-test-results.md)。


已排定的单卡下游适配实验及补充批次已完成，但论文覆盖尚未闭环。最新[逐项覆盖审计](paper-coverage-audit-2026-09-23.md)确认 G1/G3 等可补缺口，不能表述为仅剩不可复现部分。它不是两篇论文全部工业系统与所有表示模型的逐项完整复现。主实验与事后补充分开报告，负向、跨零和指标冲突均保留。

## 连续的研究问题

|阶段|痛点|改进与对照|观察|局限|
|---|---|---|---|---|
|CIKM：多模态融合|ID 交互信息不足，多模态表示如何进入 CTR？|直接多模态表示、相似度分数、SimTier 统计、MAKE 等内部对照|多种多模态方案相对 ID 基线的测试 GAUC 改善|不重训 SCL 编码器，不把下游结果当作表示训练证明|
|CIKM：优化差异|多模态分支与 ID 分支是否需要不同训练方式？|预训练/随机 × 冻结/联合，以及 ID 多轮曲线|MAKE 策略已有三个种子的验证证据；ID 多轮曲线下降|策略确认仍是验证证据；更细的预训练轮数扫描仍仅单种子|
|MUSE：长历史检索|历史太长，如何找目标相关行为？|多模态表示、类目、ID 与融合检索的共同聚合比较|多模态检索有竞争力，等权多模态表示＋ID 融合的测试结果更差|不能外推为所有学习型检索均无效|
|MUSE：聚合|选中历史后，怎样利用语义？|普通注意力、统计、SA-TA、完整组合|统计和语义注意力各自有测试收益；组合的额外收益不稳定|不强行写成每次迭代都提升|
|MUSE：历史覆盖|较长输入是否有用？|固定 K，比较 L200 与 L1000|两种聚合均有三种子正向测试证据|没有验证十万历史，也不能保证继续加长单调提升|
|补充：DIN|更复杂的短历史注意力是否更好？|局部激活与短历史点积对照|三个 GAUC 区间均跨零|不能确认优劣或等效；不是完整工业 DIN|
|补充：TWIN|两阶段评分一致是否足够？|四头一致性检索与相同聚合的多模态检索|多模态检索三个种子均更优，GAUC 区间不跨零|去重覆盖数不同；不是一致性本身的纯因果实验|
|补充：MISS|协同和多模态表示双路是否更好？|相同双路容量下的检索对照，另做共享敏感性|主比较 AUC 均提高，GAUC 无稳定优势；共享也无一致收益|不是索引树召回复现；分支重合和参数量需分别解释|

两个项目的衔接是“多模态表示如何用于 CTR → 多模态表示如何用于长历史检索与聚合”。MAKE 不被强行设为 MUSE 的前置条件；两个最终模型的分数差也不解释成单个模块收益。

## 完成清单

- 冻结主实验：54 个模型、66 项比较，完整测试集 923,211 条曝光。
- MAKE：训练策略补齐三种子；测试低频分组另作事后描述分析。
- 基线适配补充：DIN 6 次、TWIN 6 次、MISS 9 次，共 21 次完整验证训练与 12 项配对比较。
- 共同数据、词表和训练协议保持一致；补充结果不回填冻结主表。
- 收尾时核对两个项目共 20 个冻结源文件的哈希、共享协议和冻结清单，均未改变。

5% 是本研究已采用并完成的规模，不是测得的单卡最大容量。12 GB 显存支持本研究；当前实现的全量训练仍未经验证。更大规模需要重新评估 embedding 与优化器状态、数据加载和整体实验成本，不能只靠缩小 batch 推断可行性。

## 尚未覆盖的复现范围

SCL 图文编码器重训、OpenCLIP/I2I 表示对照、完整工业 DIN/TWIN/MISS、全量数据、十万历史和线上收益仍未覆盖。新增适配基线及 MAKE 策略确认仍只有验证证据。当前测试集已经被查看，后续扩展不能将其称为新的盲测。

补充实验解决了本地机制对照缺口，未消除上述边界。本轮不再为获得正向结果追加调参。

## 阅读顺序与结果入口

1. [两项目故事线](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/final-storyline.md)：先理解问题与实验对照。
2. [多模态融合主结果](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/final-test-results.md)与 [MUSE 主结果](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro/blob/main/docs/final-test-results.md)：核对冻结测试证据。
3. [MAKE 策略](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/post-main-make-seed-extension.md)：理解训练阶段差异。
4. [DIN](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro/blob/main/docs/post-main-din-results.md)、[TWIN](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro/blob/main/docs/post-main-twin-results.md)、[MISS](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro/blob/main/docs/post-main-miss-results.md)：理解补充证据与不同预算口径。

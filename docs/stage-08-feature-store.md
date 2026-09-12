# 阶段 8 · 统一特征存储与数据加载

## 数据契约

统一输入保留全部曝光，以训练划分构建可训练 ID 词表。用户、用户属性、目标商品及属性按字段分别映射；目标商品与历史商品共享词表，目标类目与历史类目共享词表。词表只包含训练曝光及训练用户历史，验证和测试数据不参与构建。

|状态|表示|处理规则|
|---|---|---|
|padding|ID 索引 0|不参与历史注意力|
|未知 ID|ID 索引 1|保留行为，使用 OOV embedding|
|缺失内容|内容行索引 0，availability=false|保留曝光和 ID 信息；提供零占位向量及独立可用标记|
|可用内容|非零内容索引，availability=true|读取 int8 向量并按 1/127 解量化|

未知 ID 可以具有可用内容；已知 ID 也可能缺内容。二者的标记独立。零占位向量仅是数据接口表示，不能把它的零相似度当作真实语义证据。检索、SimTier 和语义注意力的缺失内容分支需在模型规格中明确。

## 存储与读取

每位用户的完整 1,000 位置历史仅存一次，每条曝光保存用户行号。训练与验证引用训练用户表，测试引用测试用户表。商品/类目/内容索引使用 int32，原始有符号 ID 以排序词表保存。所有数组通过 NumPy memmap 在 CPU 侧读取，不将全表内容预载入 GPU。

[FeatureStore.batch](../src/feature_store.py) 支持 1–1,000 的历史长度，读取末尾位置作为输入；内容按需加载和解量化。短历史模型与长历史模型可以从相同输入衍生。源行号随曝光保留，便于预测文件与原数据对齐。

## 构建与核验

```powershell
python scripts/build_feature_store.py --items-dir ../user-cohort-v1-010-items --items-report reports/item-join-010.json --history-dir ../user-cohort-v1-010-histories --history-report reports/history-extraction-010.json --content-dir ../user-cohort-v1-010-content --content-report reports/content-extraction-010.json --output-dir ../shared-feature-store-v1 --report reports/feature-store-010.json
python scripts/verify_feature_store.py --store ../shared-feature-store-v1 --items-dir ../user-cohort-v1-010-items --history-dir ../user-cohort-v1-010-histories --report reports/feature-store-verification.json
```

构建入口核对各输入摘要和 revision，仅接受空输出目录。完成后发布 manifest，包含字段、词表、输入输出 SHA256。核验入口检查全部数组摘要、全部标签、源行号和目标映射，并对 50/200/1,000 长度历史进行原始记录回查。

## 实验边界

验证划分采用用户互斥，因此验证用户 ID 映射为 OOV 是预期行为；它不能单独说明模型对已有用户未来行为的泛化能力。正式训练前仍需确定公共模型结构、优化器配置和缺失内容分支。该存储契约可供 CIKM 与 MUSE 共享，但不代表模型对照已完成。

## 实测结果

存储构建约 345.5 秒，进程峰值工作集 2.86 GiB，数组合计 2.77 GiB。商品词表包含 14,997,581 个位置（含 PAD/OOV）。

|集合|曝光数|目标 ID OOV 比例|用户 ID OOV 比例|缺内容曝光|
|---|---:|---:|---:|---:|
|训练|609,902|0.00%|0.00%|12|
|验证|151,947|14.81%|100.00%|1|
|测试|185,253|15.34%|87.43%|6|

全部数组摘要、全部标签、全部源行号与目标 ID 映射检查通过；50/200/1,000 长度的抽样历史回查通过。19 条缺内容曝光全部保留。构建统计见[存储报告](../reports/feature-store-010.json)，核验见[验证报告](../reports/feature-store-verification.json)。

测试侧用户 OOV 比例较高，后续结果需分别报告已见/未见用户，并保留统一评估集合。用户 OOV 不能代替目标商品低频分组，两者的诊断目的不同。模型实验尚未运行。

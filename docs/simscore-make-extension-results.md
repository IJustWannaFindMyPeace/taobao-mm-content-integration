# SimScore＋MAKE：组合增量评估

在逐行为相似度特征上加入冻结 MAKE 特征，未进一步提高平均 GAUC。该实验在既有测试结果已知后提出，作为追加对照单独报告。

## 设置与实现

沿用同一数据划分、batch 256、单轮训练、历史窗口 200 和三个种子。SimScore 与 MAKE 两路特征同时接入；MAKE 参数冻结，接入投影参与训练。详见[固定协议](simscore-make-extension-protocol.md)。

## 三种子测试结果

|种子|AUC|GAUC|LogLoss|
|---|---:|---:|---:|
|42|0.618512|0.588875|0.389756|
|2026|0.620436|0.592011|0.391042|
|3407|0.618830|0.590184|0.389915|

|比较对象|对象平均 GAUC|组合平均 GAUC|GAUC 相对变化|
|---|---:|---:|---:|
|ID 基线|0.573708|0.590357|+2.90%|
|SimScore|0.593142|0.590357|-0.47%|
|MAKE|0.590055|0.590357|+0.05%|
|SimTier＋MAKE|0.591200|0.590357|-0.14%|

相对变化以比较对象的三种子平均 GAUC 为分母。

## 结果理解与范围

- 相对 ID 基线，三个种子的 GAUC 差值 95% 用户配对区间均高于零。
- 相对 SimScore，三个种子 GAUC 点估计均下降，两个区间低于零，一个包含零；平均 AUC 则从 0.617053 升至 0.619259。整体 AUC 与用户内排序指标 GAUC 不必同向变化。
- 相对 MAKE，平均 GAUC 仅增加约 0.05%，一个种子区间高于零，两个包含零，没有稳定增量证据。
- 相对 SimTier＋MAKE，平均 GAUC 下降约 0.14%，两个种子区间低于零，一个包含零。

两条路径都利用商品多模态信息，组合并不必然产生互补收益。本结果说明当前冻结迁移与单轮下游训练设置下，新增 MAKE 未提高 SimScore 的 GAUC；特征重叠或优化相互影响仍是可能解释，并未被单独验证。

每项比较使用 2,000 次用户配对 bootstrap，未做多重比较校正；区间以固定模型及测试队列为条件，不衡量跨训练种子的总体不确定性。没有根据本次测试调整超参数或重选模型。

既有 SimTier＋MAKE 低频商品分析保持不变；本次未评估 SimScore＋MAKE 的频次分组，不能将原组合的低频收益移用于新组合。

## 复核与运行

数值检查通过：零 MAKE 投影与 SimScore 等价、两路投影梯度有效、MAKE 参数保持冻结、缺失内容与空历史输出有限、保存重载完全一致。三个种子均完成全部训练曝光，测试前冻结模型，测试验证探针一致。

[逐项配对报告与汇总](../reports/simscore-make-extension-v1/)保留 AUC、GAUC、LogLoss、差值区间与预测摘要；不包含用户级预测或模型权重。

单种子入口为 `scripts/train_simscore_make.py`，配置为 `configs/c8-simscore-make-seed<seed>.json`；训练参数与 `train_c0.py` 一致，额外提供 `--content-checkpoint`。随后使用 `scripts/evaluate_simscore_make.py --store <store> --run <run-dir>` 评估。

完整调度入口 `scripts/run_simscore_make_extension.py <work-dir> <store>` 要求 work-dir 下的 configs/42.json、2026.json、3407.json 及 protocol.json；其父目录需包含原 MAKE 预训练目录与 final-heldout-test-v1 基线预测目录。数据与权重需在本地准备，不随仓库分发。

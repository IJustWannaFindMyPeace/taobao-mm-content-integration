# 阶段 10 · SimTier 内容融合对照

## 研究问题

在保持 ID 建模骨架和训练预算一致时，目标相关内容相似度的分布统计是否带来增量？C3 在 C0 的第一层 MLP 输入中增加直方图特征，其余分支保持一致。

依据 [CIKM 2024 §4.2 与 Algorithm 1](https://arxiv.org/html/2407.19467v1)，SimTier 将目标与历史的相似度划分为区间，并累积每个区间中的行为数量。

## 明确的实现适配

|项目|本实验规格|
|---|---|
|内容|固定 int8 SCL，经 1/127 解量化后 L2 归一化|
|分数|目标与历史归一化向量的点积，即余弦相似度|
|历史|与 C0 相同的末尾 200 个位置|
|统计|20 个等宽桶，原始计数，不使用 log1p 或可学习桶 embedding|
|边界|floor((s+1)N/2)，截断至 0…N−1，确保 −1 和 +1 均保留|
|缺失|排除 padding 和任一侧内容不可用的配对；目标缺内容时直方图全零，曝光保留|
|注入|计数经无偏置线性映射后，加到 C0 第一层 pre-activation|

论文伪代码的 ceil 写法与 0…N−1 索引存在边界处理差异；本实验采用上述闭合端点规则并标记 adapted。桶数和量化后重新归一化均为固定实现选择。该实现区别于 MUSE 源码的 log-count 与可学习桶参数。

线性注入在代数上等价于把直方图拼接后送入第一层。使用独立矩阵保留 C0 原参数初始化，新增矩阵初始化为零；测试确认两模型初始预测一致，新增矩阵能够获得非零梯度。计数中的内容缺失影响仍存在，本对照不单独隔离“可用内容数量”与“相似度分布”的贡献。

## 配置与执行

C3 保持 C0 的 seed=42、batch=32、单轮预算、样本顺序、ID embedding 与稠密优化器配置。区别仅为新增计数输入及对应权重。配置见 [c3.json](../configs/c3.json)，实现见 [simtier.py](../src/simtier.py)。

```powershell
python scripts/cache_simtier.py --store ../shared-feature-store-v1 --output-dir ../simtier-cache-v1
python scripts/train_c0.py --store ../shared-feature-store-v1 --config configs/c3.json --hist-cache ../simtier-cache-v1 --output-dir ../c3-seed42-epoch1
python scripts/compare_validation.py --baseline ../c0-seed42-epoch1/validation_predictions.npz --candidate ../c3-seed42-epoch1/validation_predictions.npz --report reports/c3-vs-c0.json
```

计数是固定内容与固定曝光的确定性函数，预计算不训练任何参数，也不使用标签。缓存仅构建训练与验证两份，关联存储 manifest 的摘要并在训练入口校验。预计算开销与训练开销分别记录。

## 验证

25 项测试通过，包含桶端点、缺失目标、零向量、历史置换不变性、C0/C3 公共初始化一致和新增参数梯度。配对比较核对标签、用户与源行号完全一致。GAUC 差值的用户 bootstrap 对用户整组重采样 2,000 次，仅衡量固定模型和当前验证集合条件下的抽样不确定性，不能替代多训练种子或独立数据抽样。

## 单种子配对结果

两模型均训练全部 609,902 条曝光，验证 151,947 条曝光，验证用户和行号完全一致。

|指标|C0|C3|C3 − C0|
|---|---:|---:|---:|
|AUC|0.591355|0.607081|+0.015725|
|GAUC|0.556495|0.573526|+0.017030|
|LogLoss|0.393647|0.392944|−0.000703|

GAUC 差值的用户 bootstrap 95% 百分位区间为 [0.012516, 0.021592]。该区间是固定模型条件下的验证用户重采样结果；训练随机性和独立子集抽样尚未覆盖，因此不作为跨种子稳定性结论。

|资源项目|实测|
|---|---:|
|固定特征预计算|37.33 秒；峰值 allocated 78.81 MiB|
|C3 一轮训练|419.09 秒|
|C3 训练峰值 allocated / reserved|2,830.80 / 2,840 MiB|
|保存重载最大输出差异|0|

缓存预计算开销不包含在训练计时中；GPU 峰值分阶段测量，不能相加当作同时占用。当前 C3 增量成本较低，不足以确定 MAKE 或完整 MUSE 的资源上限。

机器记录：[缓存统计](../reports/simtier-cache.json)、[C3 运行](../reports/c3-seed42-epoch1.json)、[配对比较](../reports/c3-vs-c0.json)。模型权重和逐行预测保留在运行目录。测试集未评估。

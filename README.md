# 多模态特征增强的广告点击率预测

基于 PyTorch 与 TAOBAO-MM 的广告 CTR 建模项目。围绕商品 ID 特征在低频场景下的学习不足，比较多模态向量融合、相似度特征、分布统计与多模态预训练迁移，研究如何将商品语义信息用于点击率预测。

项目提供模型实现、数据处理、训练配置、三随机种子评估与消融分析。输入采用公开的 128 维 SCL 商品多模态向量，重点在于下游建模与训练策略。

## 方法设计

公共预测网络使用用户与目标商品字段，以及短历史和较长历史的目标注意力表示。在相同主干上比较以下方案：

|方法|核心设计|
|---|---|
|ID 基线|用户、商品离散特征查表；聚合最近 50 条与最多 200 条历史，预测点击概率|
|Vector|将 128 维多模态向量线性映射为 32 维，与目标及逐历史商品的 ID／类目表示逐元素相加|
|SimScore|将逐行为多模态余弦相似度与有效标志映射为 32 维，加入历史表示|
|SimTier|将有效相似度转换为 20 维分桶计数，通过线性层接入预测网络|
|MAKE|用点击标签预训练多模态知识提取器，迁移 128 维兴趣向量和 32 维隐藏层输出|
|SimTier＋MAKE|在预测网络中并行接入相似度统计与预训练多模态特征|

主实验中的 MAKE 分支在接入后冻结，接入层与下游网络继续训练。另设“随机初始化／预训练 × 冻结／联合更新”对照，区分预训练与接入后更新的作用。

## 主要结果

以下为固定测试集上三个训练种子的均值。相对提升按 `(方法 GAUC − ID 基线 GAUC) / ID 基线 GAUC` 计算。

|方法|AUC ↑|GAUC ↑|GAUC 相对提升|LogLoss ↓|
|---|---:|---:|---:|---:|
|ID 基线|0.605518|0.573708|—|0.391709|
|Vector|0.607706|0.577532|+0.67%|0.391925|
|SimScore|0.617053|**0.593142**|**+3.39%**|**0.390030**|
|SimTier|0.616165|0.590215|+2.88%|0.390417|
|MAKE|0.619105|0.590055|+2.85%|0.390042|
|SimTier＋MAKE|**0.619468**|0.591200|+3.05%|0.390150|

- 五种多模态接入方案的 GAUC 均高于 ID 基线，三个种子的用户配对区间均为正。
- 显式相似度特征取得最高平均 GAUC；SimTier＋MAKE 取得最高平均 AUC。直接向量融合的 LogLoss 略差于基线，排序收益与概率预测质量需要分别评价。
- MAKE 的追加策略对照中，预训练后联合更新较随机初始化后联合训练，平均 GAUC 绝对增加 0.00404；持续增加预训练轮数未带来稳定收益。
- SimTier＋MAKE 的平均 AUC、GAUC 高于单模块，但 LogLoss 未优于单独 MAKE，额外 GAUC 收益也并非在所有种子上都明确。

[完整主测试结果](docs/final-test-results.md) · [训练策略及补充结果](docs/supplementary-test-results.md) · [结果分析](docs/research-narrative.md)

## 数据与实验设置

|项目|设置|
|---|---|
|数据|TAOBAO-MM，按用户 ID 确定性抽取名义 5% 用户队列|
|曝光规模|训练 3,052,475；验证 768,773；测试 923,211|
|划分|保留官方 train/test 来源；从官方 train 按用户划分验证集，验证用户排除于本地训练与测试|
|历史窗口|短历史 50；较长历史 200|
|下游训练|batch size 256，1 轮；种子 42、2026、3407|
|MAKE 预训练|batch size 32，各种子按前三轮验证 GAUC 选择权重|
|评估|AUC、GAUC、LogLoss；GAUC 按有效用户的曝光记录数加权|

三个种子共享同一数据划分。预训练成本单独记录；追加实验在主测试结果已查看之后开展，结果单独报告。统计区间使用 2,000 次用户配对 bootstrap，具体口径见结果文档。

[数据字段](docs/dataset-fields.md) · [抽样与划分](docs/sampling-and-splits.md) · [公共下游配置](configs/shared-downstream-protocol-v1.json)

## 代码与运行入口

|目录／入口|用途|
|---|---|
|[`src/`](src/)|ID 基线、多模态接入、MAKE、数据读取与评估指标|
|[`scripts/build_feature_store.py`](scripts/build_feature_store.py)|构建训练词表与统一特征存储|
|[`scripts/train_c0.py`](scripts/train_c0.py)|训练 ID 与多模态接入方案；通过配置选择模型|
|[`scripts/train_content.py`](scripts/train_content.py)|MAKE 多模态分支预训练|
|[`configs/`](configs/)|模型、数据及实验协议配置|
|[`reports/`](reports/)|逐种子指标、配对比较与产物摘要|
|[`tests/`](tests/)|数据、模型与评估逻辑检查|

基础 Python 环境声明见 [`environment.yml`](environment.yml)，Windows/CUDA 依赖快照见 [`requirements-win-cu132.txt`](requirements-win-cu132.txt)。`environment.yml` 仅创建 Python 与 pip 环境，训练依赖需另行安装；Windows/CUDA 环境可使用 `python -m pip install -r requirements-win-cu132.txt`。数据从 [TAOBAO-MM 官方入口](https://taobao-mm.github.io/)获取，原始数据、特征存储及模型权重不包含在仓库中。

运行顺序为：准备数据与划分 → 构建特征存储 → 按配置训练 → 验证选点 → 独立测试评估。训练脚本通过 `--store`、`--config`、`--output-dir` 指定输入与输出；SimTier 和 MAKE 还分别需要统计缓存及预训练权重。实验调度脚本含数据摘要校验，重建数据后需使用与本地数据一致的配置。

## 方法说明与参考

[Vector](docs/vector-integration.md) · [SimScore](docs/simscore-integration.md) · [SimTier](docs/simtier-integration.md) · [MAKE](docs/make-integration.md)

本项目基于 **Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights**（CIKM 2024）的研究方法，针对公开数据实现共享主干与接入层，并补充训练策略和模块对照。结构差异见[方法范围](docs/reproduction-scope.md)与[论文对照](docs/paper-comparison-and-analysis.md)。实验结论适用于当前离线数据与配置，不包含上游 SCL 重训或线上 A/B 测试。

- [论文](https://arxiv.org/abs/2407.19467)
- [关联项目：多模态两阶段用户兴趣建模](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)，研究更长历史的筛选与聚合，共享数据与评估协议。

[完整文档索引](docs/README.md)

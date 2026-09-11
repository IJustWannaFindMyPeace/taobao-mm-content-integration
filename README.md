# TAOBAO-MM · Content Integration

基于 TAOBAO-MM 的多模态内容融合机制复现，对应 CIKM 2024 论文 [Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights](https://arxiv.org/html/2407.19467v1)。本项目为非官方实现。

研究主线：**ID 表示的稀疏性 → 内容信息融合 → 训练轮数差异 → 多模态知识预训练与迁移**。

[实施路线](docs/roadmap.md) · [方法覆盖与适配边界](docs/reproduction-scope.md) · [真实数据检查](docs/stage-01-real-data.md) · [后续 MUSE 项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)

## 当前状态

|工作项|状态与证据|
|---|---|
|CUDA 基础链路|合成数据前向、反向与保存重载检查通过；见 [运行记录](reports/cuda-smoke.json)|
|官方 demo 检查|独立样例表无法按 ID 完整关联；见 [检查记录](reports/demo-audit.json)|
|真实训练分片|固定版本、下载校验及 1,890 行字段检查完成；见 [分片报告](reports/real-probe.json)|
|ID 工程原型|真实分片训练、mask 与重载检查通过；见 [阶段 2](docs/stage-02-id-probe.md)|
|统一指标|AUC、GAUC、LogLoss 与真实预测完整性检查通过；见 [阶段 3](docs/stage-03-evaluation.md)|
|正式实验协议|用户子集、验证划分、词表及内容覆盖待冻结|
|模型与结果|C0–C7 尚未完成训练；暂无论文对照成绩|

## 实验设计

所有方法计划使用统一数据、公共分支和评估流程。实验编号表示设计范围，不表示已经实现。

|编号|方法|检验目标|
|---|---|---|
|C0|ID-only|建立交互表示基线|
|C1|Content-only|比较内容分支与 ID 分支的训练曲线|
|C2|Vector-adapted|检验逐商品内容与 ID 拼接|
|C3|SimTier|检验目标相关相似度分布特征|
|C4|随机初始化 MAKE 分支联合训练|提供同结构训练策略对照|
|C5|MAKE 预训练后接入|检验预训练与迁移策略|
|C6|SimTier + MAKE|检验两类改进的互补性|
|C7|SimScore-adapted|检验逐行为相似度侧信息|

MAKE（Multimodal Knowledge Extractor）用于提取多模态知识。优先完成 C0、C1、C3、C4、C5，再补充融合对照。训练轮数、低频分组、多随机种子与资源开销用于解释收益及其稳定性；保留无提升或负向结果。

## 环境与数据

目标设备为 RTX 5070 12 GB、32 GB RAM。已验证的 Windows bootstrap 环境为 Python 3.11.15、PyTorch 2.12.1+cu132；数据检查使用 PyArrow 22.0.0。该环境复用了既有依赖，独立环境锁定和从零安装验证尚未完成。

当前脚本入口为 [CUDA 检查](scripts/smoke_train.py)、[demo 检查](scripts/audit_demo.py)和[真实分片准备](scripts/prepare_probe.py)。分片准备命令、数据版本及限制见[阶段报告](docs/stage-01-real-data.md)。数据文件与模型权重不提交至仓库。

## 评估与复现边界

正式评估计划报告 AUC、按完整用户聚合的 GAUC 和 LogLoss，保留评估末批，并采用一致的并列分数处理。当前工程分片不能用于正式方法收益结论。

本项目从公开固定内容向量研究下游融合。缺少原始图文和搜索购买配对时，不宣称复现 SCL（Semantic-aware Contrastive Learning）编码器训练、CLIP 对照或工业线上收益。SimTier 的论文统计版本与 MUSE 源码版本分别标注；MAKE 的训练与冻结策略记录为明确的实现适配。

## 与 MUSE 的衔接

本项目研究内容如何融入推荐模型；[MUSE 项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)进一步研究内容如何选择和利用长历史。两者计划共享数据版本、评估口径和可验证的公共接口，协议冲突优先对齐 MUSE，并记录差异。MAKE 不作为 MUSE 的默认前置模块。

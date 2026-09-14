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
|用户曝光子集|固定用户抽样与完整曝光扫描完成；见 [阶段 4](docs/stage-04-user-cohort.md)|
|扩容与属性|0.5% / 1% 嵌套曝光子集完成，1% 目标属性覆盖 100%；见 [阶段 5](docs/stage-05-scale-and-items.md)|
|历史与环境|源文件全部校验、选中用户历史覆盖 100%、独立 CUDA 环境通过；见 [阶段 7](docs/stage-07-environment-and-features.md)|
|统一输入|训练词表、memmap 特征与完整性核验完成；见 [阶段 8](docs/stage-08-feature-store.md)|
|正式实验协议|数据输入已版本化；公共模型、优化器与缺失内容分支待冻结|
|模型与结果|C0 与 C3 单种子一轮训练及验证完成；其余对照待实现，见 [SimTier 对照](docs/stage-10-simtier.md)|

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

目标设备为 RTX 5070 12 GB、32 GB RAM。专用 Conda 环境已完成独立安装和 CUDA 验证：Python 3.11.15、PyTorch 2.12.1+cu132、PyArrow 22.0.0。安装入口和版本快照见[阶段 7](docs/stage-07-environment-and-features.md)。

当前脚本入口为 [CUDA 检查](scripts/smoke_train.py)、[demo 检查](scripts/audit_demo.py)和[真实分片准备](scripts/prepare_probe.py)。分片准备命令、数据版本及限制见[阶段报告](docs/stage-01-real-data.md)。数据文件与模型权重不提交至仓库。

## 评估与复现边界

正式评估计划报告 AUC、按完整用户聚合的 GAUC 和 LogLoss，保留评估末批，并采用一致的并列分数处理。当前工程分片不能用于正式方法收益结论。

本项目从公开固定内容向量研究下游融合。缺少原始图文和搜索购买配对时，不宣称复现 SCL（Semantic-aware Contrastive Learning）编码器训练、CLIP 对照或工业线上收益。SimTier 的论文统计版本与 MUSE 源码版本分别标注；MAKE 的训练与冻结策略记录为明确的实现适配。

## 与 MUSE 的衔接

本项目研究内容如何融入推荐模型；[MUSE 项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro)进一步研究内容如何选择和利用长历史。两者计划共享数据版本、评估口径和可验证的公共接口，协议冲突优先对齐 MUSE，并记录差异。MAKE 不作为 MUSE 的默认前置模块。

## 数据来源与统计口径

官方数据定义、p90 覆盖率口径和指定版本的差异核验见[数据集契约](docs/dataset-contract.md)。

## 内容分支预训练

C1 内容兴趣模块与逐轮训练入口已实现，结构适配和运行协议见[阶段 11](docs/stage-11-content-pretraining.md)。单种子三轮实验完成：验证 GAUC 依次为 0.56808、0.57191、0.57095，第二轮最佳。

MAKE 接入的双因素设计、运行入口与结果边界见[阶段 12](docs/stage-12-make-factorial.md)。C0 三轮诊断和四组 MAKE 单种子对照已完成：C0 多轮验证退步，C5 验证 GAUC 为 0.57452；归因与置信区间见阶段报告。

SimTier 与 MAKE 的单种子组合对照已完成，C6 验证 GAUC 为 0.57865，比较协议见[阶段 13](docs/stage-13-combined-fusion.md)。

Vector 与 SimScore 单种子融合对照已完成，当前指标排序不完全一致，适配细节见[阶段 14](docs/stage-14-side-information.md)。

低频诊断及初始预算见[阶段 15](docs/stage-15-frequency-and-budget.md)。5% 数据准备与资源限制见[阶段 16](docs/stage-16-scale-preparation.md)。该规模 C1 三轮预训练及六组下游对照已完成并通过保存预测审计；C6 验证 GAUC 为 0.58894，C7 相对 C6 的增量区间跨零。六组总用时约 3.70 小时、峰值分配显存约 7.8 GiB，结果、配对区间与局限见[阶段 17](docs/stage-17-five-percent-core-study.md)。阶段 15 的线性预算不再作为执行预算。

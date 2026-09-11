# TAOBAO-MM上的CIKM 2024内容融合机制复现

状态：规划中，未完成环境与训练。非官方实现；没有已验证运行命令或实验成绩。

论文：[Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights](https://arxiv.org/html/2407.19467v1)。

## 研究问题

已有内容表示如何与ID模型融合？统计特征是否有效？内容推荐分支与ID分支是否存在训练轮数差异？预训练MAKE是否带来稳定收益？

## 范围

使用TAOBAO-MM公开固定向量，在RTX5070 12GB上做缩规模下游机制复现。SCL=Semantic-aware Contrastive Learning，语义感知对比学习；本项目使用其公开表示，不重训原私有搜索购买预训练任务。MAKE=Multimodal Knowledge Extractor，多模态知识提取器。

## 实验计划

C0 ID-only；C1 Content-only；C2逐商品内容与ID融合的Vector适配；C3 SimTier；C4随机初始化内容分支联合训练；C5预训练MAKE接入；C6 SimTier+MAKE；C7 SimScore逐行为相似度侧信息。C4/C5为训练策略核心对照，补预训练轮数扫描；C0/C1用于诊断轮数曲线。所有结果：未运行。

CLIP-O、CLIP-E、SCL及去Triplet/去Triplet和MoCo的原始预训练对照均已登记为数据受限；不是用公开固定向量就完成了这些实验。完整对应与开工条件见[审计文档](论文方案覆盖与开工审计.md)。

## 数据、环境与评估

共享协议protocol-v1待数据审计后冻结。保持官方测试集独立，验证来自官方训练集。标签、ID映射、历史时间边界、mask和数据清单必须固定。全量组内GAUC、全局AUC、logloss、低频分组和资源成本。

推荐WSL2+Miniconda隔离环境；Python/PyTorch与CUDA确切锁版本待5070训练试跑。原始数据与checkpoint不进入Git；本README不提供尚未验证的安装训练命令。

## 论文对齐与适配

保留SimTier和MAKE核心思想；Vector具体形式、网络宽度、隐藏层取用及冻结策略要写入配置和差异文档。公开内容版本/数据/规模与原工业实验不同。SimTier论文直方图与MUSE代码对数桶表示分开实现。

## 下一项目

后续MUSE项目在同一数据和评估协议上研究长历史检索与语义注意力。交接使用冻结公共基线；不宣称MAKE为MUSE必需模块。建议仓库名taobao-mm-content-integration，远程仓库已创建并完成首次推送。

## 完成标准

可验证数据契约、可运行配置、训练曲线、核心多种子结果、成本记录、负结果和局限。不能以离线子集提升替代工业线上收益。

## 2026-09-11 实操进度更新

本地Git骨架已建立。工作区Windows CUDA bootstrap训练检查已通过，实际Python3.11.15/PyTorch2.12.1+cu132；暂复用既有环境，最终独立依赖锁未完成。CIKM目录scripts/smoke_train.py是合成数据诊断，不是C0或论文结果。官方demo键关联不完整，真实训练等待提取一致子集。前文规划状态中的“环境未验证”已被本节更新：仅bootstrap验证完成，正式训练尚未完成。远程仓库已创建并完成首次推送。

## 分阶段学习

阅读[学习路线](docs/learning-path.md)。每阶段展示问题、关键代码、验证过的命令、产物与检查问题，通过Git提交记录进展。当前仅完成环境合成训练诊断与官方demo读取，真实数据基线尚未开始。


## 仓库链接

本项目：https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration

衔接项目：https://github.com/IJustWannaFindMyPeace/taobao-mm-muse-repro

两个仓库均为私有。首次推送已完成；正式数据训练尚未开始。

## 阶段1：真实数据工程样本

已固定官方revision并取得1,890条已关联的训练曝光。脚本、统计、命令和限制见[阶段1代码导读](docs/stage-01-real-data.md)。内容向量与正式评估划分尚未完成；不是论文结果。

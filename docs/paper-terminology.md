# 论文术语与实现命名

|术语|本文用法|
|---|---|
|Multimodal Data，多模态数据|图片、标题等原始模态数据|
|Multimodal Representations，多模态表示|编码器输出的表示；在本项目中以固定向量作为输入|
|CTR Prediction，点击率预测|预测一次曝光被点击的概率；网络原始输出 logit 经 sigmoid 转为概率|
|SCL，Semantic-Aware Contrastive Learning|语义感知对比学习，属于上游表示预训练；本项目未重训编码器|
|SimScore，Similarity Score|以目标与历史商品的语义相似度作为侧信息；本地使用归一化向量的余弦相似度|
|SimTier|按相似度区间分层计数；“分桶统计”是对计算过程的说明|
|MAKE，Multimodal Knowledge Extractor|多模态知识提取器；在固定表示上进行推荐预训练，迁移兴趣向量与 MLP 中间特征|

SCL 表示预训练与 MAKE 推荐预训练属于不同阶段。MAKE 的冻结或联合更新策略不改变输入 SCL 向量固定这一设定。

方法名称沿用论文；具体网络结构、投影相加接入和分桶边界按各方法文档标注适配。本地目标注意力基线不等同于原版 DIN。原始结果文件的标签与代码标识保留，以保证结果可追溯；读者文档和图表使用本表术语。

依据：[CIKM 2024 原论文 §3、§4、§6.1.3](https://arxiv.org/html/2407.19467v1)。中文为对应译法与计算说明，不额外扩写论文未给出的缩写全称。

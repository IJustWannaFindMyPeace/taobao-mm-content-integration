# 论文术语与实现命名

|术语|本项目用法|
|---|---|
|ID features，ID 特征|用户 ID、商品 ID、类目 ID 等离散输入字段，不是向量|
|ID embeddings，ID 嵌入向量|按离散 ID 从可训练 embedding 表中查出的向量；商品 ID embedding 与类目 ID embedding 分别查表|
|ID-based model，基于 ID 的模型|以 ID 特征的 embeddings 及历史行为建模特征进行预测；本项目简称 ID 基线|
|Multimodal Data，多模态数据|图片、标题等原始模态数据|
|Multimodal Representations，多模态表示|编码器输出的表示；在本项目中以固定向量作为输入|
|CTR Prediction，点击率预测|预测一次曝光被点击的概率；网络原始输出 logit 经 sigmoid 转为概率|
|SCL，Semantic-aware Contrastive Learning|语义感知对比学习，属于上游表示预训练；本项目未重训编码器|
|SimScore，Similarity Score|以目标与历史商品的语义相似度作为侧信息；本地使用归一化向量的余弦相似度|
|SimTier|按相似度区间分层计数；“分桶统计”是对计算过程的说明|
|MAKE，Multimodal Knowledge Extractor|多模态知识提取器；在固定多模态表示上进行推荐预训练（点击率预测任务），迁移兴趣向量与 MLP 中间特征|

SCL 通过搜索与随后购买行为构造语义相近样本，训练上游编码器；MAKE 推荐预训练使用曝光点击标签，训练固定多模态表示之上的兴趣网络和 MLP。两者属于不同阶段。MAKE 的冻结或联合更新策略不改变输入 SCL 向量固定这一设定。

方法名称沿用论文；具体网络结构、投影相加接入和分桶边界按各方法文档标注适配。本地目标注意力基线不等同于原版 DIN。原始结果文件的标签与代码标识保留，以保证结果可追溯；读者文档和图表使用本表术语。

依据：[CIKM 2024 原论文 §2、§3、§4.3、§6.1.3](https://arxiv.org/html/2407.19467v1)。中文为对应译法与计算说明，不额外扩写论文未给出的缩写全称。

结构说明中的“商品 ID 与类目 embeddings 拼接”指 32 维输入；“ID 主干特征”还包括属性 embeddings 与历史兴趣，不能一律改称单个商品 ID embedding。

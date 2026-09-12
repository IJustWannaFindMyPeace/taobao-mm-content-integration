# 方法覆盖与复现边界

范围为两篇指定 arXiv v1 的主要方法、实验对照及本地参考源码。以下清单记录实施优先级与证据限制，模型实验尚未完成。

## 1. 覆盖状态的含义

- 核心：必须实现并验证的下游实验，仍未运行。
- 条件：纳入路线，但在补齐结构/训练规格后才能运行；暂缓不等于忘记。
- 受限：缺原始数据、表示或工业系统，只保留来源与局限。
- 扩展：额外设计的实验，与原文实验分开报告。

## 2. CIKM对应表

来源：[CIKM原文](https://arxiv.org/html/2407.19467v1)。位置按该v1。

|论文位置|方案/问题|本地编号与处置|证据与局限|
|---|---|---|---|
|§3、表1|搜索购买配对、SCL|受限|不以已有向量等同重训编码器|
|表2|CLIP-O、CLIP-E|受限|缺相应原始图文/电商微调数据与匹配评估集|
|表2|SCL、去Triplet、去Triplet和MoCo|受限|不另造“只去MoCo”的原表行|
|图6(a)、§6.2|Acc@N与下游关系|受限|一个固定向量版本不足以检验跨表示相关性|
|表3|ID baseline|C0核心|公共骨架适配，非私有生产网络|
|图3、§4.1|内容-only与ID轮数差异|C1+C0核心|以训练/验证曲线检验差异|
|表3、§4开头|Vector|C2核心|目标和逐历史商品内容与ID侧信息融合，再聚合；不只池化后拼接|
|表3|SimScore|C7核心|保留逐行为相似度侧信息；插入位置须标适配|
|表3、§4.2|SimTier|C3核心|paper_hist与muse_code分开|
|表3、§4.3|MAKE|C5核心|DIN、隐藏层、冻结策略需实现规格|
|表3|SimTier+MAKE|C6核心|对C3、C5分别看增量|
|图6(c)|MAKE预训练轮数（含0）|C4/C5及P扫描核心|P=0联合训练；补P=1/3/5适配网格，按验证及预算实施|
|图6(b)|低频分组、AUC/LogLoss|核心诊断|共享频次协议；8组作为原文对齐附表，主表统一分组|
|§5、§6.4|生产链路、线上与新广告|受限|本地查表/耗时测量只是工程类比|

CLIP-O为原始通用CLIP视觉编码器，CLIP-E为电商适配版本；不可与MUSE的OpenCLIP、I2I直接混为同一组。SCL是Semantic-aware Contrastive Learning。MoCo是Momentum Contrast。MAKE是Multimodal Knowledge Extractor。

## 3. MUSE对应表

来源：[MUSE原文](https://arxiv.org/html/2512.07216v1)。

|论文位置|方案/问题|本地编号与处置|证据与局限|
|---|---|---|---|
|表1|ID、OpenCLIP、I2I、SCL检索表示|G-ID与G0核心；其余受限|缺另两套表示，不能以噪声向量替代原对照|
|表2|内容余弦|G0=R4检索核心|固定统一ESU|
|表2|多模态attention|G1条件|明确变换、分数及可训练梯度路径后运行|
|表2|内容余弦+ID余弦|G2条件|固定归一化与融合方式，记录适配|
|表2、§7.2|内容余弦+ID attention|G3条件|SA-TA式评分共享策略需明确|
|表3|TA、SimTier、TA+SimTier、SA-TA、SA-TA+SimTier|R1、R5、R2、R3、R4核心|固定检索，R5关闭被研究长历史ID注意力/均值旁路|
|图2|GSU ONLY与GSU&ESU表示替换|受限|不能用SCL一个版本完成原比较|
|表5、§7.1|DIN|B0核心，单列|R0是两阶段最近K对照，不等于独立短历史DIN|
|表5|SIM-hard、SIM-soft|B1/B2核心|MUSE论文版ESU与原版分别命名|
|表5|TWIN、MISS|B3/B4条件|原方法结构与实现规格待确认|
|表5|MUSE|B5核心|与R4一致配置时复用结果，禁止重复算作独立证据|
|图5|历史长度、增强ESU对ID-only|核心，适配1k|L扫描同时跑R1和R4，不能只扫完整模型|
|图6(a)(b)|用户长度组、商品频次组|核心诊断|9组作原文对齐附表，主表用统一分组|
|表6、§5|线上与100k链路|受限|不把1k或本机延迟当线上复现|
|图7、附录A.1引用|检索案例|受限/扩展|无原图；可用匿名ID、类目、相似度和重叠率做自建案例|

缩写：GSU=General Search Unit；ESU=Exact Search Unit；TA=Target Attention；SA-TA=Semantic-Aware Target Attention；SIM=Search-based Interest Model；TWIN=Two-stage Interest Network；MISS在本文指所引多模态索引搜索方案，复现其历史选择适配不等于实现其原始完整树检索系统；I2I=Item to Item。

## 4. 相关工作边界

MUSE正文还提及MIMN、UBR4CTR、ETA、SDIM、AlignRec、QARM等，列为背景阅读，不进入首版基准表。它们不是本论文主实验表全部实测行。CIKM的通用CTR网络及端到端多模态相关引用也属于背景。若后来加入，必须独立写来源、目的和实现规格；扩展方法需独立说明实验目的。

## 5. 实现约束与源码差异

### A1：实验覆盖

范围包含C7 SimScore、R5 SimTier-only、B0 DIN、CLIP-O/E、G1/G2/G3、MAKE预训练轮数网格以及L×ESU对照。实验编号在配置与报告中保持一致。上述实验均需独立实现和验证。

### A2：逐商品融合

C2采用逐商品拼接内容与ID/类目，再做历史建模；池化拼接如保留，另叫Vector-pooled-extension。SimScore主C7拟用r_i及有效标记作为逐行为侧信息进入匹配网络，具体宽度为本地适配。不能把SimScore替换成平均余弦而不说明。

### A3：优化器配置差异

main.py使用AdamW(fused=True)和SparseAdam；论文§7.1.2写Adam。源码默认weight_decay等也须锁定，不只记录学习率。实施主协议优先使用经实测的MUSE-code配置，另列paper-described；不把两套混写。稀疏梯度不保证优化器状态按访问行节省显存。

### A4：可学习检索器的梯度路径

G1/G3若只在no_grad的TopK索引路径使用独立MLP，其参数不会由点击损失学到有效检索。必须确定与ESU共享、辅助目标或明确预训练方案，并验证参数确实更新；自行补策略要标adapted。检索器对照要求参数已接受有效训练；补充训练策略作为适配单列。

### A5：评估记录与指标定义

utils/muse_dataset.py默认drop_last=True，main.py评估调用未覆盖该值；存在丢弃末批的路径，具体数量需实际数据确认。主评估设drop_last=False并检验输出行数与manifest一致。全局AUC不要构造P×N矩阵，使用排序/秩方法；GAUC按完整用户分组。标准并列半分、全无效组N/A；保留官方兼容口径单列。

### A6：共享基线与训练预算

CIKM的DIN式MAKE与MUSE代码点积注意力不可混称。bridge明确输入、分支、张量维度和开关，复制预测做一致性验证；仅有相同数据不够。主MUSE方法比较保留共同一轮配置作为paper-budget-adapted；多轮选最佳另开表，不混合。小batch不是官方大batch的完全等价替代。

### A7：数据版本与统计口径

MUSE v1表4与§6.2正文的公开样本/用户/商品数量不一致；README资源表也有自身版本。不同版本的统计分别引用。实际下载revision、文件清单、行数、词表与内容覆盖是本地实验依据。内容表提取可覆盖评估所需ID，但可训练ID词表若用训练构建，不得因提取内容顺便扩成测试词表。

### A8：源码版本与许可

参考源码采用 Apache-2.0 许可。ZIP SHA256：0C496337FB07A2181152DF246AC5677701ABD538BFC64D7A7AFE92FA4E3A759F。该指纹标识已检查的源码归档，对应上游 commit 尚未解析。正式引用需固定上游revision或保存带指纹归档来源，复用时保留许可；数据许可单独核查。

## 6. 当前工程状态

前序内容融合项目已完成独立环境验证、用户曝光划分、历史与内容提取及指标测试。公共模型接口和正式模型对照尚未完成。详细进度见[实施路线](roadmap.md)。

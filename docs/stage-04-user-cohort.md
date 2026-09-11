# 阶段 4 · 固定用户曝光子集

## 数据路径选择

参考预处理 notebook 通过用户 ID 和商品 ID 将曝光表与特征表左连接，再按行数写出分片。分片不构成用户完整性边界，因此本阶段直接扫描完整原始曝光表。

下载固定 revision 的 train_samples、test_samples 和 item_features 三张表，分别核对文件大小与 LFS SHA256。曝光表总计约 930 MB，比已关联历史的分片更适合用户抽样。历史特征和 SCL 向量尚需单独提取。

## 用户规则

协议名为 `user-cohort-v1-exposures`，使用明确的 64 位 SplitMix64 混合函数，不依赖 Python 进程随机 hash。

1. 原始有符号用户 ID 保留其 64 位位模式，以 seed 42 混合，结果模 1000 等于 0 时入选，目标抽样率为 0.1%。
2. 使用独立 seed 2026，结果模 5 等于 0 的入选用户作为验证用户，其余作为训练/测试用户。
3. 训练和验证曝光均来自官方 train；测试曝光来自官方 test，只保留入选的非验证用户。测试标签不参与筛选，测试用户允许在训练中未出现。
4. 完整扫描源文件，收齐对应集合内每位选中用户的曝光。每行保留 source_row 以追溯源文件位置，不去重真实的重复曝光。

该验证集按用户互斥，验证用户 ID 将是 OOV；其分布不等同于已有用户的未来曝光验证。当前缺少可核验时间字段，不能将其描述为时间划分。若未来调整抽样规模，发布新版本，不根据测试效果改变用户集合。

## 运行入口

```powershell
python scripts/fetch_source.py --manifest reports/source-manifest.json --output-dir ../taobao-mm-source --files raw/train_samples.parquet raw/test_samples.parquet raw/item_features.parquet
python scripts/select_exposures.py --source-dir ../taobao-mm-source --manifest reports/source-manifest.json --output-dir ../user-cohort-v1 --report reports/user-cohort.json
```

下载器以分块写入、大小上限和完整 SHA256 校验保护缓存；中断后重新下载未完成文件。扫描器每批 131,072 行，在 CPU 上处理曝光列，输出 Parquet 子集和机器报告。数据及原始 ID 留在仓库外。

## 验收与边界

抽样函数通过固定参考值、正负 ID、用户组互斥、跨批次及逆序一致性测试。完整扫描行数必须等于 Parquet 元数据行数；报告保留输入与输出哈希。

此产物仅冻结曝光集合。历史与商品特征连接、连接基数检查、内容覆盖和训练词表尚未完成，不能据此启动正式模型比较。最终完整数据 manifest 将在上述检查通过后发布。

## 本次产物

已完整扫描官方训练 76,015,123 行、测试 22,979,465 行。

|集合|曝光数|用户数|
|---|---:|---:|
|训练|60,509|5,518|
|验证|15,472|1,404|
|测试|18,571|1,679|

实际验证通过：源行号无重复且递增，验证用户与训练、测试用户均无交集。统计及文件指纹见 [user-cohort.json](../reports/user-cohort.json)。这些是曝光子集规模，尚未完成历史与内容连接。

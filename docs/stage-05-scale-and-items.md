# 阶段 5 · 嵌套扩容与商品属性关联

## 数据规模

扩展用户抽样规则为 `mix64(user_id, 42) % 1000 < buckets`，buckets 支持 1、5、10。原 0.1% 规则保持兼容，独立验证用户规则保持不变。实际源行号集合已验证逐级包含，扩容不改变已有用户的集合角色。

|用户抽样率|训练曝光|验证曝光|测试曝光|
|---|---:|---:|---:|
|0.1%|60,509|15,472|18,571|
|0.5%|301,470|75,844|93,925|
|1%|609,902|151,947|185,253|

[0.5% 报告](../reports/user-cohort-005.json) · [1% 报告](../reports/user-cohort-010.json) · [包含关系检查](../reports/cohort-nesting.json)

三个集合用于开发、规模敏感性和资源测试，尚未确定最终主实验规模。多训练种子仅衡量优化波动；关键结论还需独立用户抽样以及按用户重采样的置信区间。独立抽样与置信区间尚未实现。规模与训练配置根据训练/验证信息选择，测试成绩不参与选择。

## 商品属性

对 1% 子集关联原始 item_features 表中的 206、213、214 字段。关联键 205 要求全表非空且唯一；目标缺失属性时明确报错，不静默丢弃曝光。按目标索引查表，保留曝光顺序和重复曝光。

训练、验证、测试目标属性覆盖均为 100%，关联后行数与源行号顺序不变。见 [关联报告](../reports/item-join-010.json)。这不代表历史商品内容向量覆盖率为 100%。

```powershell
python scripts/select_exposures.py --source-dir ../taobao-mm-source --manifest reports/source-manifest.json --output-dir ../user-cohort-v1-010 --report reports/user-cohort-010.json --buckets 10
python scripts/attach_items.py --items ../taobao-mm-source/raw/item_features.parquet --manifest reports/source-manifest.json --cohort-report reports/user-cohort-010.json --input-dir ../user-cohort-v1-010 --output-dir ../user-cohort-v1-010-items --report reports/item-join-010.json
```

## 历史读取约束

训练用户历史文件支持 HTTP Range。读取文件尾元数据后确认：6,929,671 行用户、7 个行组；首行组包含 1,048,576 位用户，商品历史压缩列约 5.31 GB，类目历史约 1.45 GB。用户抽样不会自动使列块下载量按比例缩小。

因此不能假定远程 Parquet 的用户过滤能只下载 1% 历史。下一阶段需要具备进度和恢复能力的下载、流式解码及用户过滤，并测量实际内存。测试历史尾部的一次 Range 响应长度不符合请求，已拒绝解析；内容尾部尚未检查。此检查没有完成大文件 SHA256 校验，也没有完成历史关联。见 [布局记录](../reports/remote-parquet-layout.json)。

## 验证与状态

14 项测试通过，新增嵌套抽样、重复曝光保持、重复/缺失商品键拒绝测试。当前已完成曝光扩容和目标属性关联；用户历史、SCL 内容、训练词表和正式训练仍待完成。

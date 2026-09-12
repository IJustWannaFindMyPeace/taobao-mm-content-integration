# 阶段 6 · 历史文件传输与提取准备

本文记录该阶段的验证范围；当前项目状态见 [README](../README.md)。

## 已验证行为

下载器以固定 revision 和来源 SHA256 为身份，按 64 MiB 的精确 HTTP Range 读取。每个完整分块落盘后记录摘要；重启时逐块校验，截去未登记尾部后继续。最终完整 SHA256 匹配才将 `.part` 发布为正式 Parquet。

真实训练历史进行了两次独立启动：第一次下载 64 MiB，第二次校验已有分块后续传至 128 MiB。此结果验证续传路径，不代表完整文件已经校验通过。见 [试验记录](../reports/download-pilot.json)。

下载队列为训练用户历史、测试用户历史、SCL int8 向量，合计约 63.2 GB。三份完整文件均已通过来源 SHA256 校验，提取结果见[特征报告](stage-07-environment-and-features.md)。

```powershell
python scripts/download_large.py --manifest reports/source-manifest.json --output-dir ../taobao-mm-source --files raw/train_user_features.parquet raw/test_user_features.parquet raw/scl_embedding_int8_p90.parquet
```

同一目标目录仅运行一个下载进程。再次执行相同命令可恢复中断下载。每个分块最多尝试四次；网络持续失败时退出并保留可恢复数据。日志不保存带签名的重定向地址。仅使用完整校验后发布的 Parquet 文件。

## 历史提取入口

[extract_histories.py](../scripts/extract_histories.py) 校验源文件和曝光子集摘要，按选中用户过滤历史，检查每位用户唯一、商品/类目序列等长且无空元素，并要求选中用户覆盖完整。训练与验证读取训练历史源，测试读取测试历史源。

```powershell
python scripts/extract_histories.py --manifest reports/source-manifest.json --source-dir ../taobao-mm-source --cohort-dir ../user-cohort-v1-010 --cohort-report reports/user-cohort-010.json --output-dir ../user-cohort-v1-010-histories --report reports/history-extraction-010.json
```

脚本关闭预缓冲和多线程，每批输出 256 位用户。Parquet 解码器仍可能读取大型压缩列块，因此 batch 设置不保证内存严格限制在 256 行；真实峰值内存必须另行测量。先完成下载和校验，再运行真实提取，不自动把全量历史展开为重复曝光表。

## 验证状态

合计 17 项测试通过。历史提取的三项构造测试验证正确选取、缺失用户拒绝、重复用户拒绝。Parquet 资源在正常和异常路径均通过上下文管理器释放。真实数据提取与内存测量见[特征报告](stage-07-environment-and-features.md)。

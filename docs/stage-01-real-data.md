# 阶段1：从无法关联的demo到真实工程小样本

## 解决的问题

六张独立demo表无法按ID关联，不能按行号拼接。正式数据仓库提供已关联的训练分片，可以直接得到曝光对应的用户、目标与历史字段。

## 已执行

固定官方数据revision `f185841b6e9612edd06f5112e06416f1bdf6c6d0`。下载训练分片000159（1,352,869字节），与官方LFS SHA256核对一致。实际获得1,890条曝光、264点击、156用户、1,867目标商品；78位用户在该分片内同时有正负样本。每行商品/类目历史长度一致，保存长度均为1000；不等于1000个有效非padding行为。

产物在reports/source-manifest.json及reports/real-probe.json。原始分片和probe.parquet在仓库外，不推送GitHub。没有读取测试标签，也没有报告模型AUC。

## 代码导读

scripts/prepare_probe.py：

1. 从manifest读取固定版本与文件摘要，不用随时变化的main下载。
2. 仅允许训练分片且限制64MiB，分块写入.part，校验通过后改正式文件名。
3. 检查必要列、标签、目标ID和类目、历史对齐；最多读取4096行，保存Parquet。
4. 记录统计和限制，不能把下载成功当数据协议完成。

## 已验证的命令

从两个项目共同的工作区根目录，用项目bootstrap Python运行：

```powershell
& ./.venv-bootstrap/Scripts/python.exe projects/taobao-mm-content-integration/scripts/prepare_probe.py --manifest projects/taobao-mm-content-integration/reports/source-manifest.json --output-dir work/taobao-mm-probe --report projects/taobao-mm-content-integration/reports/real-probe.json
```

第二次会校验已有文件并复用。依赖为pyarrow22.0.0；当前bootstrap新增huggingface_hub1.31.0，后者本脚本未使用。Python3.11.15/PyTorch2.12.1+cu132，最终独立环境仍待整理。网络需要正常用户环境，未关闭TLS校验。

## 局限与下一步

这是按文件大小选的工程分片，有选择偏差；同用户可能在其他分片还有曝光。它不是之前设计的完整用户评估子集，没有validation/test划分，不适合正式比较方法收益。内容表未提取，不能立即运行SimTier/MAKE。

下一步用这些真实行完成ID映射与mask，跑C0工程训练；同时确定正式用户子集与验证规则。以后再提取所需SCL向量，不把全量35M商品向量装进GPU。

检查问题：这里统计出78位用户在分片内有正负样本，是否就能把这个分片的GAUC当作完整测试集GAUC？为什么？

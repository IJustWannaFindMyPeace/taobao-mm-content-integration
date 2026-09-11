# 阶段 1 · 真实训练分片验证

## 解决的问题

六张独立demo表无法按ID关联，不能按行号拼接。正式数据仓库提供已关联的训练分片，可以直接得到曝光对应的用户、目标与历史字段。

## 数据来源与检查结果

固定官方数据revision `f185841b6e9612edd06f5112e06416f1bdf6c6d0`。下载训练分片000159（1,352,869字节），与官方LFS SHA256核对一致。实际获得1,890条曝光、264点击、156用户、1,867目标商品；78位用户在该分片内同时有正负样本。每行商品/类目历史长度一致，保存长度均为1000；不等于1000个有效非padding行为。

来源清单见 [source-manifest.json](../reports/source-manifest.json)，统计结果见 [real-probe.json](../reports/real-probe.json)。原始分片和probe.parquet在仓库外，不推送GitHub。没有读取测试标签，也没有报告模型AUC。

## 校验流程

scripts/prepare_probe.py：

1. 从manifest读取固定版本与文件摘要，不用随时变化的main下载。
2. 仅允许训练分片且限制64MiB，分块写入.part，校验通过后改正式文件名。
3. 检查必要列、标签、目标ID和类目、历史对齐；最多读取4096行，保存Parquet。
4. 记录统计和限制，不能把下载成功当数据协议完成。

## 已验证的命令

在仓库根目录使用已安装 PyArrow 22.0.0 的 Python。下列命令与已验证的工作区调用使用相同脚本及参数，数据输出至仓库外：

```powershell
python scripts/prepare_probe.py --manifest reports/source-manifest.json --output-dir ../taobao-mm-probe --report reports/real-probe.json
```

第二次会校验已有文件并复用。依赖为pyarrow22.0.0；当前bootstrap新增huggingface_hub1.31.0，后者本脚本未使用。Python3.11.15/PyTorch2.12.1+cu132，最终独立环境仍待整理。网络需要正常用户环境，未关闭TLS校验。

## 局限与下一步

这是按文件大小选的工程分片，有选择偏差；同用户可能在其他分片还有曝光。它不是之前设计的完整用户评估子集，没有validation/test划分，不适合正式比较方法收益。内容表未提取，不能立即运行SimTier/MAKE。

下一步用这些真实行完成ID映射与mask，跑C0工程训练；同时确定正式用户子集与验证规则。以后再提取所需SCL向量，不把全量35M商品向量装进GPU。

该分片内的 78 位正负样本兼具用户仅描述局部标签覆盖。同一用户的曝光可能分布在其他分片，因此不能据此计算或替代完整测试集 GAUC。正式评估需要在固定评估集合内收集每位用户的全部预测后统一聚合。

# 阶段 7 · 专用环境与历史、内容提取

## 专用环境

工作区 `.envs/taobao-mm` 为独立 Conda prefix，Python 3.11.15，不继承旧项目 site-packages。旧 `.venv-bootstrap` 依赖的基础环境已被删除，仅保留其历史报告。

已安装并验证 PyTorch 2.12.1+cu132、PyArrow 22.0.0、NumPy 2.3.3、psutil 7.1.0。18 项测试与 pip check 通过，RTX 5070 的前后向、参数更新和保存重载通过。见 [CUDA 报告](../reports/dedicated-cuda-smoke.json)。

```powershell
conda env create --prefix ../.envs/taobao-mm --file environment.yml
conda activate ../.envs/taobao-mm
python -m pip install -r requirements-win-cu132.txt
python -m unittest discover -s tests -v
```

上述入口对应本次已执行的分步安装。依赖版本见 [requirements-win-cu132.txt](../requirements-win-cu132.txt)，这是 Windows CUDA 环境快照，不是跨平台或带 wheel 哈希的锁文件。Conda 环境声明见 [environment.yml](../environment.yml)。

## 用户历史

完整校验源 SHA256 后扫描训练 6,929,671 位用户、测试 2,063,692 位用户，分别提取训练/验证所需 69,346 位和测试所需 16,415 位。两侧覆盖率均为 100%，选中用户键唯一，商品/类目历史长度一致且无空元素。

处理批次由 256 调整为 4,096，关闭预缓冲和多线程，减少逐批过滤开销。训练侧扫描与输出约 165.5 秒，测试侧约 44.2 秒，均不含开始前完整源文件摘要校验。进程累计峰值工作集 11,797 MiB（约 11.52 GiB），两侧共用进程，因此测试侧数字不是独立峰值。见 [历史报告](../reports/history-extraction-010.json)。

输出按用户保存一条历史，尚未将历史按曝光重复展开。训练和验证共用训练历史来源，测试使用独立测试历史来源。

## 内容提取规则

所需商品为三份曝光集合的目标商品与选中用户完整历史商品的并集，排除 padding ID 0。对固定 SCL int8 表完整扫描，保留所需向量，校验选中商品键唯一、向量等长且无空元素。保留量化格式；参考代码解量化因子为 1/127。

提取所需内容可以覆盖验证和测试商品，但这不授权将它们加入可训练 ID 词表。内容缺失不能作为 padding 删除；缺失策略须在正式模型接口中统一。

```powershell
python scripts/extract_histories.py --manifest reports/source-manifest.json --source-dir ../taobao-mm-source --cohort-dir ../user-cohort-v1-010 --cohort-report reports/user-cohort-010.json --output-dir ../user-cohort-v1-010-histories --report reports/history-extraction-010.json
python scripts/extract_content.py --source ../taobao-mm-source/raw/scl_embedding_int8_p90.parquet --manifest reports/source-manifest.json --cohort-dir ../user-cohort-v1-010 --cohort-report reports/user-cohort-010.json --history-dir ../user-cohort-v1-010-histories --history-report reports/history-extraction-010.json --output-dir ../user-cohort-v1-010-content --report reports/content-extraction-010.json
```

实际工作区数据均位于 `work` 下；命令中的目录参数可按本地布局调整。正式公共数据加载、训练词表、缺失内容处理与模型消融尚未完成。

## 内容提取结果与限制

完整扫描 35,458,467 条 SCL 记录。当前用户历史与目标共请求 19,986,863 个非零商品 ID，匹配 13,792,739 个，缺失 6,194,124 个；去重商品覆盖率为 69.01%。提取向量均为 128 维 int8，无全零向量。

|目标曝光集合|匹配曝光 / 全部曝光|缺失曝光|
|---|---:|---:|
|训练|609,890 / 609,902|12|
|验证|151,946 / 151,947|1|
|测试|185,247 / 185,253|6|

内容提取峰值工作集约 2.26 GiB，约 139 秒。见 [内容报告](../reports/content-extraction-010.json)。69.01% 为历史与目标合并后的去重商品覆盖，不是历史行为加权覆盖；不能据此断言 30.99% 的历史行为缺内容。近乎完整的目标覆盖也不等于完全覆盖。

此缺失与公开 p90 内容表范围相关，仍需逐行为统计和统一 fallback。后续不得按内容存在与否删除曝光来改善成绩，也不能将内容缺失等同于 padding。当前已准备好特征子表，正式输入接口与训练词表尚未冻结。

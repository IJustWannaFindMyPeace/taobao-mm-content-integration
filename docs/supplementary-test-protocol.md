# 统一追加测试评估：范围与门槛

主实验已完成冻结测试，部分补充模型此前只有验证证据。为统一项目正式结果的评估口径，新增这一阶段；它是已看过主测试结果后的追加评估，不能重新宣称完全盲测。

## 纳入范围

完整纳入已完成的补充下游配置和种子，共73份检查点记录：DIN及短历史对照6、TWIN6、MISS9、G3及其多模态表示对照6、G1下游3、多模态表示集成项目补充31、长度补充12。多模态表示集成项目包含MAKE策略/固定轮次、ID三轮最终模型和SimTier两种边界解释。重叠主模型在核验预测对应后可复用原测试结果，不按分数选择模型。

资源探针、上游多模态表示/评分器预训练曲线和匿名案例不是最终下游推荐模型，不混入该排行榜。ID多轮诊断只保存最终模型权重，没有第2轮检查点；保留第2轮验证证据，明确无法直接补测，不用重训替换原模型。第1轮与原ID主模型的既有一致性证据单独保留。

## 执行顺序

1. 清点73份完整下游检查点，记录模型、训练报告、验证预测及数据清单SHA-256，公开候选清单。
2. 新评估入口只读载入权重，重放完整768,773条验证预测；要求与原保存预测逐元素一致、重算指标一致。资源超限或不一致即停，不放宽容差。
3. 全部重放通过后，固定测试入口、源文件、最终检查点和比较清单，另存追加测试冻结文件。原主实验冻结文件保持不变。
4. 评估同一923,211条测试曝光，统一AUC、GAUC、LogLoss；GAUC仍按同时有正负标签的完整用户计算。配对比较沿用每项2000次用户级bootstrap，不作多重校正。
5. 完整报告各配置、各种子，包括下降与跨零结果；不得用追加测试挑模型、改参数、重选轮次或长度。

## 当前状态

截至2026-09-25，73份完整验证重放全部通过，测试入口验收与冻结完成，73份测试评估及93项配对统计全部完成。见[追加测试结果](supplementary-test-results.md)。原主实验冻结和模型未修改。

本阶段不训练、无优化器、不更新参数。部分新模型入口与直方图口径需精确匹配原训练，不能直接套用只支持主模型的旧测试脚本。原96项一致性检查继续有效，但不代表本阶段已完成。

[候选检查点清单](../reports/supplementary-test-candidates-v1.json)只记录产物相对位置与指纹，不包含权重或原始数据。

## 测试前设计审计

已完成[设计与执行审计](supplementary-pretest-audit.md)。辅助依赖冻结、显式test报告、操作系统单实例锁、原子完成记录与续跑校验均已落实；执行约束6项自动化测试和比较契约3项测试通过。全部73份验证重放通过后才冻结并启动测试。

## 执行与复核入口

以下命令从同时包含两个项目与本地产物的工作目录运行。需要既有数据、候选检查点、比较契约和验证重放产物；仓库仅发布代码与摘要，不包含数据和权重。

```powershell
python projects/taobao-mm-content-integration/scripts/run_supplementary_test.py --work work --output-dir work/supplementary-heldout-test-v1 --resume
python projects/taobao-mm-content-integration/scripts/summarize_supplementary_test.py --work work
python projects/taobao-mm-muse-repro/scripts/summarize_adapted_baseline_test.py --work work
python -m unittest discover -s projects/taobao-mm-content-integration/tests -p "test_supplementary*.py"
```

首次执行新输出目录时省略 `--resume`。续跑核验已完成文件与冻结摘要；已完成目录不会自动重新训练或覆盖。汇总入口重新校验预测与报告指纹，并生成两个仓库共享的报告。公开冻结清单是移除机器绝对路径后的投影，不能直接替代原执行冻结文件启动续跑。

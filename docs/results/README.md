# 实验结果

本目录只保存已经完成、数字已经固定的人工可读实验报告。原始JSON/CSV、模型和逐样本预测
保存在本地`artifacts/runs/`，不在这里重复提交。

| 版本 | 状态 | 入口 |
|---|---|---|
| v1 | 已完成 | [`v1/README.md`](v1/README.md) |
| v2 | Hidden-State PLP 已完成 | [`v2/plp_v2_results.md`](v2/plp_v2_results.md) |
| v3 | PLP-only 消融与 OOF 选择已完成，最终 Test 未打开 | [`v3/README.md`](v3/README.md) |

v2 Test 是复用已打开的 family holdout，因此属于开发性对照；严格确认性结果仍需要新
holdout。只有当某一版本完成冻结评价后，才创建对应的版本目录。

v3 当前只冻结开发阶段结论：5 折 family-grouped OOF 已选择 `plp_terminal_zero_v3`，但
12-family 最终 holdout 仍未使用，因此文档不会把它写成最终 Test 结论。

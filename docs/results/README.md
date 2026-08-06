# 实验结果

本目录只保存已经完成、数字已经固定的人工可读实验报告。原始JSON/CSV、模型和逐样本预测
保存在本地`artifacts/runs/`，不在这里重复提交。

跨阶段总览：[`baseline_alps_plp_v3_comparison.md`](baseline_alps_plp_v3_comparison.md)。

| 版本 | 状态 | 入口 |
|---|---|---|
| v1 | 已完成 | [`v1/README.md`](v1/README.md) |
| v2 | Hidden-State PLP 已完成 | [`v2/plp_v2_results.md`](v2/plp_v2_results.md) |
| v3 | PLP-only 三消融与最终 Test 已完成 | [`v3/plp_terminal_zero_v3_results.md`](v3/plp_terminal_zero_v3_results.md) |

v2 Test 是复用已打开的 family holdout，因此属于开发性对照；严格确认性结果仍需要新
holdout。只有当某一版本完成冻结评价后，才创建对应的版本目录。

v3 已通过 5 折 family-grouped OOF 从三个单因素消融中选择 `plp_terminal_zero_v3`，并完成
12-family 一次性最终 Test。Test MAE 点估计改善约 5.35%，但配对 95% CI 略微跨 0，因此
记录为“观察到小幅改善，严格优越性声明未通过”。

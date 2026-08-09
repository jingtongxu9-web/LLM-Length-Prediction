# Hybrid concat v1、residual v2 与 gated v2.1 实验结果

## 1. 实验问题

Hybrid 阶段研究 ALPS 的生成前全局先验和 PLP 的生成中动态状态应如何融合。三种版本使用相同
Train family、相同逐步 trace、相同五折 family-grouped OOF 和相同评价指标，差异只在融合结构。

## 2. 三种结构

| 版本 | 核心结构 | 研究假设 |
|---|---|---|
| concat v1 | 将 PLP 7168 维状态与 ALPS 5 维摘要拼接，由 MLP 直接预测剩余长度 | 两类信息应自由联合建模 |
| residual v2 | `ALPS countdown + PLP residual` | ALPS 给主路线，PLP 只负责动态纠偏 |
| gated residual v2.1 | 对 residual 加进度门、置信 gate 和有界修正 | 只有可靠时才允许 PLP 修改 ALPS |

## 3. 同口径 OOF 总体结果

| 方法 | Family-macro sequence-balanced MAE | 相对 concat v1 | 结论 |
|---|---:|---:|---|
| **Hybrid concat v1** | **49.916** | — | 选中 |
| residual v2 | 57.934 | +8.018 | 淘汰 |
| gated residual v2.1 | 56.067 | +6.151 | 比 v2 小幅改善，仍淘汰 |

作为解释参照，同一 OOF 数据上的 ALPS countdown MAE 为 `55.724`，PLP terminal-zero v3 为
`59.778`。concat v1 同时优于二者，说明两类信息存在互补性。

## 4. 分组与稳定性分析

### 4.1 按生成进度

| 解码进度 | ALPS MAE | concat v1 MAE | gated v2.1 MAE | v2.1 平均 effective gate |
|---|---:|---:|---:|---:|
| 0–10% | **57.529** | 58.964 | 57.143 | 0.038 |
| 10–25% | **58.588** | 59.068 | 58.893 | 0.144 |
| 25–50% | 60.428 | **58.068** | 62.528 | 0.313 |
| 50–75% | 57.708 | **50.445** | 59.222 | 0.527 |
| 75–100% | 47.134 | **32.689** | 45.126 | 0.795 |

concat v1 的总体收益主要出现在生成中后段，尤其是最后 25%；早期 ALPS 已提供较强全局先验，
concat 并未稳定胜过 ALPS。这个趋势符合两类信息的角色：decode state 随生成推进才逐渐包含更多
可用于修正长度的信息。

v2.1 的 gate 随进度机械增大，但 25%–75% 区间的 MAE 反而比 ALPS 更高，说明“后期更允许
修正”不等于“修正方向可靠”。

### 4.2 按任务与预设长度

| 分组 | ALPS MAE | concat v1 MAE | gated v2.1 MAE |
|---|---:|---:|---:|
| Code | 82.619 | **68.514** | 84.361 |
| QA | 55.481 | **53.010** | 54.253 |
| Summarization | 29.072 | **28.224** | 29.588 |
| Long | 75.448 | **66.168** | 75.496 |
| Medium | 70.788 | **63.010** | 72.728 |
| Short | 20.937 | 20.570 | **19.978** |

concat 对难度最高的 Code、Long 和 Medium 改善最明显；Short 本身误差较低，三种方法差距很小。
这说明 Hybrid 的主要价值不是所有样本平均微调，而是在长输出和复杂代码生成中融合动态证据。

### 4.3 任务×长度九宫格

| 单元 | ALPS MAE | concat v1 MAE | gated v2.1 MAE |
|---|---:|---:|---:|
| Code–Long | 107.45 | **85.66** | 111.07 |
| Code–Medium | 94.84 | **76.16** | 99.18 |
| Code–Short | 45.57 | 43.73 | **42.83** |
| QA–Long | 69.42 | 66.34 | **64.64** |
| QA–Medium | 89.00 | **82.90** | 90.04 |
| QA–Short | **8.02** | 9.79 | 8.07 |
| Summarization–Long | 49.47 | **46.51** | 50.78 |
| Summarization–Medium | **28.51** | 29.98 | 28.96 |
| Summarization–Short | 9.23 | **8.19** | 9.03 |

concat 在 9 个单元中的 6 个优于 ALPS，最大收益集中在 Code–Long 和 Code–Medium。它不是每个
单元都占优，因此论文应表述为“总体及复杂长输出上更强”，而不是无条件支配所有任务。

### 4.4 五折稳定性

concat v1 在 5 个 outer fold 中有 4 折优于 ALPS；唯一例外为 fold 4：ALPS `44.451`，concat
`52.426`。各折存在明显 family 难度差异，因此总体结论必须结合 family 配对 bootstrap，而不能
只看某一个折。gated v2.1 的表现也随折波动，进一步支持不冻结该版本。

## 5. v1 为什么有效

concat v1 没有强制规定 ALPS 和 PLP 谁是主、谁是辅。MLP 可以在生成早期更多使用 ALPS 全局
先验，在中后期更多使用 decode hidden state，也可以学习二者的非线性交互。虽然 ALPS 摘要只有
5 维，但 OOF 改善表明这些特征没有被 7168 维 PLP 状态稀释。

相对 ALPS，concat v1 的 MAE 减少 `5.808 token`，约 `10.4%`；相对 PLP v3 减少
`9.862 token`，约 `16.5%`。

## 6. v2 为什么失败

residual v2 看起来更符合“ALPS 先预测、PLP 再微调”的直觉，但它把 PLP 限制为 ALPS 的纠偏器。
实验中 v2 平均给出约 `+5.75 token` 的修正，而 ALPS 平均误差只有约 `+0.41 token`，说明残差
头形成了不需要的方向性偏移。PLP 的动态状态适合参与完整剩余长度判断，不一定适合只学习一个
附着在 ALPS 上的残差。

## 7. v2.1 改善了什么、为什么仍未选中

v2.1 使用：

```text
bounded_delta = B * tanh(raw_delta)
gate = progress * sigmoid(gate_logit)
prediction = max(0, ALPS_countdown + gate * bounded_delta)
```

它相对 v2 改善约 `1.87 token`。普通配对 95% CI 为 `[-3.71, -0.09]`，支持小幅数值改善；
但预先规定的严格 99% familywise CI 为 `[-4.20, 0.49]`，跨过 0，不能宣称多重比较后仍稳定优于 v2。

更重要的是，v2.1 相对 concat v1 仍差 `6.151 token`，严格区间为 `[+0.47, +13.01]`，明确
劣于 concat v1。其 learned confidence 中位数约 `0.989`，但修正成功率只有约 `49.5%`，gate
与 MAE 改善的相关性接近 0；即 gate 经常打开，却没有学会可靠判断什么时候应修改 ALPS。

## 8. 冻结结论

- 当前 Hybrid 代表方法冻结为 `alps_plp_concat_v1`。
- residual v2 和 gated residual v2.1 保留为结构消融和负向证据，不再继续用同一 OOF 调参。
- concat v1 已完成全量 Train 拟合；OOF 用于方法选择，全量模型用于未来新 holdout。
- 旧 PLP-only Test 已被打开，不能再冒充全新的 Hybrid Test；最终确认需要新建未参与选择的 holdout。

## 9. 冻结模型

| 方法 | 文件 | SHA-256 |
|---|---|---|
| ALPS | `alps_prior.json` | `7c446512e4c83d9e4332c8cfad50a148bbf0b505f271966ee5ece778ae5d2828` |
| PLP v3 | `plp_terminal_zero_v3.pt` | `8e460067150a0507389c1afedce1178d6445c4e43604598896d10d146acc2708` |
| concat v1 | `alps_plp_concat_v1.pt` | `f22bb194e7c827a3c657f82b17eb33adcdeaa37359ed0235163def93b5fdeae3` |
| residual v2 | `alps_plp_residual_v2.pt` | `2412636efa42c019f9ee8f54b846ebd5b639f3e9f930b7ca56ae48e61081d660` |

## 10. 原始证据位置

```text
artifacts/runs/alps_plp_hybrid_versions/oof/oof_report.json
artifacts/runs/alps_plp_hybrid_versions/oof/oof_predictions.csv
artifacts/runs/alps_plp_gated_residual_v2_1/oof/oof_report.json
artifacts/runs/alps_plp_gated_residual_v2_1/oof/oof_predictions.csv
artifacts/runs/alps_plp_hybrid_versions/models/model_registry.json
```

前两项支持 concat v1 与 residual v2；中间两项支持 gated v2.1 的增量同折诊断；模型注册表支持
全量 Train 权重、协议状态和 SHA-256。报告中的任务、长度、九宫格、进度和 fold 数字均来自上述
OOF 文件，而不是 Train 拟合指标。

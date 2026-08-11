# 实验结果总索引

本目录按**方法家族**保存已经完成、数字已经固定的人工可读报告。方法原理放在
[`../methods/`](../methods/README.md)，机器生成的 JSON/CSV、逐样本预测和模型放在本地
`artifacts/runs/`，不在这里重复提交。

## 1. 目录结构

```text
results/
├── baseline/       # 最小信息基线：Global mean、Prompt-token Ridge、Metadata 诊断
├── alps/           # ALPS 各版本的完整结果
├── plp/            # Dynamic v1、Hidden-State v2、Terminal-Zero v3
├── hybrid/         # concat v1、residual v2、gated residual v2.1
├── bayesian_sequential/ # PDF-aligned proposed method；Stage 3 工程 pilot 已通过，尚无效果结果
└── comparisons/    # 跨方法综合比较与论文主表
```

旧目录名 `v1/v2/v3` 同时混合了 ALPS 和 PLP 的不同阶段，容易误认为是同一模型连续升级。本次只
改变文档归档位置，不修改已经冻结的实验数字。

## 2. Baseline

入口：[`baseline/README.md`](baseline/README.md)

| 报告 | 内容 |
|---|---|
| [`baseline/prompt_token_ridge_results.md`](baseline/prompt_token_ridge_results.md) | 输入 token Ridge、全局均值、Metadata 诊断和当前同口径 countdown 补充实验 |

## 3. ALPS

入口：[`alps/README.md`](alps/README.md)

| 版本 | 报告 | 状态 |
|---|---|---|
| ALPS v1 / Layer 14 Ridge | [`alps/alps_v1_results.md`](alps/alps_v1_results.md) | Train、五折 OOF、Test、九宫格和概率校准分析均已完成 |

## 4. PLP

入口：[`plp/README.md`](plp/README.md)

| 版本 | 报告 | 状态 |
|---|---|---|
| Dynamic-Signal MLP v1 | [`plp/dynamic_signal_mlp_v1_results.md`](plp/dynamic_signal_mlp_v1_results.md) | 已完成，作为早期动态 baseline |
| Hidden-State PLP v2 | [`plp/hidden_state_plp_v2_results.md`](plp/hidden_state_plp_v2_results.md) | 已完成开发性 Test |
| Terminal-Zero PLP v3 | [`plp/terminal_zero_v3_results.md`](plp/terminal_zero_v3_results.md) | 三项消融、OOF 选择和一次性 Test 已完成 |

## 5. Hybrid

入口：[`hybrid/README.md`](hybrid/README.md)

| 版本 | 报告 | 状态 |
|---|---|---|
| concat v1 / residual v2 / gated v2.1 | [`hybrid/hybrid_v1_v2_v2_1_results.md`](hybrid/hybrid_v1_v2_v2_1_results.md) | concat v1 已选中并完成全量 Train；v2/v2.1 已淘汰 |

## 6. Bayesian Sequential

入口：[`bayesian_sequential/README.md`](bayesian_sequential/README.md)

数学与实验合同、CPU 核心和真实 Qwen Stage 3 工程 pilot 已完成；当前可报告内容仅限采集与
数据合同验收，不含 Bayesian OOF 或 final-Test 效果数字。见
[`bayesian_sequential/stage3_pilot_20260811.md`](bayesian_sequential/stage3_pilot_20260811.md)。
旧 PLP/Hybrid 结果不能复制到该目录或改名为 Bayesian posterior。

## 7. 跨方法对比

入口：[`comparisons/README.md`](comparisons/README.md)

| 报告 | 内容 |
|---|---|
| [`comparisons/stage1_alps_baselines_dynamic.md`](comparisons/stage1_alps_baselines_dynamic.md) | 第一阶段 ALPS、静态 baseline 与旧 Dynamic-Signal MLP 的完整总结 |
| [`comparisons/baseline_alps_plp_v3_comparison.md`](comparisons/baseline_alps_plp_v3_comparison.md) | Baseline、ALPS、PLP v3 的跨阶段对比 |
| [`comparisons/four_method_main_comparison.md`](comparisons/four_method_main_comparison.md) | PDF 主线回归前的历史四方法比较；concat v1 是 discriminative baseline |

四方法报告不属于 `hybrid/`：它横跨四个方法家族。当前 ALPS、PLP v3 和 concat v1 已有相同
Hybrid Train trace 上的 OOF 数字；Prompt-token countdown 的同口径数字仍需在 AutoDL 现有
trace 上补算。

## 8. 论文写作建议

论文的 Methods 部分从 [`../methods/README.md`](../methods/README.md) 取技术原理；Experiments
部分按本目录组织：

1. Baseline 验证最小输入长度信息；
2. ALPS 验证生成前静态语义先验；
3. PLP 验证生成中动态 hidden state；
4. Hybrid 比较不同判别式融合结构；
5. Bayesian Sequential 验证 prior + incremental evidence + posterior update；
6. 最终主表加入 selected Bayesian method；
7. PLP 与 Hybrid 的其他版本作为 baseline/消融。

## 9. Test 边界

PLP-only v3 已使用原协议的 12-family Test，因此该 Test 不能再次称为未见的 Hybrid Test。
Hybrid concat v1 的最终确认需要新建从未参与方法选择的 holdout。OOF 负责开发与选型，全量
Train 模型负责未来 holdout，两者不能混称为最终 Test。

# ALPS + PLP Hybrid 实验结果

Hybrid 阶段比较三种融合结构。当前选中的是自由特征拼接 `concat v1`；`residual v2` 和
`gated residual v2.1` 是用于检验串行修正假设的消融版本。

| 报告 | 内容 |
|---|---|
| [`hybrid_v1_v2_v2_1_results.md`](hybrid_v1_v2_v2_1_results.md) | 三个 Hybrid 版本的同折 OOF 结果、统计比较、失败原因和冻结结论 |

融合原理见 [`../../methods/hybrid_concat_residual_gated.md`](../../methods/hybrid_concat_residual_gated.md)。
四种核心路线的总比较位于
[`../comparisons/four_method_main_comparison.md`](../comparisons/four_method_main_comparison.md)。

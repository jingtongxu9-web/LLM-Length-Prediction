# PLP 实验结果

这里按照 PLP 方法自身的版本演进保存报告。三个版本不能互相覆盖：它们使用的输入和研究问题不同。

| 版本 | 方法身份 | 状态 | 报告 |
|---|---|---|---|
| Dynamic-Signal MLP v1 | 五个人工动态信号的早期项目基线 | 已完成 | [`dynamic_signal_mlp_v1_results.md`](dynamic_signal_mlp_v1_results.md) |
| Hidden-State PLP v2 | Prompt/decode hidden-state PLP | 已完成开发性 Test | [`hidden_state_plp_v2_results.md`](hidden_state_plp_v2_results.md) |
| Terminal-Zero PLP v3 | 对 v2 做三项单因素消融后选中的 PLP-only | 已完成一次性 Test | [`terminal_zero_v3_results.md`](terminal_zero_v3_results.md) |

详细原理见 [`../../methods/plp_only_explained.md`](../../methods/plp_only_explained.md)，版本关系见
[`../../methods/plp_versions.md`](../../methods/plp_versions.md)。

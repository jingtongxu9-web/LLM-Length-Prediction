# 方法原理总索引

本目录解释模型为什么这样设计，不记录会随实验运行变化的结果数字。结果统一位于
[`../results/`](../results/README.md)。

## 从简单到复杂阅读

| 顺序 | 文档 | 主要内容 |
|---:|---|---|
| 1 | [`prompt_token_baseline.md`](prompt_token_baseline.md) | 只用输入 token 数的 Ridge、`log1p` 目标、countdown 与五折逻辑 |
| 2 | [`alps.md`](alps.md) | Layer 14、最后 Prompt token、Ridge、静态总长度预测与概率区间 |
| 3 | [`plp_versions.md`](plp_versions.md) | Dynamic v1、Hidden-State v2、Terminal-Zero v3 的版本关系 |
| 4 | [`plp_only_explained.md`](plp_only_explained.md) | 面向初学者完整解释 entropy pooling、`h_prompt`、`h'_t`、7168 维输入、20-bin 和 MLP |
| 5 | [`hybrid_concat_residual_gated.md`](hybrid_concat_residual_gated.md) | concat v1、residual v2、gated v2.1 的融合结构、训练和 OOF |
| 6 | [`bayesian_sequential_inference.md`](bayesian_sequential_inference.md) | **项目核心主线**：ALPS prior、增量 evidence、Bayes update、hazard、uncertainty 与实验边界 |

## 项目核心方法与历史 baseline

| 路线 | 身份 | 生成前/生成中 | 输入 | 预测器 |
|---|---|---|---|---|
| Prompt-token baseline | Baseline | 生成前 | Prompt token 数 | Ridge |
| ALPS | Prior / baseline | 生成前 | Layer-14 Prompt hidden state | Ridge + shifted-lognormal prior |
| Dynamic-Signal MLP v1 | Dynamic baseline | 生成中 | step、entropy、EOS 等标量 | 点回归 MLP |
| PLP terminal-zero v3 | Hidden-state baseline | 生成中 | Prompt/decode final-layer hidden states | Progressive MLP |
| Hybrid concat v1 | Discriminative-fusion baseline | 生成中 | PLP states + ALPS 摘要 | Progressive MLP |
| Bayesian Sequential v1 | **Proposed method** | 生成前 + 生成中 | ALPS prior + 非重叠 decode evidence | Likelihood-ratio head + Bayesian filter |

现有 PLP softmax 或 Hybrid concat 输出不是由 `prior × likelihood` 递归更新得到的 posterior。
因此旧结果继续有效，但不得把它们改名为 Bayesian sequential inference。

## 结果与实施合同

- 完成后的实验数字：[`../results/README.md`](../results/README.md)
- 尚未执行或后续改进：[`../planning/`](../planning/)
- 机器可读冻结合同：[`../../configs/experiments/`](../../configs/experiments/)
- 实际运行入口：[`../../scripts/README.md`](../../scripts/README.md)

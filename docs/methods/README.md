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

## 四条论文主路线

| 路线 | 生成前/生成中 | 输入 | 预测器 |
|---|---|---|---|
| Prompt-token baseline | 生成前 | Prompt token 数 | Ridge |
| ALPS | 生成前 | Layer-14 Prompt hidden state | Ridge |
| PLP terminal-zero v3 | 生成中 | Prompt/decode final-layer hidden states | Progressive MLP |
| Hybrid concat v1 | 生成中 | PLP states + ALPS 摘要 | Progressive MLP |

## 结果与实施合同

- 完成后的实验数字：[`../results/README.md`](../results/README.md)
- 尚未执行或后续改进：[`../planning/`](../planning/)
- 机器可读冻结合同：[`../../configs/experiments/`](../../configs/experiments/)
- 实际运行入口：[`../../scripts/README.md`](../../scripts/README.md)

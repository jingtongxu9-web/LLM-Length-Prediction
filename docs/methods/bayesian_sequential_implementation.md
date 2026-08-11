# Bayesian Sequential v1 第二阶段实现说明

## 状态与边界

第二阶段已于 2026-08-11 完成不依赖 Qwen 推理或新 GPU trace 的概率与训练核心。它不包含新实验数字，也不
打开 final holdout。科学定义仍以
[`bayesian_sequential_inference.md`](bayesian_sequential_inference.md) 和
[`../../configs/experiments/bayesian_sequential_v1.json`](../../configs/experiments/bayesian_sequential_v1.json)
为准。

## 模块

| 模块 | 责任 |
|---|---|
| `models/prior.py` | family-grouped OOF log1p residual variance 校准与 full-Train Ridge 重拟合 |
| `models/bayesian_filter.py` | shifted-lognormal 半整数离散化、overflow、存活条件化、log-space update、posterior summary |
| `models/hazard.py` | posterior 与离散 stopping hazard 的可逆转换 |
| `data/sequential.py` | 冻结更新点、非重叠 block features、hidden delta、exact/censored sequence target |
| `models/bayesian_scorer.py` | scalar/hidden-delta 共享 scorer、rollout-balanced sequence loss、AdamW、checkpoint |
| `evaluation/sequential.py` | NLL、CRPS、coverage/width、点指标、stable-time、overflow 与更新耗时 |

## 已实现的不变量

- prior 的 exact integer states 为 `0..4096`，额外一格是 `L > 4096` 的 overflow；
- transition 只把 exact state 左移实际生成的 `delta`，overflow 始终保持为 tail state；
- evidence block 使用 `(t_prev, t]`，不会重复乘入历史 token；
- scorer 对所有候选长度共享参数，输出增量 log-likelihood-ratio，不直接回归长度；
- hidden-delta 候选的随机投影是固定 buffer，不是可训练参数；
- 一条 rollout 内先平均 timestep loss，再在 rollout 间平均；
- `max_new_tokens` 停止使用 `P(R_t > censor_boundary)`，其中包含 overflow；
- terminal token 以 `R_t=0` 进入训练与评价；
- checkpoint 固定 method ID、scorer spec、合同 digest 与训练报告。

## 第二阶段测试门

合成序列测试必须覆盖：

1. shifted-lognormal prior 的整数质量与上尾保存；
2. countdown/survival transition 方向；
3. 极端 likelihood score 下的 log-space 数值稳定性；
4. posterior/hazard 往返；
5. `1,5,10,...,+terminal` 和非重叠 block；
6. exact 与 right-censored sequence loss；
7. scalar 与 hidden-delta scorer 身份隔离；
8. checkpoint 保存、读取和重建；
9. 概率、点、coverage、收敛与耗时指标。

通过这些测试只表示 CPU 核心实现正确。进入第三阶段前仍需做统一 collector pilot，验证真实
token、EOS、hidden-state layer、temperature softmax 和 terminal semantics。

# Bayesian Sequential v1 第七阶段：Train-family OOF error feedback

## 0. 结论边界

第七阶段按预先冻结的 `>100 token` 与序列 MAE 最差 5% 规则，审计第五阶段选中的
`bayesian_entropy_scalar_v1`。输入仍是 60 个 Train family、1,620 条真实 Qwen trace 和
137,957 个 OOF 更新点；1,620 个 trace SHA-256 与 Stage-5 archive 的 51 个文件再次验证通过。

本阶段没有 refit、没有重新选择方法、没有调标签阈值，也没有创建或访问 final holdout。
自动标签表示“trace 中观察到该模式”，不是已经证明它导致误差。

## 1. Error cohorts

| Cohort | Sequence | Mean sequence MAE | Mean bias | Negative-bias rate | Mean max under / over |
|---|---:|---:|---:|---:|---:|
| 全部 OOF | 1,620 | 70.11 | -3.38 | 47.78% | 62.88 / 87.52 |
| 任一保存点 `|error| > 100` | 839 | 111.01 | -13.73 | 54.95% | 105.59 / 130.88 |
| 序列 MAE 最差 5% | 81 | 245.44 | -113.87 | 72.84% | 278.43 / 119.54 |

最差 5% 的冻结 `higher` 阈值为 `191.9036 token`。这 81 条全部已包含在 839 条 absolute-error
cohort 中，所以人工 review queue 的并集仍为 839 条。最差 5% 的平均 bias 明显偏负，说明最严重
的序列级失败主要呈低估方向；这与第六阶段 long-tail 低估相符。

## 2. 自动 trace 标签

| 自动标签 | 全部 1,620 | Review 839 | Review rate | 解释边界 |
|---|---:|---:|---:|---|
| entropy rebound | 550 | 349 | 41.60% | 后 25% entropy 相对此前最低分块均值回升至少 0.25 nat |
| entropy oscillation | 1,432 | 831 | 99.05% | 高频信号；过于常见，只能作为 review cue |
| sampling divergence | 39 | 15 | 1.79% | 同 Prompt/温度三 seed 长度 range 与 CV 同时越界 |
| repetition | 184 | 163 | 19.43% | token 4-gram 重复或同 token 连续运行；非语义重复判定 |
| early stop | 1 | 0 | 0.00% | peer-relative 规则，没有解释 review cohort |
| posterior variance increase | 0 | 0 | 0.00% | 进度分箱 median 未出现冻结的 25% 相邻增幅 |
| posterior premature collapse | 12 | 9 | 1.07% | 早期窄 95% interval 且至少两点未覆盖真值 |
| posterior oscillation | 770 | 659 | 78.55% | 总长度预测范围至少 100 且方向改变至少 3 次 |

`entropy_oscillation` 在全部序列也有 88.40%，不能据此声称它是 error-specific failure cause。
`posterior_oscillation` 在 review queue 更集中，但这里没有做预注册因果检验，也不把相关性写成
机制结论。

## 3. Temperature 与任务模式

Review queue 在 temperature `0.3 / 0.7 / 1.0` 分别有 `300 / 279 / 260` 条。虽然高温条数较少，
但 temperature `1.0` 的平均序列 MAE 最高（`120.57`）、平均 bias 最负（`-62.82`）、负 bias
比例最高（`73.85%`），且 entropy rebound 达 `81.15%`；temperature `0.3` 的平均 bias 则为
`+28.60`。这支持把 sampling-policy shift 和高温低估作为下一方法的设计问题，但不允许给
现有 v1 偷加 temperature 特征。

按任务，review queue 为 code `379`、QA `308`、summarization `152`。QA 的平均 MAE 最高
（`120.59`）；code 的 token-level repetition 标记最常见（`32.98%`）。按长度，long/medium/
short 分别为 `465 / 349 / 25`；short 数量少但 sampling-divergence 与 premature-collapse 比例较高，
应谨慎解释小样本。

## 4. 无法自动判定的语义标签

`open_ended_prompt` 和 `hallucination` 均保持 `unresolved`。数值 trace 没有足够证据可靠回答：

- Prompt 是否真正开放式、允许多种合理长度；
- 输出中的具体事实是否被参考答案或来源证伪。

因此 839 条 review queue 只列出这两项人工复核需求，不把缺少标签误记为阴性。以后若进行语义
复核，应冻结 annotation guideline、盲法和一致性协议；结果仍只能来自 Train-family OOF。

## 5. 下一方法候选，不修改 v1

本阶段只形成设计假设：如果继续开发，应以新 ID（例如
`bayesian_entropy_temperature_stability_v2`）显式建模 temperature / entropy stability，并重新执行
完整五折 family-grouped OOF。候选至少需要同时检验：

1. high-temperature negative bias 与 long-tail early underestimation；
2. posterior total oscillation；
3. NLL/CRPS/coverage 是否改善，而非只改善点 MAE；
4. 不把人工语义标签或 final holdout 用于调参。

当前证据不充分到可以直接实现并宣称 v2；`bayesian_entropy_scalar_v1` 与第五阶段选择结论保持冻结。

## 6. 阶段决策

第七阶段的自动 OOF error feedback 已完成。它确认最严重 5% 主要偏向低估，并定位了高温 entropy
rebound、重复模式与 posterior oscillation 等可复核现象，同时明确排除了“自动判断幻觉”的不可靠
做法。进入一次性 final benchmark 之前，仍需决定是否预注册并完整 OOF 验证新 method ID；如果
不做新方法，则应直接冻结现有方法、schema、比较列表和 holdout 协议，不能在 final 结果之后返工。

## 7. Provenance

| 项目 | SHA-256 |
|---|---|
| Stage-7 config | `168906452bddce9725a822e575d8130d9360878002820e2bc7e28466c8b08f51` |
| Stage-7 report schema | `09c10c0d5c7d0bdc8ef918179106d3e77119f339f559d71b01fa60f5a3b33a6a` |
| Stage-7 full report | `c0e144af839bcb234dda7c2ed88d790c3f390432b5c1618492ccc876ea40df75` |
| Sequence audit | `88b009dcf3774a5767625a6d41b898840186ff4a902398b3b1d17c7bcd56f32b` |
| Manual review queue | `456197fc57cd85b6c3128395ca8c1b0a6d14db53948af40c8746d0f023f21d71` |

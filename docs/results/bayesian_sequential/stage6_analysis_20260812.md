# Bayesian Sequential v1 第六阶段：不确定性、收敛与 serving

## 0. 结论边界

2026-08-12 基于第五阶段冻结的 5-fold family-grouped OOF 结果完成第六阶段分析。输入覆盖 60 个
Train family、1,620 条真实 Qwen trace 和 137,957 个逐更新点。Stage 5 archive 中 51 个文件
重新通过 SHA-256；五折 scalar checkpoint 在 CPU 上精确重放，posterior mean 与服务器冻结预测
的最大差为 `0.000104 token`。

本阶段没有训练或修改任何模型、没有调阈值、没有 robustness refit，也没有创建或访问 final
holdout。报告 `status=pass` 表示分析完整和边界合规，不表示下述每一项科学假设都通过。

## 1. 不确定性收缩与校准

主温度 `0.7` 下，选中 Bayesian scalar 的 sequence-balanced 进度结果为：

| Decode progress | Posterior variance | Entropy | 50% coverage | 90% coverage | 95% coverage |
|---|---:|---:|---:|---:|---:|
| 0–10% | 27,882.49 | 5.7953 | 56.68% | 92.61% | 95.66% |
| 10–25% | 20,450.65 | 5.7473 | 54.31% | 91.39% | 95.35% |
| 25–50% | 15,575.28 | 5.6289 | 52.77% | 88.62% | 92.93% |
| 50–75% | 10,899.25 | 5.4119 | 57.86% | 91.00% | 95.34% |
| 75–100% | 6,665.01 | 5.1201 | 44.73% | 84.97% | 91.49% |

方差从首段降到末段的 `23.90%`，entropy 降到 `88.35%`，所以“生成证据使 posterior 收缩”
得到数值支持。但是最后 25% 的名义 90%/95% coverage 分别只有 `84.97%/91.49%`，均超出冻结
的 ±2 percentage-point 校准容差。结论只能写成：**posterior 在收缩，但后段存在过度自信；
方差下降不能单独视为成功。**

从五折 checkpoint 精确重放的 uncertainty cone 已按 temperature × progress 保存。主温度早期
真实剩余长度的 sequence-balanced 均值约 `397.95`，posterior median 约 `378.12`，2.5%/97.5%
边界约 `191.45/716.20`；后段真实剩余约 `50.24`，median 约 `76.87`，边界约 `4.75/265.60`。
后段 cone 仍较宽且 mean/median 偏高，与 coverage、bias 需要联合解释。

## 2. 严格 5% 稳定收敛

冻结定义要求总长度预测在某个保存点进入 5% 相对误差后，**所有后续保存点**都保持在 5% 内。
始终未达到者记为失败，不能用 terminal point 冒充成功。

| 分组 | Sequence | 成功 | 成功率 | 成功样本 median step | median progress |
|---|---:|---:|---:|---:|---:|
| 全部温度 | 1,620 | 30 | 1.85% | 980 | 89.71% |
| temperature 0.3 | 540 | 2 | 0.37% | 1,060 | 89.72% |
| temperature 0.7 | 540 | 11 | 2.04% | 990 | 89.43% |
| temperature 1.0 | 540 | 17 | 3.15% | 950 | 89.84% |

Summarization 540 条全部未达到严格稳定条件；short intended-length 也全部失败。这里没有预注册
额外 pass threshold，因此不做事后二元门控，但 `1.85%` 且成功通常晚至约 90% progress，明显
不支持“Bayesian scalar 很早就稳定达到 5% 总长度误差”的强主张。

## 3. 长尾早期低估

主温度真实总长度经验 top 10% 使用冻结 `higher` 分位数，阈值为 `869 token`；只评价前 25%
生成进度。

| 方法 | MAE | Bias | Positive underestimation | >100 token underestimation rate |
|---|---:|---:|---:|---:|
| ALPS countdown | 136.36 | -56.04 | 96.20 | 50.00% |
| Bayesian scalar | 198.61 | -190.05 | 194.33 | 76.89% |
| PLP terminal-zero v3 | 240.02 | -237.33 | 238.68 | 77.31% |
| ALPS+PLP concat v1 | 187.49 | -183.99 | 185.74 | 68.91% |

选中 scalar 的 early-tail positive underestimation 比 ALPS 多 `98.13 token`，没有改善该风险。
在 temperature `1.0` 的对应尾部，scalar early bias 进一步降到约 `-208.01 token`。这与第五
阶段高温负 bias 的 robustness 结果一致，是第七阶段 error feedback 的核心对象。

## 4. 在线更新开销

以下使用第五阶段 RTX 4090 D 实际记录的逐更新时延，并与第四阶段同一请求真实 Qwen 生成时延
比较；没有把本地 checkpoint 重放时延混进来。

| 指标 | 结果 |
|---|---:|
| Mean / p50 update | 0.962 / 0.946 ms |
| p95 / p99 update | 1.069 / 1.181 ms |
| Mean cumulative update per sequence | 81.93 ms |
| p95 cumulative update per sequence | 179.63 ms |
| Mean predictor / Qwen generation duration | 1.087% |
| p95 ratio | 1.359% |
| Peak predictor state | 93,864 bytes |

这支持“当前 scalar posterior update 本身开销较小”，但不是含网络、调度、Qwen decode 和生产
框架的端到端延迟结论。

## 5. Deterministic serving replay

在 step 1 使用真实 Qwen rollout duration、batch size 8、16-token KV quantum 和固定长度 bucket
重放。Qwen2.5-7B-Instruct 冻结结构对应 output-token 增量 KV `57,344 bytes/token`。这里不包含
Prompt KV、模型权重或 activation，因此不等于整机显存。

| 策略 | Throughput tok/s | KV overreservation rate | Underallocation rate | Peak batch output-KV |
|---|---:|---:|---:|---:|
| Oracle + quantum | 345.29 | 1.76% | 0.00% | 497.0 MiB |
| 固定 4096 | 266.69 | 878.17% | 0.00% | 1,792.0 MiB |
| ALPS mean | 326.09 | 13.23% | 35.56% | 518.0 MiB |
| PLP v3 | 331.25 | 5.42% | 52.47% | 360.5 MiB |
| concat v1 | 331.38 | 5.43% | 49.26% | 361.4 MiB |
| Bayesian scalar mean | 326.44 | 12.46% | 36.79% | 505.7 MiB |
| Bayesian scalar q97.5 | 326.99 | 89.33% | 1.36% | 959.9 MiB |

Posterior mean 较省 KV，但 36.79% request underallocate；97.5% 上界将该比例降到 1.36%，代价
是 overreservation 升至 89.33%。没有一个 Bayesian 策略同时支配两端。因此本阶段记录的是明确
的容量—风险 tradeoff，不声称生产 serving superiority。

## 6. Provenance

| 项目 | SHA-256 / 值 |
|---|---|
| Stage-6 config | `700cf2c944eb886cb90c6ff8a8279c948fde941e68f04c4c12c13ff4357d42ee` |
| Scientific contract | `10187cda101bd364418d9b999568006dc75625c222eefee3c65e371c51e4dcb4` |
| Stage-4 dataset digest | `636b48c6fc3a94fcf0aac60696ad938b328beed568cf281ffd1834d1fdb3d328` |
| Stage-5 OOF report | `51e4545a7a154b3af29885093a20bf649240fefa97b438d54bee0955c2358559` |
| Stage-5 file manifest | `fc53dc2a693b114b1db71cd19fb89b49d2927dcca3b7203d14fdbf1e1111381c` |
| Stage-6 full report | `aa9b1ed93fccb46bad3ad2efd2517fbccf0d4aa2926470b00ecfae25be497cff` |
| Uncertainty curves | `b60f1baf66d60c46b87e6fb0eef6a02c104dfb13a5eedbbd6f1ee34f14c3cece` |
| Uncertainty cone | `7e4ce114cfe39bf2c4ebd6db4cf6e7f58a5450e3a4c3f120d87ab755ea3388f2` |
| Serving replay | `654c7b1d902f8119ed743099300a39214487480c22c22aea4bffea5879b75e21` |

KV 结构参数来自冻结 revision 的[官方 Qwen2.5-7B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)。

## 7. 阶段决策

第六阶段完成。证据支持 Bayesian scalar 相对 ALPS 的总体概率/点指标改善和低成本更新，但同时
否定三种过强表述：不确定性并非全进度校准、5% 总长度误差并非早期稳定收敛、长输出早期低估
并未解决。下一步只在 Train-family OOF 上进入第七阶段 error feedback；仍不得访问 final
holdout。

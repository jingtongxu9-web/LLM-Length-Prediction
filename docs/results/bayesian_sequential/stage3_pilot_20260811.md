# Bayesian Sequential v1 第三阶段真实 Qwen Pilot

## 0. 结论边界

2026-08-11 在 AutoDL 单卡 RTX 4090 上完成冻结的 9-rollout Train-only pilot，机器验收、
统一 trace、断点续跑、terminal 语义和本地二次校验全部通过。第三阶段工程门状态为
**pass**，允许进入第四阶段 full-Train collector 的冻结与实现。

本报告**不构成 Bayesian 方法效果结果**：pilot 没有训练 Bayesian scorer，没有计算 OOF
NLL/CRPS/coverage，也没有访问 final holdout。它只证明真实 Qwen 数据采集链路满足合同。

机器可读摘要见
[`stage3_pilot_20260811_summary.json`](stage3_pilot_20260811_summary.json)。

## 1. 冻结范围

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| 数据边界 | 仅已打开的 Train design families；未访问 final holdout |
| 任务 | QA、Summarization、Code |
| 长度 | Short、Medium、Long |
| Rollout | 3 family × 3 length × 1 temperature × 1 seed = 9 |
| Sampling | temperature `0.7`、top-p `0.95`、seed `42` |
| 最大生成 | `4096` tokens，EOS 计入输出长度 |
| Trace stride | `1,5,10,...,+terminal` |
| 概率信号 | temperature-scaled full softmax，位于 top-p 截断之前 |
| Hidden states | zero-based Layer 14 prior、Prompt pooled、initial/final decode state |

## 2. 服务器环境

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，标称 25.26 GB / PyTorch 23.53 GiB |
| Compute capability | 8.9 |
| BF16 | 支持 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.48.3 |
| CUDA runtime | 12.8 |
| 峰值 allocated | 14.42 GiB |
| 峰值 reserved | 14.68 GiB |

4090 的 23.53 GiB 是厂商标称 24 GB 显卡的正常二进制显示值。preflight 已改为按厂商十进制
24 GB 合同判断，同时保留 GB 与 GiB 两种报告值。

## 3. 验收结果

| 指标 | 结果 |
|---|---:|
| Expected / valid traces | 9 / 9 |
| Missing selected jobs | 0 |
| Task × length cells | 9 / 9 |
| EOS / censored | 9 / 0 |
| Censoring rate | 0.0 |
| Total observed tokens | 3,843 |
| Summed per-trace collector duration | 77.237 s |
| Aggregate observed-token throughput | 49.76 tokens/s |
| Total compressed trace bytes | 5,652,103 |
| Warnings / failures | 0 / 0 |
| Final holdout accessed | false |

`status=pass`、`real_qwen_pilot_complete=true`。本地下载后重新校验 archive SHA、9 个 NPZ、
9 行 collection index、每条 trace SHA 和 frozen pilot contract，全部通过。

## 4. 九宫格人工审计

| Cell | Observed tokens | Saved points | Terminal step | Stop |
|---|---:|---:|---:|---|
| QA / Short | 41 | 10 | 41 | EOS |
| QA / Medium | 487 | 99 | 487 | EOS |
| QA / Long | 711 | 144 | 711 | EOS |
| Summarization / Short | 48 | 11 | 48 | EOS |
| Summarization / Medium | 217 | 45 | 217 | EOS |
| Summarization / Long | 385 | 78 | 385 | EOS |
| Code / Short | 157 | 33 | 157 | EOS |
| Code / Medium | 849 | 171 | 849 | EOS |
| Code / Long | 948 | 191 | 948 | EOS |

全部 trace 满足：token/entropy/EOS 数组长度等于 observed tokens；hidden-state steps 严格遵循
冻结 stride 并包含真实 terminal；最后 token 是合法 EOS；所有数组有限；Layer 14、final-layer、
model/tokenizer revision 和 probability source 均与合同一致。

## 5. Provenance 与归档策略

| 项目 | 值 |
|---|---|
| Collector server HEAD | `64fe01de7319edb1dc7868bae86eff506c18d2e6` |
| Nominal-memory preflight fix | `cedccb6b6890181aa57c504b0d067f407282c2cd` |
| Runtime preflight SHA-256 | `84069031f29aac02c5847533351e2f9060e6656c75b3e26e052df79cafe278f1` |
| Local archive | `bayesian_stage3_pilot_20260811.tar.gz` |
| Archive SHA-256 | `fce730b46d8dd8f860ea7dcbdcadb95d91b663eb83a1031b03999ff6316e2c79` |

服务器因 GitHub 网络超时停留在 collector commit `64fe01d`；实际执行的 preflight 文件与远程
修复 commit `cedccb6` 中的 runtime 文件 SHA 完全一致。原始 9 个 NPZ、archive、完整
`artifacts/`、模型权重和 `pip freeze` 只保存在本地/备份介质，不提交 Git。Git 仅保存本报告、
脱敏摘要和哈希。

## 6. 阶段决策

第三阶段完成并冻结。第四阶段开始前需要新建 full-Train 配置、预算 preflight、完整
acceptance schema 和 resumable collector；不得直接把 pilot 配置扩写后开跑，也不得访问
final holdout。

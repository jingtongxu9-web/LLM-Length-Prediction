# 项目文档

`docs/` 将论文所需内容分为五类：方法原理、实验结果、研究计划、部署手册和参考资料。可执行命令
位于 `scripts/`，机器可读实验合同位于 `configs/experiments/`，大体积 trace、模型和逐样本预测
位于本地 `artifacts/runs/`。

## 1. 最推荐的阅读顺序

1. [`methods/bayesian_sequential_inference.md`](methods/bayesian_sequential_inference.md)：阅读项目核心 Bayesian Sequential 数学合同；
2. [`methods/bayesian_sequential_implementation.md`](methods/bayesian_sequential_implementation.md)：查看第二阶段 CPU 核心实现；
3. [`methods/README.md`](methods/README.md)：理解 Baseline、ALPS、PLP、判别式 Hybrid 与 proposed method 的关系；
4. [`results/README.md`](results/README.md)：按方法家族找到每一个冻结历史实验报告；
5. [`planning/pdf_alignment_recovery_plan.md`](planning/pdf_alignment_recovery_plan.md)：查看回到 PDF 主线的实施阶段；
6. [`planning/research_plan.md`](planning/research_plan.md)：了解后续 holdout 与 serving 路线；
7. [`deployment/autodl_5090.md`](deployment/autodl_5090.md)：在 AutoDL 运行。

## 2. 目录结构

```text
docs/
├── methods/                       # 模型原理，不混入实验数字
│   ├── prompt_token_baseline.md
│   ├── alps.md
│   ├── plp_versions.md
│   ├── plp_only_explained.md
│   ├── hybrid_concat_residual_gated.md
│   ├── bayesian_sequential_inference.md
│   └── bayesian_sequential_implementation.md
├── results/                       # 已完成实验的人工可读报告
│   ├── baseline/
│   ├── alps/
│   ├── plp/
│   ├── hybrid/
│   └── comparisons/
├── planning/                      # 尚未执行或仍在设计的工作
├── deployment/                    # AutoDL、学校 Docker 与隔离服务器手册
└── references/                    # 背景 PDF 与论文资料
```

## 3. 方法原理

| 文档 | 内容 |
|---|---|
| [`methods/prompt_token_baseline.md`](methods/prompt_token_baseline.md) | 输入长度 Ridge baseline |
| [`methods/alps.md`](methods/alps.md) | ALPS Layer-14 hidden state 与 Ridge |
| [`methods/plp_versions.md`](methods/plp_versions.md) | PLP v1/v2/v3 版本关系 |
| [`methods/plp_only_explained.md`](methods/plp_only_explained.md) | PLP 的 entropy pooling、decode state、20-bin 和 MLP 细节 |
| [`methods/hybrid_concat_residual_gated.md`](methods/hybrid_concat_residual_gated.md) | Hybrid concat v1、residual v2 与 gated residual v2.1 |
| [`methods/bayesian_sequential_inference.md`](methods/bayesian_sequential_inference.md) | **项目 proposed method**：ALPS prior、增量 evidence、posterior update 与 uncertainty |
| [`methods/bayesian_sequential_implementation.md`](methods/bayesian_sequential_implementation.md) | Bayesian Sequential v1 的第二阶段 CPU 核心实现说明 |

## 4. 实验报告

完整索引见 [`results/README.md`](results/README.md)。核心入口：

| 方法家族 | 报告入口 |
|---|---|
| Baseline | [`results/baseline/README.md`](results/baseline/README.md) |
| ALPS | [`results/alps/README.md`](results/alps/README.md) |
| PLP | [`results/plp/README.md`](results/plp/README.md) |
| Hybrid | [`results/hybrid/README.md`](results/hybrid/README.md) |
| 跨方法比较 | [`results/comparisons/README.md`](results/comparisons/README.md) |

这里保留了此前生成的全部实质性报告。旧的 `results/v1`、`v2`、`v3` 只是重新按方法语义移动：

- 旧 `v1/alps.md` → `alps/alps_v1_results.md`；
- 旧 `v1/dynamic_signal_mlp.md` → `plp/dynamic_signal_mlp_v1_results.md`；
- 旧 `v2/plp_v2_results.md` → `plp/hidden_state_plp_v2_results.md`；
- 旧 `v3/plp_terminal_zero_v3_results.md` → `plp/terminal_zero_v3_results.md`；
- 旧 `v1/README.md` → `comparisons/stage1_alps_baselines_dynamic.md`。

## 5. 计划与部署

- [`planning/alps_improvement_plan.md`](planning/alps_improvement_plan.md)：ALPS 概率区间校准；
- [`planning/pdf_alignment_recovery_plan.md`](planning/pdf_alignment_recovery_plan.md)：PDF 主线回归阶段、保留资产与禁止事项；
- [`planning/hidden_state_plp_v2.md`](planning/hidden_state_plp_v2.md)：PLP v2 实施边界；
- [`planning/research_plan.md`](planning/research_plan.md)：整体研究路线；
- [`planning/v2_experiment_plan.md`](planning/v2_experiment_plan.md)：新 holdout、校准和公平比较计划；
- [`planning/alps_plp_hybrid_v3.md`](planning/alps_plp_hybrid_v3.md)：Hybrid 确认性实验规划；
- [`deployment/autodl_5090.md`](deployment/autodl_5090.md)：AutoDL RTX 5090；
- [`deployment/docker_4090.md`](deployment/docker_4090.md)：学校 RTX 4090 Docker；
- [`deployment/isolated_server.md`](deployment/isolated_server.md)：隔离服务器检查清单。
- [`deployment/alps_plp_hybrid_v3_direct_server.md`](deployment/alps_plp_hybrid_v3_direct_server.md)：Hybrid 服务器逐步运行手册；
- [`deployment/bayesian_sequential_stage3_pilot.md`](deployment/bayesian_sequential_stage3_pilot.md)：Bayesian 第三阶段 9-rollout GPU pilot；
- [`deployment/bayesian_sequential_stage4_full_train.md`](deployment/bayesian_sequential_stage4_full_train.md)：Bayesian 第四阶段 1,620-rollout Train-only 分块采集与归档；
- [`deployment/bayesian_sequential_stage5_oof.md`](deployment/bayesian_sequential_stage5_oof.md)：Bayesian 第五阶段 family-grouped OOF、robustness 和预注册方法选择；
- [`references/大模型输出长度预测.pdf`](references/大模型输出长度预测.pdf)：仓库早期背景版本，保留用于 provenance；
- [`references/大模型输出长度预测_2026-07-09_authoritative.pdf`](references/大模型输出长度预测_2026-07-09_authoritative.pdf)：当前权威项目需求基线。

AutoDL 和学校 Docker 是两条并列部署路径。AutoDL 不替换已有的 `Dockerfile`、
`docker-compose.yml`、`.env` 和 `requirements-docker.lock`。

## 6. 结果与原始证据的边界

`docs/results/` 保存适合论文撰写的冻结解释；`artifacts/runs/` 保存可审计原始证据。报告中的每个
关键数字应能追溯到对应 JSON/CSV、模型注册表和数据摘要，但 Git 不提交大体积 trace 与权重。

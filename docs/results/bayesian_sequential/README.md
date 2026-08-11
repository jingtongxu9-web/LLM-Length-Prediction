# Bayesian Sequential 实验结果

当前尚无 Bayesian Sequential **方法效果结果**。数学合同、第二阶段 CPU 核心，以及第三阶段
真实 Qwen 9-rollout Train-only pilot 均已完成。第四阶段的 1,620-rollout full-Train 配置、
预算、preflight、collector 和验收 schema 已冻结，等待服务器执行。第三阶段与第四阶段采集
都只验证数据与工程合同，不构成 Bayesian 预测效果、OOF 方法选择或最终 Test 结果。

第三阶段可提交证据：

- 人工可读报告：[`stage3_pilot_20260811.md`](stage3_pilot_20260811.md)；
- 脱敏机器摘要：[`stage3_pilot_20260811_summary.json`](stage3_pilot_20260811_summary.json)。

在此之前：

- 数学合同见 [`../../methods/bayesian_sequential_inference.md`](../../methods/bayesian_sequential_inference.md)；
- CPU 核心实现见 [`../../methods/bayesian_sequential_implementation.md`](../../methods/bayesian_sequential_implementation.md)；
- 实施计划见 [`../../planning/pdf_alignment_recovery_plan.md`](../../planning/pdf_alignment_recovery_plan.md)；
- 机器合同见 [`../../../configs/experiments/bayesian_sequential_v1.json`](../../../configs/experiments/bayesian_sequential_v1.json)；
- 旧 ALPS、PLP 和 Hybrid 数字仍位于各自历史结果目录，不迁移、不改写。

禁止在实现前填入模拟或预期结果，也禁止把 concat/residual 的旧指标复制为 Bayesian 结果。

阶段门：真实 RTX 4090/Qwen2.5-7B-Instruct pilot 已通过 9/9 trace、9 个 task-length cell、
terminal/EOS 语义、provenance、显存和本地归档复验。下一步按
[`../../deployment/bayesian_sequential_stage4_full_train.md`](../../deployment/bayesian_sequential_stage4_full_train.md)
采集并本地复验完整 Train trace；在 OOF 方法选择完成前不得访问 final holdout。

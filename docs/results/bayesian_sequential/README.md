# Bayesian Sequential 实验结果

数学合同、第二阶段 CPU 核心、第三阶段真实 Qwen pilot、第四阶段 1,620-rollout full-Train
采集，以及第五阶段 Train-family OOF 均已完成。五折全部通过，预注册 paired family NLL 规则
选择 `bayesian_entropy_scalar_v1`。这是 Train-family OOF 方法选择证据，不是 final Test 结果；
新的 final holdout 尚未创建或访问。

第三阶段可提交证据：

- 人工可读报告：[`stage3_pilot_20260811.md`](stage3_pilot_20260811.md)；
- 脱敏机器摘要：[`stage3_pilot_20260811_summary.json`](stage3_pilot_20260811_summary.json)。

第五阶段可提交证据：

- 人工可读报告：[`stage5_oof_20260812.md`](stage5_oof_20260812.md)；
- 脱敏机器摘要：[`stage5_oof_20260812_summary.json`](stage5_oof_20260812_summary.json)。

在此之前：

- 数学合同见 [`../../methods/bayesian_sequential_inference.md`](../../methods/bayesian_sequential_inference.md)；
- CPU 核心实现见 [`../../methods/bayesian_sequential_implementation.md`](../../methods/bayesian_sequential_implementation.md)；
- 实施计划见 [`../../planning/pdf_alignment_recovery_plan.md`](../../planning/pdf_alignment_recovery_plan.md)；
- 机器合同见 [`../../../configs/experiments/bayesian_sequential_v1.json`](../../../configs/experiments/bayesian_sequential_v1.json)；
- 旧 ALPS、PLP 和 Hybrid 数字仍位于各自历史结果目录，不迁移、不改写。

禁止把 concat/residual 的旧指标复制为 Bayesian 结果，也禁止把第五阶段 OOF 选型写成 final
holdout 泛化结论。

阶段门：第五阶段已覆盖 60 family、1,620 sequence、137,957 个逐步预测点，五折均为 `pass`，
robustness 没有 refit，完整 archive 和内部文件均在本地通过 SHA-256 复验。下一步进入第六阶段
不确定性、收敛与 serving；在第六/七阶段完成并冻结前不得访问 final holdout。

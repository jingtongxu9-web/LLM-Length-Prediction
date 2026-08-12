# Bayesian Sequential 实验结果

当前尚无 Bayesian Sequential **方法效果结果**。数学合同、第二阶段 CPU 核心、第三阶段真实
Qwen 9-rollout Train-only pilot，以及第四阶段 1,620-rollout full-Train 采集和本地逐文件复验
均已完成。第五阶段 Train-family OOF 的配置、数据接入、五折训练、概率评价和预注册选择代码
已经实现，等待完整模型训练。第三与第四阶段采集仍不构成 Bayesian 预测效果或最终 Test 结果。

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

阶段门：第四阶段已通过 1,620/1,620 trace、60 family、9 个 task-length cell、三个 temperature、
三个 seed、terminal/EOS、provenance、archive 和本地文件级 SHA-256 复验。下一步按
[`../../deployment/bayesian_sequential_stage5_oof.md`](../../deployment/bayesian_sequential_stage5_oof.md)
运行五折 OOF；在方法选择完成前不得访问 final holdout。

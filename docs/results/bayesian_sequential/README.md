# Bayesian Sequential 实验结果

数学合同、第二阶段 CPU 核心、第三阶段真实 Qwen pilot、第四阶段 1,620-rollout full-Train
采集、第五阶段 Train-family OOF、第六阶段不确定性/收敛/serving 分析，以及第七阶段
OOF error feedback 均已完成。五折全部通过，预注册 paired family NLL 规则选择
`bayesian_entropy_scalar_v1`。这是 Train-family OOF
诊断证据，不是 final Test 结果；
新的 final holdout 尚未创建或访问。

第三阶段可提交证据：

- 人工可读报告：[`stage3_pilot_20260811.md`](stage3_pilot_20260811.md)；
- 脱敏机器摘要：[`stage3_pilot_20260811_summary.json`](stage3_pilot_20260811_summary.json)。

第五阶段可提交证据：

- 人工可读报告：[`stage5_oof_20260812.md`](stage5_oof_20260812.md)；
- 脱敏机器摘要：[`stage5_oof_20260812_summary.json`](stage5_oof_20260812_summary.json)。

第六阶段可提交证据：

- 人工可读报告：[`stage6_analysis_20260812.md`](stage6_analysis_20260812.md)；
- 脱敏机器摘要：[`stage6_analysis_20260812_summary.json`](stage6_analysis_20260812_summary.json)。

第七阶段可提交证据：

- 人工可读报告：[`stage7_error_feedback_20260812.md`](stage7_error_feedback_20260812.md)；
- 脱敏机器摘要：[`stage7_error_feedback_20260812_summary.json`](stage7_error_feedback_20260812_summary.json)。

在此之前：

- 数学合同见 [`../../methods/bayesian_sequential_inference.md`](../../methods/bayesian_sequential_inference.md)；
- CPU 核心实现见 [`../../methods/bayesian_sequential_implementation.md`](../../methods/bayesian_sequential_implementation.md)；
- 实施计划见 [`../../planning/pdf_alignment_recovery_plan.md`](../../planning/pdf_alignment_recovery_plan.md)；
- 机器合同见 [`../../../configs/experiments/bayesian_sequential_v1.json`](../../../configs/experiments/bayesian_sequential_v1.json)；
- 旧 ALPS、PLP 和 Hybrid 数字仍位于各自历史结果目录，不迁移、不改写。

禁止把 concat/residual 的旧指标复制为 Bayesian 结果，也禁止把第五阶段 OOF 选型写成 final
holdout 泛化结论。

阶段门：第七阶段已在相同的 60 family、1,620 sequence、137,957 个逐步 OOF 点完成错误审计；
没有 model/method reselection 或 threshold tuning，final holdout 未创建或访问。任何修正必须使用
新 method ID 并重跑完整 OOF，否则下一步只能冻结现有方法后进入一次性 final benchmark。

Stage-8A 最终冻结代码现已准备完成，但尚未产生最终模型或 final-holdout 结果。其顺序固定为：
先在 540 条主温度 Train trace 上拟合并复验七方法 checkpoint，再冻结 registry/提交/新 manifest，
最后才允许一次性采集与评价。当前 Stage-8B gate 默认阻塞；执行边界见
[`../../deployment/bayesian_sequential_stage8_final_benchmark.md`](../../deployment/bayesian_sequential_stage8_final_benchmark.md)。

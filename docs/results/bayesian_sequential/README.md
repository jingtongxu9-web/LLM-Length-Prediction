# Bayesian Sequential 实验结果

数学合同、第二阶段 CPU 核心、第三阶段真实 Qwen pilot、第四阶段 1,620-rollout full-Train
采集、第五阶段 Train-family OOF、第六阶段不确定性/收敛/serving 分析、第七阶段 OOF error
feedback 和第八阶段一次性 final benchmark 均已完成。五折预注册规则选择
`bayesian_entropy_scalar_v1`；随后在 12 个全新 family、324 条 trace 上一次性评价，没有进行
final-holdout 重选、调阈值或 refit。

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

第八阶段 Stage-8A 可提交证据：

- 人工可读报告：[`stage8a_final_models_20260816.md`](stage8a_final_models_20260816.md)；
- 脱敏机器摘要：[`stage8a_final_models_20260816_summary.json`](stage8a_final_models_20260816_summary.json)。

第八阶段 Stage-8B 最终可提交证据：

- 人工可读报告：[`stage8_final_benchmark_20260816.md`](stage8_final_benchmark_20260816.md)；
- 脱敏机器摘要：[`stage8_final_benchmark_20260816_summary.json`](stage8_final_benchmark_20260816_summary.json)；
- 七方法总体指标：[`stage8_final_method_metrics_20260816.csv`](stage8_final_method_metrics_20260816.csv)。

在此之前：

- 数学合同见 [`../../methods/bayesian_sequential_inference.md`](../../methods/bayesian_sequential_inference.md)；
- CPU 核心实现见 [`../../methods/bayesian_sequential_implementation.md`](../../methods/bayesian_sequential_implementation.md)；
- 实施计划见 [`../../planning/pdf_alignment_recovery_plan.md`](../../planning/pdf_alignment_recovery_plan.md)；
- 机器合同见 [`../../../configs/experiments/bayesian_sequential_v1.json`](../../../configs/experiments/bayesian_sequential_v1.json)；
- 旧 ALPS、PLP 和 Hybrid 数字仍位于各自历史结果目录，不迁移、不改写。

禁止把 concat/residual 的旧指标复制为 Bayesian 结果，也禁止把第五阶段 OOF 选型写成 final
holdout 泛化结论。

阶段门：第七阶段已在相同的 60 family、1,620 sequence、137,957 个逐步 OOF 点完成错误审计；
Stage-8B 随后按 ready lock 一次性采集和评价。Final holdout 已永久打开，不能再用于任何模型、
方法或阈值选择；若有新理论修正，必须使用新 method ID、全新开发流程和另一批从未打开的
future holdout。

Stage-8A 已在 540 条主温度 Train trace 上完成七方法最终拟合，服务器 CPU 恢复和本地下载
哈希均通过。Stage-8B candidate 包含 12 个全新 family、36 条 Prompt；相对两个历史 manifest
的 396 条记录、72 个 family 完成 exact 与去模板相似度审计，未发现重叠。ready lock 已固定
Stage-8A 配置、合并提交、registry、七个模型、manifest 和审计报告哈希；该锁随后合并到
`main`，服务器完成了 324 条采集与唯一一次 benchmark。审计证据见
[`stage8b_final_holdout_overlap_review.json`](stage8b_final_holdout_overlap_review.json)，执行边界见
[`../../deployment/bayesian_sequential_stage8_final_benchmark.md`](../../deployment/bayesian_sequential_stage8_final_benchmark.md)。

最终结果的核心边界是：Bayesian sequential inference 已真正实现并完成独立验证，但预注册
scalar primary 没有取得总体泛化优势。hidden-delta 的概率 NLL 描述性优于 scalar，concat
baseline 的点误差最低；这些观察不触发 final-holdout 重选。

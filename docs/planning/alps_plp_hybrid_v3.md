# ALPS+PLP Hybrid v3 冻结实验设计

## 研究问题

v3 只回答一个预先声明的问题：在从未用于开发的新 Prompt family 上，`ALPS+PLP Hybrid`
的 progressive remaining-length prediction 是否优于全部冻结对照。PLP v2 已经完成，其
结果不会被覆盖；v3 使用新的 method ID、trace schema、Prompt manifest 和输出目录。

## 为什么仍然需要 v3

v1/v2 的 Test 已在开发过程中打开，不能再承担确认性结论。v2 还是 PLP-only，无法直接回答
“ALPS prior 与 PLP hidden state 组合后是否最好”。v3 因此同时解决四件事：

1. 新增 12 个从未打开的 family holdout；
2. 所有方法使用同一次 Qwen generation 和完全相同的保存点；
3. 训练阶段以 family 为 group 做 OOF，Hybrid 的 ALPS 输入再做内层 grouped cross-fit；
4. 在一次性 Test 上进行家族级配对 bootstrap 和七重比较校正。

## 数据与目标

- Train/design：旧 60 family，180 prompts，3 seeds，共 540 rollouts；
- Test/confirmatory：新 12 family，36 prompts，3 seeds，共 108 rollouts；
- Qwen/tokenizer revision：`a09a35458c702b33eeacc393d103063234e8bc28`；
- ALPS target：`log1p(output_tokens)`；
- ALPS 输出：残差方差与 shifted log-normal prior；
- progressive target：每个保存点的 `remaining_tokens`；
- terminal：专用 zero bin，不把终点强行映射到正长度箱。

## 八种冻结方法

1. `step_only_ridge`：只用生成步数；
2. `alps_countdown`：静态 ALPS 总长度减当前步；
3. `dynamic_ridge`：五个动态标量的 Ridge；
4. `dynamic_signal_mlp_v1`：已经定义的项目版动态 MLP；
5. `plp_v2_frozen`：在共享 v3 trajectory 上原样运行 v2 算法；
6. `plp_small_terminal_v3`：小容量、正确 terminal 的消融；
7. `alps_dynamic_ridge`：prior summaries 加五个动态标量；
8. `alps_plp_hybrid_v3`：prior summaries 加 Prompt/decode hidden state 的主方法。

## 泄漏控制

外层 5-fold OOF 的 group 是 `prompt_family_id`，同一 family 的三种长度、三个 seed 和所有
时间点永远在同一折。对 Hybrid 和 `alps_dynamic_ridge`，每个外层训练折中的 ALPS summary
来自额外的 4-fold family-grouped cross-fit；外层 validation 的 prior 只能由外层 Train
拟合。最终全 Train 模型也用 cross-fitted prior summaries 训练 stacking head，再另行拟合
全 Train ALPS 供 Test 推断。

## 删失与统计规则

`stop_reason == max_new_tokens` 是右删失，主分析完整排除，不把未知 remaining 当作 0。删失率
达到 5% 发出 warning，达到 10% 终止实验。主指标是 family-macro、sequence-balanced MAE。
置信区间以 family 为 bootstrap 单位，固定 2000 次。主方法与七个对照的配对差为
`Hybrid MAE - comparator MAE`；七个 Bonferroni 置信区间的上界都小于 0，才支持预注册的
“预测效果优于全部对照”结论。

OOF 是开发阶段稳定性证据，不是最终 claim。最终 Test 只打开一次；预测 superiority 与
serving superiority 分开表述。后者还必须通过冻结的离线 serving replay，且不得称为真实
生产系统测量。

## 冻结文件

- `configs/experiments/alps_plp_hybrid_v3_base.json`
- `configs/experiments/alps_plp_hybrid_v3.json`
- `configs/experiments/alps_plp_hybrid_v3_protocol.json`
- `configs/reports/alps_plp_hybrid_v3_report_schema.json`
- `data/prompts/alps_plp_hybrid_v3_prompts.jsonl`

服务器运行顺序见
[`../deployment/alps_plp_hybrid_v3_direct_server.md`](../deployment/alps_plp_hybrid_v3_direct_server.md)。

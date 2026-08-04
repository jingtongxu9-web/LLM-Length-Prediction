# v2 实验计划

ALPS概率区间的专项实施步骤见
[`alps_improvement_plan.md`](alps_improvement_plan.md)。

## 1. 版本边界

ALPS v1 的 Final Test 已经打开，只能作为历史结果保留。v2 不是继续在同一 Test 上调整
Layer、Ridge alpha、MLP 结构或区间宽度，而是使用新的 family-level holdout 完成确认性
评价。

## 2. 研究目标

1. 在新 Prompt family 上确认固定 Layer 14、`Ridge(alpha=1.0)` 的点预测能力；
2. 使用 Train 内 family-grouped OOF residual 建立独立概率校准；
3. 在相同 timestep 评价 ALPS-minus-step、Dynamic Ridge、Dynamic-Signal MLP 和
   ALPS+dynamic hybrid；
4. 使用已经实现的 Hidden-State PLP v2 单独采集生成 token hidden state，并使用论文的
   length-bin 输出头，不与 Dynamic-Signal MLP v1 混名。

Hidden-State PLP v2 的机器可读合同是
[`../../configs/experiments/plp_v2_manifest.json`](../../configs/experiments/plp_v2_manifest.json)。
其采集、训练和评估脚本已经完成，当前状态是“待 GPU pilot/完整运行”，不是“已有结果”。

## 3. 数据与评价顺序

所有 Prompt 变体和 rollout 必须按 `prompt_family_id` 保持在同一 split，防止语义泄漏。
正式采集前冻结 family 数量、Prompt schema、模型 revision、tokenizer、chat template、解码
参数和 seeds。

```text
Design/Train
    -> family-grouped CV 与方法诊断
    -> 冻结点预测模型和校准方法
    -> 全 Train 一次性训练最终模型
    -> Final Test 只打开一次
```

如果需要独立 Calibration split，应在采集前与 Train/Test 一起冻结；不得看到 Test 后再从
Test 中切出校准数据。

## 4. 本地与 GPU 分工

本地可以完成：

- Prompt schema、family split 和 manifest 检查；
- 单元测试、合成数据测试和评价脚本；
- grouped-CV、conformal calibration 和报告生成代码；
- v1 trace 上的探索性诊断，但不能把它写成新的独立 Test 结论。

GPU 服务器负责：

- 按冻结 manifest 采集新的 Train/Calibration/Test trace；
- 为 Hidden-State PLP v2 保存所需的 Prompt pooled state 与生成期 hidden state；
- 最终配置冻结后执行一次 Final Test；
- 保存环境报告、日志、模型和结果校验和。

## 5. 进入下一阶段的条件

只有同时满足以下条件，才进入 serving benchmark：

- ALPS 在 family-grouped OOF 和新 Final Test 上稳定优于 Prompt-token 与 Metadata baseline；
- 点预测误差和长输出低估风险得到分组报告；
- 95%区间的 Coverage 与宽度达到可解释的平衡；
- 动态方法在同一 remaining-token 目标、同一 timestep 权重下完成公平比较；
- Test 没有被用于反复选择模型或校准参数。

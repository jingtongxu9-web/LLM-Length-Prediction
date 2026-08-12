# PDF 主线回归实施计划

## 1. 目标

本计划把项目主线从“ALPS + Hidden-State PLP 的判别式融合”纠正为参考 PDF 要求的：

```text
ALPS 静态概率 prior
        +
非重叠 decode evidence
        ↓
Bayesian sequential update
        ↓
动态剩余长度 posterior 与 uncertainty cone
```

现有实验不删除、不改写历史结论，但重新定位为 baseline、消融或历史证据。正式数学定义见
[`../methods/bayesian_sequential_inference.md`](../methods/bayesian_sequential_inference.md)。

## 2. 现有资产如何处理

| 资产 | 决定 | 理由 |
|---|---|---|
| ALPS Layer-14 Ridge | 保留 | 与 PDF 的 pre-planning 主体一致 |
| ALPS in-sample residual variance | 替换 | 未见 family 区间明显欠校准 |
| Prompt/family/seed split | 保留 | 已有分组防泄漏基础 |
| Hugging Face hooks | 保留并扩展 | 需要新增非重叠 evidence block 原始量 |
| Dynamic-Signal MLP v1 | 保留为 scalar dynamic baseline | 特征方向接近 PDF，但不读取 prior、也不做 Bayes update |
| Hidden-State PLP v2/v3 | 保留为 hidden-state baseline | 是有效对照，但不是 PDF 核心方法 |
| concat v1 | 保留为 discriminative-fusion baseline | 已有强 OOF 结果，但不是 posterior update |
| residual/gated residual | 保留为负向结构消融 | 证明点残差修正不等于可靠动态推断 |
| serving replay | 保留 | 可直接消费新 posterior point/interval 输出 |
| 已打开 Test | 历史证据 only | 不再用于特征、loss 或结构选择 |

## 3. 阶段一：数学和实验合同

状态：**已完成并确认（2026-08-11）。**

- [x] 保存用户指定的权威 PDF 副本与 SHA-256；
- [x] 冻结 latent state `R_t = L - t`；
- [x] 冻结 ALPS shifted-lognormal prior 与 OOF 方差校准来源；
- [x] 冻结 stride transition；
- [x] 冻结 non-overlapping evidence unit；
- [x] 冻结 log-space posterior update；
- [x] 解决 PDF 中未来 hazard 不可观测的问题；
- [x] 冻结 censored rollout 口径；
- [x] 冻结 baseline、主指标、收敛指标和选择规则；
- [x] 新增机器可读 JSON 合同；
- [x] 新增合同自动测试；
- [x] 导师/项目负责人已于 2026-08-11 确认合同，状态已改为
  `phase1_approved_for_implementation`。此状态变化只表示批准实现，不允许改变科学内容。

阶段门：在合同确认前，不采集新的 Bayesian full Train 或 final holdout。

## 4. 阶段二：无 GPU 的核心实现

状态：**已完成（2026-08-11）。**

新增模块：

```text
src/llm_length_prediction/data/sequential.py
src/llm_length_prediction/models/bayesian_filter.py
src/llm_length_prediction/models/hazard.py
src/llm_length_prediction/evaluation/sequential.py
```

最低实现顺序：

1. [x] shifted-lognormal integer mass 与 overflow；
2. [x] posterior shift/condition transition；
3. [x] log-space likelihood update；
4. [x] posterior summaries、credible intervals 和 derived hazard；
5. [x] synthetic sequence builder；
6. [x] right-censoring likelihood；
7. [x] scalar evidence scorer；
8. [x] hidden-delta scorer；
9. [x] rollout-balanced sequence NLL；
10. [x] checkpoint schema 与数值稳定性测试。

阶段门：已通过 36 个专项测试，覆盖 prior、transition、evidence、posterior、censoring、scorer、
checkpoint 和 metrics。下一阶段是统一 collector pilot；本阶段测试不构成真实模型效果证据。

## 5. 阶段三：统一 trace collector 与小型 pilot

状态：**已于 2026-08-11 完成真实 RTX 4090/Qwen 9-rollout pilot，并通过本地复验。**

- [x] 新建独立、pickle-free 的 unified trace v1；
- [x] 保存 Layer-14、Prompt pooled、initial decode 与更新点 decode state；
- [x] 保存逐 token ID、entropy 和 EOS probability；
- [x] 冻结 3 task × 3 length、Train-only 的 9-rollout pilot；
- [x] 实现原子写入、严格校验、断点续跑、index 和 acceptance report；
- [x] 实现 CUDA/BF16/model revision/disk preflight；
- [x] fake causal-LM 本地端到端测试；
- [x] 服务器执行 1-rollout smoke；
- [x] 服务器执行 3-rollout task smoke；
- [x] 服务器补齐 9 rollout 并通过 acceptance；
- [x] 人工抽查 short/medium/long 的 terminal、censoring、显存、磁盘与耗时。

Collector 必须一次保存所有 frozen methods 所需信息：

- Layer-14 `h_0`；
- Prompt final-layer pooled state；
- 每个更新点的 decode hidden state；
- 非重叠 block 的逐 token entropy/EOS 原始量或无损聚合；
- token IDs、step、temperature、seed、stop reason；
- terminal point；
- censored 标记与完整 provenance。

Pilot 覆盖 QA、Summarization、Code 与 Short、Medium、Long，先确认字段、对齐、磁盘、耗时、
断点续跑和 terminal semantics，再允许全量 Train 采集。

服务器命令和带回产物见
[`../deployment/bayesian_sequential_stage3_pilot.md`](../deployment/bayesian_sequential_stage3_pilot.md)。
已完成的工程验收结果见
[`../results/bayesian_sequential/stage3_pilot_20260811.md`](../results/bayesian_sequential/stage3_pilot_20260811.md)。

## 6. 阶段四：一次性完整 Train 采集

状态：**已完成（2026-08-12）。**

- [x] 冻结 180 Train Prompt、60 family、3 temperature、3 seed，共 1,620 rollout；
- [x] 冻结 Stage 3 实测外推的 GPU 时间、磁盘和显存预算；
- [x] 实现 100-job 默认分块、原子 NPZ、严格校验和断点续跑；
- [x] 实现完整覆盖、censoring、CUDA provenance 和 final-holdout 边界验收；
- [x] 实现 4090/BF16/revision/动态剩余磁盘 preflight；
- [x] 冻结机器可读 full-Train report schema；
- [x] 在服务器通过 full-Train preflight；
- [x] 采集并复验全部 1,620 trace；
- [x] 将 archive 下载到本地并完成逐 trace SHA-256 二次复验。

主开发 temperature 为 `0.7`；robustness temperature 为 `0.3/1.0`。同一 family 的全部
temperature、seed、长度版本和 timestep 保持在同一 fold。

Full Train trace 冻结后，各方法只做离线训练和评价，不为每个模型重新运行 Qwen。
服务器手册见
[`../deployment/bayesian_sequential_stage4_full_train.md`](../deployment/bayesian_sequential_stage4_full_train.md)。

## 7. 阶段五：Train-family OOF 与方法选择

状态：**代码、真实数据 preflight 和五折合同已实现；等待完整模型训练。**

统一比较：

1. Prompt-token Ridge countdown；
2. ALPS countdown；
3. Dynamic-Signal MLP v1；
4. Hidden-State PLP terminal-zero v3；
5. concat v1；
6. Bayesian scalar；
7. Bayesian hidden-delta。

所有 scaler、ALPS prior、variance calibration 和 neural model 都在折内拟合。按预注册 paired
NLL 规则选择 Bayesian scalar 或 hidden-delta，不允许在同一 OOF 上继续增加候选。

冻结实现见
[`../../configs/experiments/bayesian_sequential_stage5_oof_v1.json`](../../configs/experiments/bayesian_sequential_stage5_oof_v1.json)，
运行手册见
[`../deployment/bayesian_sequential_stage5_oof.md`](../deployment/bayesian_sequential_stage5_oof.md)。
主训练严格只使用 `temperature=0.7`；`0.3/1.0` 只由每折冻结模型评价，禁止 robustness refit。

## 8. 阶段六：不确定性、收敛与 serving

生成：

- uncertainty cone；
- posterior variance/entropy curve；
- NLL、CRPS、coverage/width；
- stable time-to-5%-accuracy；
- long-tail underestimation；
- online update overhead；
- batching/KV-cache serving replay。

方差下降不是单独成功标准，必须与 coverage 一起解释。

## 9. 阶段七：OOF error feedback

只在 Train-family OOF 上分析绝对误差大于 100 token 和最差 5%。分类 entropy rebound、
oscillation、open-endedness、sampling divergence、repetition、hallucination、early stop 和
posterior failure。任何理论修正形成新的 method ID，并重新执行完整 OOF；不得查看 final
holdout 后再修补。

## 10. 阶段八：一次性 final benchmark

在以下内容全部冻结后才创建并打开新 family holdout：

- git commit；
- JSON 配置及 SHA-256；
- Train dataset digest；
- checkpoints 及 SHA-256；
- 指标与报告 schema；
- comparison list；
- serving replay 规则；
- statistical tests。

最终运行一次性生成所有冻结方法的预测、概率指标、点指标、uncertainty cone、收敛速度、
robustness 和 serving replay。Final holdout 不选择任何模型或阈值。

## 11. 明确禁止事项

- 不删除或改写旧实验结果；
- 不把 concat/residual 称为 Bayesian posterior update；
- 不把 PLP softmax probabilities 未经 prior 更新就称为 posterior；
- 不在递归更新中重复乘入包含完整历史的 causal state；
- 不把 censored rollout 当 terminal zero；
- 不用旧 Test 校准 ALPS variance；
- 不用新 final holdout 调特征、epoch、loss、temperature 或报告口径；
- 不为了得到“锥形收敛”而人工压缩 posterior variance。

# 四方法主比较：Prompt-token Baseline、ALPS、PLP 与 Hybrid

本文是跨方法的论文主比较报告，不是单独的 Hybrid 报告。各方法自身的完整结果分别位于：

- [`../baseline/prompt_token_ridge_results.md`](../baseline/prompt_token_ridge_results.md)；
- [`../alps/alps_v1_results.md`](../alps/alps_v1_results.md)；
- [`../plp/terminal_zero_v3_results.md`](../plp/terminal_zero_v3_results.md)；
- [`../hybrid/hybrid_v1_v2_v2_1_results.md`](../hybrid/hybrid_v1_v2_v2_1_results.md)。

对应技术原理统一由 [`../../methods/README.md`](../../methods/README.md) 导航。

## 1. 当前结论与证据边界

本项目最终主比较固定为四种技术路线：

1. `prompt_token_ridge_countdown`：只用输入 token 数；
2. `alps_countdown`：只用生成前的 Layer-14 语义先验；
3. `plp_terminal_zero_v3`：只用生成中的 Prompt/decode hidden states；
4. `alps_plp_concat_v1`：联合使用 ALPS 与 PLP。

截至当前，后三种方法已经在相同的 60 个 Train family、540 条 rollout、45,119 个逐步
预测点上完成五折 family-grouped OOF。`concat v1` 的 OOF MAE 最低，并已在全部 Train 上
训练和冻结。输入-token baseline 历史上已运行，但旧结果属于早期静态 ALPS 数据，不能直接
放进当前动态排行榜；同口径的补充 OOF 已冻结协议，尚待在已有 trace 上执行。最终确认仍需
一批从未用于 PLP Test 或方法选择的新 Hybrid holdout。

## 2. 四个主方法分别在研究什么

| 方法 | 输入 | 训练目标 | 生成时输出 | 科学问题 |
|---|---|---|---|---|
| Prompt-token Ridge | 格式化 Prompt 的 token 数，1维 | `log1p(T)` | `max(T_hat-step,0)` | 单纯“输入更长所以输出更长”能解释多少？ |
| ALPS | Prompt 最后 token 的 Layer-14 hidden state，3584维 | `log1p(T)` | ALPS countdown | 生成前语义表示是否包含输出规划信息？ |
| PLP terminal-zero v3 | entropy-pooled Prompt state + 当前 decode state，7168维 | 剩余长度20-bin分布 | 每个采样 step 的剩余长度 | 生成路径的实时状态能解释多少？ |
| Hybrid concat v1 | PLP 7168维 + 标准化 ALPS 5维摘要，共7173维 | 剩余长度20-bin分布 | 每个采样 step 的剩余长度 | 全局路线与实时状态能否互补？ |

其中 `T` 是最终输出 token 总数。Prompt-token Ridge 和 ALPS 都先预测一次总长度；为了与
PLP/Hybrid 在同一个 timestep 公平比较，评价时都减去当前 `step` 并截断到非负值。

### 2.1 为什么必须保留 Prompt-token baseline

这是最小信息基线。若它已经接近 ALPS 或 Hybrid，模型可能只是在利用输入长度，而不是语义
规划或生成状态。历史静态实验中，Prompt-token Ridge 的 family-grouped OOF MAE 为
`260.60`、Log R² 为 `0.018`，说明输入长度本身解释力很弱；但该数字不能与当前逐步 OOF
直接比较，因此需要新的 `prompt_token_ridge_countdown` 补充分析。

### 2.2 ALPS 的技术方案

ALPS 从 Qwen 的零基 Layer 14 取 Prompt 最后一个 token 的 3584维 hidden state，用
`alpha=1.0` 的 Ridge 拟合 `log1p(T)`。在某个生成 step，ALPS 提供五个摘要：

```text
mu_log_total
residual_variance
mean_total_tokens
mean_remaining_countdown
median_remaining_countdown
```

它的优势是生成前即可得到全局路线；局限是生成开始后不会根据实际回答路径主动更新。

### 2.3 PLP terminal-zero v3 的技术方案

PLP 为每条 Prompt 形成一个3584维 entropy-pooled Prompt state，并在每个保存 step 读取
当前 token 的3584维 causal decode state。二者拼成7168维，交给20-bin progressive MLP
预测剩余长度分布。v3 在 v2 的正长度 bins 外增加 terminal zero 机制，解决真实剩余长度为0
时仍被迫预测正数的问题。

### 2.4 Hybrid concat v1 的技术方案

concat 是 concatenation，即向量拼接：

```text
[PLP prompt/decode state: 7168维 | ALPS summaries: 5维]
                            ↓
                       7173维输入
                            ↓
                  progressive MLP直接预测剩余长度
```

ALPS 不是不可修改的起点，而是五个可被 MLP 使用、变换或忽略的输入特征。这个自由度使模型
能够在生成早期依赖全局先验，在生成后期更多使用 decode state。

## 3. 当前同口径 OOF 结果

| 方法 | Family-macro sequence-balanced MAE | RMSE | Bias | R² |
|---|---:|---:|---:|---:|
| Prompt-token Ridge countdown | 待补充同口径 OOF | — | — | — |
| ALPS countdown | 55.724 | 83.922 | +0.411 | 0.846 |
| PLP terminal-zero v3 | 59.778 | 95.810 | -6.397 | 0.799 |
| **Hybrid concat v1** | **49.916** | **77.565** | -5.398 | **0.868** |

concat v1 相对 ALPS 的 MAE 降低 `5.808` token，约 `10.4%`；相对 PLP 降低 `9.862`
token，约 `16.5%`。因此当前证据支持“ALPS 与 PLP 的信息具有互补性”。这仍是 Train 内部
family-grouped OOF 选择结论，不是独立 Test 的最终声明。

## 4. 为什么还实验了 residual v2

concat v1 回答的是“自由联合是否有效”，但不能检验更符合直觉的串行设计：

```text
先由 ALPS 预测全局剩余长度
再由 PLP 根据实时生成状态纠正 ALPS
```

因此 residual v2 定义为：

```text
R_hat_t = ALPS_countdown_t + Delta_PLP_t
```

它使用 PLP 7168维状态以及 ALPS countdown、step、entropy、entropy mean/slope、EOS
probability 六个控制量，共7174维，训练 MLP 预测有符号残差。该实验有明确理论作用：判断
PLP 更适合“重算剩余长度”还是“只修正静态先验”。

## 5. v2 与 gated v2.1 的结果

| 方法 | OOF MAE | 相对 concat v1 | 结论 |
|---|---:|---:|---|
| concat v1 | **49.916** | — | 选中 |
| residual v2 | 57.934 | +8.018 | 淘汰 |
| gated residual v2.1 | 56.067 | +6.151 | 部分修复 v2，仍淘汰 |

v2 的平均预测修正为 `+5.75` token，而 ALPS 实际平均只需约 `-0.41` token，说明直接残差
网络产生了方向偏差。v2.1 增加：

```text
bounded_delta = B * tanh(raw_delta)
gate = progress * sigmoid(gate_logit)
prediction = max(0, ALPS_countdown + gate * bounded_delta)
```

它将 MAE 改善约 `1.87` token。v2.1-minus-v2 的普通95%配对区间为
`[-3.71,-0.09]`，支持小幅数值改善；预先采用的严格99% familywise CI 为
`[-4.20,0.49]`，仍跨过0，不能声称在多重比较后稳定优于v2。

更关键的是，v2.1 相对 concat v1 的严格 CI 为 `[+0.47,+13.01]`，明确更差。learned gate
confidence 中位数达到 `0.989`，但修正成功率只有 `49.5%`，gate 与 MAE 改善的相关性接近
0。说明 gate 大多处于打开状态，却没有学会可靠识别“什么时候应该修改 ALPS”。

这个负向结果提供了重要解释：当前数据支持 ALPS 与 PLP 联合建模，但不支持把 PLP 限制为
ALPS 的串行残差控制器。问题不是两类特征无效，而是融合形式不合适。

## 6. PLP 内部消融为何不进入四方法主表

PLP v3 选择前还比较了三个单因素消融：

| PLP方案 | OOF MAE | 作用 |
|---|---:|---|
| v2 frozen | 61.037 | Hidden-State PLP 控制组 |
| terminal-zero v3 | **59.778** | 增加终点0机制，选中 |
| 512 small head | 70.965 | 检验缩小 MLP，明显退化 |
| rollout-balanced target range | 60.816 | 检验加权区间，改善不稳定 |

这些实验用于选择“纯 PLP”代表版本；最终主表只保留被选中的 terminal-zero v3，避免把同一
方法家族的调试版本与四条核心技术路线混在一起。

## 7. 已冻结的全量模型

全部模型使用60个 Train family、540条完整 rollout，截断率为0。注册表验证结果：

| 方法 | 文件 | SHA-256 |
|---|---|---|
| ALPS | `alps_prior.json` | `7c446512e4c83d9e4332c8cfad50a148bbf0b505f271966ee5ece778ae5d2828` |
| PLP v3 | `plp_terminal_zero_v3.pt` | `8e460067150a0507389c1afedce1178d6445c4e43604598896d10d146acc2708` |
| concat v1 | `alps_plp_concat_v1.pt` | `f22bb194e7c827a3c657f82b17eb33adcdeaa37359ed0235163def93b5fdeae3` |
| residual v2 | `alps_plp_residual_v2.pt` | `2412636efa42c019f9ee8f54b846ebd5b639f3e9f930b7ca56ae48e61081d660` |

concat v1 是当前正式 Hybrid 候选。v2/v2.1 作为方法消融冻结，不再继续使用当前 OOF 调参。

## 8. 剩余工作

1. 在现有 Train trace 上运行同 family folds 的 Prompt-token Ridge countdown OOF；
2. 在全部540条 Train rollout 上冻结该 baseline；
3. 用四方法报告替换跨数据阶段的非公平排行榜；
4. 创建新的、未被 PLP-only Test 使用的 Hybrid holdout；
5. 一次性比较 Prompt-token baseline、ALPS、PLP v3 和 concat v1；
6. 只有新 holdout 才用于最终确认性结论。

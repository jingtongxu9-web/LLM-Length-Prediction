# Baseline、ALPS 与 PLP v3 综合对比

本报告汇总项目当前已经完成的长度预测方法，回答三个问题：

1. 只使用简单输入信息的 baseline 能做到什么程度？
2. ALPS 在生成前提供了什么增量能力？
3. PLP terminal-zero v3 在生成过程中提供了什么增量能力？

ALPS 和 PLP 不是相互替代的两个同目标模型。ALPS 在生成前预测最终总长度，PLP 在生成
过程中持续预测当前剩余长度。因此本报告先在各自公平口径内比较，再讨论为什么下一步需要
ALPS+PLP。

详细结果见：

- ALPS 与静态 baseline：[`stage1_alps_baselines_dynamic.md`](stage1_alps_baselines_dynamic.md)；
- ALPS 完整报告：[`../alps/alps_v1_results.md`](../alps/alps_v1_results.md)；
- Hidden-State PLP v2：[`../plp/hidden_state_plp_v2_results.md`](../plp/hidden_state_plp_v2_results.md)；
- PLP 三消融与 terminal-zero v3：
  [`../plp/terminal_zero_v3_results.md`](../plp/terminal_zero_v3_results.md)。

## 0. 核心结论

1. **简单输入长度 baseline 很弱。**Prompt token Ridge 的 family-grouped OOF MAE 为
   `260.60`，Log R² 仅为 `0.018`；输入有多长无法可靠推断输出会有多长。
2. **ALPS 是有效的生成前静态预测器。**Layer-14 Ridge 的 OOF MAE 为 `60.87`、Log R² 为
   `0.953`，相比 Prompt token baseline 将 MAE 降低约 `76.6%`。
3. **ALPS 的主要问题不是点预测，而是概率校准。**名义 95% 区间在 OOF 只覆盖 `71.1%`，
   在旧 Final Test 只覆盖 `63.9%`，说明区间过窄、模型过度自信。
4. **旧五标量动态 baseline 不足。**Dynamic-Signal MLP v1 的 Test sequence-balanced MAE
   为 `136.66`、Raw R² 为 `0.089`，早期严重低估、后期严重高估。
5. **Hidden-State PLP 显著强于旧动态 baseline。**PLP v2 在旧开发性 Test 上的 MAE 为
   `60.03`、Raw R² 为 `0.790`，证明 Prompt/decode hidden states 包含关键动态长度信息。
6. **PLP terminal-zero v3 对 v2 有小幅总体改善。**在新的 12-family Test 上，MAE 从
   `75.06` 降至 `71.04`，相对改善约 `5.35%`；终点 MAE 从 `23.11` 降至 `1.50`。
7. **PLP v3 的严格显著性声明未通过。**配对 95% CI 为 `[-9.03,0.57]`，略微跨 0；QA
   明显改善，但 Code 退化。
8. **下一步结合具有明确动机。**ALPS 提供生成前的全局长度路线，PLP 提供生成中的实时
   纠偏。二者需要在同一 trace、同一 timestep 和新 holdout 上比较，不能直接用现有静态
   MAE 与动态 MAE 排名。

## 1. 三类方法分别研究什么

| 方法 | 预测时机 | 预测目标 | 核心输入 | 每条 rollout 预测次数 |
|---|---|---|---|---:|
| 静态 baseline | 生成前 | 最终输出总长度 `T` | 全局均值、Prompt token 数或元数据 | 1 |
| ALPS | 生成前 | 最终输出总长度 `T` | Prompt 最后一个 token 的 Layer-14 hidden state | 1 |
| 动态 baseline | 生成中 | 剩余长度 `R_t=T-t` | step、entropy、entropy mean/slope、EOS probability | 多次 |
| PLP v3 | 生成中 | 剩余长度 `R_t=T-t` | Prompt pooled hidden state + 当前 decode hidden state | 多次 |

最重要的比较边界是：

```text
ALPS MAE 约 61  ≠  PLP v3 MAE 约 71 的直接胜负
```

ALPS 的一个样本是一条 rollout 的最终总长度；PLP 的一个样本是 rollout 中某个 timestep 的
剩余长度。目标分布、样本数量和权重均不同，因此两者的数字不能直接作为排行榜。

## 2. 静态方法：Baseline 与 ALPS 的公平比较

以下为同一批 60 个 design family 上的 family-grouped 五折 OOF rollout-level 结果。所有
方法都在生成前预测最终总长度，因此可以直接比较。

| 方法 | MAE | RMSE | Raw R² | Log R² | 95% Coverage | 区间均宽 |
|---|---:|---:|---:|---:|---:|---:|
| Global mean | 284.12 | 324.78 | -0.091 | -0.002 | 98.4% | 2514.11 |
| Prompt tokens Ridge | 260.60 | 313.97 | -0.020 | 0.018 | 96.5% | 2486.80 |
| Metadata | 92.95 | 140.88 | 0.795 | 0.925 | 94.7% | 547.68 |
| Metadata + Prompt tokens | 92.94 | 140.08 | 0.797 | 0.925 | 94.9% | 545.99 |
| **ALPS Layer 14** | **60.87** | **91.31** | **0.914** | **0.953** | 71.1% | **180.81** |

### 2.1 Prompt token baseline

Prompt token 数几乎没有解释能力。其 Log R² 只有 `0.018`，与全局均值接近。虽然 Coverage
达到 96.5%，但平均区间宽度接近 2487 token，主要是依靠非常宽的区间覆盖真实结果，而不是
点预测准确。

该结果排除了一个简单解释：ALPS 的优势不是因为它间接“数出了 Prompt 有多少 token”。

### 2.2 Metadata baseline

Metadata 使用任务类型和预设 short/medium/long 条件，MAE 降到约 `92.95`。这说明当前
Prompt 设计中的组间差异本身能够解释相当一部分输出长度。

但 Metadata 加入 Prompt token 数后几乎没有变化；而 ALPS 又将 MAE 从约 93 降到 61，说明
Layer-14 hidden state 还捕获了具体内容、语义结构和回答规划信息，而不仅是类别标签。

### 2.3 ALPS

相对 Prompt token baseline，ALPS：

- MAE 降低约 **76.6%**；
- RMSE 降低约 **70.9%**；
- Log R² 从 `0.018` 提高到 `0.953`。

ALPS 的生成前点预测能力已经得到 family-grouped OOF 和旧 Final Test 支持：

| 评价位置 | MAE | RMSE | Log R² | 95% Coverage |
|---|---:|---:|---:|---:|
| Train | 32.19 | 48.74 | 0.991 | 94.7% |
| Family-grouped OOF | 60.87 | 91.31 | 0.953 | 71.1% |
| 旧 Final Test | 66.97 | 97.11 | 0.929 | 63.9% |

OOF 与 Test 的点预测接近，说明 ALPS 没有在未见 family 上失效。当前最明显的问题是概率
区间：点预测越准确，模型给出的区间反而越窄，导致名义 95% Coverage 只有约 64%–71%。

## 3. 动态方法：旧 Baseline 与 Hidden-State PLP

动态方法在生成过程中每隔若干 token 预测 `remaining_tokens`。以下历史结果用于说明特征
升级的作用；v1/v2 使用旧开发性 Test，v3 使用后来一次性打开的新 Test，因此不能把所有
数值视为完全相同数据上的确认性排名。

| 方法 | Test 数据身份 | Sequence-balanced MAE | RMSE | Raw R² | Bias |
|---|---|---:|---:|---:|---:|
| Dynamic-Signal MLP v1 | 旧开发性 Test | 136.66 | 190.19 | 0.089 | +38.20 |
| Hidden-State PLP v2 | 旧开发性 Test | **60.03** | **91.60** | **0.790** | +12.65 |
| PLP v2 frozen control | 新 12-family Test | 75.06 | 120.17 | 0.761 | -20.69 |
| **PLP terminal-zero v3** | 新 12-family Test | **71.04** | **115.26** | **0.780** | **-10.28** |

### 3.1 Dynamic-Signal MLP v1

旧动态 baseline 只使用五个标量：

```text
step, entropy, entropy_mean, entropy_slope, eos_probability
```

它不读取 Prompt 内容或 hidden state，因而无法在生成早期区分任务本身可能需要几十、几百
还是上千 token。其 Test MAE 为 `136.66`，Raw R² 只有 `0.089`：

- 0–10% 平均低估约 212 token；
- 25–50% 是唯一相对有效的阶段；
- 75–100% 平均高估约 145 token。

这说明“只观察生成进度和概率不确定性”不足以可靠预测剩余长度。

### 3.2 Hidden-State PLP v2

PLP v2 将输入升级为：

```text
3584 维 entropy-pooled Prompt state
+
3584 维当前 decode causal state
=
7168 维 PLP 输入
```

相对 Dynamic-Signal MLP v1，旧 Test MAE 从 `136.66` 降至 `60.03`，下降约 `56.1%`；Raw
R² 从 `0.089` 提高到 `0.790`。这证明性能提升主要来自语义和生成路径 hidden states，而
不是简单更换 MLP。

PLP v2 的主要问题是临近 EOS 仍存在预测下限：20 个正长度 bins 没有精确的 0 类别，真实
剩余长度为 0 时仍会预测约 23 token。

### 3.3 PLP terminal-zero v3

PLP v3 以 v2 为 baseline 完成三个单因素消融：

| 消融 | OOF MAE | 相对 v2 | 结论 |
|---|---:|---:|---|
| Terminal zero bin | **59.78** | **-1.26** | 保留 |
| 512 small head | 70.96 | +9.93 | 淘汰 |
| Rollout-balanced target range | 60.82 | -0.22 | 区间跨 0，淘汰 |

最终 Test 只比较预先选定的 terminal-zero v3 与 v2 frozen control：

| 指标 | PLP v2 control | Terminal-zero v3 | 变化 |
|---|---:|---:|---:|
| Family-macro MAE | 75.06 | **71.04** | **-4.01** |
| RMSE | 120.17 | **115.26** | -4.91 |
| Raw R² | 0.761 | **0.780** | +0.019 |
| Bias | -20.69 | **-10.28** | 低估减轻 10.41 |
| Terminal MAE | 23.11 | **1.50** | **-21.61** |
| Nonterminal MAE | 75.26 | **72.03** | -3.23 |

MAE 点估计相对改善约 `5.35%`。配对 95% CI 为 `[-9.03,0.57]`，略微跨 0，因此结果应写为
“观察到小幅整体改善并明确修复终点”，不能写为“已显著优于 v2”。

任务差异仍然明显：QA MAE 改善约 14.99 token，Summarization 基本持平，Code 退化约
3.37 token。PLP v3 改善了输出头的终点结构，但没有解决所有任务的 family-level 泛化。

## 4. ALPS countdown 与 PLP v3 的同 Trace 公平比较

v3 的 60-family OOF 已经在相同生成输出、相同 seed 和相同 timestep 上保存了 ALPS 与 PLP
预测。因此可以把 ALPS 的总长度预测转换为动态 countdown，再与 PLP v3 直接比较：

\[
\widehat{R}^{ALPS}_t=\max(0,\widehat{T}_{ALPS}-t)
\]

### 4.1 总体结果

| 方法 | Family-macro MAE | RMSE | Raw R² | Bias |
|---|---:|---:|---:|---:|
| Step-only Ridge | 235.48 | 276.04 | -0.669 | +132.02 |
| **ALPS countdown** | **55.72** | **83.94** | **0.846** | +0.38 |
| PLP terminal-zero v3 | 59.78 | 95.81 | 0.799 | -6.40 |

PLP v3 相对 ALPS countdown 的 family 配对差值为：

```text
PLP v3 MAE − ALPS countdown MAE
estimate = +4.054 token
95% CI   = [-2.970, 10.951]
family_count = 60
```

就点估计而言，ALPS countdown 总体领先约 4.05 token；但配对区间跨 0，当前 OOF 不能证明
ALPS 在所有未见 family 上稳定优于 PLP。两者远远优于只使用 step 的动态 baseline。

### 4.2 任务与长度差异

下表的变化为 `PLP v3 − ALPS countdown`，负数表示 PLP 更好。

| 分组 | ALPS MAE | PLP v3 MAE | PLP−ALPS |
|---|---:|---:|---:|
| QA | **55.45** | 67.49 | +12.05 |
| Summarization | **29.07** | 31.90 | +2.84 |
| Code | 82.66 | **79.94** | -2.72 |
| Short | **20.94** | 25.55 | +4.61 |
| Medium | **70.75** | 73.79 | +3.04 |
| Long | **75.48** | 80.00 | +4.52 |

ALPS 在 QA、Summarization 和三个预设长度条件上整体占优；PLP 在 Code 上略好。这里使用
的是 Train-family OOF，而 PLP 最终 Test 中 Code 相对 v2 出现退化，两者并不矛盾：前者比较
PLP 与 ALPS，后者比较 terminal-zero 与旧 PLP v2，回答的是不同问题。

### 4.3 解码阶段差异

| 解码进度 | ALPS countdown MAE | PLP v3 MAE | PLP−ALPS |
|---|---:|---:|---:|
| 0–10% | **59.71** | 78.74 | +19.03 |
| 10–25% | **60.41** | 77.76 | +17.35 |
| 25–50% | **59.54** | 67.82 | +8.27 |
| 50–75% | 58.47 | **52.84** | -5.63 |
| 75–100% | 45.51 | **41.10** | -4.41 |
| 精确终点 | 28.05 | **1.36** | **-26.69** |
| 非终点整体 | **56.05** | 60.71 | +4.66 |

这张表比总体 MAE 更能说明二者的关系：

- **生成早期 ALPS 明显更好。**此时实际 decode 路径信息很少，而 ALPS 已经提供完整 Prompt
  的全局长度 prior；
- **生成后半程 PLP 反超。**当前 causal hidden state 已经包含实际生成路径，能够修正静态
  prior 无法预知的 seed 与路径差异；
- **终点 PLP 明显更合理。**Terminal zero bin 几乎消除了 ALPS countdown 和旧 PLP 都存在
  的正剩余长度下限。

所以更准确的结论不是“ALPS 胜过 PLP”或“PLP 胜过 ALPS”，而是：

> ALPS 擅长早期全局尺度，PLP 擅长后期路径纠偏和终点判断；两者的优势出现在不同阶段。

### 4.4 现有 Hybrid OOF 作为下一步信号

同一 OOF 中，开发版 `alps_plp_hybrid_v3` 的 family-macro MAE 为 `49.87`：

- 相对 ALPS countdown 改善 `5.86` token，普通配对 95% CI 为 `[-11.60,-0.84]`；
- 相对 PLP terminal-zero v3 改善 `9.91` token，95% CI 为 `[-12.94,-7.17]`；
- 对 ALPS 的九比较 familywise 修正区间上界为 `+0.93`，因此不能提前作最终确认性声明。

这些只是 design-family OOF 开发证据，不是新的 Final Test。但它已经支持研究优先级：结合
方向比继续单独修改 PLP-only 更有希望。由于原 v3 holdout 已由 PLP-only 使用，Hybrid 必须
在新 holdout 上重新做一次冻结后的确认性评价。

## 5. 当前方法能力矩阵

| 维度 | 简单 baseline | ALPS | PLP terminal-zero v3 |
|---|---|---|---|
| 使用时机 | 生成前 | 生成前 | 生成中持续更新 |
| 预测目标 | 最终总长度 | 最终总长度 | 当前剩余长度 |
| Prompt 语义 | 无或仅类别 | Layer-14 state | Entropy-pooled final-layer state |
| 已生成路径 | 无 | 无 | 当前 causal decode state |
| 模型类型 | 均值/Ridge | StandardScaler + Ridge | 20-bin MLP head |
| 非线性 | 无 | 无 | 有 |
| 生成前可用 | 是 | **是** | 否，至少需要开始生成 |
| 生成中纠偏 | 否 | 只能做静态 countdown | **是** |
| 当前主要优势 | 计算简单 | 静态点预测强 | 动态、终点准确 |
| 当前主要限制 | 信息不足 | 区间欠校准 | Code 泛化与显著性不足 |

从服务系统角度看：

- Baseline 只能作为“没有 hidden-state 方法时”的参照；
- ALPS 适合请求刚到达时做初始 batching、KV-cache 或资源预算；
- PLP 适合生成开始后不断更新剩余长度和结束时间估计。

## 6. 为什么需要 ALPS+PLP

可以把长度预测类比为导航：

- ALPS 是出发前根据目的地和路线得到的初始预计总时长；
- PLP 是行驶过程中根据当前位置和实时路况更新的剩余时间；
- ALPS+PLP 是让实时导航同时知道原始全局路线与当前局部状态。

纯 ALPS 看不到实际生成路径。不同 seed 可能让同一 Prompt 产生不同长度，静态 prior 无法
提前知道本次生成会选择哪条路径。

纯 PLP 虽然看到当前路径，但在生成最早期可观察 token 很少；它需要仅凭 Prompt/decode
representation 自行恢复整体长度尺度。PLP v3 在 QA 上受益明显，但 Code family 的尺度泛化
仍不稳定。

ALPS+PLP 的核心假设是：

> ALPS 提供全局总长度 prior，PLP 提供生成路径的动态修正；二者融合后，尤其在解码早期，
> 应比任何一者单独使用更稳定。

## 7. 下一阶段怎样公平比较

新的 Hybrid 实验不能把现有 ALPS MAE `60.87` 与 PLP MAE `71.04` 直接放在同一排行榜。
应在同一批新 Test trace 的每个 timestep 上，把所有方法统一成“预测剩余长度”：

### 7.1 ALPS-only countdown

先用 ALPS 预测最终总长度，再减去已经生成的 token 数：

\[
\widehat{R}^{ALPS}_t=\max(0,\widehat{T}_{ALPS}-t)
\]

### 7.2 PLP-only

使用本阶段冻结的 terminal-zero PLP：

\[
\widehat{R}^{PLP}_t=f(h_{prompt},h'_t)
\]

### 7.3 ALPS+PLP

将 ALPS prior summary 与 PLP hidden-state features 共同输入动态 head：

\[
\widehat{R}^{Hybrid}_t=
g(h_{prompt},h'_t,\widehat{T}_{ALPS},\sigma^2_{ALPS},t)
\]

三种方法必须满足：

- 使用同一批 Prompt family、seed、生成输出和 timestep；
- 同一 family 的所有长度、seed 和 timestep 保持在同一折；
- 使用相同 sequence-balanced、family-macro 指标；
- 同时报 overall、task、3×3、seed、decode progress 和 terminal/nonterminal；
- 只在 design family 上开发与做 OOF；
- 模型和指标全部冻结后，一次性打开新的 family holdout。

主要比较应为：

| 比较 | 回答的问题 |
|---|---|
| ALPS countdown vs 简单 step baseline | 静态语义 prior 是否提供动态价值 |
| PLP v3 vs ALPS countdown | 实际 decode state 是否超越静态倒计时 |
| ALPS+PLP vs PLP v3 | ALPS prior 是否提供 PLP 之外的增量信息 |
| ALPS+PLP vs ALPS countdown | 动态 hidden state 是否提供静态 prior 之外的增量信息 |

## 8. 数据与 Test 边界

当前结果不是来自一套完全相同的最终 Test：

- ALPS、Prompt token baseline 和 Dynamic-Signal MLP v1 使用 v1 数据/旧 Test；
- Hidden-State PLP v2 的旧结果属于开发性 Test；
- PLP terminal-zero v3 使用后来准备并一次性打开的 12-family Test。

因此本报告可以比较方法是否学到了相应信号、总结各阶段能力与限制，但不能把跨数据版本的
Final Test MAE 写成一个统一显著性排行榜。

更重要的是，PLP v3 已经消耗当前 12-family holdout。这批 family 不能继续作为 ALPS+PLP
的未见 Test。下一阶段必须新建并冻结 Hybrid holdout，否则所谓“Hybrid 最终结果”实际上是
在已看过的 Test 上继续开发。

## 9. 当前阶段最终判断

### Baseline

Prompt token 数和五个动态标量都不足以独立完成可靠的长度预测。它们的价值是提供必要的
低信息参照，证明后续性能并非来自简单长度或进度捷径。

### ALPS

ALPS 已经证明生成前 hidden state 中存在强输出长度信号。其点预测在未见 family 上保持较高
解释力，适合作为静态 prior；主要待解决问题是预测区间欠校准，而不是重新证明点预测有效。

### PLP terminal-zero v3

Hidden-State PLP 显著强于旧动态标量 baseline。Terminal-zero v3 进一步修复了 EOS 预测下限，
并获得约 5.35% 的新 Test MAE 点估计改善；但严格优越性区间略跨 0，Code 任务仍存在退化。
它应作为下一阶段冻结的 PLP-only 候选，而不是继续使用当前 Test 调整。

### 下一步

> 当前证据已经分别支持“ALPS 能提供有效静态 prior”和“PLP hidden states 能提供有效动态
> 剩余长度预测”。下一阶段的核心不再是继续分别优化二者，而是在新的公平数据边界下检验：
> ALPS prior 能否为 PLP 提供稳定、可量化的增量收益。

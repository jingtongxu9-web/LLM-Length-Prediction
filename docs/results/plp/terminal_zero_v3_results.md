# PLP 第三阶段：三个消融与 Terminal-Zero v3 结果

本报告记录 PLP-only 第三阶段的完整实验结果。本阶段以 **Hidden-State PLP v2** 为冻结
baseline，在相同 Train trace 上进行三个单因素消融，通过 family-grouped OOF 选择
`plp_terminal_zero_v3`，随后在 12 个全新 Prompt family 上执行一次性最终 Test。

Dynamic-Signal MLP v1 属于早期五标量工程 baseline，不进入本轮正式消融。本文中的“PLP
baseline”均指 Hidden-State PLP v2。

方法原理见 [`../../methods/plp_only_explained.md`](../../methods/plp_only_explained.md)，冻结
协议见
[`../../../configs/experiments/plp_terminal_v3_protocol.json`](../../../configs/experiments/plp_terminal_v3_protocol.json)。

## 0. 核心结论

1. 三个单因素消融中，只有 **terminal zero bin** 在 60-family OOF 上得到稳定改善；
2. `plp_terminal_zero_v3` 的 OOF MAE 为 `59.78`，相对 PLP v2 改善 `1.26` token，修正后的
   98.33% familywise CI 为 `[-2.02, -0.51]`；
3. 最终 Test 上，family-macro MAE 从 `75.06` 降至 `71.04`，点估计改善 `4.01` token，
   相对下降约 **5.35%**；
4. 最终配对 95% CI 为 `[-9.03, 0.57]`，上界略微跨 0，因此预注册的严格优越性判定
   `passed=False`；这表示**观察到改善但证据不足以声明显著优于 baseline**，不表示 v3
   比 v2 更差；
5. Terminal 真实剩余长度为 0 时，平均预测从 `23.11` 降至 `1.50` token，说明独立 zero bin
   实质修复了终点预测下限；
6. 改善主要来自 QA，Code 任务略有退化。三个 seed 都是改善方向，任务差异而非随机 seed
   是最终置信区间较宽的主要来源；
7. PLP-only 阶段至此完成并冻结。下一步是 ALPS+PLP，但当前 12-family holdout 已由 PLP
   使用，Hybrid 确认性实验必须准备新的 holdout。

## 1. 实验身份与数据边界

| 项目 | 内容 |
|---|---|
| 代码版本 | Git commit `eac30b0` |
| 运行环境 | AutoDL，NVIDIA RTX 5090，PyTorch 2.8 / CUDA 12.8 |
| 生成模型 | `Qwen/Qwen2.5-7B-Instruct`，冻结 revision |
| 方法身份 | PLP-only；模型输入不含 ALPS prior、任务标签或 family 标签 |
| Train / design data | 60 个 Prompt family、540 条 rollout、45,119 个逐步预测点 |
| 开发验证 | 5 折 family-grouped OOF；同一 family 的长度、seed 和 timestep 不跨折 |
| 最终 Test | 12 个全新 family、108 条 rollout、10,110 个逐步预测点 |
| Seeds | `42/43/44` |
| 采样频率 | 第 1 个 token、每 5 个 token、终止 token |
| 删失 | Train/Test 均为 0；所有 rollout 均由 EOS 结束 |
| Test gate | `2026-08-06T23:00:08Z` 一次性打开 |
| 主指标 | Family-macro sequence-balanced MAE |

Sequence-balanced 指标让每条 rollout 的全部预测点总权重相同，避免长回答因为记录点更多而
主导结果；family-macro 再让每个 Prompt family 对主指标具有相同权重。最终 Test 的统计独立
单位是 12 个 family，而不是 10,110 个高度相关的 timestep。

原始结果保存在 AutoDL 的 `artifacts/runs/plp_terminal_v3/final_test/`。本次归档副本哈希为：

| 文件 | SHA-256 |
|---|---|
| `OPENED.json` | `3cd39fd1d3580b022dc57614e7aca2ab16ac2c3df9aa5911ecadd4516b964802` |
| `final_report.json` | `fa7038bc8f9469c5ed478c7cb6d716a546909dbc847a6a388114f85eeaed1bfb` |
| `predictions.csv` | `d2cf5ca5e51c57e92a777b7f71827dd449a0c9569643c59d4ec32cf731ba7438` |

## 2. PLP baseline 与三个单因素消融

PLP v2 的输入是一个 3584 维 entropy-pooled Prompt 表征与一个 3584 维当前 decode hidden
state 的拼接，共 7168 维。Qwen 权重完全冻结，只训练 PLP prediction head。

本阶段没有同时改多个因素，而是以 PLP v2 为共同 baseline，分别只改变一项：

| 方法 | Hidden dim | Zero bin | Target range | 目的 |
|---|---:|---|---|---|
| `plp_v2_frozen` | 3584 | 无 | 所有 target，普通分位数 | 冻结 baseline |
| `plp_terminal_zero_v3` | 3584 | **有** | 正长度 target，普通分位数 | 检验独立终点类别 |
| `plp_small_head_v3` | **512** | 无 | 与 v2 相同 | 检验降低模型容量 |
| `plp_weighted_range_v3` | 3584 | 无 | **rollout-balanced** 分位数 | 检验长 rollout 对范围的影响 |

这三个消融分别回答三个不同问题：

1. v2 临近 EOS 的高估是否来自 20-bin 输出没有精确的零点？
2. 25.8M 参数的 prediction head 是否过大？
3. target range 是否被长回答的大量 timestep 主导？

除表中单个变量外，Prompt/decode 表征、20-bin soft labels、loss、dropout、epochs、batch
size、learning rate 和 seed 均保持一致。因此可以把性能变化归因于相应消融，而不是训练条件
同时变化。

## 3. 指标解释

| 指标 | 含义 | 判断方式 |
|---|---|---|
| MAE | 预测剩余 token 与真实剩余 token 的平均绝对差 | 越低越好；本实验主指标 |
| RMSE | 对少数大误差施加更高惩罚 | 越低越好 |
| Bias | `prediction - actual` | 负数表示整体低估，正数表示整体高估 |
| Raw R² | 解释真实剩余长度方差的比例 | 越接近 1 越好 |
| Paired CI | 对每个 family 的“候选 MAE−baseline MAE”做配对 bootstrap | 区间完全低于 0 才支持稳定优越性 |

OOF 阶段同时比较三个候选，因此使用 Bonferroni 修正后的 98.33% familywise CI。最终 Test
只比较一个预先选定的候选和 baseline，因此使用配对 95% CI。

## 4. OOF 消融结果与候选选择

以下全部来自 60 个 Train family 的五折 OOF；每个 family 的预测都由未见过该 family 的折内
模型产生，不是最终模型在自身 Train 上的拟合分数。

| 方法 | Family-macro OOF MAE | 相对 v2 | 95% paired CI | 98.33% familywise CI | 决策 |
|---|---:|---:|---|---|---|
| `plp_v2_frozen` | 61.037 | — | — | — | baseline |
| `plp_terminal_zero_v3` | **59.778** | **-1.259** | `[-1.875,-0.643]` | `[-2.022,-0.513]` | **保留** |
| `plp_small_head_v3` | 70.965 | +9.928 | `[8.059,11.777]` | `[7.599,12.180]` | 淘汰 |
| `plp_weighted_range_v3` | 60.816 | -0.221 | `[-0.854,0.348]` | `[-1.005,0.511]` | 淘汰 |

### 4.1 Terminal zero bin

Terminal-zero 是唯一在修正后区间中仍稳定优于 v2 的消融。它在 5 个 fold 和 seeds
42、43、44 上方向一致，因此被预先冻结为最终候选。

### 4.2 Small head

将 hidden dim 从 3584 缩到 512 后 MAE 上升约 9.93 token，且置信区间完全高于 0。当前数据
不支持“单纯缩小 prediction head 即可改善泛化”的假设。较小模型在本任务上表现为容量不足，
因此不进入最终 Test。

### 4.3 Rollout-balanced target range

加权 target range 的 MAE 仅改善 0.22 token，配对区间跨 0。该变化既不能证明有益，也没有
充分证据说明有害，因此按照预注册规则不进入最终 Test。

## 5. 最终 Test 总体结果

最终 Test 只比较 OOF 后冻结的 `plp_terminal_zero_v3` 与 `plp_v2_frozen`。

| 指标 | PLP v2 | Terminal-zero v3 | 变化 |
|---|---:|---:|---:|
| **Family-macro MAE** | 75.055 | **71.040** | **-4.015** |
| Sequence-balanced RMSE | 120.168 | **115.259** | -4.909 |
| Sequence-balanced Raw R² | 0.761 | **0.780** | +0.019 |
| Sequence-balanced Bias | -20.691 | **-10.285** | +10.407，低估减轻 |

Terminal-zero v3 的 MAE 相对下降约 **5.35%**，RMSE 相对下降约 **4.08%**，并将整体平均
低估幅度缩小约一半。就点估计而言，它在未见 family 上取得了小幅但一致方向的总体改善。

配对 family bootstrap 为：

```text
Terminal-zero v3 MAE − PLP v2 MAE
estimate = -4.015 token
95% CI   = [-9.034, 0.571]
family_count = 12
```

区间上界比 0 高 `0.571` token，因此预注册规则判定 `passed=False`。正确解释是：

> Test 点估计支持 terminal-zero v3，但 12 个独立 family 下的不确定性仍不足以排除零改善或
> 极轻微退化，故不能作“已显著优于 PLP v2”的确认性声明。

这不是性能方向反转。OOF 改善 `1.26` token，Test 改善 `4.01` token，Test 的点估计反而更大；
未通过的主要原因是 Test family 数量少且任务间效果异质，使置信区间变宽。

## 6. Terminal 与非终点结果

| 位置 | PLP v2 MAE | Terminal-zero v3 MAE | 变化 |
|---|---:|---:|---:|
| `remaining_tokens = 0` | 23.112 | **1.502** | **-21.610** |
| `remaining_tokens > 0` | 75.257 | **72.030** | -3.227 |

在真实终点，v2 即使已经生成 EOS，平均仍预测约 23 个剩余 token；独立 zero bin 将该值降至
约 1.5 token，直接验证了本消融的设计目标。同时，非终点 MAE 也改善 3.23 token，说明总体
收益并非仅来自每条 rollout 的最后一个样本点。

终点平均预测按任务分别约为：QA `0.99`、Summarization `1.01`、Code `2.51` token，三类任务
都远优于 v2 约 `23.1` token 的终点下限。

## 7. 按任务分析

| 任务 | PLP v2 MAE | Terminal-zero v3 MAE | 变化 | 解释 |
|---|---:|---:|---:|---|
| QA | 83.303 | **68.312** | **-14.991** | 主要收益来源 |
| Summarization | 32.443 | **32.023** | -0.420 | 基本持平 |
| Code | **109.419** | 112.786 | **+3.366** | 出现退化 |

进一步排除 terminal 点后，QA 非终点 MAE 仍改善约 `14.03` token；Summarization 非终点退化
约 `0.66`，Code 非终点退化约 `3.68`。因此 terminal zero bin 对终点的修复具有跨任务一致性，
但它对正剩余长度分布的影响并不均匀：QA 显著受益，Code 未受益。

## 8. 任务×预设长度九宫格

`intended_length` 只用于分组分析，不作为 PLP 输入特征。

| 任务 | 长度 | PLP v2 MAE | Terminal-zero v3 MAE | v3−v2 |
|---|---|---:|---:|---:|
| Code | Short | **98.374** | 106.468 | +8.094 |
| Code | Medium | **121.254** | 121.520 | +0.266 |
| Code | Long | **108.630** | 110.369 | +1.738 |
| QA | Short | 12.965 | **11.693** | -1.272 |
| QA | Medium | 106.413 | **87.504** | -18.909 |
| QA | Long | 130.531 | **105.740** | -24.791 |
| Summarization | Short | 11.496 | **8.177** | -3.319 |
| Summarization | Medium | 34.424 | **32.860** | -1.564 |
| Summarization | Long | **51.408** | 55.032 | +3.624 |

最大的收益来自 QA/Medium 和 QA/Long；最大的退化来自 Code/Short。按长度汇总时，Short
整体退化 `1.17` token，Medium 和 Long 分别改善 `6.74` 与 `6.48` token。该结果说明 v3 更有
利于较长 QA 剩余长度估计，但不能据此声称它对所有 Prompt 类型普遍改善。

每个九宫格 Test 单元只有 4 个独立 family，因此这里的单元差异属于机制解释，不单独进行
显著性声明。

## 9. 按解码进度分析

| 解码进度 | PLP v2 MAE | Terminal-zero v3 MAE | 变化 |
|---|---:|---:|---:|
| 0–10% | 113.529 | **106.113** | -7.416 |
| 10–25% | 99.375 | **91.448** | -7.927 |
| 25–50% | 83.094 | **78.173** | -4.921 |
| 50–75% | 66.881 | **64.623** | -2.259 |
| 75–100% | 44.801 | **44.262** | -0.539 |

最终 Test 的五个阶段全部是改善方向。早期改善最大，随后随真实剩余长度缩短而逐渐减小；
独立 terminal 分析则显示 EOS 点获得约 21.61 token 的大幅改善。两者共同说明 v3 同时修正了
一部分全程低估以及旧 head 的终点预测下限。

## 10. Seed 与 family 稳定性

| Seed | PLP v2 MAE | Terminal-zero v3 MAE | 变化 |
|---:|---:|---:|---:|
| 42 | 69.739 | **65.918** | -3.821 |
| 43 | 81.463 | **77.628** | -3.835 |
| 44 | 73.963 | **69.575** | -4.388 |

三个 seed 的改善幅度都约为 4 token，结论不依赖某一个采样 seed。

12 个 Test family 中有 8 个改善、4 个退化：

- 4 个 QA family 全部改善，单 family 改善约 `13.16–17.72` token；
- 4 个 Summarization family 中 3 个改善、1 个退化；
- 4 个 Code family 中只有 1 个改善，`dependency_injection` family 退化约 `12.20` token。

Family 差异解释了为何平均改善达到 4.01 token，但配对区间仍略微跨 0：改善不是由 seed
不稳定造成，而是由任务和具体 Prompt family 的泛化差异造成。

## 11. 对三个消融假设的最终判断

| 假设 | 证据 | 判断 |
|---|---|---|
| 精确 zero bin 能修复 EOS 高估 | OOF 稳定改善；Test 终点 MAE `23.11→1.50` | **成立** |
| 缩小 head 能缓解过拟合 | Small-head OOF MAE 恶化 `9.93` token | **不成立** |
| Rollout-balanced range 能稳定改善 | OOF 改善仅 `0.22`，区间跨 0 | **证据不足** |
| Terminal-zero 能在所有任务上普遍改善 | QA 改善，Code 退化；最终 CI 略跨 0 | **不成立** |

因此，本阶段最可靠的结论不是“terminal-zero 已全面解决 PLP”，而是：

> 显式终点类别是正确且必要的结构修正，能够消除旧版终点预测下限，并带来约 5.35% 的总体
> Test MAE 点估计改善；但它没有解决任务间泛化差异，当前样本量下也未通过严格优越性声明。

## 12. PLP 阶段结论与下一步

### PLP-only 已完成

本项目现已完成：

1. PLP v2 hidden-state baseline；
2. 三个独立的 PLP v3 消融；
3. 60-family 五折 OOF 候选选择；
4. 全部 Train 上的最终模型冻结；
5. 12-family 一次性最终 Test；
6. 总体、任务、九宫格、seed、解码进度和 terminal/nonterminal 分析。

`plp_terminal_zero_v3` 可以作为后续工程比较中的 **PLP-only 候选**，但论文表述必须保留
“最终 Test 置信区间略跨 0”的边界。不得根据本次 Test 修改 v3 后继续使用同一批 family
重新宣称改进；若要专门解决 Code 退化，应建立 PLP v4 并准备新 holdout。

### 下一步：ALPS+PLP

下一阶段比较：

1. ALPS-only：生成前的静态总长度 prior；
2. PLP-only：本报告冻结的 terminal-zero v3；
3. ALPS+PLP：将静态 prior 与动态 hidden-state 信息结合。

当前 12 个 Test family 已由 PLP-only gate 正式消耗，不能再作为 Hybrid 的未见 Test。下一步
应先创建并冻结新的 Hybrid holdout，再只用现有 60 个 design family 进行开发与 OOF，最后
一次性打开新的 Test。这样才能判断 ALPS prior 是否为 PLP 提供了超出 terminal-zero v3 的
稳定增益。

> 最终判断：PLP 第三阶段获得了方向正确、幅度有限的总体改善，terminal-zero 的机制目标
> 得到明确验证；但严格显著性声明未通过，主要剩余问题是 Code 与特定 family 的泛化。
> PLP-only 阶段至此结束，后续研究转入 ALPS+PLP。

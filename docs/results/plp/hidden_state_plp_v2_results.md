# Hidden-State PLP v2 实验结果

本报告记录 Hidden-State PLP v2 的冻结 Train/Test 结果。该方法在生成过程中每隔 5 个 token
预测一次剩余长度，输入由 entropy-guided Prompt 表征和当前生成 token 的最终层 causal
hidden state 拼接而成。它不读取 ALPS prior，也不使用任务类型、预设长度或 Prompt family
作为模型输入。

方法原理见 [`../../methods/plp_only_explained.md`](../../methods/plp_only_explained.md)，实现
合同见 [`../../../configs/experiments/plp_v2_manifest.json`](../../../configs/experiments/plp_v2_manifest.json)。

## 0. 核心结论

Hidden-State PLP v2 的必要评估已经完成，现有 Test 报告覆盖整体性能、3×3 Prompt 类别、
五个生成进度阶段以及 Train/Test 泛化差距，不需要为当前 v2 继续增加新的评估维度。

- **总体能力：**Test sequence-balanced MAE 为 `60.03` token，RMSE 为 `91.60`，Bias 为
  `+12.65`，Raw R² 为 `0.790`。模型能够在未见 Prompt family 上预测剩余生成长度，但精度
  明显低于 Train。
- **3×3 Prompt 类别：**Summarization/Short 最好，MAE 为 `12.23`；主要薄弱区域是
  Code/Medium、Code/Long、QA/Medium 和 QA/Long，MAE 约为 `85–91` token。说明 PLP 的
  效果随任务与 Prompt 长度条件明显变化，并非所有 Prompt 种类表现一致。
- **渐进预测能力：**随着生成推进，Test MAE 从前 0–10% 的 `73.37` 下降到最后 75–100%
  的 `49.72`，证明生成过程中持续更新预测是有效的；但最后四分之一平均高估 `39.16`
  token，临近 EOS 时仍不够可靠。
- **相对旧基线：**相比 Dynamic-Signal MLP v1，Test MAE 降低 `56.1%`，说明 Prompt 与
  decode hidden states 确实提供了旧版五个标量特征中缺失的长度信息。
- **泛化边界：**Train/Test MAE 从 `27.99` 上升到 `60.03`，存在明显的 family-level 泛化
  缺口。当前 Test 只有 12 个独立 family；每个 3×3 单元虽然有 12 条 rollout，但本质上只有
  4 个独立 family。因此这些数据足够作为 v2 阶段性结果和类别差异分析，但九宫格中的细粒度
  排名仍属于探索性结论，不能视为大样本确认性结论。

## 1. 结果身份与实验边界

| 项目 | 内容 |
|---|---|
| 代码版本 | Git commit `21aa7db`（`add hidden-state PLP v2 and reorganize docs`） |
| 运行环境 | AutoDL，NVIDIA RTX 5090，PyTorch 2.8 / CUDA 12.8 |
| 生成模型 | `Qwen/Qwen2.5-7B-Instruct`，固定 revision |
| 方法身份 | PLP-only，paper-aligned、non-exact replication |
| Train | 48 个 family、144 个 Prompt、432 条 rollout、36,472 个预测点 |
| Test | 12 个 family、36 个 Prompt、108 条 rollout、8,647 个预测点 |
| Seeds | `42/43/44` |
| 删失情况 | Train/Test 均为 0 条；所有 rollout 都在 4096 token 上限前结束 |
| 更新频率 | 第 1 个 token、每 5 个 token 和终止 token |
| 主指标 | Sequence-balanced MAE/RMSE/Bias/Raw R² |

现有 v1 Test 在本实验前已经打开，因此本报告中的 Test 是**开发性对照**，不能再用于修改
v2 后重新声称确认性结果。严格的后续结论需要新的 family-level holdout。

作者公开仓库没有完整 PLP 源码，论文也没有说明可变长历史 hidden states 如何进入同一个
固定维度 prediction head。本项目使用“entropy-pooled Prompt state + 当前 causal decode
state”的 7168 维实现；当前 causal state 已经注意到此前生成的全部 token。因此本结果不能
表述为逐行复现论文 PLP。

原始评价文件保存在 AutoDL 的 `artifacts/runs/plp_v2/`，本次归档副本哈希为：

| 文件 | SHA-256 |
|---|---|
| `train_evaluation.json` | `468f559d821e61aff387ff7f7c959a55230adec89508e73b176a390786aa9074` |
| `test_evaluation.json` | `9d52465194a86be51332af4daecbe49ac7a5e81b315680a519c3858183a052d7` |

## 2. 模型与训练设置

| 项目 | 固定值 |
|---|---|
| Prompt representation | 全部 chat-template Prompt token 的 entropy-softmax pooled final-layer state |
| Decode representation | 当前生成 token 的 final-layer causal state |
| 输入维度 | `3584 + 3584 = 7168` |
| Prediction head | `Linear(7168,3584) → LayerNorm → ReLU → Dropout(0.1) → Linear(3584,20)` |
| 可训练参数 | 25,772,564 |
| 目标 | 未变换的 `remaining_tokens = T - t` |
| 输出 | 20-bin softmax 分布的 bin-center 期望值 |
| Loss | `0.95 × soft-target CE + 0.05 × token-space MSE` |
| Optimizer | AdamW，learning rate `2e-5`，weight decay `0` |
| 训练 | 10 epochs，batch size 16，seed 42 |
| 样本权重 | 每条 rollout 的全部预测点总权重相同 |

Qwen 权重完全冻结；只有约 25.8M 参数的 PLP head 被训练。10 个 epoch 的 sequence-balanced
联合 loss 为：

```text
659.77 → 354.28 → 270.12 → 215.70 → 183.50
       → 161.14 → 136.23 → 123.10 → 111.61 → 105.04
```

Loss 单调下降，没有发散或 NaN。第 10 轮仍在下降，说明优化过程尚未完全进入平台期；但
epochs 已经冻结，不能根据 Train 或已打开的 Test 临时增加训练轮数。

## 3. 指标口径

| 指标 | 含义 | 判断方式 |
|---|---|---|
| MAE | 预测剩余 token 与真实剩余 token 的平均绝对误差 | 越低越好 |
| RMSE | 对少数大误差惩罚更重 | 越低越好；明显高于 MAE 代表仍有大误差长尾 |
| Bias / Mean error | `prediction - actual` | 负数为低估，正数为高估 |
| Raw R² | 模型解释当前评价集合剩余长度方差的比例 | 1 最好，0 等同均值，负数比均值更差 |

普通 timestep 指标把每个预测点视为一个样本，因此长回答因为记录点更多而权重更大。
Sequence-balanced 指标先让每条 rollout 的全部点总权重相同，是本报告的主要口径。

## 4. Train/Test 总体结果

| 指标 | Train | Test | 解释 |
|---|---:|---:|---|
| 普通 MAE | 38.00 | 78.14 | 长回答预测点更多，因而数值更高 |
| 普通 RMSE | 54.06 | 110.02 | Test 仍存在较大的尾部误差 |
| 普通 Bias | -6.12 | +9.24 | Train 略低估，Test 略高估 |
| 普通 Raw R² | 0.950 | 0.748 | Test 保留明显解释能力 |
| **Sequence-balanced MAE** | **27.99** | **60.03** | Test 是 Train 的 2.14 倍 |
| **Sequence-balanced RMSE** | **41.59** | **91.60** | Test 是 Train 的 2.20 倍 |
| **Sequence-balanced Bias** | **-2.01** | **+12.65** | Test 平均高估约 13 token |
| **Sequence-balanced Raw R²** | **0.963** | **0.790** | 泛化下降明显，但 Test 仍有效 |

Test sequence-balanced MAE 为 60.03 token、R² 为 0.790，说明 hidden-state PLP 在未见
Prompt family 上确实学到了可泛化的剩余长度信号。但 Train MAE 27.99 到 Test MAE 60.03
的差距也非常明确，不能把高 Train R² 当成模型已经解决问题。

## 5. 与 Dynamic-Signal MLP v1 对比

两种方法都逐步预测 `remaining_tokens`，并采用 sequence-balanced 口径，因此可以直接比较：

| 方法 | Test MAE | Test RMSE | Test Raw R² | Test Bias |
|---|---:|---:|---:|---:|
| Dynamic-Signal MLP v1 | 136.66 | 190.19 | 0.089 | +38.20 |
| **Hidden-State PLP v2** | **60.03** | **91.60** | **0.790** | **+12.65** |

相对旧动态标量 MLP，Hidden-State PLP v2：

- MAE 降低 **56.1%**；
- RMSE 降低 **51.8%**；
- Raw R² 从 `0.089` 提高到 `0.790`；
- 平均高估从 38.20 token 降到 12.65 token。

这证明旧 v1 的主要限制是 `step/entropy/EOS probability` 五个标量信息不足。加入 Prompt
语义表征和当前 causal hidden state 后，动态剩余长度预测能力出现实质提升。

ALPS 预测的是生成前的**最终总长度**，PLP 预测的是生成过程中的**当前剩余长度**，二者的
样本位置和目标不同，因此不能直接用 ALPS MAE 约 61 与 PLP MAE 60.03 排名。

## 6. 按解码进度分析

以下均为 sequence-balanced 指标：

| 解码进度 | Train MAE | Test MAE | Test Bias | Test Raw R² |
|---|---:|---:|---:|---:|
| 0–10% | 37.86 | 73.37 | -12.11 | 0.824 |
| 10–25% | 29.53 | 72.49 | -13.08 | 0.798 |
| 25–50% | 27.23 | 62.82 | -2.13 | 0.748 |
| 50–75% | 26.92 | 55.45 | +24.68 | 0.430 |
| 75–100% | 24.31 | **49.72** | **+39.16** | **-1.695** |

### 6.1 渐进预测能力成立

Test MAE 随生成推进总体单调下降：

```text
73.37 → 72.49 → 62.82 → 55.45 → 49.72
```

这验证了 PLP 的核心目标：模型看到越来越多的实际生成路径后，剩余长度点预测逐步改善。
旧 Dynamic-Signal MLP 在后期反而恶化到 147.98 MAE；v2 后期为 49.72，改进尤其明显。

### 6.2 后期出现系统性高估

前 25% 平均低估约 12–13 token；25–50% 基本无偏；50% 后转为高估，最后四分之一平均
高估 39.16 token。后期 MAE 虽然最低，但真实剩余长度的范围也更窄，负 R² 表明模型没有
充分解释后期单元内部差异，甚至不如使用该阶段的均值。

一个待核验机制是全局 20-bin head：预测值只能是 bin centers 的加权平均，若最小 bin center
明显大于 0，模型临近 EOS 时便存在不可消除的预测下限。当前评价文件没有保存
`target_range`，因此本报告只把它列为解释假设；需要结合 `training_report.json` 后才能确认。

## 7. 按任务分析

| 任务 | Train MAE | Test MAE | Test Bias | Test Raw R² |
|---|---:|---:|---:|---:|
| Code | 37.61 | **79.72** | +15.90 | 0.779 |
| QA | 27.79 | 64.89 | **+22.51** | 0.716 |
| Summarization | **18.58** | **35.48** | -0.48 | 0.786 |

Summarization 最稳定，Test 几乎无系统偏差。Code 的绝对误差最大；QA 的整体高估最明显。
三类任务的 Test R² 都在 0.71 以上，说明模型仍能区分相对长短，但 Code/QA 的绝对尺度
估计需要改进。

## 8. 按预设长度条件分析

`intended_length` 是实验预设的 Prompt 条件，只用于分组评价，不进入模型输入。

| 条件 | Train MAE | Test MAE | Test Bias | Test Raw R² |
|---|---:|---:|---:|---:|
| Short | **14.59** | **31.14** | +15.21 | -0.598 |
| Medium | 32.24 | 70.95 | +7.17 | 0.713 |
| Long | 37.14 | 78.00 | +15.55 | 0.734 |

Short 的绝对误差最低，但 Test R² 为负。原因是 Short 单元的真实剩余长度方差较小，R² 对
误差非常敏感；不能仅凭负 R² 得出“Short 最差”。Medium/Long 的主要问题是绝对误差仍在
71–78 token。

## 9. 任务×长度九宫格

以下仍为 sequence-balanced 指标：

| 任务 | 长度 | Train MAE | Test MAE | Test Bias | Test Raw R² |
|---|---|---:|---:|---:|---:|
| Code | Short | 20.66 | 60.60 | +25.19 | -1.683 |
| Code | Medium | 41.66 | 87.78 | +0.59 | 0.685 |
| Code | Long | 50.51 | **90.79** | +21.93 | 0.755 |
| QA | Short | 11.76 | 20.59 | +16.91 | -2.524 |
| QA | Medium | 35.51 | **89.19** | +16.82 | 0.336 |
| QA | Long | 36.09 | 84.89 | **+33.82** | 0.579 |
| Summarization | Short | 11.36 | **12.23** | +3.53 | 0.211 |
| Summarization | Medium | 19.55 | 35.86 | +4.11 | 0.512 |
| Summarization | Long | 24.82 | 58.33 | -9.09 | 0.654 |

最强单元是 Summarization/Short，MAE 仅 12.23；其次是 QA/Short 和
Summarization/Medium。主要薄弱区域集中在 Code 的 Medium/Long，以及 QA 的 Medium/Long。
Short 单元中的负 R²应结合较小样本和窄方差解释；Code/Short 同时有 60.60 MAE 和明显正
Bias，属于真实薄弱单元，而 Summarization/Short 虽 R²不高但绝对误差很小。

## 10. Seed 稳定性

| Seed | Test MAE | Test RMSE | Test Bias | Test Raw R² |
|---:|---:|---:|---:|---:|
| 42 | 60.39 | 90.53 | +18.03 | 0.793 |
| 43 | **58.87** | **90.33** | +8.43 | **0.808** |
| 44 | 60.83 | 93.88 | +11.48 | 0.767 |

三个 seed 的 MAE 只相差约 2 token，说明总体结论不是由某一个采样 seed 偶然驱动。当前
主要误差来源是未见 Prompt family、任务和长度结构，而不是随机种子不稳定。

## 11. 过拟合与泛化判断

本实验存在明显的 Train/Test 泛化缺口：

```text
Sequence-balanced MAE: 27.99 → 60.03
Sequence-balanced R²:  0.963 → 0.790
```

可以把它描述为明显的过拟合风险或 family-level 泛化不足，但不能说模型完全只记住了
Train，因为 Test R² 仍达到 0.790，且分阶段 MAE 保持渐进下降。

36,472 个训练点并非 36,472 个独立样本：同一 rollout 内的 timestep 高度相关，同一 Prompt
的三个 seed 也共享语义，真正独立的 Train family 只有 48 个。与此同时 prediction head 有
25.8M 个可训练参数，因此有效样本规模与模型容量并不匹配，这是泛化缺口的重要风险来源。

目前无法仅凭本实验把差距完全拆分为“纯参数过拟合”和“未见 Prompt family 的分布偏移”。
要区分两者，应在 Train family 内做 grouped validation，而不是继续使用已打开的 Test 调参。

## 12. 结论与后续边界

### 已验证的能力

1. Hidden-State PLP v2 在 Test 上达到 sequence-balanced MAE `60.03`、Raw R² `0.790`；
2. 相比 Dynamic-Signal MLP v1，MAE 降低 56.1%，证明 hidden states 提供了关键动态信息；
3. Test MAE 随解码进度从 73.37 降至 49.72，渐进预测目标成立；
4. 三个 seed 结果稳定；
5. Summarization 尤其是 Short/Medium 表现最好。

### 当前限制

1. Train/Test MAE 相差约 2.14 倍，存在明显 family-level 泛化缺口；
2. 生成后半段从低估转为高估，最后四分之一平均高估 39.16 token；
3. Code 和 QA 的 Medium/Long 是主要薄弱单元；
4. 25.8M 参数相对于 48 个独立 Train family 偏大；
5. Test 已用于开发性分析，不能继续作为 v2 调参依据；
6. 当前实现是论文对齐的固定维度解释，不是公开源码逐行复现。

### 下一版建议

保留当前 v2 checkpoint 和结果不变。后续以新方法 ID 开展 v3：

1. 只在 Train family 内做 grouped validation，用于选择 epoch、head width 和正则强度；
2. 比较更小的 bottleneck head，降低有效样本不足时的过拟合风险；
3. 读取并报告 `target_range` 与最小 bin center，验证后期预测下限假设；
4. 将显式 zero bin、包含端点的 bin centers 或额外连续回归项作为独立消融；
5. 使用新的 family-level holdout 得到确认性 Test，避免继续复用当前 Test。

> 最终判断：Hidden-State PLP v2 是一次有效且显著优于旧动态 baseline 的实验，但尚未达到
> “稳定泛化且临近 EOS 可靠”的状态。它应作为当前 paper-aligned PLP 基线冻结保留。

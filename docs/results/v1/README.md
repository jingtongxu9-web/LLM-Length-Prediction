# v1 综合实验结果

本文件是 v1 实验的唯一总入口。ALPS 的完整分组、五折和校准分析见
[`alps.md`](alps.md)；Dynamic-Signal MLP 的方法与结果见
[`dynamic_signal_mlp.md`](dynamic_signal_mlp.md)。

本报告汇总当前 v1 的三组实验：

1. ALPS Layer-14 Ridge 静态长度预测；
2. Prompt 输入 token 数 Ridge baseline；
3. Dynamic-Signal MLP v1 动态剩余长度预测。

## 1. 结果身份与实验状态

| 项目 | 内容 |
|---|---|
| 代码版本 | Git commit `136c12c`（`add dynamic length prediction v1`） |
| 运行环境 | AutoDL，NVIDIA RTX 5090，PyTorch 2.8 / CUDA 12.8 |
| 数据规模 | 60 个 prompt family、180 个 Prompt、540 条 rollout |
| Train | 48 个 family、144 个 Prompt、432 条 rollout |
| Test | 12 个 family、36 个 Prompt、108 条 rollout |
| Seeds | `42/43/44` |
| 生成模型 | `Qwen/Qwen2.5-7B-Instruct`，固定 revision |
| ALPS | zero-based Layer 14，`Ridge(alpha=1.0)` |
| Dynamic v1 | 五个动态标量，MLP `[128,64]`，9089 个参数 |

540 条 rollout 已全部采集，三组方法均已完成训练和评价。ALPS 五折使用 Train 内
family-grouped out-of-fold 预测，不选择 Layer 或 alpha；最终 Test 在此前已经打开，因此
新增 baseline 和 Dynamic-Signal MLP 的 Test 结果属于事后对照，不得继续用于 v1 调参。

## 2. 指标说明

| 指标 | 含义 | 判断方式 |
|---|---|---|
| MAE | 平均绝对 token 误差 | 越低越好 |
| RMSE | 对大误差更敏感的 token 误差 | 越低越好；明显高于 MAE 表示有长尾大误差 |
| Raw R² | 原始 token 空间的解释能力 | 1 最好；0 等同均值预测；负数比均值预测更差 |
| Log R² | `log1p(length)` 空间的解释能力 | 1 最好；更重视相对长度尺度 |
| Mean error / Bias | `prediction - actual` | 负数为低估，正数为高估 |
| NLL | 真实长度在预测分布下的负对数似然 | 同一目标和数据上越低越好 |
| 95% Coverage | 实际值落入名义 95% 区间的比例 | 应与区间宽度一起解释 |

ALPS 的 Prompt-mean 指标先对同一 Prompt 的三个 seed 求均值，用于评价 Prompt 的期望
输出长度；rollout-level 指标评价每一次随机生成。Dynamic-Signal MLP 中长回答包含更多
timestep，因此以 **sequence-balanced** 指标作为主要结论：每条 rollout 的所有动态点总权重
相同，避免长回答仅因采样点多而支配结果。

## 3. ALPS 固定五折结果

### 3.1 Rollout-level out-of-fold 对比

| 方法 | MAE | RMSE | Raw R² | Log R² | NLL | 95% Coverage | 区间均宽 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global mean | 284.12 | 324.78 | -0.091 | -0.002 | 7.123 | 98.4% | 2514.11 |
| Prompt tokens | 260.60 | 313.97 | -0.020 | 0.018 | 7.113 | 96.5% | 2486.80 |
| Metadata | 92.95 | 140.88 | 0.795 | 0.925 | **5.830** | 94.7% | 547.68 |
| Metadata + Prompt tokens | 92.94 | 140.08 | 0.797 | 0.925 | **5.829** | 94.9% | 545.99 |
| **ALPS Layer 14** | **60.87** | **91.31** | **0.914** | **0.953** | 6.868 | 71.1% | **180.81** |

### 3.2 Prompt-mean out-of-fold 对比

| 方法 | Prompts | MAE | RMSE | Raw R² | Log R² |
|---|---:|---:|---:|---:|---:|
| Global mean | 144 | 282.06 | 321.11 | -0.093 | -0.002 |
| Prompt tokens | 144 | 257.11 | 310.18 | -0.020 | 0.019 |
| Metadata | 144 | 87.26 | 132.21 | 0.815 | 0.932 |
| Metadata + Prompt tokens | 144 | 87.02 | 131.36 | 0.817 | 0.933 |
| **ALPS Layer 14** | **144** | **54.03** | **77.27** | **0.937** | **0.962** |

### 3.3 ALPS 结果解释

与 Prompt token baseline 相比，ALPS 的 rollout MAE 降低 **76.6%**，RMSE 降低
**70.9%**。这证明 Layer-14 hidden state 包含的长度信息远超“输入有多少 token”。

Metadata 已经能够利用 `qa/summarization/code` 和 `short/medium/long` 解释大量组间差异，
但 ALPS 相比 Metadata 仍将 MAE 降低 **34.5%**、RMSE 降低 **35.2%**，说明 hidden
state 还包含 Prompt 内容和语义层面的增量信息。Metadata 加入 Prompt token 数后几乎没有
改善，输入长度不是当前性能的主要来源。

ALPS 的 Prompt-mean MAE 比 rollout-level 更低，说明它更擅长预测一个 Prompt 的平均输出
尺度，而无法在生成前预知 seed 引起的随机路径差异。

### 3.4 Train、五折与最终 Test

| 评价位置 | MAE | RMSE | Log R² | 95% Coverage |
|---|---:|---:|---:|---:|
| Train rollout | 32.19 | 48.74 | 0.991 | 94.7% |
| 五折 OOF rollout | 60.87 | 91.31 | 0.953 | 71.1% |
| Final Test rollout | 66.97 | 97.11 | 0.929 | 63.9% |

Train 指标明显过于乐观，但五折与 Test 接近，说明高维 Ridge 虽然在 Train 上接近插值，
却没有在未见 family 上失效。ALPS v1 的**点预测泛化能力得到支持**。

主要不足是概率校准：五折名义 95% 区间只覆盖 71.1%，Test 只覆盖 63.9%。ALPS 区间
很窄，点预测准确，但对错误过于自信。Metadata 的 NLL 反而更低，原因是它的点预测虽然
较差，却给出了更宽、覆盖更充分的区间。

该问题的模型输入边界、论文评价范围和 OOF conformal 修正建议见
[`alps.md`](alps.md)。

## 4. Prompt 输入 token Ridge baseline

| Split | N | MAE | RMSE | Log R² | NLL | 95% Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Train | 432 | 260.15 | 313.66 | 0.0198 | 7.112 | 96.5% |
| Test | 108 | 246.77 | 297.52 | 0.0106 | 7.074 | 91.7% |

Train 与 Test 都接近零解释力，因此这不是“Train 好、Test 差”的过拟合，而是输入特征
本身很弱。高 Coverage 也不表示预测准确：五折中该方法的平均区间宽度约 2487 tokens，
主要依靠极宽区间覆盖真实值。

结论是：**Prompt 输入长度不能作为本数据上的有效输出长度预测器，也不能解释 ALPS 的
优势。**

## 5. Dynamic-Signal MLP v1

Dynamic-Signal MLP v1 使用：

```text
step, entropy, entropy_mean, entropy_slope, eos_probability
```

每 5 个生成 token 形成一个非终止样本；它不读取 ALPS prior、Prompt hidden state 或逐
token hidden state。Train 包含 36,040 个动态点，Test 包含 8,539 个动态点。

### 5.1 总体结果

| 指标 | Train | Test |
|---|---:|---:|
| 动态点数 | 36,040 | 8,539 |
| Rollout 数 | 432 | 108 |
| 普通 MAE | 169.89 | 159.96 |
| 普通 RMSE | 231.95 | 216.82 |
| 普通 Raw R² | 0.064 | 0.013 |
| Log R² | -0.028 | 0.004 |
| Sequence-balanced MAE | 140.12 | **136.66** |
| Sequence-balanced RMSE | 193.62 | 190.19 |
| Sequence-balanced Raw R² | 0.198 | **0.089** |
| Sequence-balanced Bias | +25.44 | +38.20 |
| Sequence-balanced 95% Coverage | 93.5% | 92.8% |

Test Sequence-balanced Raw R² 只有 0.089，Log R² 接近 0，说明五个动态标量只解释了
很少的剩余长度变化。Train 本身也不强，因此主要问题不是经典的严重过拟合，而是特征、
目标或校准方式不足；Train/Test 之间存在一定差距，但不是当前失败的主因。

完整方法和分阶段结果见
[`dynamic_signal_mlp.md`](dynamic_signal_mlp.md)。

### 5.2 Test 按解码进度

以下以 sequence-balanced 指标为主。Bias 为 `prediction - actual`。

| 解码进度 | Points | MAE | RMSE | Raw R² | Bias | 95% Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 0–10% | 904 | 241.27 | 319.10 | -0.402 | **-212.23** | 74.9% |
| 10–25% | 1,270 | 152.63 | 206.39 | 0.206 | -56.19 | 99.0% |
| 25–50% | 2,123 | **90.93** | **130.48** | **0.476** | +31.19 | 100.0% |
| 50–75% | 2,127 | 120.56 | 155.06 | -0.927 | +99.90 | 100.0% |
| 75–100% | 2,115 | 147.98 | 193.43 | -14.187 | **+145.15** | 81.1% |

动态模型呈现清晰的阶段性错误：

1. **早期严重低估。**0–10% 平均低估 212 tokens。此时仅凭少量 entropy/EOS 信号难以
   判断回答最终属于短输出还是长输出，模型又没有 Prompt 语义或 ALPS prior。
2. **中段最有效。**25–50% 的 Sequence-balanced R² 达到 0.476，说明生成路径发展到
   中段后，当前动态信号开始包含可用信息。
3. **后期严重高估。**75–100% 平均高估 145 tokens，预测甚至明显差于该阶段直接使用
   均值的参照。

### 5.3 后期高估的技术原因

模型在 `log1p(remaining_tokens)` 上训练，并使用一个全局残差方差：

```text
residual_variance = 1.3360
```

从 log 空间还原 token 均值时使用：

\[
\exp(\mu+\sigma^2/2)-1
\]

其方差修正因子为：

\[
\exp(1.3360/2)\approx1.95
\]

这会把 log-space 中位数对应的剩余长度近似提高到约 1.95 倍。该修正有助于表达长尾，
但在生成后期真实剩余长度已经很小时容易产生明显向上偏差。其他可能因素包括：

- log-MSE 压缩大长度的绝对误差，早期长回答低估受到的惩罚不足；
- 一个全局残差方差无法描述早期、中期和后期不同的误差分布；
- terminal point 被排除，模型没有直接的 `remaining_tokens=0` 锚点；
- EOS probability 记录于 top-p 过滤前，不完全等同实际采样 EOS 的概率；
- 五个动态标量不含 Prompt 内容和任务语义。

### 5.4 为什么总体 Bias 和 Coverage 看起来尚可

普通 Test Bias 为 -2.61，表面接近 0，但它来自早期低估与后期高估互相抵消，不代表各
阶段准确。总体 Coverage 约 94% 同样掩盖了分阶段问题：早期只有约 75%、后期约 81%，
中段却达到 100%。当前概率分布是阶段间错配，而不是稳定校准。

## 6. 三个主要研究结论

### 6.1 ALPS 是否有效？

**有效。**固定 Layer-14 / alpha=1.0 的 family-grouped 五折 MAE 为 60.87、Log R² 为
0.953，并与独立 Test 接近。ALPS 在未见 Prompt family 上具有稳定点预测能力。

### 6.2 ALPS 是否只是利用输入长度？

**不是。**Prompt token baseline 的 Test Log R² 只有 0.011，五折 MAE 为 260.60；
ALPS 五折 MAE 为 60.87。Metadata 加入 Prompt token 后也几乎没有改善。

### 6.3 当前五个动态信号是否足以可靠预测剩余长度？

**不足。**Dynamic-Signal MLP v1 只在 25–50% 阶段表现出中等能力，整体 R² 接近 0，
并存在明显的早期低估、后期高估。它应作为已跑通的工程 baseline 和负向结果保留，不能
称为可靠的论文 PLP 复现。

## 7. 比较边界

ALPS 的 MAE 约 61 tokens，Dynamic-Signal MLP 的 Sequence-balanced Test MAE 约 137
tokens，但两者不能直接据此排名：

- ALPS 在生成前、每条 rollout 预测一次最终总长度；
- Dynamic-Signal MLP 在生成过程中、每条 rollout 的多个 timestep 预测剩余长度；
- 两者目标、采样单位和评价分布不同。

后续公平动态比较至少需要加入：

\[
\widehat{remaining}_t=\max(0,\widehat{total}_{ALPS}-t)
\]

即把 ALPS 静态总长度预测减去已经生成的 token 数，作为同 timestep 的动态 baseline。
还应在相同五特征上比较 Dynamic Ridge，判断 MLP 的非线性是否真正产生价值。

由于 v1 Test 已经查看，以上新增设计只能作为 Train-only 诊断；任何确认性结论应使用新的
holdout 或 v2 数据划分。

## 8. v1 最终结论

> ALPS v1 在未见 Prompt family 上获得稳定且显著优于静态 baseline 的长度点预测能力，
> 证明 Qwen Layer-14 hidden state 含有有效的输出长度信息；但其概率区间明显欠校准。
> Prompt 输入 token 数几乎没有预测力。Dynamic-Signal MLP v1 虽成功完成动态训练与评价，
> 但五个标量信号无法在整个解码过程中稳定预测剩余长度，仅在生成中段表现出一定能力，
> 并呈现早期低估、后期高估。因此它应作为 v1 工程 baseline 保留。独立的 Hidden-State
> PLP v2 代码已经实现，需重新采集 hidden-state trace 后才能产生可比较结果。

## 9. 原始结果路径

```text
artifacts/runs/alps_v1/diagnostics/grouped_cv/summary.csv
artifacts/runs/alps_v1/diagnostics/grouped_cv/results.json
artifacts/runs/alps_v1/stage1/train_evaluation.json
artifacts/runs/alps_v1/stage1/test_evaluation.json
artifacts/runs/alps_v1/comparisons/input_token_ridge/train_evaluation.json
artifacts/runs/alps_v1/comparisons/input_token_ridge/test_evaluation.json
artifacts/runs/alps_v1/comparisons/plp_only/training_report.json
artifacts/runs/alps_v1/comparisons/plp_only/train_evaluation.json
artifacts/runs/alps_v1/comparisons/plp_only/test_evaluation.json
```

原始 trace、模型和运行结果不提交 Git；本报告记录可提交的汇总结果。

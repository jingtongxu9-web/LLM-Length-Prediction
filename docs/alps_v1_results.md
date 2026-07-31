# ALPS v1 实验结果

## 1. 统计指标

### 1.1 点预测指标

| 指标 | 含义 | 判断方式 |
|---|---|---|
| Bias | 平均预测长度减去平均实际长度 | 接近0最好；正数为高估，负数为低估 |
| MAE | 平均每条预测与实际长度相差多少tokens | 越低越好 |
| RMSE | 对大误差更敏感的长度误差 | 越低越好；明显高于MAE表示存在较大错误 |
| Raw R² | 在原始token尺度上的解释能力 | 1为完美，0等同组内均值参照，负数表示低于该参照 |
| Log R² | 在`log1p(output_tokens)`尺度上的解释能力 | 1为完美，更侧重比例尺度 |
| Pearson r | 预测长度与实际长度的线性相关性 | 越接近1，长度排序关系越强 |

本报告用两种单位评价点预测：

- **Prompt-mean**：先对同一个Prompt的三个seed实际长度求均值，再与ALPS点预测比较，
  主要用于评价Prompt平均输出长度的预测能力。
- **Rollout-level**：直接评价每一次实际生成，用于描述单次请求误差。

### 1.2 概率指标

| 指标 | 含义 | 判断方式 |
|---|---|---|
| NLL | 真实长度在ALPS预测分布中的负对数似然 | 相同数据上越低越好 |
| 95% Coverage | 实际rollout落入ALPS 95%预测区间的比例 | 应接近95% |

NLL和Coverage使用全部rollout评价，因为它们需要衡量三个seed产生的实际长度波动。

## 2. 冻结实验设置

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| ALPS特征 | zero-based Layer 14，最后一个Prompt token |
| 精度 | BF16 |
| Temperature / Top-p | `0.7` / `0.95` |
| Max new tokens | `4096`，包含EOS |
| Seeds | `42/43/44` |
| Ridge | Train-only StandardScaler，`alpha=1.0` |
| 训练目标 | `log1p(output_tokens)` |
| Train | 144 Prompts，432 rollouts |
| Test | 36 Prompts，108 rollouts |
| 任务类型 | QA、Summarization、Code |
| 长度条件 | Short、Medium、Long |

## 3. 总体结果

### 3.1 Prompt-mean点预测

| Split | Prompts | Actual mean | Predicted mean | Bias | MAE | RMSE | Raw R² | Log R² | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 144 | 415.06 | 415.58 | +0.53 | 1.83 | 2.83 | 0.9999 | 0.9999 | 1.0000 |
| Test | 36 | 392.98 | 411.77 | +18.79 | 58.01 | 86.14 | 0.9050 | 0.9354 | 0.9680 |

Test Prompt-mean的Raw R²为0.9050、Log R²为0.9354、Pearson相关为0.9680，说明ALPS
能够较强地预测未见Prompt的总体平均输出长度和相对排序。

Train几乎被完美拟合，而Test MAE上升到58.01，表明当前Ridge存在明显的Train/Test
泛化差距。

### 3.2 Rollout-level点预测与概率结果

| Split | N | Bias | MAE | RMSE | Raw R² | Log R² | NLL | 95% Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 432 | +0.53 | 32.19 | 48.74 | 0.9754 | 0.9910 | 4.7692 | 94.7% |
| Test | 108 | +18.79 | 66.97 | 97.11 | 0.8823 | 0.9286 | 8.0641 | 63.9% |

Prompt-mean相对rollout-level使Test MAE从66.97下降到58.01，说明seed随机性贡献了部分
单次生成误差，但ALPS在未见Prompt上的误差仍然存在。Test Coverage只有63.9%，概率
区间明显过窄。

## 4. 按长度条件分析

以下为Test Prompt-mean结果。

| Condition | Prompts | Actual mean | Predicted mean | Bias | MAE | RMSE | Raw R² | Log R² | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 12 | 81.75 | 66.37 | −15.38 | 27.92 | 39.11 | 0.6081 | 0.6741 | 0.8216 |
| Medium | 12 | 476.28 | 512.98 | +36.70 | 71.38 | 104.74 | 0.7623 | 0.8549 | 0.9224 |
| Long | 12 | 620.92 | 655.96 | +35.04 | 74.73 | 98.79 | 0.6567 | 0.7656 | 0.9161 |

ALPS能够明显区分三种长度条件，但存在稳定偏差：

- Short平均低估15.38 tokens；
- Medium平均高估36.70 tokens；
- Long平均高估35.04 tokens。

Medium的R²最高；Short绝对MAE最小，但相对于自身长度的误差比例最大。

## 5. 按任务类型分析

以下为Test Prompt-mean结果。

| Task | Prompts | Actual mean | Predicted mean | Bias | MAE | RMSE | Raw R² | Log R² | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QA | 12 | 364.33 | 373.50 | +9.16 | 47.74 | 67.51 | 0.9246 | 0.9237 | 0.9648 |
| Summarization | 12 | 236.78 | 238.43 | +1.65 | 26.75 | 40.90 | 0.9328 | 0.9678 | 0.9659 |
| Code | 12 | 577.83 | 623.38 | +45.54 | 99.55 | 126.61 | 0.8210 | 0.8634 | 0.9598 |

Summarization预测最好，QA次之，Code误差最大。三类任务的Pearson相关均约为0.96，说明
跨长度条件的总体排序较强；Code的主要问题是绝对误差较大并且存在系统性高估。

## 6. 九宫格结果

每个Test单元包含4个独立Prompt。以下为三个seed求均值后的Prompt-mean点预测。

| Cell | Prompts | Actual mean | Predicted mean | Bias | MAE | RMSE | Raw R² | Log R² | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QA / Short | 4 | 34.33 | 31.01 | −3.32 | 19.55 | 25.02 | −0.2025 | −0.2560 | −0.9962 |
| QA / Medium | 4 | 445.33 | 459.98 | +14.65 | 77.43 | 101.23 | −2.6870 | −2.2522 | −0.9337 |
| QA / Long | 4 | 613.33 | 629.49 | +16.16 | 46.22 | 52.90 | −4.0411 | −3.8736 | −0.3690 |
| Summarization / Short | 4 | 48.25 | 41.77 | −6.48 | 9.05 | 9.58 | −1.1628 | −1.1063 | −0.5143 |
| Summarization / Medium | 4 | 235.83 | 262.46 | +26.62 | 28.03 | 35.59 | −2.4089 | −2.2865 | −0.8412 |
| Summarization / Long | 4 | 426.25 | 411.06 | −15.19 | 43.17 | 60.51 | −0.3150 | −0.2711 | 0.0722 |
| Code / Short | 4 | 162.67 | 126.32 | −36.35 | 55.17 | 62.22 | −2.1457 | −3.1344 | −0.1426 |
| Code / Medium | 4 | 747.67 | 816.49 | +68.82 | 108.68 | 146.29 | −6.3445 | −4.1111 | 0.7971 |
| Code / Long | 4 | 823.17 | 927.32 | +104.16 | 134.79 | 151.05 | −6.4314 | −5.6151 | −0.4560 |

九个单元的Prompt-mean R²均小于0。该结果说明，当前ALPS的总体高R²主要来自任务类型和
长度条件之间的大尺度差异；在同一任务、同一长度条件内部，尚未表现出稳定的细粒度Prompt
预测能力。

每个单元只有4个Prompt，R²和相关系数对单个样本非常敏感，因此不能把具体数值视为稳定
估计。但九个单元方向一致，说明组内泛化是当前ALPS v1需要重点验证的问题。

## 7. 长度变化感知

同一Prompt family包含Short、Medium、Long三个版本。

| 指标 | All | QA | Summarization | Code |
|---|---:|---:|---:|---:|
| Test实际长度严格递增率 | 91.7% | 100.0% | 100.0% | 75.0% |
| ALPS预测长度严格递增率 | 91.7% | 100.0% | 100.0% | 75.0% |

ALPS在11/12个Test family中正确反映了Short < Medium < Long的整体方向。

| Contrast | Actual Δ | Predicted Δ | Δ Bias | Δ MAE | Δ RMSE |
|---|---:|---:|---:|---:|---:|
| Short→Medium | 394.53 | 446.61 | +52.08 | 75.68 | 101.00 |
| Medium→Long | 144.64 | 142.98 | −1.66 | 77.31 | 98.46 |
| Short→Long | 539.17 | 589.59 | +50.42 | 81.12 | 117.22 |

ALPS对“是否变长”的方向判断较好，但对“变长多少”的预测仍存在明显误差。

## 8. Seed与概率分布

同一Prompt跨seed的ALPS预测跨度为0，因为ALPS在生成前读取固定的预填充隐藏状态。Test
实际长度跨seed的平均标准差为32.69 tokens。

| 分组 | 95% Coverage |
|---|---:|
| Overall | 63.9% |
| Short | 30.6% |
| Medium | 83.3% |
| Long | 77.8% |
| QA | 63.9% |
| Summarization | 72.2% |
| Code | 55.6% |

当前ALPS概率分布没有充分覆盖实际生成波动，其中Short和Code的区间校准最弱。

## 9. 结果分析

### 9.1 已验证的能力

1. **总体长度预测有效。**
   Test Prompt-mean Raw R²为0.9050、Log R²为0.9354，Pearson相关为0.9680。

2. **能够识别任务与长度条件带来的大尺度差异。**
   三种任务和三种长度条件的总体预测均保持较高相关性。

3. **能够感知同一family的长度变化方向。**
   11/12个Test family的预测满足Short < Medium < Long。

4. **Summarization和QA表现较好。**
   两类任务的Prompt-mean Raw R²均超过0.92。

### 9.2 当前不足

1. **Train/Test差距明显。**
   Train Prompt-mean MAE仅1.83，Test为58.01，当前Ridge存在明显过拟合迹象。

2. **小单元内预测能力不足。**
   九宫格Prompt-mean R²全部为负，当前总体性能主要来自跨组差异。

3. **Code任务误差较大。**
   Code Prompt-mean MAE为99.55，并平均高估45.54 tokens。

4. **概率区间欠校准。**
   Test 95% Coverage仅为63.9%，当前概率分布过度自信。

### 9.3 结论

> ALPS v1能够利用Layer 14隐藏状态预测未见Prompt的总体输出长度尺度，并较好识别任务和
> 长度条件引起的变化；但当前优势主要体现在组间区分，同组内部的细粒度泛化尚未得到
> 支持。Ridge在Train Prompt均值上接近完美拟合，Test误差明显上升，同时概率区间覆盖
> 不足。因此，ALPS v1可以作为有效的静态长度预测基线，但仍需要对论文派生的固定配置
> 进行family-grouped泛化验证，并改进概率校准。

## 10. 后续重点

1. 在Train内部按`prompt_family_id`进行五折交叉验证，固定Layer 14和`alpha=1.0`，
   只检查泛化能力并比较baseline，不据此修改v1。
2. 增加每个任务—长度单元的独立Prompt数量，稳定估计组内能力。
3. 使用out-of-fold残差校准ALPS概率分布。
4. 保留当前ALPS v1结果；任何新配置使用新的未见holdout进行最终评价。

## 11. 结果文件

```text
artifacts/runs/alps_v1/stage1/train_breakdown.{json,csv,md}
artifacts/runs/alps_v1/stage1/test_breakdown.{json,csv,md}
artifacts/runs/alps_v1/stage1/train_prompt_mean_breakdown.csv
artifacts/runs/alps_v1/stage1/test_prompt_mean_breakdown.csv
artifacts/runs/alps_v1/stage1/train_length_contrasts.csv
artifacts/runs/alps_v1/stage1/test_length_contrasts.csv
```

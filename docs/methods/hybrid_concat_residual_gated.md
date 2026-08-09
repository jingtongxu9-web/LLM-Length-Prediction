# ALPS + PLP Hybrid 技术原理：concat v1、residual v2 与 gated v2.1

## 1. 文档范围

本文只解释三个 Hybrid 算法的技术结构，不报告实验结果：

1. `alps_plp_concat_v1`：把 ALPS 摘要作为额外特征交给 PLP；
2. `alps_plp_residual_v2`：以 ALPS countdown 为基线，让 PLP 学习有符号修正；
3. `alps_plp_gated_residual_v2_1`：限制修正幅度，并用进度与 gate 控制是否采用修正。

这里的 v1/v2/v2.1 是 **Hybrid 方法版本**。历史名称 `alps_plp_hybrid_v3` 是一套包含多种候选
方法的实验协议，不是第四个 Hybrid 模型。三个版本复用同一批 Qwen trace，不重新生成回答，
也不训练 Qwen 权重。

## 2. 两个基础信息源

### 2.1 ALPS：生成前的全局路线

ALPS 根据 Prompt 的 Layer-14 表征预测最终输出总长度 `T_ALPS`。已经生成 `t` 个 token 时，
ALPS countdown 为：

```text
A_t = max(0, T_ALPS - t)
```

它的优点是生成前即可得到；局限是生成开始后不会根据实际回答内容更新。

### 2.2 PLP：生成中的当前状态

PLP 在每个记录点使用：

- 3584 维 entropy-pooled Prompt hidden state `h_prompt`；
- 3584 维当前 causal decode hidden state `h_decode_t`。

二者拼成 7168 维状态。PLP 能读取实际生成路径，但 PLP-only 没有显式的 ALPS 总长度先验。

## 3. concat v1：自由特征融合

### 3.1 输入

ALPS 在每个 step 提供五个摘要：

1. `mu_log_total`：log 空间总长度位置预测；
2. `residual_variance`：ALPS 训练残差方差；
3. `mean_total_tokens`：token 空间的总长度均值；
4. `mean_remaining_countdown`：均值总长度减去当前 step；
5. `median_remaining_countdown`：中位数总长度减去当前 step。

五维摘要只能用相应训练折的统计量标准化，再与 PLP 状态拼接：

```text
x_v1 = [h_prompt, h_decode_t, standardized(ALPS summaries)]
dim(x_v1) = 3584 + 3584 + 5 = 7173
```

### 3.2 Prediction head

```text
7173-dimensional input
  -> Linear(7173, 1024)
  -> LayerNorm(1024)
  -> ReLU
  -> Dropout(0.1)
  -> 20-bin remaining-length distribution
```

20 个 bin 的概率期望转换为具体剩余 token 数，terminal zero 使用独立零长度表达。

### 3.3 设计含义

concat 不规定 ALPS 和 PLP 谁是主、谁是辅。网络可以在早期更多依赖 ALPS，在生成中后期更多
依赖 decode state，也可以学习二者的非线性交互。代价是最终预测不能直接拆解成“ALPS 原值 +
多少修正”。

## 4. residual v2：ALPS 基线加动态修正

### 4.1 核心公式

```text
delta_t = g_theta(h_prompt, h_decode_t, controls_t)
R_hat_t = max(0, A_t + delta_t)
```

`A_t` 是 ALPS countdown，`delta_t` 是 PLP 学到的正负修正。如果动态状态没有提供可靠信息，
理想行为是 `delta_t` 接近 0，使模型退回纯 ALPS。

### 4.2 输入

在 7168 维 PLP 状态之外加入六个标准化控制量：

```text
[A_t, step, entropy, entropy_mean, entropy_slope, eos_probability]
```

总维度为 `7174`。其中 `A_t` 不仅是网络输入，还在网络外直接加到最终预测上，因此不会因只有
一维而被高维 hidden state 稀释。

### 4.3 Prediction head 与目标

```text
7174-dimensional input
  -> Linear(7174, 1024)
  -> LayerNorm(1024)
  -> ReLU
  -> Dropout(0.1)
  -> correction scalar + terminal logit
```

非终点监督为：

```text
delta_true = true_remaining - A_t
```

残差使用 Smooth-L1，terminal 分支使用类别平衡 BCE。冻结组合为 80% 残差损失和 20% terminal
损失。修正输出层以零初始化，使训练开始时模型等同于纯 ALPS。每条 rollout 的总训练权重相同，
避免长回答因 step 更多而支配训练。

## 5. gated residual v2.1：保守地决定是否修正

v2.1 不覆盖 v2，而是一个独立消融。它保留同样的 7174 维输入，但把 correction hidden layer
缩小为 512，并增加三层限制。

### 5.1 有界候选修正

```text
B_t = max(16, 0.5 * (A_t + 1))
bounded_delta_t = B_t * tanh(raw_delta_t)
```

`tanh` 保证修正不会无限增大；允许范围随 ALPS 当前剩余长度变化。

### 5.2 进度约束与 learned gate

```text
progress_t = step / max(step + A_t, 1)
gate_t = progress_t * sigmoid(gate_logit_t)
applied_delta_t = gate_t * bounded_delta_t
R_hat_t = max(0, A_t + applied_delta_t)
```

`sigmoid(gate_logit_t)` 是网络主动学习的信任程度，`progress_t` 是硬约束；因此实际 gate 永远
不超过生成进度。gate bias 初始化为 `-3`，使训练初期只允许很小修正。

### 5.3 损失与 terminal 分支

```text
normalized Smooth-L1 final-prediction loss
+ 0.05 * correction magnitude penalty
+ 0.01 * gate usage penalty
+ 0.10 * class-balanced terminal BCE
```

terminal 分类不共享 correction backbone，只读取六个控制量的独立线性分支，避免容易学习的 EOS
判断干扰较弱的 residual 信号。weight decay 固定为 `1e-4`。

## 6. 为什么需要 grouped cross-fitting

ALPS 本身也是从 Train family 训练出来的。如果用全量 Train ALPS 对同一批样本产生特征，再训练
Hybrid，第二层会看到过度乐观的先验，形成 stacking leakage。

项目采用两层分组交叉拟合：

1. 外层五折按完整 `prompt_family_id` 留出验证 family；
2. 每个外层训练折内部再交叉拟合 ALPS，产生该训练折的无泄漏 ALPS 特征；
3. Hybrid 只使用这些 cross-fitted 特征训练；
4. 外层验证 family 由只见过外层训练 family 的 ALPS 和 Hybrid 共同预测。

同一 family 的三个长度、三个 seed 和所有 step 始终在同一折。

## 7. 公平比较与诊断

三个 Hybrid 版本必须使用完全相同的 family folds、trace、step 和主指标。核心对照还包括纯
`alps_countdown` 与纯 `plp_terminal_zero_v3`。主指标为 family-macro、rollout-balanced MAE。

Residual 类版本还应报告：

- 候选修正与实际施加修正的分布；
- 修正方向一致率和修正成功率；
- gate 在不同生成进度的使用率；
- gate 与逐点 MAE 改善的相关性；
- terminal precision、recall 与 F1；
- 按任务、预设长度、3×3 单元、seed 和 fold 的稳定性。

这些诊断用来回答 gate 是否真正学会“何时相信动态修正”，而不只是频繁改写 ALPS。

## 8. OOF 与全量训练的区别

- 五折 OOF：每折训练一个临时模型，只用于无泄漏方法比较与选择；
- 全量 Train：选择方法后，用全部 Train family 拟合一个可保存模型；
- 最终 Test：只能使用从未参与结构选择的新 holdout。

五个 OOF 模型不会平均成最终模型；最终模型也不能用自己的 Train 指标代替 OOF 证据。

## 9. 代码与配置

| 内容 | 位置 |
|---|---|
| concat v1 / residual v2 模型 | `src/llm_length_prediction/models/hybrid_versions.py` |
| v1/v2 OOF | `scripts/evaluate_hybrid_versions_oof.py` |
| v1/v2 全量训练 | `scripts/train_hybrid_versions.py` |
| gated v2.1 OOF | `scripts/evaluate_gated_residual_v2_1_oof.py` |
| gated v2.1 全量训练 gate | `scripts/train_gated_residual_v2_1.py` |
| v1/v2 合同 | `configs/experiments/alps_plp_hybrid_versions.json` |
| v2.1 合同 | `configs/experiments/alps_plp_gated_residual_v2_1.json` |

实际实验数字与选型结论见
[`../results/hybrid/hybrid_v1_v2_v2_1_results.md`](../results/hybrid/hybrid_v1_v2_v2_1_results.md)。

# ALPS + PLP Hybrid v1 / v2 技术原理

## 1. 文档范围

本文只解释当前项目中两个明确的 Hybrid 算法：

- `alps_plp_concat_v1`：把 ALPS 信息作为额外特征交给 PLP；
- `alps_plp_residual_v2`：先用 ALPS 给出剩余长度基线，再让 PLP 只学习修正量。

这里的 v1/v2 是 **Hybrid 方法版本**。仓库中旧名称 `alps_plp_hybrid_v3` 是一套历史实验协议，
其中包含十种方法；二者不是同一层级的版本号。旧协议和旧结果保持不变。

两个版本复用同一批 Qwen 统一 trace，不需要重新生成回答。它们都只在 Train family 上做分组
OOF 开发比较；PLP-only 已经使用过的旧 Test 不能再被称为 Hybrid 的独立 Test。

## 2. 两个基础预测器

### 2.1 ALPS：出发前给出全程估计

ALPS 在生成开始前读取 Prompt 的 Layer-14 表征，并预测最终输出总长度。设其总长度均值预测为
`T_ALPS`，已经生成 `t` 个 token，则 ALPS 的朴素剩余长度倒计时为：

```text
A_t = max(0, T_ALPS - t)
```

`A_t` 有明确含义：如果生成过程没有提供任何新证据，目前还估计剩余多少 token。

### 2.2 PLP：行驶中读取当前状态

PLP 在每个记录点使用：

- 3584 维 Prompt 汇总隐藏状态；
- 3584 维当前 decode 隐藏状态。

二者拼成 7168 维状态向量。它能够看到生成过程已经走到哪里，但 PLP-only 没有一个显式的
ALPS 总长度先验。

## 3. Hybrid v1：特征拼接

### 3.1 输入

ALPS 对每个生成点提供五个摘要：

1. `mu_log_total`：对数总长度的位置预测；
2. `residual_variance`：ALPS 训练残差方差；
3. `mean_total_tokens`：转换回 token 空间的总长度均值；
4. `mean_remaining_countdown`：均值总长度减去当前 step；
5. `median_remaining_countdown`：中位数总长度减去当前 step。

这五维只在 Train 上标准化，再与 7168 维 PLP 状态拼接：

```text
x_v1 = [h_prompt, h_decode_t, standardized(ALPS summaries)]
dim(x_v1) = 3584 + 3584 + 5 = 7173
```

### 3.2 输出

`x_v1` 进入一个 1024 隐层的渐进预测头，输出 20 个剩余长度 bin 的概率，并用概率期望得到
一个具体剩余 token 数。零长度有独立 terminal bin。

### 3.3 优点与限制

优点是网络能自由学习 ALPS 和 PLP 信息的任意组合；旧 OOF 中已经证明这条路线有效。限制是
ALPS 的五维信息可能被当成普通特征忽略，输出也不能直接解释为“ALPS 基线被修正了多少”。

## 4. Hybrid v2：ALPS 基线 + PLP 残差修正

### 4.1 核心公式

v2 不让网络从零重新预测剩余长度，而是预测 ALPS 当前误差：

```text
delta_t = g_theta(h_prompt, h_decode_t, controls_t)
R_hat_t = max(0, A_t + delta_t)
```

其中：

- `A_t` 是 ALPS 剩余长度倒计时；
- `delta_t` 是 PLP 学到的有正负号修正；
- `R_hat_t` 是最终剩余长度预测。

如果 PLP 没学到可靠的新信息，`delta_t` 接近 0，模型自然退回 ALPS，而不是输出一个任意值。

### 4.2 v2 输入

除 7168 维 PLP 状态外，v2 加入六个标准化控制量：

```text
[A_t, step, entropy, entropy_mean, entropy_slope, eos_probability]
```

因此输入维度为：

```text
3584 + 3584 + 6 = 7174
```

这六维不会承担完整语义表示；它们告诉网络当前 ALPS 基线、生成进度、不确定性趋势和结束信号。
更重要的是，`A_t` 还在网络外直接加到最终输出，因此不会因为“只有一维”而被高维隐藏状态稀释。

### 4.3 残差头骨架

```text
7174-dimensional input
  -> Linear(7174, 1024)
  -> LayerNorm(1024)
  -> ReLU
  -> Dropout(0.1)
  -> correction scalar
  -> terminal logit
```

两个输出分别负责：

- `correction scalar`：还应该在 ALPS 倒计时上加减多少 token；
- `terminal logit`：当前是否已经到达真正的零剩余长度。

修正输出层以全零初始化，因此训练开始时 `delta_t = 0`，模型的初始行为就是纯 ALPS。

### 4.4 训练目标

对非终止点，监督信号为：

```text
delta_true = true_remaining - A_t
```

使用 Smooth-L1 损失学习这个残差。它在小误差附近像平方误差，在异常大误差处像绝对误差，
比直接用 MSE 更不容易被少数超长输出支配。

终止分支使用带类别平衡的二元交叉熵。推理时终止概率达到 0.5，就把剩余长度置为 0。两个损失
的冻结组合为 80% 残差损失和 20% 终止损失。

每条 rollout 的所有记录点权重之和相同。长回答虽然产生更多 step，但不会因此支配训练。

## 5. 为什么不能直接看 Train 指标

ALPS 本身也是从 Train family 学出的。若先用全部 Train 拟合 ALPS，再把它对同一批样本的预测交给
Hybrid，第二层会看到过于乐观的先验，形成 stacking leakage。

本项目使用两层分组交叉拟合：

1. 外层五折把完整 `prompt_family_id` 留作 OOF 验证；
2. 在每个外层训练折内部，再用四折产生该训练折的 ALPS cross-fitted 特征；
3. v1/v2 只用这些未见 family 的 ALPS 特征训练；
4. 外层验证 family 由只见过外层训练 family 的 ALPS 和 Hybrid 共同预测。

同一 family 的三种长度、三个 seed 和所有 step 永远位于同一折，避免近重复 Prompt 泄漏。

## 6. 比较设计

新 OOF 报告固定比较四种方法：

| 方法 | 回答的问题 |
|---|---|
| `alps_countdown` | 只靠生成前先验能做到什么程度？ |
| `plp_terminal_zero_v3` | 只靠生成中状态能做到什么程度？ |
| `alps_plp_concat_v1` | 自由拼接融合是否有效？ |
| `alps_plp_residual_v2` | 显式先验加动态修正是否更稳、更可解释？ |

主指标是 family-macro、rollout-balanced MAE，并报告五组配对 family bootstrap 区间。v2 还额外
报告修正成功率、平均修正量和终止判定率，用于判断网络是否真的在改善 ALPS，而不只是整体指标
偶然变好。

## 7. 文件与运行入口

| 文件 | 作用 |
|---|---|
| `src/llm_length_prediction/models/hybrid_versions.py` | v1/v2 训练、预测、保存和加载 |
| `configs/experiments/alps_plp_hybrid_versions.json` | 两个版本及 OOF 比较合同 |
| `scripts/evaluate_hybrid_versions_oof.py` | 五折 family-grouped OOF，不读取 Test |
| `scripts/train_hybrid_versions.py` | OOF 完成后，在全部设计 Train 上拟合可保存模型 |

运行顺序：

```bash
python scripts/evaluate_hybrid_versions_oof.py --device auto
python scripts/train_hybrid_versions.py --device auto
```

第二条命令只训练最终模型，不会产生新的独立证据。方法选择必须依据第一条命令的 OOF 报告；
最终确认性结论仍需要另建一个从未看过的新 Hybrid holdout。

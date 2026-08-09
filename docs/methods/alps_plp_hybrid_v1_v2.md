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

## 8. Residual v2 的 OOF 结果与 v2.1 修正

原始 `alps_plp_residual_v2` 必须保留，不能用新实现覆盖。它在 60-family OOF 中的
family-macro MAE 为 57.93，而 concat v1 为 49.92；v2-minus-v1 的 99% familywise CI 为
`[2.24, 14.94]`，说明当前 v2 明确更差。其平均修正为 `+5.75` token，但 ALPS 实际平均只需
`-0.41` token 修正，修正成功率仅 43.47%。

这并不能否定 residual 结构本身，因为 v2 同时改变了融合方式、输入、损失和 terminal 结构。
因此新增开发方法 `alps_plp_gated_residual_v2_1`，而不是篡改 v2。

### 8.1 v2.1 的保守更新

令 ALPS 倒计时仍为 `A_t`。网络先产生有界候选修正：

```text
bounded_delta_t = B_t * tanh(raw_delta_t)
```

其中：

```text
B_t = max(16, 0.5 * (A_t + 1))
```

再计算生成进度和信任门：

```text
progress_t = step / max(step + A_t, 1)
gate_t = progress_t * sigmoid(gate_logit_t)
```

最终预测为：

```text
applied_delta_t = gate_t * bounded_delta_t
R_hat_t = max(0, A_t + applied_delta_t)
```

这个结构有三个安全边界：

1. `tanh` 限制候选修正幅度；
2. `gate_t <= progress_t`，生成早期不能过度改写 ALPS；
3. gate bias 固定初始化为 -3，训练开始时只允许很小的动态修正。

### 8.2 模型与损失

v2.1 复用 v2 的 7174 维输入，但把隐层从 1024 缩小为 512，并使用 `weight_decay=1e-4`。
terminal 分类不再共享 correction backbone，而是只读取最后六个标准化控制量的独立线性分支，
避免已经很容易学会的 EOS 判断干扰弱 residual 信号。

训练损失由四部分组成：

```text
normalized Smooth-L1 final-prediction loss
+ 0.05 * correction magnitude penalty
+ 0.01 * gate usage penalty
+ 0.10 * class-balanced terminal BCE
```

这些惩罚表达一个明确先验：没有稳定动态证据时，保持 ALPS 比主动修正更安全。

### 8.3 增量 OOF

v2.1 必须使用与原 v1/v2 完全相同的五个 family folds，但无需重新训练旧控制模型。
增量脚本会验证原 OOF 的数据 digest、family-fold 映射、方法列和每个逐点键，然后只训练五个
v2.1 折模型。同时拟合一个计算成本很低的 `alps_scalar_residual_ridge`：它只用 ALPS
countdown、step、entropy、entropy mean/slope 和 EOS probability 六个标量预测有符号残差。
这个诊断用于区分“残差信号本身不可预测”和“高维 MLP 训练方式不合适”。

运行命令：

```bash
python scripts/evaluate_gated_residual_v2_1_oof.py --device auto
```

结果写入：

```text
artifacts/runs/alps_plp_gated_residual_v2_1/oof/
```

报告不只给总体 MAE。`gated_correction_diagnostics` 专门检验 gate 是否真的学会了“何时信任
动态修正”，包含：

- learned gate confidence 与乘上 progress 后 effective gate 的加权分位数，以及
  `0.05/0.10/0.25/0.50` 阈值使用率；
- 按生成进度、gate 强度、任务、预设长度、3×3 任务-长度单元、terminal 状态和 outer fold
  分组的指标；
- v2.1 相对 ALPS 和 concat v1 的逐点 MAE 改善、修正方向一致率和修正成功率；
- gate 与改善幅度的加权相关性，以及修正幅度与真实所需修正幅度的相关性；
- 候选修正是否频繁撞到幅度上限；
- terminal 分类的 TP、FP、FN、TN、precision、recall 和 F1。

learned gate confidence 是网络主动输出的信任程度，effective gate 才是实际施加修正的比例；
后者还受到生成进度的硬约束。这里“gate 使用率”不能单独证明模型有效。平均 gate 很大可能
只是网络频繁改写 ALPS；只有
高-gate 区间同时产生正的 MAE improvement，并且在不同 fold 与 3×3 单元中保持一致，才说明
gate 提供了有用的选择机制。逐点明细 CSV 还保存 progress、correction bound、相对两种控制
方法的改善量、修正是否成功以及是否触碰边界，便于后续画图和复核。

只有当 v2.1-minus-concat-v1 的 familywise CI 上界严格小于 0，最终全量训练入口才会放行：

```bash
python scripts/train_gated_residual_v2_1.py --device auto
```

否则脚本会停止并要求继续保留 concat v1。

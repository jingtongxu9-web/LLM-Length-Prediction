# Prompt-token Ridge Baseline 原理

## 1. 它要回答什么

Prompt-token baseline 是整个项目的信息下限：只知道输入有多少个 token，能否预测输出会有
多少个 token。它不理解文本内容，也不读取 Qwen hidden state。

如果这个简单方法已经很强，那么 ALPS、PLP 或 Hybrid 的改善可能只是来自输入长度；如果它很弱，
复杂方法的收益才更可能来自语义和生成状态。

## 2. 输入与目标

每条 rollout 只有一个输入特征：

```text
x = 应用 Qwen 官方 chat template 后的 Prompt token 数
```

训练目标是总输出长度的对数：

```text
y = log1p(total_output_tokens)
```

使用 `log1p` 是因为输出长度右偏且存在长尾。模型预测后再用 `expm1` 还原为 token 数。

## 3. 为什么使用 Ridge

这里只有一个输入特征，不需要神经网络。Ridge 是带 L2 权重惩罚的线性回归：

```text
min ||y - Xw||² + alpha * ||w||²
```

当前 `alpha=1.0` 与项目的冻结线性基线保持一致。它没有 epoch、学习率或反向传播；参数由
线性代数一次求解。

## 4. 总长度怎样变成逐步剩余长度

静态比较直接使用预测总长度 `T_hat`。为了与 PLP 和 Hybrid 在第 `t` 个 decode step 公平比较：

```text
remaining_hat_t = max(T_hat - t, 0)
```

这称为 Prompt-token Ridge countdown。它不会在生成过程中获得新信息，只是把同一个总长度预测
按已经生成的 token 数机械递减。

## 5. 数据划分

不能按 rollout 随机打散，因为同一个 `prompt_family_id` 的 short/medium/long 和三个 seed 高度
相关。项目使用 family-grouped 五折：一个 family 的所有变体和 rollout 必须在同一折。

五折产生五个仅用于 OOF 诊断的 Ridge；完成方法选择后，再用全部 Train family 拟合一个最终
Ridge，供未来新 holdout 使用。五个折模型不会投票合并成最终模型。

## 6. 它不是什么

- 不是 ALPS：没有 Layer-14 hidden state；
- 不是 PLP：没有 decode state，也不会随生成内容更新；
- 不是 Metadata baseline：不输入任务类型或 short/medium/long 标签；
- 高 Coverage 不一定代表好，如果区间极宽，同样可以覆盖大量真实值。

结果见 [`../results/baseline/prompt_token_ridge_results.md`](../results/baseline/prompt_token_ridge_results.md)。

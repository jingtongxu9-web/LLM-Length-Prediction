# ALPS 原理与项目实现

## 1. 一句话理解

ALPS 在生成回答之前读取 Prompt 的内部语义表示，预测这次回答最终会生成多少个 token。它类似
导航出发前根据目的地和路线给出的全程时间估计。

## 2. 冻结的大模型

Qwen2.5-7B-Instruct 只负责产生 hidden state，LLM 权重和 tokenizer 完全冻结。项目没有微调
Qwen；真正被训练的只是一个很小的 Ridge 回归器。

冻结条件包括模型 revision、BF16、官方 chat template、temperature `0.7`、top-p `0.95`、
max new tokens `4096`、seed `42/43/44`、输出长度定义和 Layer 14。

## 3. ALPS 输入如何得到

格式化 Prompt 输入 Qwen 后，每个 token、每一层都有一个 hidden state。当前版本取：

```text
Layer 14
Prompt 最后一个 token
3584 维 hidden state
```

最后一个 Prompt token 的状态不是随机选的。因果 Transformer 中，它已经汇聚此前全部 Prompt
token 的上下文；Layer 14 则是当前冻结的中层语义位置。该层是实验合同，不在 Test 上重新选择。

每个 Prompt 得到一个 3584 维向量，不是每个生成 step 得到一个新 ALPS 向量。

## 4. Ridge 学习什么

训练标签是 rollout 最终输出 token 数 `T`，实际拟合：

```text
Layer-14 vector -> log1p(T)
```

Ridge 在普通线性回归的平方误差上增加 L2 权重惩罚，限制 3584 个系数整体过大。当前
`alpha=1.0` 已冻结。Ridge 由闭式/数值线性代数求解，没有神经网络 epoch 和学习率。

## 5. 推理与 countdown

生成前，Ridge 给出总长度预测 `T_hat`。当与逐步模型比较时，第 `t` 个 step 使用：

```text
ALPS_countdown_t = max(T_hat - t, 0)
```

这不会吸收生成中的新证据，只是把初始总长度估计随进度递减。因此 ALPS 擅长提供全局路线，
PLP 则用于读取当前行驶状态。

## 6. 点预测与概率区间

项目还根据 log 空间残差方差构造名义 95% 预测区间。95% 是模型的目标覆盖率，不是“预测值有
95% 概率正确”。若在大量未见样本中区间只覆盖约 64%–71% 的真实值，说明区间过窄、模型对
不确定性过度自信；这与点预测 MAE 是否优秀是两个不同问题。

## 7. 评价设计

- Train 指标用于检查拟合；
- family-grouped 五折 OOF 用于估计未见 family 的开发性能；
- Test 只用于冻结后的最终评价；
- 任务、预设长度和九宫格只用于分组诊断，不输入 ALPS 模型。

结果见 [`../results/alps/alps_v1_results.md`](../results/alps/alps_v1_results.md)，区间校准的后续方案见
[`../planning/alps_improvement_plan.md`](../planning/alps_improvement_plan.md)。

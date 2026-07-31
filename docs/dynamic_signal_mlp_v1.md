# Dynamic-Signal MLP v1（项目版 PLP）

## 版本身份

本方法是当前仓库可直接运行的动态剩余长度预测基线：

```text
正式名称：Dynamic-Signal MLP v1
内部标识：dynamic-signal-mlp-v1 / project_plp_only
实验角色：项目版 PLP、动态信号 MLP baseline
论文原版 PLP 复现：否
```

它研究的问题是：**在回答已经生成一部分后，能否利用当前生成进度、token 分布不确定性和
结束概率，预测还会生成多少 token？**

它借鉴了 PLP“在解码过程中反复预测剩余长度”的研究目标，但没有使用论文 PLP 的动态
hidden-state 表示和 soft-label length-bin 输出头。因此，报告和演示中应使用
“Dynamic-Signal MLP v1”或“项目版 PLP”，不能写成“论文 PLP 复现”。

## 为什么先保留这一版

ALPS v1 已经保存了每 5 个生成 token 的 entropy、entropy rolling statistics 和 EOS
probability。Dynamic-Signal MLP v1 可以直接复用这些 trace：

- 不重新下载或加载 Qwen；
- 不重新生成 540 条 rollout；
- 不读取 ALPS Layer-14 特征或 Ridge prior；
- 可以先验证简单动态信号是否包含剩余长度信息。

这使它适合作为当前 v1 的工程基线。论文复现需要重新保存逐 token hidden states，属于数据
合同和模型结构都不同的新实验，应作为 v2 单独冻结。

## 五个输入特征

在每个非终止 trace point 构造一个训练样本：

| 特征 | 当前实现中的定义 | 作用 |
|---|---|---|
| `step` | 已经生成的新 token 数 | 表示当前解码进度 |
| `entropy` | 当前 next-token 完整概率分布的熵 | 表示当前预测不确定性 |
| `entropy_mean` | 最近最多 20 个 token 的 entropy 均值 | 平滑单 token 波动 |
| `entropy_slope` | 最近窗口首尾 entropy 的平均变化率 | 表示不确定性的上升或下降趋势 |
| `eos_probability` | 当前完整 softmax 分布分配给 EOS token 的概率和 | 表示模型当前结束生成的倾向 |

`entropy` 和 `eos_probability` 来自 temperature 缩放后的完整 softmax，记录位置在 top-p
过滤之前。模型每 5 个生成 token 更新一次预测，终止点因为真实剩余长度为 0 且没有进一步
调度价值而不进入训练。

目标是：

\[
\log(1 + remaining\_tokens)
\]

模型输出经 shifted log-normal 均值变换后还原为预测的剩余 token 数。

## MLP 结构与参数量

冻结结构为：

```text
5 features -> Linear(5, 128) -> ReLU -> Dropout(0.1)
           -> Linear(128, 64) -> ReLU -> Dropout(0.1)
           -> Linear(64, 1)
```

可训练参数均为全连接层的权重和偏置：

\[
(5\times128+128)
+(128\times64+64)
+(64\times1+1)
=9089
\]

ReLU 和 Dropout 没有可训练参数。9089 只描述模型容量，不证明 MLP 一定优于线性模型；
当前结构是 v1 冻结的工程假设。后续若比较相同五特征的 Dynamic Ridge 或更小 MLP，必须
作为新的 Train-only 诊断或新实验版本，不能依据已经打开的 v1 Test 调参。

## 冻结条件

机器可读合同是
[`configs/experiments/plp_v1_manifest.json`](../configs/experiments/plp_v1_manifest.json)。

| 条件 | v1 固定值 |
|---|---|
| 基础 rollout | ALPS v1 的固定 Train/Test trace |
| Trace stride | 每 5 个生成 token |
| Entropy window | 20 |
| 输入 | 上述 5 个动态标量 |
| ALPS prior / hidden state | 不使用 |
| 目标 | `log1p(remaining_tokens)` |
| 隐藏层 | `[128, 64]` |
| Dropout | `0.1` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Epochs / batch size | `50` / `512` |
| Seed | `42` |
| 样本权重 | 每条 rollout 的所有非终止点总权重相同 |
| 超参数选择 | 无；不得根据 v1 Test 调参 |

同一 rollout 内的 timestep 高度相关，不能把大量 trace point 当作同等数量的独立序列。
sequence-balanced loss 保证长回答不会仅因包含更多采样点而在训练损失中占更大总权重，但
它不会创造新的独立 prompt family。

## 运行

以下命令只读取已有 trace，不会加载 Qwen：

```bash
python scripts/train_dynamic.py
python scripts/evaluate_dynamic.py --split train
python scripts/evaluate_dynamic.py --split test --confirm-final-test
```

默认输出：

```text
artifacts/runs/alps_v1/comparisons/plp_only/
├── model.json
├── training_report.json
├── train_evaluation.json
├── train_evaluation.csv
├── test_evaluation.json
└── test_evaluation.csv
```

评价报告包含总体剩余长度误差以及按 decode progress 分组的误差。现有 v1 Test 已经在
ALPS 分析中打开，因此本方法的 v1 Test 比较属于事后诊断；不能根据这些结果继续调整结构、
学习率或特征后再次宣称是独立最终测试。

## 与论文 PLP 的边界

论文 PLP 在第 \(t\) 步组合 Prompt 表示和已经生成 token 的 hidden states，并使用与论文
静态方法相同的 soft-label length-bin prediction head。论文实验还采用 20 个 length bins
和 CE + MSE 联合损失。当前 v1 则只使用五个标量动态信号，由 MLP 直接回归
`log1p(remaining_tokens)`；两者输入、输出头和损失函数均不同。

论文来源：

- [Predicting LLM Output Length via Entropy-Guided Representations，Sections 3.2–3.3](https://arxiv.org/html/2602.11812v2)
- [论文公开仓库 LP_Bench](https://github.com/xiehuanyi/LP_Bench)

未来 v2 若决定复现论文版，需要重新设计并冻结：

1. Prompt 与生成 token hidden-state 的采集格式；
2. 动态表示的聚合和存储方案；
3. 20-bin soft-label prediction head；
4. CE + MSE 联合损失；
5. 独立的 Train/Validation/Test 或新 holdout；
6. 与 Dynamic-Signal MLP v1 和 ALPS v1 的公平比较。

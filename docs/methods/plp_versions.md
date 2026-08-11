# PLP 版本谱系与技术逻辑

> 本文记录三个历史 PLP baseline 的版本关系。它们均不接收上一时刻 posterior，因此不等同于
> 项目权威 PDF 所要求的 Bayesian sequential inference。项目 proposed method 见
> [`bayesian_sequential_inference.md`](bayesian_sequential_inference.md)。

## 1. 为什么有三个版本

项目历史上出现过三个都带“PLP”含义的版本，但它们不是同一个模型反复改名。必须在论文中明确
区分输入信息和研究目的。

| 版本 | 输入 | 预测器 | 研究作用 |
|---|---|---|---|
| Dynamic-Signal MLP v1 | step、entropy、entropy mean/slope、EOS probability | 小型回归 MLP | 早期人工动态信号 baseline |
| Hidden-State PLP v2 | 3584 维 Prompt 表征 + 3584 维 decode state | 20-bin progressive MLP | 当前论文对齐、非精确复现的 PLP 主体 |
| Terminal-Zero PLP v3 | v2 输入 + 独立终点零机制 | progressive MLP | 修正剩余长度为 0 时仍输出正数的问题 |

## 2. Dynamic-Signal MLP v1

v1 没有直接读取 hidden state，只使用五个低维、人工挑选的生成信号。它验证 entropy、EOS 概率
和生成进度是否已经足够预测剩余长度。结果显示信号在中段有一定作用，但早期和后期误差较大，
所以它只保留为 baseline，不代表真正的 Hidden-State PLP。

## 3. Hidden-State PLP v2

v2 对 Prompt 内每个 token 的最后层 hidden state 按 entropy softmax 加权，得到一个 3584 维
`h_prompt`。生成过程中，每个保存 step 读取当前 token 的 3584 维 causal decode state
`h'_t`。二者拼接为 7168 维：

```text
[h_prompt ; h'_t] -> MLP -> 20 个剩余长度区间概率
```

20-bin 输出再取区间代表值的概率期望，得到一个具体的剩余 token 数。Qwen 仍完全冻结，只训练
prediction head。

## 4. Terminal-Zero PLP v3

v2 的长度 bins 主要表示正剩余长度，终点附近可能持续输出小正数。v3 增加明确的 zero/terminal
机制，使真实剩余长度为 0 成为可直接表达的状态。

v3 选择前还进行两个对照：缩小 MLP head，以及 rollout-balanced target-range weighting。
前者明显退化，后者没有稳定收益，因此最终只冻结 terminal-zero 改动。

## 5. 为什么仍使用 MLP

PLP 输入是两组高维 hidden state，Prompt 语义、当前生成状态与剩余长度之间很可能存在非线性
交互；而输出又是 20-bin 分布，不是单一连续值。因此使用含 Linear、LayerNorm、ReLU 和
Dropout 的 MLP prediction head。它不是训练 Qwen，只是训练约百 MB 的外接预测头。

## 6. 与 ALPS 的根本区别

- ALPS：生成前一次性预测总长度，输入 Layer-14 Prompt 表征，Ridge。
- PLP：生成中反复预测剩余长度，输入 Prompt 与当前 decode hidden state，MLP。
- Hybrid：同时利用 ALPS 全局先验和 PLP 动态状态。

完整逐步原理见 [`plp_only_explained.md`](plp_only_explained.md)，结果索引见
[`../results/plp/README.md`](../results/plp/README.md)。

# PLP-only 原理：把输出长度预测理解成“导航剩余时间”

本文面向没有机器学习背景的读者，解释本项目 Hidden-State PLP v2 的基本原理。重点回答：

- 一个 Prompt 为什么会产生很多个 3584 维隐藏状态？
- entropy 是什么，为什么能用于给 Prompt token 加权？
- 一个 Prompt 最后会得到几个 `h_prompt`？
- 生成过程中的 `h'_t` 是什么？
- PLP 为什么不显式输入已经生成的 token 数？
- 20 个长度区间和 soft label 怎样变成一个具体的 token 预测值？
- PLP 与 DDPM、ALPS、Dynamic-Signal MLP 有什么区别？
- 为什么必须重新让 Qwen 跑一遍 Prompt？

## 1. 一句话理解 PLP

PLP（Progressive Length Prediction）在回答生成过程中不断预测：

> **从现在开始，Qwen 还会继续生成多少个 token？**

假设某次回答最终有 `300` 个 token，那么：

| 已生成 token 数 `t` | 真实剩余长度 `R_t = T - t` |
|---:|---:|
| 1 | 299 |
| 5 | 295 |
| 50 | 250 |
| 200 | 100 |
| 295 | 5 |
| 300 | 0 |

PLP 不是只在生成前猜一次总长度，而是随着回答发展持续更新预测。

## 2. 导航类比

把一次回答生成想成一次开车导航：

| 导航系统 | PLP |
|---|---|
| 出发前输入目的地和路线要求 | 用户输入 Prompt |
| 路线本身的复杂程度 | Prompt 表征 `h_prompt` |
| 车辆当前的位置、方向和路况 | 当前解码隐藏状态 `h'_t` |
| 已经行驶的距离 | 已经生成的 token 数 `t` |
| 预计还要行驶多久 | 预测剩余 token 数 `R_t` |
| 行驶过程中不断更新预计到达时间 | 每隔若干 token 更新剩余长度预测 |

这个类比里有两个核心信息：

1. **路线信息**：任务本身可能要求短答、长篇分析或复杂代码；
2. **当前行驶状态**：回答可能刚开始、正在展开、准备总结或已经接近结束。

PLP 将这两类信息结合起来预测剩余长度。

```text
一个 Prompt（n 个 token）
        |
        v
最终层 token states（n × 3584）
        |
        v
按每个 Prompt token 的 entropy 加权
        |
        v
h_prompt（3584维，整个 Prompt 只有一个）
        |
        +--------------------------+
                                   |
已生成 y1...yt                    |
        |                          |
        v                          v
h'_t（3584维） -------------> 拼接（7168维）
                                   |
                                   v
                           PLP prediction head
                                   |
                                   v
                           20个长度区间概率
                                   |
                                   v
                           预计剩余 token 数
```

## 3. 先认识几个基础概念

### 3.1 Token

Qwen 不直接按“一个汉字”或“一个单词”处理文本，而是先把文本切成 token。

例如下面只是一个概念性示意：

```text
请解释量子纠缠
        ↓
[请] [解释] [量子] [纠缠]
```

真实 tokenizer 的切分结果可能不同。输出长度也是按新生成的 token 数计算，而不是按字数或
句子数计算。

### 3.2 Hidden state

hidden state 是 Qwen 在处理某个位置时形成的内部数字表示。对于 Qwen2.5-7B-Instruct，
本项目使用的隐藏向量维度是 `3584`。

一个 3584 维向量可以写成：

```text
[0.12, -0.38, 1.07, ..., 0.24]
```

人不能直接为每一维命名，但整个向量会编码语义、结构、位置、上下文和生成状态等信息。

### 3.3 Rollout

一次 `Prompt + seed` 的完整随机生成叫一个 rollout。同一个 Prompt 使用不同 seed，可能得到
不同文本和不同输出长度。

### 3.4 Remaining length

如果回答最终长度为 `T`，当前已经生成 `t` 个 token，那么真实剩余长度是：

```text
R_t = T - t
```

PLP 的学习目标就是预测这个 `R_t`。

## 4. 第一步：把整个 Prompt 压缩成一个 `h_prompt`

### 4.1 一个 Prompt 会先产生多少个隐藏状态？

假设一个格式化后的 Prompt 有 `100` 个 token：

```text
x_1, x_2, ..., x_100
```

Qwen 最终 Transformer 层会为每一个位置产生一个 3584 维隐藏状态：

```text
h_1, h_2, ..., h_100
```

因此原始 Prompt hidden states 的形状是：

```text
100 × 3584
```

可以理解为：

```text
Prompt token x₁   → h₁，3584维
Prompt token x₂   → h₂，3584维
...
Prompt token x₁₀₀ → h₁₀₀，3584维
```

这里的 `h_i` 不是孤立的词向量。由于 Qwen 使用因果注意力，它表示模型读到当前位置时，对
此前 Prompt 前缀的内部理解：

```text
h_i = f(x_1, x_2, ..., x_i)
```

### 4.2 为什么不能直接保留全部隐藏状态？

不同 Prompt 的 token 数不同：

```text
Prompt A：50 × 3584
Prompt B：100 × 3584
Prompt C：500 × 3584
```

而后面的固定结构 MLP 需要固定输入尺寸。所以必须把一个 Prompt 的多个 token hidden
states 压缩成一个固定长度向量。

最简单的方法是直接平均，但这等于假设所有 token 同样重要。论文采用 entropy-guided
pooling，让模型更关注可能携带重要长度信息的位置。

## 5. Entropy 是什么？

### 5.1 它衡量“模型对下一个 token 有多不确定”

Qwen 处理到 Prompt 的某个位置后，会预测下一个 token 的概率。

这里发生在 **Prompt 输入（prefill）阶段**。Prompt 虽然是一次性送入模型的，但因果语言模型
仍会在每个位置形成“如果从这里继续，下一个 token 可能是什么”的概率分布：

```text
Prompt位置 i 的隐藏状态 h_i
        ↓ 经过 Qwen 自己冻结的输出层（LM head）
词表中每个候选 token 的 logits
        ↓ softmax
概率分布 p_i(v)
        ↓ 计算 entropy
一个标量 H_i
```

因此，每个 Prompt token 位置都有两类不同的量：

| 符号 | 含义 | 形状 |
|---|---|---:|
| `h_i` | 第 `i` 个位置的隐藏状态 | 3584 维向量 |
| `H_i` | 该位置对下一个 token 的预测 entropy | 1 个标量 |

大写 `H_i` 和小写 `h_i` 不是同一个东西。更准确地说，Qwen 会用 `h_i` 算出下一个 token
的概率分布，再从这个概率分布算出 `H_i`。

情形一：概率非常集中。

```text
token A：0.90
token B：0.05
其他：  0.05
```

模型非常确定，entropy 较低。

情形二：很多候选概率接近。

```text
token A：0.20
token B：0.18
token C：0.17
token D：0.15
其他：  0.30
```

模型不太确定，entropy 较高。

数学定义是：

```text
H_i = - sum over v in V of:
      P(v | x_1...x_i) × log P(v | x_1...x_i)
```

其中 `V` 是整个词表，`H_i` 是第 `i` 个 Prompt 位置对应的 next-token entropy。

### 5.2 手算一个 entropy

为了能手算，暂时假设词表里只有三个候选 token，并使用自然对数 `ln`。

情形 A：

```text
p = [0.90, 0.05, 0.05]

H = -(0.90×ln0.90 + 0.05×ln0.05 + 0.05×ln0.05)
  ≈ 0.394
```

其中一个候选占了 90%，模型很确定，所以 entropy 较低。

情形 B：

```text
p = [0.70, 0.20, 0.10]

H = -(0.70×ln0.70 + 0.20×ln0.20 + 0.10×ln0.10)
  ≈ 0.802
```

情形 C：

```text
p = [1/3, 1/3, 1/3]

H = -3×(1/3×ln(1/3))
  = ln3
  ≈ 1.099
```

三个候选同样可能时，分布最散、模型最不确定，entropy 最高。真实 Qwen 会对整个词表
的概率求和，计算方法相同，只是候选 token 多得多。

### 5.3 Entropy 不是什么？

entropy 不是：

- 回答正确率；
- 文本质量分数；
- Prompt 的字数；
- 模型已经生成的 token 数；
- “这个 token 本身有多难”的人工标签。

它只是模型内部概率分布的不确定程度。

尤其需要区分两件事：

- **高 entropy**：模型给出的候选概率分布比较分散；
- **随机采样**：根据这个分布和 seed 实际抽出一个 token。

即使还没有执行随机采样，概率分布和 entropy 也已经存在。在当前 PLP v2 中，Prompt
pooling 使用的是 Prompt prefill 阶段的 entropy；生成阶段当然也能计算 entropy，但当前
PLP-only 的动态输入使用 `h'_t`，不额外输入解码 entropy 标量。

### 5.4 为什么高 entropy token 可能更重要？

当模型在某个位置面临很多可能的后续方向时，这个位置可能决定回答将如何展开。例如任务
要求里出现“分五章”“详细比较”“给出完整代码和测试”等结构信息，可能影响后续回答路径和
输出长度。

这不表示高 entropy 永远更重要，而是论文采用的一种 **设计假设（inductive bias）**：模型
越不确定的位置，越可能处在语义分叉点，因此在汇总 Prompt 时给予该位置更高权重。

它不是数学定理，也不表示：

- 高 entropy 一定对应重要指令；
- 高 entropy 一定导致更长的回答；
- 低 entropy token 一定没有长度信息。

高 entropy 只会让该位置的 `h_i` 在 `h_prompt` 中占比更高。最后是更长还是更短，仍由
后面的 PLP prediction head 根据训练数据学习。这个假设是否适合我们的 Qwen 数据，应通过
entropy pooling、mean pooling、last-token pooling 等消融实验比较，而不能只靠直觉认定。

## 6. Entropy 怎样变成加权系数？

如果 Prompt 有 `n` 个 token，就会得到：

```text
H_1, H_2, ..., H_n
```

对这些 entropy 做 softmax：

```text
w_i = exp(H_i) / [exp(H_1) + exp(H_2) + ... + exp(H_n)]
```

`exp` 是指数函数。它始终为正，而且单调递增，因此 entropy 越大，得到的未归一化权重
也越大。分母把所有值归一化，使权重总和等于 1。

沿用上一节的三个 entropy：

```text
H_1 = 0.394，exp(H_1) ≈ 1.483
H_2 = 0.802，exp(H_2) ≈ 2.230
H_3 = 1.099，exp(H_3) ≈ 3.001

总和 ≈ 6.714

w_1 ≈ 1.483 / 6.714 ≈ 0.221
w_2 ≈ 2.230 / 6.714 ≈ 0.332
w_3 ≈ 3.001 / 6.714 ≈ 0.447
```

第三个位置的概率分布最散，所以它的 hidden state 获得最高权重。

所有权重满足：

```text
w_1 + w_2 + ... + w_n = 1
```

然后对**同一个 Prompt 内部的 token hidden states**加权：

```text
h_prompt = w_1 × h_1 + w_2 × h_2 + ... + w_n × h_n
```

这里不是对多个 Prompt 加权，而是对一个 Prompt 里的多个 token 加权。

因为每个 `h_i` 都是 3584 维，加权求和后的 `h_prompt` 仍然是 3584 维。

### 6.1 这些权重是训练出来的吗？

不是。当前 entropy-guided pooling 没有额外可训练参数：

```text
冻结的 Qwen → p_i(v) → H_i → softmax → w_i
```

当 Qwen、tokenizer、chat template 和 Prompt 都相同时，`H_i` 与 `w_i` 都是确定的。训练
数据不会直接修改这些权重。真正被训练的是后面的 PLP prediction head，它学习如何把
`[h_prompt; h'_t]` 映射为剩余长度。

因此，当前方法是先人为规定“更高 entropy 获得更高 pooling 权重”，再让 MLP 学习这个
汇总表征与剩余长度之间的关系。如果以后改成 learned-attention pooling，权重才会由训练
目标直接学习；那将是另一种模型变体。

### 6.2 一个 Prompt 最终得到多少个 `h_prompt`？

答案是：**一个。**

```text
Prompt A 的 n 个 token states → 一个 h_prompt_A，3584维
Prompt B 的 m 个 token states → 一个 h_prompt_B，3584维
Prompt C 的 k 个 token states → 一个 h_prompt_C，3584维
```

同一个 rollout 生成期间，Prompt 不变，所以它的 `h_prompt` 也保持不变。
在 Qwen 权重、tokenizer 和 chat template 都冻结时，同一个 Prompt 使用 seeds 42、43、44
进行三次生成，Prompt prefill 部分是确定的，因此三次 rollout 的 `h_prompt` 也相同；不同
seed 主要影响后续抽样出来的 token、`h'_t` 和最终输出长度。

### 6.3 三维简化例子

真实向量有 3584 维。为了方便理解，假设只有三维：

```text
h_1 = [1, 2, 0]
h_2 = [3, 0, 1]
h_3 = [0, 1, 4]
```

entropy softmax 得到：

```text
w_1 = 0.2
w_2 = 0.5
w_3 = 0.3
```

那么：

```text
h_prompt = 0.2 × h_1 + 0.5 × h_2 + 0.3 × h_3
         = [1.7, 0.7, 1.7]
```

这就是从多个 token hidden states 得到一个 Prompt feature 的过程。

## 7. 第二步：生成过程中的 `h'_t` 是什么？

Prompt 处理结束后，Qwen 开始生成回答：

```text
y_1, y_2, ..., y_T
```

当第 `t` 个输出 token 已经生成并被模型处理后，最终 Transformer 层会产生一个新的 3584
维向量：

```text
h'_t = f(x_1, ..., x_n, y_1, ..., y_t)
```

撇号 `'` 只是为了与 Prompt 阶段的 `h_i` 区分。

### 7.1 `h'_t` 只表示当前那个词吗？

不是。它虽然位于当前生成 token 的位置，但因果注意力使它能够读取：

```text
完整Prompt
+ 之前已经生成的token
+ 当前生成token
```

所以 `h'_t` 更像“Qwen 生成到当前阶段后的内部状态”，而不是当前 token 的普通词向量。

例如：

```text
h′₁   ：刚开始回答时的状态
h′₅   ：已经生成前5个token后的状态
h′₅₀  ：回答展开到第50个token后的状态
h′₂₀₀ ：回答发展到第200个token后的状态
```

这些向量维度都相同，但数值和包含的信息不同。

### 7.2 每生成一个 token 都要额外计算一次隐藏状态吗？

Qwen 为了预测下一个 token，本来就需要在每一步计算当前内部状态。PLP 只是顺便读取最终层
向量，不是每一步从头再运行一遍整个 Prompt。

实际生成时会使用 KV cache。模型在第 `t` 步只处理新加入的 token，同时通过缓存读取此前
Prompt 和已生成前缀的信息：

```text
第t步的新token
    + KV cache中的Prompt与y_1...y_(t-1)
                    ↓
最终层当前位置状态 h'_t（3584维）
                    ↓
Qwen原本就用它预测 y_(t+1)
                    ↓
PLP同时读取它来预测剩余长度
```

所以每一步确实都会产生一个新的 3584 维 `h'_t`，但这不是为了 PLP 额外重算整段文本。
`h'_t` 也不是“第 `t` 个词的 3584 维词典释义”，而是模型看到当前全部可见上下文后的
状态摘要。

论文描述为每一步更新。为了遵守本项目已经冻结的 PLP 更新频率，当前实现读取并保存：

```text
t = 1, 5, 10, 15, ...，以及最终token
```

没有保存的步骤仍然会被 Qwen 正常计算，只是不形成 PLP 数据点。

## 8. 第三步：PLP 的输入到底有哪些？

当前 Hidden-State PLP-only 的输入是：

```text
z_t = [h_prompt ; h'_t]
```

两个 3584 维向量拼接后得到：

```text
3584 + 3584 = 7168维
```

其中：

- `h_prompt`：任务和 Prompt 路线信息，在同一 rollout 中固定；
- `h'_t`：回答当前生成状态，随着解码不断变化。

### 8.1 当前 PLP-only 没有显式输入什么？

它没有额外输入：

```text
ALPS预测值
step
entropy标量
entropy_mean
entropy_slope
eos_probability
Prompt family标签
short / medium / long标签
任务类型标签
```

Prompt entropy 只用于计算 pooling 权重，之后进入预测头的是加权后的 `h_prompt`，不是一串
entropy 标量。

### 8.2 不输入 `step`，模型怎么知道生成到了哪里？

`h'_t` 由完整 Prompt 和此前生成前缀共同计算，并且 Qwen 使用位置信息，所以它会隐式编码
当前生成位置和回答阶段。

系统外部仍然知道 `t`。PLP 预测的是剩余长度 `R_t`；如果需要估计最终总长度，则计算：

```text
预测最终总长度 = 已生成长度 + 预测剩余长度
T_hat_t = t + R_hat_t
```

例如：

```text
已经生成 t = 100
PLP预测还剩 180
预计最终总长度 = 100 + 180 = 280
```

显式加入 `step` 可以作为后续消融实验，但那会形成 `PLP + step` 变体，不是当前先测试的
纯 hidden-state PLP-only。

## 9. 第四步：为什么输出 20 个长度区间？

先澄清：“20 个长度区间”是模型内部的 **20 个分类 bins**，不是 ALPS 评估里提到的
“95% prediction interval”。两者含义不同：

- 20 bins：prediction head 的内部输出形式，最后会换算成一个具体的剩余 token 点预测；
- 95% prediction interval：模型对预测误差不确定性的统计范围，用 coverage 检查校准程度。

本节只讨论前者。

### 9.1 输出长度是长尾数据

大量回答可能较短，少量回答特别长。如果直接用普通 MSE 回归，极长回答的平方误差可能主导
训练。论文因此结合分类与回归。

直接输出一个实数当然也可以，并且应该作为回归消融基线。采用 bins 的原因不是“长度只能
预测成范围”，而是希望 prediction head 同时学会“更可能落在哪一段”以及“具体大约是多少”。

### 9.2 把剩余长度分成 20 个 bins

为了演示，假设 Train 中使用的剩余长度范围是 `0～1000`，分成 20 段：

```text
Bin 0：0～50，中心25
Bin 1：50～100，中心75
Bin 2：100～150，中心125
...
Bin 19：950～1000，中心975
```

真实实现会使用 Train 剩余长度的 1%～99% 分位范围建立 20 个 bins；Test 不参与确定范围。

### 9.3 Soft label 表达“相邻区间比远处区间更接近”

假设真实剩余长度是 `230`，落在中心为 `225` 的 Bin 4。

普通 one-hot 只会写成：

```text
[0, 0, 0, 0, 1, 0, ...]
```

这无法表达“预测到相邻 Bin 3 比预测到 Bin 19 更接近”。所以论文使用：

```text
p_j 与 exp(-|j-i|) 成正比
```

其中 `i` 是真实 bin，距离越远，目标概率越低：

```text
Bin 2：较小概率
Bin 3：中等概率
Bin 4：最高概率
Bin 5：中等概率
Bin 6：较小概率
```

### 9.4 从 20 个概率得到一个具体 token 数

MLP 输出 20 个 logits，softmax 后得到 20 个概率：

```text
p_hat_1, p_hat_2, ..., p_hat_20
```

最终点预测是概率与 bin center 的加权平均：

```text
预测剩余长度 = 所有区间的（预测概率 × 区间中心）之和
R_hat_t = p_hat_1 × c_1 + ... + p_hat_20 × c_20
```

例如：

```text
中心175：概率0.15
中心225：概率0.60
中心275：概率0.25
```

则：

```text
R_hat_t = 0.15 × 175 + 0.60 × 225 + 0.25 × 275
        = 230
```

所以模型内部使用区间概率，最终仍输出一个具体的剩余 token 数。

## 10. Prediction head 和损失函数

对于 Qwen2.5-7B 的 `d=3584`，当前 prediction head 是：

```text
输入：[h_prompt ; h′_t]，7168维
  ↓
Linear(7168, 3584)
  ↓
LayerNorm
  ↓
ReLU
  ↓
Dropout(0.1)
  ↓
Linear(3584, 20)
  ↓
20个剩余长度区间概率
```

联合损失为：

```text
总损失 = 0.95 × CE损失 + 0.05 × MSE损失
```

- CE 让预测的 20-bin 分布接近 soft label；
- MSE 让最终期望长度接近真实剩余 token 数。

Qwen 的参数完全冻结。训练的只是后面的 PLP prediction head。

### 10.1 逐层理解这套 MLP 骨架

MLP（Multi-Layer Perceptron，多层感知机）可以先理解成“若干个全连接层，中间加入非线性
处理”。当前 PLP head 的数据流如下：

```text
[h_prompt; h'_t]：7168维输入
        ↓
Linear(7168, 3584)：学习怎样混合两类隐藏状态
        ↓
LayerNorm(3584)：把这一条样本的特征尺度整理稳定
        ↓
ReLU：加入非线性表达能力
        ↓
Dropout(0.1)：训练时随机遮掉10%的中间激活
        ↓
Linear(3584, 20)：得到20个bin的原始分数logits
        ↓
Softmax：把20个logits变成总和为1的概率
        ↓
按bin center加权求和：得到一个具体的剩余token点预测
```

#### Linear 在做什么？

第一层会为每个输出维度学习一组权重，将 `h_prompt` 和 `h'_t` 的 7168 个数重新组合成
3584 个中间特征。它不是简单截断向量，而是在训练中学习“哪些 Prompt 信息和解码状态组合
对剩余长度有用”。第二层再把 3584 个中间特征映射为 20 个 bin logits。

#### LayerNorm 在做什么？

对一条样本的 3584 个中间特征，LayerNorm 先计算这一条向量自己的均值和方差，再进行标准化：

```text
normalized_j = (x_j - 本条向量均值) / sqrt(本条向量方差 + epsilon)
output_j = gamma_j × normalized_j + beta_j
```

其中 `gamma` 和 `beta` 是会训练的缩放与平移参数。LayerNorm：

- 不会把 3584 维压缩成更少维；
- 不会删除这条样本；
- 主要避免不同隐藏状态维度的尺度差异让优化过程忽大忽小；
- 训练和推理时都使用当前样本自身的统计量，不依赖整个 batch 的均值，因此小 batch 也能使用。

可以把它类比成：不同传感器的数字量纲不同，先把它们整理到更稳定的尺度，再交给后面的
判断器。

#### ReLU 在做什么？

```text
ReLU(x) = max(0, x)
```

负数变成 0，正数保留。它给网络加入非线性；如果只有连续的 Linear 而没有非线性，中间堆
多层通常仍可合并成一个线性变换，表达能力会受限。

#### Dropout(0.1) 在做什么？

`0.1` 表示训练时，每次前向传播会随机将约 10% 的中间激活设为 0，其余激活会按比例放大，
使整体期望基本不变：

```text
训练：每个激活有10%概率被临时遮掉
推理：不再随机遮掉，使用完整网络
```

因此 Dropout 不是永久删除 10% 参数，也不是每次推理都随机丢信息。它是一种正则化方法，
迫使网络不要过度依赖某几个中间神经元，从而降低记住训练样本细节的风险。

#### 为什么 LayerNorm 在 ReLU 和 Dropout 前面？

当前顺序先稳定 Linear 输出的尺度，再经过 ReLU 形成非线性激活，最后对这些激活做 Dropout。
这与作者公开 EGTP prediction head 的实现顺序一致。论文的 PLP 部分说明复用同一 prediction
head，因此当前主实验保留这个顺序，不在首次实验前任意调整。

### 10.2 7168 维输入是否太大？

`7168` 是每条 PLP 数据的 **特征维度**，不是模型层数，也不是训练样本数：

```text
h_prompt：3584维
h'_t：    3584维
拼接：    7168维
```

当前第一层 `Linear(7168, 3584)` 的参数量约为：

```text
7168 × 3584 + 3584 ≈ 25.69 million
```

再加上 LayerNorm 和 `Linear(3584, 20)`，整个 PLP head 大约有 `25.77 million` 个可训练参数。
它远小于被冻结的 70 亿参数 Qwen，但相对当前 PLP 训练数据仍不算小。

因此需要区分两种判断：

- **显存和计算上**：Qwen 表征预先采集后，训练 head 时不需要同时加载 Qwen，5090 可以处理；
- **统计上**：参数较多且同一 rollout 的步骤高度相关，仍可能过拟合，必须看按 Prompt family
  分组的 Train/Test 表现，而不能只看 Train loss。

当前 v2 先保留这一结构，便于检验论文式 hidden-state PLP。若出现明显过拟合，可以另做
较小 bottleneck（例如 `7168 → 512 → 20`）、PCA/随机投影或更强正则化的消融实验；这些应
标成独立变体，不能在未记录的情况下悄悄替换主实验。

### 10.3 当前是否需要修改 LayerNorm 或 Dropout？

结论是：**当前主实验不修改。** 理由有三点：

1. 论文指定 PLP 使用 Section 3.2 的同一个 prediction head；
2. 作者公开的 EGTP head 使用的正是
   `Linear(d,d/2) → LayerNorm → ReLU → Dropout(0.1) → Linear(d/2,20)`；
3. 我们的输入是两个 3584 维状态拼接后的 7168 维，所以按同一规则得到
   `Linear(7168,3584) → ... → Linear(3584,20)`。

保留结构不等于假定它一定不会过拟合。训练完成后需要比较 Train/Test MAE、按解码 step 的
误差和 epoch loss。如果出现明显泛化差距，再把以下方案作为单独消融，而不是改写主结果：

- `Dropout(0.0)`：判断 Dropout 是否真的有帮助；
- `Dropout(0.2)`：测试更强正则化；
- `7168 → 512 → 20`：测试更小 prediction head；
- 保留 LayerNorm 与去掉 LayerNorm：判断标准化是否改善稳定性。

在没有 PLP v2 实际 Train/Test 结果之前，直接修改结构没有数据依据，还会让我们难以判断
效果来自 PLP 表征还是来自额外调参。

## 11. 一个 rollout 怎样产生多条训练数据？

假设一次回答最终有 `300` 个 token。同一个 rollout 只有一个固定 `h_prompt`，但会产生多个
不断变化的 `h'_t`：

| 步骤 | 输入 | 训练标签 |
|---:|---|---:|
| 1 | `[h_prompt; h′₁]` | 299 |
| 5 | `[h_prompt; h′₅]` | 295 |
| 10 | `[h_prompt; h′₁₀]` | 290 |
| 50 | `[h_prompt; h′₅₀]` | 250 |
| 100 | `[h_prompt; h′₁₀₀]` | 200 |
| 200 | `[h_prompt; h′₂₀₀]` | 100 |
| 295 | `[h_prompt; h′₂₉₅]` | 5 |
| 300 | `[h_prompt; h′₃₀₀]` | 0 |

因此 PLP 学到的是：回答状态怎样从“刚开始”逐渐发展到“接近结束”。

长回答会产生更多 PLP 点。为了避免长回答仅仅因为点多而支配训练，本项目让每条 rollout
的所有点总权重相同。

## 12. PLP 与 DDPM、ALPS、旧动态基线的区别

### 12.1 PLP 与 DDPM

两者都随步骤更新预测，但本质不同：

| DDPM | PLP |
|---|---|
| 人为给数据加入噪声 | 不加噪声 |
| 学习预测噪声或干净样本 | 学习预测剩余 token 数 |
| 通过反向过程逐步去噪生成数据 | 观察 Qwen 正常的自回归生成过程 |
| 本身是生成模型 | 是在线长度预测器 |

所以 PLP 更像导航预计到达时间，而不是图像去噪。

### 12.2 PLP-only 与 ALPS-only

| ALPS-only | PLP-only |
|---|---|
| 回答生成前预测一次最终总长度 | 回答生成中持续预测剩余长度 |
| 使用 Layer 14 最后一个 Prompt token state | 使用最终层 Prompt pooled state 和 decode state |
| Ridge 回归 | 20-bin soft-label MLP head |
| 输入不随生成变化 | `h'_t` 随生成不断变化 |

### 12.3 Hidden-State PLP 与 Dynamic-Signal MLP v1

| Dynamic-Signal MLP v1 | Hidden-State PLP v2 |
|---|---|
| 输入 5 个标量 | 输入两个 3584 维 hidden-state features |
| 使用 step、entropy trend、EOS probability | 使用 Prompt 语义和当前 causal decode state |
| 直接预测 `log1p(remaining)` | 20-bin soft label + token-space MSE |
| 可复用旧 ALPS trace | 必须重新采集 hidden states |

Dynamic-Signal MLP v1 是工程 baseline；本文解释的是单独的 Hidden-State PLP-only。

## 13. 为什么必须重新让 Qwen 跑一遍 Prompt？

旧 `data/interim/alps_v1/` 主要保存：

```text
Layer-14 Prompt特征
step
entropy
entropy_mean
entropy_slope
eos_probability
```

它没有保存 PLP 所需的：

```text
完整Prompt的最终层 entropy-pooled state
每个PLP更新点的最终层 decode hidden state
```

hidden state 是 Qwen 在推理当时产生的内部激活，不能只根据最终文本可靠恢复。因此需要按照
相同 Prompt、temperature、top-p 和 seeds 重新执行 Qwen 推理并采集这些向量。

这不是重新训练或微调 Qwen：

```text
Qwen权重：冻结
重新做的事情：生成回答并记录旧trace没有保存的内部状态
之后训练的东西：PLP prediction head
```

新trace使用schema v2。除Prompt和decode隐藏状态外，它还保存完整生成token ids，因此可以
用冻结的tokenizer重建回答，并核验每个stride点记录的token是否与完整生成序列一致。磁盘上
不重复保存7168维拼接输入；拼接只在训练时发生。

## 14. 与论文方法的实现边界

论文描述在第 `t` 步组合 Prompt 表征和已经生成 token 的隐藏状态集合：

```text
Aggregate(h_prompt, {h'_1, ..., h'_t})
```

并称 Aggregate 为简单拼接。但如果把全部历史向量直接平铺，输入维度会随着 `t` 增长，无法
直接进入同一个固定维度 MLP。作者公开仓库当前也没有 PLP 源代码解释这一点。

另一个需要明确的边界是：作者当前公开EGTP脚本默认截取前4个原始Prompt token，而论文
公式描述输入序列的entropy pooling。当前项目使用官方chat template格式化后的全部Prompt
token，遵循论文公式语义，但不属于作者EGTP脚本逐行复现。

本项目将其冻结为：

```text
z_t = [h_prompt ; h'_t]
```

理由是当前 causal state `h'_t` 已经注意并编码此前生成前缀。这保留了论文的核心输入语义、
剩余长度目标、最终层表示、20-bin head 和联合损失，但不能声称是作者代码的逐行 exact
replication。

参考：

- [Predicting LLM Output Length via Entropy-Guided Representations](https://arxiv.org/html/2602.11812v2)
- [作者公开仓库 LP_Bench](https://github.com/xiehuanyi/LP_Bench)
- [作者公开的 EGTP prediction head 实现](https://github.com/xiehuanyi/LP_Bench/blob/main/EGTP/egtp/model.py)

## 15. 最后记住这七句话

1. 一个有 `n` 个 token 的 Prompt，先产生 `n` 个 3584 维 hidden states。
2. 小写 `h_i` 是 3584 维隐藏向量；大写 `H_i` 是从 next-token 概率分布算出的一个 entropy
   标量。
3. 当前 entropy 权重 `w_i` 是确定性算出的，不是训练参数；“高 entropy 获得更高权重”是
   需要用实验验证的设计假设。
4. entropy 权重作用于同一个 Prompt 内部的 token，最终一个 Prompt 只得到一个 3584 维
   `h_prompt`。
5. 每个生成步骤都有一个新的 3584 维 `h'_t`，它表示模型读完 Prompt 和当前生成前缀后的
   内部状态。
6. PLP-only 拼接 `[h_prompt; h'_t]`，不输入 ALPS 预测值，也不输入 Prompt family 或人工
   short/medium/long 标签。
7. prediction head 输出 20 个长度 bins 的概率，再计算加权平均，得到具体的剩余 token 点预测。

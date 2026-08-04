# Dynamic-Signal MLP v1 实验结果

## 1. 方法身份

本报告记录当前项目版 PLP，即 **Dynamic-Signal MLP v1**。它借鉴“生成过程中持续预测
剩余长度”的问题定义，但不是论文 hidden-state PLP 的完整复现。

```text
正式名称：Dynamic-Signal MLP v1
内部标识：dynamic-signal-mlp-v1 / project_plp_only
实验角色：项目版PLP、动态信号MLP baseline
论文原版PLP复现：否
```

它研究的是：回答已经生成一部分后，能否利用生成进度、token分布不确定性和结束概率，
预测还会生成多少token。v1保留这条方法，是因为它能直接复用ALPS每5个token保存的trace，
不需要重新加载Qwen或重新生成540条rollout。

模型输入五个已有 trace 标量：

```text
step
entropy
entropy_mean
entropy_slope
eos_probability
```

模型不读取 ALPS prior、Prompt hidden state 或生成 token hidden state。结构为
`5 -> 128 -> 64 -> 1`，共9089个可训练参数，目标是
`log1p(remaining_tokens)`。

### 1.1 五个输入特征

| 特征 | 当前实现 | 作用 |
|---|---|---|
| `step` | 已经生成的新token数 | 表示当前解码进度 |
| `entropy` | 当前next-token完整概率分布的熵 | 表示当前预测不确定性 |
| `entropy_mean` | 最近最多20个token的entropy均值 | 平滑单token波动 |
| `entropy_slope` | 最近窗口首尾entropy的平均变化率 | 表示不确定性变化趋势 |
| `eos_probability` | 完整softmax分布分配给EOS的概率和 | 表示模型当前结束生成的倾向 |

`entropy`和`eos_probability`来自temperature缩放后的完整softmax，记录于top-p过滤前。
每5个生成token构造一个非终止样本，终止点不进入训练。

### 1.2 MLP结构

```text
5 features -> Linear(5, 128) -> ReLU -> Dropout(0.1)
           -> Linear(128, 64) -> ReLU -> Dropout(0.1)
           -> Linear(64, 1)
```

可训练参数为：

\[
(5\times128+128)+(128\times64+64)+(64\times1+1)=9089
\]

9089只描述模型容量，不证明MLP一定优于线性模型；该结构是v1冻结的工程假设。

## 2. 数据与训练

| 项目 | Train | Test |
|---|---:|---:|
| Rollout | 432 | 108 |
| 非终止动态点 | 36,040 | 8,539 |
| Prompt family | 48 | 12 |

每5个生成 token 保存一个动态点，终止点不参与训练。训练使用 sequence-balanced loss，
使每条 rollout 的所有动态点总权重相同。冻结设置为 AdamW、50 epochs、batch size 512、
learning rate `1e-3`、weight decay `1e-4`、dropout `0.1`、seed 42。

机器可读合同位于
[`configs/experiments/plp_v1_manifest.json`](../../../configs/experiments/plp_v1_manifest.json)。

| 冻结项 | v1固定值 |
|---|---|
| 基础rollout | ALPS v1固定Train/Test trace |
| Trace stride / entropy window | `5` / `20` |
| 输入 | 上述5个动态标量 |
| ALPS prior / hidden state | 不使用 |
| 目标 | `log1p(remaining_tokens)` |
| 隐藏层 / dropout | `[128,64]` / `0.1` |
| Optimizer | AdamW |
| Learning rate / weight decay | `0.001` / `0.0001` |
| Epochs / batch size | `50` / `512` |
| Seed | `42` |
| 样本权重 | 每条rollout的所有非终止点总权重相同 |
| 超参数选择 | 无；不得根据v1 Test调参 |

## 3. 指标说明

普通 timestep 指标把每个动态点视为一个样本，因此长回答因包含更多点而权重更大。
Sequence-balanced 指标让每条 rollout 总权重相同，是本报告的主要解释口径。

Bias 定义为：

\[
bias=prediction-actual
\]

负数表示低估剩余长度，正数表示高估。R² 为1表示完美预测，0等同当前评价集合的均值
预测，负数表示比均值预测更差。

## 4. 总体结果

| 指标 | Train | Test |
|---|---:|---:|
| 普通 MAE | 169.89 | 159.96 |
| 普通 RMSE | 231.95 | 216.82 |
| 普通 Raw R² | 0.064 | 0.013 |
| Log R² | -0.028 | 0.004 |
| 普通 Bias | -29.89 | -2.61 |
| Sequence-balanced MAE | 140.12 | **136.66** |
| Sequence-balanced RMSE | 193.62 | 190.19 |
| Sequence-balanced Raw R² | 0.198 | **0.089** |
| Sequence-balanced Bias | +25.44 | +38.20 |
| Sequence-balanced NLL | 6.172 | 6.161 |
| Sequence-balanced 95% Coverage | 93.5% | 92.8% |

Test Sequence-balanced Raw R² 只有0.089，Log R²接近0，说明当前五个动态标量只解释了
很少的剩余长度变化。Train 本身也不强，因此主要问题不是“MLP 把 Train 记住了”，而是
特征、目标和全局概率假设没有形成足够有效的映射。Train/Test 存在一定差距，但不是失败
的主要原因。

普通 Bias 接近0并不表示模型准确。普通指标让长 rollout 的大量 timestep 权重更高，且
早期低估和后期高估会互相抵消；Sequence-balanced Test Bias 仍为 +38.20 tokens。

## 5. Test 分阶段结果

以下以 sequence-balanced 指标为主：

| 解码进度 | Points | Rollouts | MAE | RMSE | Raw R² | Bias | 95% Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 - 10% | 904 | 108 | 241.27 | 319.10 | -0.402 | **-212.23** | 74.9% |
| 10 - 25% | 1,270 | 105 | 152.63 | 206.39 | 0.206 | -56.19 | 99.0% |
| 25 - 50% | 2,123 | 108 | **90.93** | **130.48** | **0.476** | +31.19 | 100.0% |
| 50 - 75% | 2,127 | 108 | 120.56 | 155.06 | -0.927 | +99.90 | 100.0% |
| 75 - 100% | 2,115 | 108 | 147.98 | 193.43 | -14.187 | **+145.15** | 81.1% |

## 6. 分阶段分析

### 6.1 生成早期严重低估

0 - 10%平均低估212 tokens，R²为负。刚开始生成时，`step` 几乎都很小，少量 entropy、
entropy trend 和 pre-top-p EOS probability 尚不足以判断回答最终会走向短输出还是长输出。
由于没有 Prompt 内容或 ALPS prior，不同任务和回答意图在输入上难以区分。

训练目标使用 log-MSE，也会压缩长回答的绝对误差：把一个很长的剩余长度低估几百 token，
在 log 空间受到的惩罚远小于原始 token MSE。

### 6.2 生成中段出现有效信号

25 - 50%阶段表现最好：Sequence-balanced MAE 为90.93、Raw R²为0.476。生成路径发展到
中段后，累计 step、entropy 均值/趋势和 EOS probability 开始反映当前 rollout 的具体路径。

这说明五个动态标量并非完全无效，但其可用信息主要集中在生成中段，不能支持从早期到
结束的稳定渐进预测。

### 6.3 生成后期严重高估

75 - 100%阶段平均高估145 tokens，R²为 -14.19。模型接近结束时仍预测较多剩余 token。

当前训练得到的全局 log residual variance 为：

```text
1.3360
```

点预测从 log 空间转换为 token 均值时使用：

\[
\exp(\mu+\sigma^2/2)-1
\]

对应方差修正因子约为：

\[
\exp(1.3360/2)\approx1.95
\]

它会把 log-space 中位数对应的剩余长度显著向上修正。一个全局方差无法同时适应早期的大
剩余长度与后期的小剩余长度，容易在后期产生系统性高估。终止点被排除也使模型缺少明确的
`remaining_tokens=0` 锚点。

## 7. 概率结果

总体 Test Sequence-balanced Coverage 为92.8%，表面接近95%，但分阶段分别为74.9%、
99.0%、100%、100%和81.1%。总体数值掩盖了明显阶段错配：早期和后期覆盖不足，中段又
可能过宽。当前动态分布不能称为稳定校准。

NLL 随生成推进下降不能单独解释为模型持续变好，因为当前 log-normal NLL 含有目标长度
相关项，后期真实 remaining tokens 本身更小；应结合 MAE、Bias、R²和 Coverage 判断。

## 8. 是否过拟合

当前结果不支持“严重过拟合是主要问题”：

```text
Train sequence-balanced R² = 0.198
Test  sequence-balanced R² = 0.089
```

Train 高于 Test，存在一定泛化差距，但 Train 本身也很弱。增加网络容量未必能解决问题；
更可能需要更有信息的特征、静态 prior、分阶段误差模型或论文式 dynamic hidden state。

## 9. 与 ALPS 的比较边界

不能直接用 ALPS MAE 60.87 和 Dynamic-Signal MLP MAE 136.66 排名：

- ALPS 在生成前对每条 rollout 预测一次最终总长度；
- Dynamic-Signal MLP 在多个 timestep 预测剩余长度；
- 两者目标、样本单位和评价分布不同。

公平动态 baseline 应在同一 timestep 计算：

\[
\widehat{remaining}_t=\max(0,\widehat{total}_{ALPS}-t)
\]

还应使用相同五特征训练 Dynamic Ridge，以判断 MLP 的非线性是否带来增量。

## 10. 结论

> Dynamic-Signal MLP v1 已完成训练和 Test 评价，但整体剩余长度解释力较弱，只在生成
> 25% - 50%阶段表现出中等能力，并存在明显的早期低估、后期高估和分阶段概率错配。
> 它应作为项目 v1 的工程 baseline 和负向实验结果保留，而不能作为可靠 PLP 或论文原版
> PLP 复现。下一版应优先比较 ALPS-minus-step、Dynamic Ridge、ALPS+动态信号 hybrid，
> 后续应运行已独立实现的 Hidden-State PLP v2，并重新采集论文所需的生成 token hidden
> state。

## 11. 与论文PLP的边界

论文PLP在第`t`步组合Prompt表示和已经生成token的hidden states，并使用soft-label
length-bin prediction head；公开方法还采用20个length bins和CE + MSE联合损失。当前v1
只使用五个标量信号，由MLP直接回归`log1p(remaining_tokens)`。二者输入、输出头和损失
函数均不同，不能把当前结果表述成“论文PLP复现”。

参考：

- [Predicting LLM Output Length via Entropy-Guided Representations](https://arxiv.org/html/2602.11812v2)
- [论文公开仓库LP_Bench](https://github.com/xiehuanyi/LP_Bench)

这些内容已经在 `configs/experiments/plp_v2_manifest.json` 与对应采集、训练、评估脚本中
实现；当前仍需在GPU上重新采集和训练，尚无v2结果。公开仓库没有PLP源码，因此v2对
可变长“简单拼接”采用了固定维度解释，并非逐行exact replication。

## 12. 运行与原始结果

以下命令只读取已有trace，不加载Qwen：

```bash
python scripts/train_dynamic.py
python scripts/evaluate_dynamic.py --split train
python scripts/evaluate_dynamic.py --split test --confirm-final-test
```

原始输出：

```text
artifacts/runs/alps_v1/comparisons/plp_only/model.json
artifacts/runs/alps_v1/comparisons/plp_only/training_report.json
artifacts/runs/alps_v1/comparisons/plp_only/train_evaluation.json
artifacts/runs/alps_v1/comparisons/plp_only/train_evaluation.csv
artifacts/runs/alps_v1/comparisons/plp_only/test_evaluation.json
artifacts/runs/alps_v1/comparisons/plp_only/test_evaluation.csv
```

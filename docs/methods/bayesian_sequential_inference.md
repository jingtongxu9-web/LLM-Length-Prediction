# Bayesian Sequential v1 数学与推断合同

## 0. 合同身份

本文是项目后续实现的唯一数学基线。权威需求来源为
[`../references/大模型输出长度预测_2026-07-09_authoritative.pdf`](../references/大模型输出长度预测_2026-07-09_authoritative.pdf)，
SHA-256 为：

```text
b3c39e785a4a9217fd5d3c580b127c5a25093314f82bacb86f64f98feed02e62
```

机器可读合同位于
[`../../configs/experiments/bayesian_sequential_v1.json`](../../configs/experiments/bayesian_sequential_v1.json)。
本合同冻结后，任何改变 latent state、更新频率、prior、evidence、损失、数据边界或主指标的
实现都必须使用新的 method ID，不能静默覆盖 `bayesian-sequential-v1`。

当前 Dynamic-Signal MLP、Hidden-State PLP、concat、residual 和 gated residual 均保留为
baseline 或结构消融；它们不是本文定义的贝叶斯序列模型。

## 1. 研究问题

给定 Prompt `x` 和冻结的大模型，在回答生成前先预测最终输出总长度，在生成过程中每隔固定
token 数吸收新证据，持续更新剩余长度的完整概率分布。系统必须同时回答：

1. 当前预计还会生成多少 token？
2. 该预测的不确定性是多少？
3. 新生成内容是否使预测区间逐步收窄？
4. 与 ALPS-only、dynamic-only 和判别式 Hybrid 相比，显式贝叶斯更新是否带来增量价值？

## 2. 记号与长度口径

| 符号 | 含义 |
|---|---|
| `x` | 格式化后的 Prompt |
| `L` | rollout 最终生成 token 总数，包含 EOS |
| `t` | 当前已经生成的 token 数，终点满足 `t = L` |
| `R_t` | 当前剩余 token 数，`R_t = L - t` |
| `h_0` | Prefill 结束时 zero-based Layer 14 的最后一个 Prompt token hidden state |
| `E_t` | 当前 next-token 完整概率分布的 entropy |
| `e_t` | 从上一更新点到当前更新点的新证据 block |
| `p_t(r)` | 观察到第 `t` 步证据后，`R_t = r` 的 posterior probability |

长度支持集冻结为整数 token：

```text
R_t in {0, 1, ..., max_new_tokens - t}
max_new_tokens = 4096
```

实现中另保留一个不参与整数点预测伪装的 `overflow` 状态，表示真实总长度大于 4096。它用于
保存 shifted-lognormal 的上尾质量，并使 `max_new_tokens` 停止的 rollout 能贡献正确的
右删失生存概率。摘要必须单独报告 overflow probability；若所求 quantile 落入 overflow，区间
上界报告为无穷而不是 4096。

不使用 Test 长度分位数确定支持集或 bins。被 `max_new_tokens` 截断的 rollout 属于右删失数据，
不能把未知剩余长度标为 0。

## 3. 生成前 ALPS prior

### 3.1 均值模型

Qwen 权重完全冻结。ALPS 读取 `h_0`，用 Train-only StandardScaler 和固定
`Ridge(alpha=1.0)` 预测：

```text
mu_0 = W_mu^T standardize(h_0) + b_mu
```

项目实现使用 shifted log-normal：

```text
log(1 + L) ~ Normal(mu_0, sigma_0^2)
```

这与旧代码一致，也避免在数值实现中对 0 取对数。论文和项目文档统一使用这一口径，不再一处
写 `log(L)`、另一处写 `log1p(L)`。

### 3.2 方差校准

`sigma_0^2` 不得继续使用全量 Train 的 in-sample residual MLE 作为最终未见 family 方差。
主方法冻结为：

```text
variance_source = family-grouped out-of-fold log1p residual MLE
fold_count = 5
group_unit = prompt_family_id
```

外层验证 family 的 prior 只能来自外层训练 family 拟合的 ALPS。最终全量模型的方差校准参数
来自 Train family 的 cross-fitted residual，不能读取已打开的旧 Test 或未来 final holdout。

### 3.3 离散化

ALPS 连续分布通过 CDF 区间积分映射为整数总长度质量：

```text
P(L = k) = F(k + 0.5) - F(max(0.5, k - 0.5))
```

在 `1 <= k <= 4096` 上重新归一化。初始剩余长度 prior 为：

```text
p_0(R_0 = r) = P(L = r | h_0)
```

若采集协议允许长度超过 4096，则必须显式加入 overflow/censored mass；v1 不允许静默丢弃尾部
概率。

## 4. 序列状态转移

更新点冻结为：

```text
t = 1, 5, 10, 15, ...，以及 terminal token
```

设上一更新点为 `t_prev`，本次间隔为 `delta = t - t_prev`。在吸收新动态证据前，先执行确定性
倒计时与已存活条件化：

```text
p_t^-(r) proportional to p_t_prev(r + delta)
```

所有 `r + delta` 超出上一 posterior 支持集的项为 0，然后重新归一化。这个步骤同时利用了
“序列确实存活到 t”这一事实，淘汰原本预测会更早结束的概率质量。

终点 `t = L` 必须包含 `R_t = 0` 监督；不能像 Dynamic-Signal MLP v1 一样排除 terminal point。

## 5. 增量 evidence 合同

### 5.1 禁止重复计算历史证据

当前 causal decode hidden state `h_t` 已经包含 Prompt 和全部生成前缀。如果把每个完整 `h_t`
当作独立 likelihood 递归相乘，会重复使用历史信息。因此 v1 的 evidence 单位冻结为：

```text
从 t_prev + 1 到 t 的新生成 token block
```

collector 必须保存足以构造非重叠 block 统计量的逐 token 或 block-level 原始量，不能只保存
相互重叠的 rolling window 后直接当作独立 likelihood。

### 5.2 主 evidence features

主模型输入分为三组：

```text
prior context:
  mu_0
  sigma_0^2
  prior mean total tokens

time context:
  t
  delta
  t / max_new_tokens

new block evidence:
  last entropy
  non-overlapping block entropy mean
  non-overlapping block entropy slope
  last EOS probability
  block maximum EOS probability
  delta entropy
  delta EOS probability
  terminal-observed flag
```

所有 entropy 和 EOS probability 均来自 temperature 缩放后的完整 softmax、top-p 截断前。

### 5.3 Hidden-state 扩展

PDF 的理论式写作 `lambda_t = f(E_t, h_t)`，微基准文字则冻结了轻量标量 MLP。为避免在首次
实现前混淆两种问题，v1 预声明两个候选，但不得再增加未登记变体：

1. `bayesian_entropy_scalar_v1`：只使用上述标量 evidence；
2. `bayesian_entropy_hidden_delta_v1`：在标量 evidence 上增加冻结投影后的
   `h_t - h_t_prev`，不直接拼接两个 3584 维完整状态。

两个候选使用完全相同的 folds、prior、状态转移、损失和指标。只能在 Train-family OOF 中按
预注册选择规则决定最终候选。

## 6. Likelihood-ratio head 与 Bayes update

Evidence network 不直接回归一个剩余长度点预测。它对每个候选剩余长度 `r` 计算增量
log-likelihood-ratio score：

```text
s_theta(e_t, r) = g_theta(evidence_features_t, candidate_features(r))
```

候选特征至少包含：

```text
r
log1p(r)
r / max_new_tokens
candidate_total = t + r
overflow indicator
```

使用共享 scorer，而不是为 4097 个长度各自训练完全独立的参数。Posterior 更新为：

```text
log p_t(r) = log p_t^-(r) + s_theta(e_t, r) - log Z_t
```

其中 `log Z_t` 由 `logsumexp` 计算。所有 posterior 操作必须在 log space 完成，并满足：

```text
p_t(r) >= 0
sum_r p_t(r) = 1
P(R_t < 0) = 0
P(L < t) = 0
```

网络参数 `theta` 只在离线 Train/OOF 训练中更新。正式生成时 `theta` 完全冻结，每一步改变的
是当前请求的 posterior，不是模型参数。

## 7. Hazard 与停时分布

PDF 连续式使用未来 `lambda_(t+r)`，但未来 entropy/hidden state 在第 `t` 步不可观测。v1 不
直接读取未来状态，而是从当前 posterior 导出离散停时 hazard：

```text
q_t(r) = P(R_t = r | R_t >= r, evidence up to t)
       = p_t(r) / sum_(u >= r) p_t(u)
```

于是：

```text
p_t(r) = q_t(r) * product_(j < r) (1 - q_t(j))
```

`q_t(r)` 是 PDF 中 EOS stopping intensity 的离散实现。Entropy/hidden-state evidence 通过
Bayes update 改变 `p_t`，从而动态改变 hazard curve。未来版本若直接参数化 hazard，必须使用
新的 method ID 并证明其与 prior 的概率组合没有重复使用 base rate。

## 8. 训练目标

每条 rollout 内的所有更新点总权重相同，避免长回答因 timestep 更多而支配训练：

```text
L_nll = mean_over_rollouts(
  mean_over_saved_steps(-log p_t(R_t_true))
)
```

v1 的主损失冻结为 posterior NLL。允许的辅助项只有：

```text
terminal BCE weight = 0.10
posterior total-variation stability penalty weight = 0.01
```

Soft label 不在 v1 主方法中默认开启。它属于 PDF 第四环的理论修正候选；若 OOF error analysis
支持使用，必须创建新 method ID，不能利用 final holdout 选择权重。

## 9. 输出与不确定性

每个更新点至少输出：

```text
posterior mean remaining
posterior median remaining
posterior mode remaining
posterior variance
50%, 90%, 95% equal-tail credible intervals
posterior entropy
derived hazard curve summary
```

总长度点预测为：

```text
L_hat_t = t + E_p_t[R_t]
```

PDF 所说“Var(R_t) 快速收敛到 0”是待验证假设，不是强制约束。只有当 interval coverage 保持
合理时，方差和区间宽度下降才代表有效收敛。

## 10. 评价合同

### 10.1 概率指标

- family-macro、sequence-balanced posterior NLL（模型选择主指标）；
- CRPS；
- 50% / 90% / 95% interval coverage；
- interval mean width；
- posterior variance 与 entropy 随 decode progress 的变化；
- uncertainty cone：真实剩余长度、posterior median、2.5% 和 97.5% quantile。

### 10.2 点预测指标

- MAE、RMSE、Bias、Raw R-squared；
- long-output underestimation；
- terminal 与 nonterminal 分开报告；
- task、intended length、temperature、seed、family fold 与 progress breakdown。

### 10.3 收敛速度

相对误差定义为：

```text
relative_error_t = abs(L_hat_t - L) / max(L, 1)
```

`stable_time_to_5pct` 是第一个满足“该点及所有后续保存点都不超过 5%”的 step。若始终未达到，
记为未收敛，不能把最后一个 step 当作成功。必须同时报告成功率和成功样本的 token 数分布。

### 10.4 推理开销

报告每次 posterior update 的 CPU/GPU wall time、峰值内存、相对基础生成延迟和保存 posterior
摘要所需磁盘，不把离线 trace 采集开销混入在线 predictor overhead。

## 11. 方法比较与预注册选择

同一 trace、同一 timestep、同一 family folds 比较：

1. `prompt_token_ridge_countdown`；
2. `alps_countdown`；
3. `dynamic_signal_mlp_v1`；
4. `plp_terminal_zero_v3`；
5. `alps_plp_concat_v1`；
6. `bayesian_entropy_scalar_v1`；
7. `bayesian_entropy_hidden_delta_v1`。

两个 Bayesian 候选以 family-macro sequence-balanced posterior NLL 选择。只有当 hidden-delta
候选相对 scalar 候选的 family-paired 95% bootstrap CI 全部低于 0，才选择更复杂的 hidden
版本；否则保留 scalar 版本。Final holdout 不参与模型、特征、epoch、loss 或 threshold 选择。

## 12. 数据与防泄漏边界

- family 是唯一分组单位；同一 family 的任务长度版本、seed、temperature 和全部 timestep
  必须处于同一 fold；
- ALPS prior 对外层训练样本使用 inner grouped cross-fitting，对外层验证样本只用 outer-train
  family 拟合；
- scaler、方差校准、candidate normalization 和任何统计量均只从相应训练 fold 计算；
- 当前已打开的 ALPS/PLP Test 只保留为历史证据；
- 新 Bayesian final holdout 必须包含从未用于方法设计的新 family，且只允许在所有代码、配置、
  checkpoint、指标和报告 schema 冻结后打开一次；
- censored rollout 用生存/右删失口径处理，不允许当成真实终点。

## 13. Temperature 协议

主开发条件保持与现有仓库兼容：

```text
temperature = 0.7
top_p = 0.95
```

冻结后的 robustness 条件为：

```text
temperature in [0.3, 1.0]
top_p = 0.95
```

同一 Prompt family 的全部 temperature 必须留在同一 fold。v1 不允许根据 robustness 条件重新
拟合模型；它检验固定方法对 sampling policy shift 的稳定性。未来若显式把 temperature 输入
prior 或 evidence model，必须使用新 method ID。

## 14. Error feedback 边界

在 Train-family OOF 上审查：

- 绝对误差大于 100 token；
- 最差 5%；
- entropy rebound 或高频振荡；
- repetition、hallucination、open-ended prompt、sampling divergence、early stop；
- posterior 方差反常增大、过早塌缩或来回振荡。

Error analysis 只能产生预注册的新消融或下一 method ID，不能根据 final holdout 修补 v1。

## 15. 第一阶段完成条件

进入实现阶段前必须满足：

1. 本数学合同与 JSON 合同字段一致；
2. 权威 PDF 的路径和 SHA-256 固定；
3. latent state、transition、evidence unit、posterior update 和 censoring 口径无歧义；
4. baseline 与 proposed method 身份明确；
5. Train/OOF/final holdout 边界明确；
6. 主指标、收敛指标和选择规则固定；
7. 自动测试能够检查上述关键不变量；
8. 尚未采集或打开任何新的 Bayesian final holdout。

以上是进入第二阶段前的历史门槛，并已在实现开始时满足。Stage-8B 于 2026-08-16 按冻结合同
一次性打开 final holdout；完成后的结果与永久 no-reselection 边界见
[`../results/bayesian_sequential/stage8_final_benchmark_20260816.md`](../results/bayesian_sequential/stage8_final_benchmark_20260816.md)。

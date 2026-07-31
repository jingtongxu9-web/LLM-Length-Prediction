# ALPS v1 五折验证与过拟合解释

## 结论

当前 v1 只需要在已有 432 条 Train trace 上补跑一次：

```bash
python scripts/evaluate_grouped_cv.py
```

这条命令同时验证固定的 Layer 14 ALPS、`prompt_tokens` baseline 和其他诊断 baseline。
每一折临时训练一个 Ridge，在未见过的 `prompt_family_id` 上预测；五个临时 Ridge 随后全部
丢弃，不会合并，也不会替换 `stage1/prior.json`。最终可部署的 Ridge 仍然是使用全部 Train
一次性拟合的那个模型。

现有 v1 Test 已经打开，因此这次五折属于**补充的、回顾性 Train 内部诊断**。它不能用于
修改 Layer、alpha 或其他配置后反复查看同一 Test。下一轮严格实验应按“五折验证 → 全 Train
最终拟合 → 一次性打开新 Test/holdout”的顺序执行。

## 官方 ALPS 的 Qwen 结果

官方 Qwen 结果文件记录的 Layer 14 指标为：

| 指标 | 官方结果 |
|---|---:|
| Train R² | 0.9999998 |
| 5-fold CV R² | 0.8484 |
| Test R² | 0.8568 |
| Test MAE | 80.20 tokens |

输入 token 数 baseline 的 Train/Test/CV R² 分别为 0.2744、0.3563 和 0.2746。

来源：

- <https://raw.githubusercontent.com/glenfmessenger/alps/main/alps_qwen/results/probe_results.json>
- <https://raw.githubusercontent.com/glenfmessenger/alps/main/RESULTS_SUMMARY.md>

官方材料没有把该现象明确命名为“过拟合故障”，但数值本身确实显示 hidden-state Ridge
在 Train 上近乎插值。关键是 Layer 14 的 CV R² 与 Test R² 很接近（0.848 与 0.857）：
Train 分数不能代表泛化能力，但独立数据上的效果是稳定的。

## 我们为什么也出现很大的 Train/Test 差距

Qwen2.5-7B 的单层 hidden state 有 3584 维，而当前 Train 虽有 432 条 rollout，实际上只有：

- 144 个不同 Prompt；
- 48 个独立 `prompt_family_id`；
- 每个 Prompt 的三个 seed 共享同一个生成前 Layer-14 特征。

三个 seed 增加的是输出随机性观测，不是三个独立的输入特征。因此有效的独立输入数量远小于
3584 个特征维度。即使 Ridge 使用 `alpha=1.0`，高维线性模型仍然可以非常贴近 Train 中
每个 Prompt 的平均输出长度。

这意味着：

1. Train R² 接近 1 不能作为 ALPS 有效的证据；
2. family-grouped CV 和 Test 才是主要泛化证据；
3. 当前 Test Prompt-mean R² 仍为 0.9354、rollout-level Log R² 为 0.9286，说明 ALPS
   的总体长度排序和预测信号仍然较强；
4. Test MAE 与区间覆盖明显劣于 Train，说明逐 rollout 精度和随机性校准仍有改进空间。

五折不会“消除”过拟合，它只会让我们看到模型在未见 family 上的真实水平。如果补跑的
CV 与现有 Test 接近，可以认为当前 Test 结果不是偶然；如果 CV 远低于 Test，则说明当前
Test 可能偏容易或样本较小，需要在下一轮增加独立 family/新 holdout。

## 后续对照方法

在不重新生成 Qwen rollout、不改变 temperature、top-p、seed、prompt split 和 tokenizer
的前提下，仓库增加两条比较路径：

- `input_token_ridge`：只使用 Prompt 输入 token 数预测最终输出 token 数；
- `dynamic-signal-mlp-v1`（内部兼容标识 `project_plp_only`）：只使用生成中的 step、
  entropy、entropy trend 和 EOS probability 预测剩余 token 数，不读取 ALPS prior
  或 Layer-14 hidden state。

Dynamic-Signal MLP v1 可以直接使用现有 trace。论文原版 PLP 使用逐 token hidden state，
而当前 v1 trace 没有保存该字段，因此两者不能写成完全复现。完整版本边界见
[`dynamic_signal_mlp_v1.md`](dynamic_signal_mlp_v1.md)。若未来要复现原版 PLP，需要扩展
采集器并重新生成 Train/Test trace，作为 v2 单独执行。

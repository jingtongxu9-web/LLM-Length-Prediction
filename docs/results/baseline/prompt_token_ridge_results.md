# Prompt-token Ridge Baseline 实验结果

## 1. 实验目的

该基线只使用格式化 Prompt 的 token 数预测输出 token 数，用来检验“输入越长，输出也越长”
这一简单规律能解释多少性能。它不读取 Prompt 语义、任务类型、hidden state 或生成中状态。

## 2. 静态总长度结果

在 ALPS v1 的 60 个 design family 上，family-grouped 五折 OOF 结果为：

| 方法 | MAE | RMSE | Raw R² | Log R² | 95% Coverage | 区间均宽 |
|---|---:|---:|---:|---:|---:|---:|
| Global mean | 284.12 | 324.78 | -0.091 | -0.002 | 98.4% | 2514.11 |
| **Prompt tokens Ridge** | **260.60** | **313.97** | **-0.020** | **0.018** | **96.5%** | **2486.80** |
| Metadata | 92.95 | 140.88 | 0.795 | 0.925 | 94.7% | 547.68 |
| Metadata + Prompt tokens | 92.94 | 140.08 | 0.797 | 0.925 | 94.9% | 545.99 |

普通 Train/Test 拆分中的 Prompt-token Ridge 结果：

| Split | N | MAE | RMSE | Log R² | NLL | 95% Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Train | 432 | 260.15 | 313.66 | 0.0198 | 7.112 | 96.5% |
| Test | 108 | 246.77 | 297.52 | 0.0106 | 7.074 | 91.7% |

## 3. 结果解释

- Train 与 Test 都接近零解释力，因此问题不是过拟合，而是输入 token 数本身信息不足。
- 96.5% Coverage 不表示预测准确；其平均区间宽度接近 2487 token，主要靠极宽区间覆盖真实值。
- Metadata 明显更强，说明 Prompt 设计里的任务类型和 short/medium/long 条件包含组间长度信息。
- Metadata 加入 Prompt token 数几乎不变，再次说明单纯输入长度贡献很小。

因此，ALPS 的优势不能由“它只是间接数出了 Prompt token 数”解释。

## 4. 当前四方法比较中的补充 Baseline

上面的 `260.60` 是生成前总长度预测，不能直接与当前逐步剩余长度 Hybrid OOF 排行榜混合。
为保证公平，当前补充实验在完全相同的 60 个 family、540 条 rollout、五折划分和保存 step 上：

1. 每折只用训练 family 拟合 `prompt_tokens -> log1p(total_output_tokens)` Ridge；
2. 在验证 family 上得到总长度预测 `T_hat`；
3. 第 `t` 个生成 step 使用 `max(T_hat-t, 0)` 得到剩余长度预测；
4. 与 ALPS countdown、PLP v3 和 Hybrid concat v1 计算同一指标。

该补充实验的脚本已经实现，但数值尚需在保存完整 trace 的 AutoDL 环境运行。它不会重新运行
Qwen，也不会改变已经完成的 Hybrid 方法选择。

## 5. 相关文件

- 方法原理：[`../../methods/prompt_token_baseline.md`](../../methods/prompt_token_baseline.md)
- 旧阶段综合报告：[`../comparisons/stage1_alps_baselines_dynamic.md`](../comparisons/stage1_alps_baselines_dynamic.md)
- 四方法主比较：[`../comparisons/four_method_main_comparison.md`](../comparisons/four_method_main_comparison.md)

# PLP-only terminal-zero v3

## 当前状态

PLP-only 的开发阶段已经完成，最终 Test 尚未打开。

- 数据：60 个 Train prompt family、540 个 rollout、45,119 个逐步预测点；
- 验证：按 `prompt_family_id` 分组的 5 折 OOF，Test 未参与；
- 对照：`plp_v2_frozen`；
- 三个单因素消融：terminal zero bin、512 小头、rollout-balanced target range；
- 选择结果：`plp_terminal_zero_v3`。

这里的 v3 仍然是 PLP-only：输入只有 Prompt 表征和当前 decode hidden state，不读取 ALPS
预测或 prompt family 标签。

## 消融结果

主指标是 family-macro sequence-balanced MAE，单位为 token，越低越好。

| 方法 | OOF MAE | 相对 PLP v2 | 结论 |
|---|---:|---:|---|
| `plp_v2_frozen` | 61.037 | — | 冻结对照 |
| `plp_terminal_zero_v3` | **59.778** | **-1.259** | 保留 |
| `plp_weighted_range_v3` | 60.816 | -0.221 | 区间跨 0，不保留 |
| `plp_small_head_v3` | 70.965 | +9.928 | 明显退化，不保留 |

Terminal-zero 相对 v2 的 family 配对 bootstrap：

- 95% CI：`[-1.875, -0.643]` token；
- 新独立入口按冻结 seed 重算的 98.33% familywise CI：`[-2.022, -0.513]` token。

两个区间都完全低于 0，说明改善并非只由少数 family 偶然造成。它在 5 个 fold 和 seeds
42、43、44 上方向一致。

## 改善发生在哪里

| 解码进度 | PLP v2 MAE | Terminal-zero v3 MAE | v3-v2 |
|---|---:|---:|---:|
| 0–10% | 77.92 | 78.72 | +0.80 |
| 10–25% | 76.34 | 77.56 | +1.22 |
| 25–50% | 67.09 | 67.84 | +0.75 |
| 50–75% | 54.37 | 52.92 | -1.45 |
| 75–100% | 45.55 | **41.13** | **-4.42** |

Terminal-zero 的作用主要在生成后半程。真实剩余长度为 0 的终点上，旧 v2 平均仍预测
`23.25` token，新版本降到 `1.36` token；非终点 MAE 也从 `61.21` 小幅降到 `60.71`。
因此它不是只把最后一个点“做对”，而是改善了接近结束时的剩余长度判断。代价是前 50%
有约 0.8–1.2 token 的轻微退化，最终 Test 必须继续报告分阶段结果，不能只看总体平均数。

## 冻结的 PLP-only v3

`plp_terminal_zero_v3` 只相对 v2 改一个因素：增加独立的零长度 bin。

| 条件 | 冻结值 |
|---|---|
| PLP 输入 | 3584 维 Prompt 表征 + 3584 维 decode hidden state |
| MLP hidden dim | 3584 |
| 输出 | 20 bins，其中 1 个是精确的 terminal zero bin |
| 正长度 target range | 未按 rollout 加权的 1%–99% 分位数 |
| Loss | 0.95 soft-label CE + 0.05 raw-token MSE |
| Dropout | 0.1 |
| Epochs / batch / LR | 10 / 16 / `2e-5` |
| Optimizer / weight decay | AdamW / 0.0 |
| Seed | 42 |

512 小头和 rollout-balanced range 都不进入最终模型，也不再根据 Test 修改上述设置。

## 下一步：先冻结模型，再决定是否消耗 holdout

现有统一 Train trace 已经包含 PLP 所需的 Prompt/decode hidden states，不需要重新运行 Qwen。

如果十方法最终训练已经完成，可校验并复用其中字节完全一致的两个 checkpoint：

```bash
python scripts/train_plp_v3_models.py --reuse-hybrid-models
```

如果十方法训练没有完成，直接只训练 PLP v2 对照和 terminal-zero v3：

```bash
python scripts/train_plp_v3_models.py --device auto
```

这一步会生成：

```text
artifacts/runs/plp_terminal_v3/
├── selection/oof_selection.json
└── models/
    ├── plp_v2_frozen.pt
    ├── plp_terminal_zero_v3.pt
    └── model_registry.json
```

当前 12 个全新 family 同时是原 Hybrid 协议预留的 holdout。若现在用于完成 PLP-only，便不能
再作为 Hybrid 的“未见 Test”；以后做 Hybrid 必须重新设计一套 holdout。确认接受这一点后，
才运行：

```bash
python scripts/open_plp_v3_test_gate.py --confirm-final-test
python scripts/collect_hybrid_v3_dataset.py \
  --splits test \
  --test-owner plp-terminal-v3 \
  --confirm-final-test
python scripts/evaluate_plp_v3_final.py
```

最终报告会同时输出总体指标、task、short/medium/long、3×3、seed、解码进度和
terminal/nonterminal 分析，以及 terminal-zero 相对 v2 的 family 配对 bootstrap 区间。

# Bayesian Sequential 第六阶段：不确定性、收敛与 serving replay

## 1. 科学边界

第六阶段只读取第四阶段真实 Qwen trace 与第五阶段冻结的 family-grouped OOF 产物。它不训练
模型、不调阈值、不做 robustness refit，也不创建或访问 final holdout。

分析包括：

- ALPS countdown 与选中 Bayesian scalar 的 NLL、CRPS、coverage/width、方差和熵进度曲线；
- 从五折 checkpoint 精确重放得到 2.5%/50%/97.5% uncertainty cone；
- “当前及后续所有保存点都保持总长度相对误差不超过 5%”的严格收敛时间；
- 主温度真实总长度经验 top 10% 的早期低估；
- 第五阶段 RTX 4090 逐更新计时相对第四阶段真实 Qwen 生成时延的开销；
- 固定 bucket、batch、KV quantum 与真实生成时延的确定性离线 serving replay。

方差下降不自动等于不确定性成功，必须与 coverage 和 interval width 联合解释。Serving replay
也不是 vLLM 或生产系统实测，不形成生产 serving superiority claim。

## 2. 本地路径

示例：

```bash
export STAGE4_ROOT="/absolute/path/to/stage4_rsync/extracted"
export STAGE5_ROOT="/absolute/path/to/stage5_oof/extracted/artifacts/runs/bayesian_sequential_v1/stage5_oof"
```

Stage 4 根目录必须包含 1,620 个 trace、collection report 和 index。Stage 5 根目录必须包含
五折 checkpoint、逐更新点预测、OOF report、selection 和 `file_sha256.txt`。

## 3. 首次强校验

```bash
python scripts/preflight_bayesian_stage6.py \
  --stage4-root "$STAGE4_ROOT" \
  --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files
```

必须看到：

- `status: pass`、`ready: true`；
- Stage 4 trace 为 1,620；
- Stage 5 为 5 folds、1,620 sequence、137,957 observation；
- selected method 为 `bayesian_entropy_scalar_v1`；
- 51 个 Stage 5 manifest 文件重新通过 SHA-256；
- `model_refit_performed: false`、`final_holdout_accessed: false`。

## 4. 完整分析

```bash
python scripts/run_bayesian_stage6_analysis.py \
  --stage4-root "$STAGE4_ROOT" \
  --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files
```

若希望同时重新计算 1,620 个 Stage 4 NPZ 的 SHA，可增加 `--verify-trace-hashes`。该开关不
改变分析结果，只增加输入完整性检查时间。

输出位于：

```text
artifacts/runs/bayesian_sequential_v1/stage6_analysis/
├── environment/preflight.json
├── uncertainty_curves.csv
├── uncertainty_cone.csv
├── serving_replay.json
├── stage6_report.json
└── stage6_summary.json
```

`uncertainty_cone.csv` 是按 temperature × decode-progress 聚合的 sequence-balanced cone，避免
把 137,957 个逐请求点再次复制进 Git。精确逐点 posterior 仍可由冻结 checkpoint 重放。

## 5. Serving replay 口径

在第一保存点 `step=1` 比较七种策略：oracle、固定 4096、ALPS mean、PLP v3、concat v1、
Bayesian mean 和 Bayesian 97.5% 上界。所有预测按 16-token quantum 分配，并按冻结长度 bucket
排序后以 batch size 8 重放。Batch 时延取该 batch 内真实 Qwen rollout 时延最大值。

Qwen2.5-7B-Instruct 的冻结结构为 28 层、28 attention heads、4 KV heads、head dimension 128、
BF16；因此只计算 output-token 增量 KV 时，每 token 为 57,344 bytes。Prompt KV、权重、临时
activation 和调度器开销不在本 replay 中，不能把这里的 KV 数字写成整台服务器显存峰值。

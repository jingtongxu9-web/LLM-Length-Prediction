# Bayesian Sequential 第七阶段：OOF error feedback

第七阶段只审计第五阶段冻结的 Train-family OOF 预测和第四阶段统一 trace。它不训练、不重新选择
方法、不调阈值，也不创建或打开 final holdout。`status=pass` 仅表示审计完整且边界合规。

## 数据位置

```bash
cd "/Users/mininetfly/Documents/LLM Length Prediction/work/LLM-Length-Prediction-main"

export STAGE4_ROOT="/Users/mininetfly/Desktop/LLM Length Prediction/第四阶段实验结果/stage4_rsync/extracted"
export STAGE5_ROOT="/Users/mininetfly/Desktop/LLM Length Prediction/第五阶段实验结果/extracted/artifacts/runs/bayesian_sequential_v1/stage5_oof"
export PYTHONPATH=src
```

## 预检与运行

```bash
python scripts/preflight_bayesian_stage7_error_feedback.py \
  --stage4-root "$STAGE4_ROOT" \
  --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files

python scripts/run_bayesian_stage7_error_feedback.py \
  --stage4-root "$STAGE4_ROOT" \
  --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files \
  --verify-trace-hashes
```

本阶段主要是 CPU/磁盘读取和统计，不需要 RTX 4090。全量逐 trace SHA-256 校验在 Mac 上约十几
秒完成；如果以后只在 AutoDL 保留数据，同样命令也能运行，但没有科学上的 GPU 必要。

## 冻结 cohort

- `absolute-error cohort`：一条序列任一保存点 `|error| > 100 token`；
- `worst-5% cohort`：按每条序列所有保存点的 MAE，使用 `higher` 95% 分位数；
- 人工 review queue：以上两者的并集；
- 分析单元始终是 `(prompt_id, temperature, seed)`，不把保存点当独立样本。

自动标签的固定阈值均在
[`../../configs/experiments/bayesian_sequential_stage7_error_feedback_v1.json`](../../configs/experiments/bayesian_sequential_stage7_error_feedback_v1.json)
中。自动标签是可复核的 trace 现象，不是因果诊断。

`open_ended_prompt` 需要阅读 Prompt；`hallucination` 需要解码输出并结合任务参考或人工判断。
当前 frozen trace 没有可靠完成这两件事所需的完整语义证据，因此这两个字段必须保留
`unresolved`，不能自动填 `false`。

## 输出

```text
artifacts/runs/bayesian_sequential_v1/stage7_error_feedback/
├── environment/preflight.json
├── sequence_audit.jsonl
├── manual_review_queue.jsonl
├── stage7_report.json
└── stage7_summary.json
```

完整 JSONL 是本地可重建产物，不提交 Git；Git 只保存聚合报告。任何理论修正必须创建新的
method ID 并重新执行完整 family-grouped OOF，不能修改 `bayesian_entropy_scalar_v1`。

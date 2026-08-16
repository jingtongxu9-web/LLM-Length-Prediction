# Bayesian Sequential 第八阶段：最终模型冻结与一次性盲测

## 当前允许执行的边界

第八阶段分为两个不可交换的子阶段：

1. **Stage-8A**：只使用已经打开的 60 个 Train family、`temperature=0.7` 的 540 条
   trace，拟合七个比较方法所需的最终模型并生成 checkpoint registry；
2. **Stage-8B**：Stage-8A 分支合并、最终模型哈希冻结、12 个新 family 经过独立语义重叠
   复核后，才创建 ready lock，随后一次性采集 324 条 final-holdout trace 并评测。

Stage-8B candidate、审计证据和 ready lock 现已在锁定分支生成；模板文件仍故意保持不可运行。
锁定分支合并到 `main` 且服务器外层 preflight 通过之前，不读取或采集新的 final holdout。

## Stage-8A：服务器最终训练

从已合并到 `main` 的 Stage-8A 提交开始，并使用和 Stage 5 完全相同的环境：Python 3.12.3、
PyTorch 2.8.0+cu128、NumPy 2.2.2、scikit-learn 1.6.1、CUDA 12.8、RTX 4090 D、BF16。

```bash
cd /root/autodl-tmp/LLM-Length-Prediction
source /root/autodl-tmp/venvs/llm-length/bin/activate

export STAGE4_DATA_ROOT=/root/autodl-tmp/LLM-Length-Prediction
export OMP_NUM_THREADS=8

python -m pip install --no-deps --editable .

python scripts/preflight_bayesian_stage8a.py \
  --dataset-root "$STAGE4_DATA_ROOT" \
  --verify-trace-hashes \
  --verify-training-environment
```

preflight 必须显示 540 条训练 trace、60 个 family、主温度 0.7、环境已验证、final holdout
未打开。然后运行：

```bash
nohup env OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  /root/autodl-tmp/venvs/llm-length/bin/python -u \
  scripts/train_bayesian_final_models.py \
  --dataset-root "$STAGE4_DATA_ROOT" \
  --device auto --verify-trace-hashes \
  > artifacts/runs/bayesian_sequential_v1/final_models_train.log 2>&1 &

echo $! > artifacts/runs/bayesian_sequential_v1/final_models_train.pid
```

进度查看：

```bash
PID=$(cat artifacts/runs/bayesian_sequential_v1/final_models_train.pid)
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
tail -n 30 artifacts/runs/bayesian_sequential_v1/final_models_train.log
nvidia-smi
```

训练结束后必须完整恢复全部模型：

```bash
python scripts/preflight_bayesian_final_models.py

sha256sum \
  artifacts/runs/bayesian_sequential_v1/final_models/checkpoint_registry.json
```

输出目录包含 ALPS prior、Prompt Ridge、Dynamic MLP、PLP v3、concat v1、Bayesian scalar、
Bayesian hidden-delta 以及训练报告。不得只保留 scalar；七方法最终对比需要完整 registry。

## Stage-8B：ready lock 的合并边界

Stage-8A 结果带回本地后，先在隔离于模型拟合和选择的流程中执行：

```bash
python scripts/audit_bayesian_stage8b_candidate.py
python scripts/finalize_bayesian_stage8b_lock.py
```

审计同时覆盖所有历史 JSONL manifest 的 exact ID、family、规范化全文，以及去掉长度模板后的
字符序列与 trigram 相似度；逐 family 人工主题判断另存于
`configs/reviews/bayesian_sequential_stage8b_semantic_review_v1.json`。生成器只有在以下内容全部
通过时才写 `bayesian_sequential_stage8b_lock_v1.json`：

- Stage-8A 代码已合并到远端 `main`；
- checkpoint registry 及每个模型 SHA-256 已带回并复验；
- 12 个全新 family、36 个 Prompt 已由与模型训练隔离的流程编写；
- 与仓库内所有历史 manifest 完成 exact ID、family、规范化文本和人工语义重叠复核；
- `bayesian_sequential_stage8b_lock_v1.json` 填入 config、Git commit、registry、七个模型和
  manifest 哈希；
- `prompt_semantic_overlap_review_complete=true`，同时仍保持
  `final_holdout_opened=false` 和 `final_holdout_selects_nothing=true`。

ready lock 在功能分支上出现不代表可以采集。`load_final_holdout_contract` 还要求工作区干净且
`HEAD == origin/main`，因此必须先审查并合并锁定分支。合并后，4090 服务器重新拉取 `main`、
恢复 Stage-8A final-model 目录，并运行：

```bash
python scripts/preflight_bayesian_stage8b_ready.py \
  --verify-model-loading
```

只有报告同时显示 `ready=true`、七模型成功恢复、`final_holdout_opened=false` 和
`final_holdout_accessed=false`，才进入下面的一次性采集入口。

## 解锁后的唯一采集与评测入口

只有 gate 报告 `ready=true` 后，才允许在 4090 上执行：

```bash
python scripts/collect_bayesian_final_holdout.py \
  --model "$MODEL_PATH" --max-new-jobs 25

python scripts/collect_bayesian_final_holdout.py --report-only
```

重复 25-job 命令直到 `valid=324 missing=0 status=pass`。collector 原子写入、严格验证并可
断点续跑；已有无效文件会中止且不会覆盖。最后执行一次：

```bash
python scripts/run_bayesian_final_benchmark.py \
  --device auto --verify-trace-hashes
```

final benchmark 同时生成七方法预测、概率与点指标、分组结果、uncertainty cone、严格稳定
5% 收敛、描述性 paired-family bootstrap 和冻结 serving replay。它不选择模型、不调整阈值、
不根据 final holdout 重新训练。

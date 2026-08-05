# ALPS+PLP Hybrid v3：直接 Python 服务器完整手册

适用于已经配置 CUDA-enabled PyTorch、**不使用 Docker** 的服务器。以下命令默认项目位于
`/data/home/mulei/Summer_Camp`，从项目根目录执行。不要在完成 OOF 和冻结最终模型之前运行
任何带 `test` 的 v3 命令。

## 0. 设置路径并检查机器

```bash
mkdir -p /data/home/mulei/Summer_Camp
cd /data/home/mulei/Summer_Camp

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h /data
```

意义：确认当前 shell、GPU、PyTorch/CUDA 和大盘空间正确。v3 trace、模型与日志都写到独立
目录，不覆盖 ALPS v1 或 PLP v2。

## 1. 获取合并后的固定代码

```bash
# 仅当目录还是空目录时执行这一条：
git clone https://github.com/jingtongxu9-web/LLM-Length-Prediction.git .

# 如果目录已经是该仓库，从这里开始：
git pull --ff-only origin main
mkdir -p data/interim/alps_plp_hybrid_v3 artifacts/runs/alps_plp_hybrid_v3
export MODEL_PATH=/data/home/mulei/Summer_Camp/models/Qwen2.5-7B-Instruct
git rev-parse HEAD | tee artifacts/runs/alps_plp_hybrid_v3/environment_repo_commit.txt
git status --short
```

如果目录已是仓库，只执行后三条。`git status --short` 应为空；把 commit SHA 留作结果溯源。

## 2. 安装锁定依赖，不替换服务器已有 CUDA PyTorch

```bash
python -m pip install --requirement requirements-autodl.lock
python -m pip install --no-deps --editable .
python -m pip freeze > artifacts/runs/alps_plp_hybrid_v3/environment_pip_freeze.txt
```

意义：固定 Transformers、scikit-learn、SciPy、pytest 和 ruff，同时保留服务器镜像提供的
CUDA PyTorch。

## 3. 校验 Prompt、模型 revision 和环境

如果模型目录尚不存在，先下载冻结 snapshot；已经完整下载过则跳过：

```bash
python scripts/download_model.py \
  --experiment configs/experiments/alps_plp_hybrid_v3_base.json \
  --output "$MODEL_PATH"
```

随后执行全部校验：

```bash
python scripts/build_hybrid_v3_manifest.py --check

python scripts/preflight_server.py \
  --experiment configs/experiments/alps_plp_hybrid_v3_base.json \
  --model "$MODEL_PATH"

test "$(tr -d '\r\n' < "$MODEL_PATH/.frozen_revision")" = \
  "a09a35458c702b33eeacc393d103063234e8bc28"
```

意义：确认 216 条 Prompt 可重复生成且 SHA-256 未变；模型和 tokenizer 都指向冻结 commit。
任何失败都应在采集前解决。

## 4. 运行本地检查

```bash
python -m pytest
python -m ruff check --no-cache .
```

意义：验证 schema、family 隔离、terminal bin、删失门槛和统计函数。测试不加载 7B 模型，
不能替代下一步 GPU pilot。

## 5. 只采集 6 条 Train pilot

```bash
set -o pipefail
python scripts/collect_hybrid_v3_dataset.py \
  --model "$MODEL_PATH" \
  --splits train \
  --limit 6 \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/pilot.log
```

意义：用真实 Qwen 同时验证 Layer 14 prior、最终层 Prompt/decode hidden state、entropy、EOS
概率、完整 token provenance 和原子 NPZ。命令可安全重跑；已通过校验的 trace 会跳过。

检查 pilot：

```bash
python -c "import json; p='artifacts/runs/alps_plp_hybrid_v3/collection_summary.json'; print(json.load(open(p)))"
find data/interim/alps_plp_hybrid_v3/train -name '*.npz' | wc -l
```

## 6. 完成全部 540 条 Train rollout

```bash
set -o pipefail
python scripts/collect_hybrid_v3_dataset.py \
  --model "$MODEL_PATH" \
  --splits train \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/train_collection.log
```

意义：生成一次共享 trajectory；全部 8 种方法都读取这些相同 trace。完成后 summary 中
`by_split.train` 应为 540。若断线，重复同一命令即可续跑。

## 7. 跑严格 grouped OOF

```bash
set -o pipefail
python scripts/evaluate_hybrid_v3_oof.py --device auto \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/oof.log
```

意义：执行 5-fold family-grouped OOF，并在每个外层 Train 内对 ALPS prior 再做 4-fold
cross-fit；同时生成八种方法的 OOF prediction、家族级指标和置信区间。它不会读取 Test。

检查：

```bash
python -c "import json; p='artifacts/runs/alps_plp_hybrid_v3/oof/oof_report.json'; r=json.load(open(p)); print(r['censoring']); print(list(r['methods'])); print(r['hybrid_paired_differences'])"
```

若 censoring status 为 `abort`，代码会直接停止；不要靠删除失败样本绕过门槛。OOF 结果仅用于
检查稳定性，不根据它修改冻结超参数，否则必须创建新 protocol/version。

## 8. 用全部 Train 冻结最终八种模型

```bash
set -o pipefail
python scripts/train_hybrid_v3_models.py --device auto \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/final_training.log
```

意义：用全 Train 训练最终模型；stacking head 仍使用 cross-fitted ALPS summaries。脚本保存
每个模型文件的 SHA-256 到 `models/model_registry.json`。此后不要改代码、配置或模型。

## 9. 最后检查并一次性打开 Test gate

这是不可逆的实验边界。确认不再调参后才执行：

```bash
python scripts/open_hybrid_v3_test_gate.py --confirm-final-test
```

意义：重新运行 pytest 与 ruff；验证 OOF、Train digest、八种模型、全部哈希，并确认不存在
任何 v3 Test trace；然后创建 `final_test/OPENED.json`。gate 打开后只能完成预注册流程，不能
根据 Test 结果返回修改模型再测。

## 10. 一次采集全部 108 条 Test rollout

```bash
set -o pipefail
python scripts/collect_hybrid_v3_dataset.py \
  --model "$MODEL_PATH" \
  --splits test \
  --confirm-final-test \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/test_collection.log
```

意义：只对 12 个新 family 生成 Test；重复命令仅用于断点续跑，不能改变模型或 protocol。
完成后 collection summary 的 `by_split.test` 应为 108。

## 11. 生成唯一的最终统计报告

```bash
set -o pipefail
python scripts/evaluate_hybrid_v3_final.py \
  2>&1 | tee artifacts/runs/alps_plp_hybrid_v3/final_evaluation.log
```

主要输出：

- `final_test/predictions.csv`：每个保存点的真实值和八种预测；
- `final_test/final_report.json`：总体/家族宏平均指标、absolute-step 分组、配对 CI；
- `primary_claim_result.passed`：七个 Bonferroni CI 上界是否全部小于 0。

`passed=true` 才支持“Hybrid 在此冻结 holdout 的 prediction MAE 上优于全部七个对照”。它不
自动证明真实 serving 更好，也不代表所有模型/数据分布。

## 12. 跑冻结的离线 serving replay

```bash
python scripts/run_hybrid_v3_serving_benchmark.py
```

意义：按冻结 bucket、batch size 与 KV allocation quantum 比较完成时间、吞吐、KV
over-reservation 和 underallocation。输出为 `serving/serving_report.json`。这是 deterministic
offline replay，不应写成真实生产 serving 测量。

## 13. 备份完整证据

```bash
tar -czf alps_plp_hybrid_v3_artifacts.tar.gz \
  artifacts/runs/alps_plp_hybrid_v3
sha256sum alps_plp_hybrid_v3_artifacts.tar.gz \
  > alps_plp_hybrid_v3_artifacts.tar.gz.sha256
```

至少带走压缩包与 SHA 文件。原始 trace 体积更大，若需完整审计，再单独备份
`data/interim/alps_plp_hybrid_v3/`。不要把模型权重、trace 或未脱敏的生成内容提交 GitHub。

## 故障恢复原则

- Train/Test collection：重复原命令，自动跳过合法文件；
- OOF/训练失败：在未打开 Test 前可按相同配置重跑；
- gate 已打开：不得修改协议或模型；只能续采 Test、最终评价和备份；
- censoring ≥10%：实验按协议终止，先报告原因；若改变 `max_new_tokens`，必须命名新版本；
- CUDA OOM：先确认没有其他进程占显存；不要偷偷改 BF16、模型或生成上限。必须改变条件时
  建立新配置和实验 ID。

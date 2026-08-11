# Bayesian Sequential 第三阶段服务器 Pilot

## 1. 哪些工作必须上服务器

第三阶段分成两部分：

| 工作 | 本地即可 | GPU 服务器 |
|---|---:|---:|
| trace schema、配置、断点续跑、合同测试 | 是 | 否 |
| fake causal-LM collector 测试 | 是 | 否 |
| 加载 Qwen2.5-7B-Instruct BF16 | 否 | 是 |
| 真实逐 token entropy/EOS 与 hidden-state 采集 | 否 | 是 |
| 9-rollout pilot 的显存、磁盘、terminal 对齐验收 | 否 | 是 |

本阶段不训练 Bayesian scorer，不采集全量 Train，不创建或打开 final holdout。

## 2. 冻结的 Pilot 范围

配置：
[`../../configs/experiments/bayesian_sequential_pilot_v1.json`](../../configs/experiments/bayesian_sequential_pilot_v1.json)

- 只使用已打开的 Train design families；
- family：`qa_001_gradient_descent`、`summarization_001_urban_transit`、
  `code_001_lru_cache`；
- 每个 family 使用 short、medium、long；
- temperature `0.7`、seed `42`；
- 共 9 个 rollout；
- `max_new_tokens=4096`，不为 pilot 缩短长度口径；
- 输出到 `data/interim/bayesian_sequential_v1_pilot/`；
- 验收报告输出到 `artifacts/runs/bayesian_sequential_v1/pilot/`。

## 3. 服务器要求

- NVIDIA GPU，建议 32 GiB；最低合同为 24 GiB；
- BF16；
- PyTorch `>=2.6`、Transformers `>=4.48`；
- Blackwell GPU 使用 CUDA `>=12.8` 的 PyTorch build；
- 建议至少 5 GiB 可写磁盘用于 pilot、安全余量和临时文件；
- Qwen 模型目录包含匹配 revision 的 `.frozen_revision`。

不要把模型权重、`data/interim/` 或 `artifacts/` 提交到 Git。代码通过 Git 分支转移，模型权重和
实验产物通过服务器磁盘、对象存储或 `rsync` 管理。

## 4. 服务器命令

在服务器仓库根目录执行：

```bash
python -m pip install -e '.[dev,model]'
export MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct

python scripts/preflight_bayesian_pilot.py
python scripts/collect_bayesian_pilot.py --limit 1
python scripts/collect_bayesian_pilot.py --limit 3
python scripts/collect_bayesian_pilot.py
```

`--limit 1` 与 `--limit 3` 是显存、字段和断点续跑 smoke test，会把状态报告为 incomplete，
这是预期行为。最后一个无 `--limit` 命令会复用已经验证通过的 trace，只补齐缺失的 rollout。

如果服务器模型不在 `MODEL_PATH`，可显式传入：

```bash
python scripts/preflight_bayesian_pilot.py --model /absolute/model/path
python scripts/collect_bayesian_pilot.py --model /absolute/model/path
```

## 5. 必须带回的文件

```text
artifacts/runs/bayesian_sequential_v1/pilot/environment/preflight.json
artifacts/runs/bayesian_sequential_v1/pilot/collection_index.jsonl
artifacts/runs/bayesian_sequential_v1/pilot/pilot_acceptance.json
data/interim/bayesian_sequential_v1_pilot/train/.../*.npz
```

其中 `pilot_acceptance.json` 必须满足：

- `status == "pass"`；
- `valid_trace_count == 9`；
- `missing_selected_job_count == 0`；
- `by_task_length` 有 9 个 cell；
- `real_qwen_pilot_complete == true`；
- `final_holdout_accessed == false`；
- censoring rate 未达到 `0.34` abort threshold；
- model/tokenizer revision、CUDA peak memory 和 trace SHA 全部存在。

## 6. Pilot 后的人工检查

抽查 short、medium、long 各一条：

1. `generated_token_ids` 数量等于 `observed_tokens`；
2. entropy/EOS 数组覆盖每一个 token，而不是只覆盖更新点；
3. hidden-state steps 严格为 `1,5,10,...,+EOS terminal`；
4. EOS rollout 的最后一个 token 是合法 EOS，terminal 对应 `R_t=0`；
5. `max_new_tokens` rollout 没有伪造 terminal zero；
6. Layer-14、final-layer、temperature-before-top-p 和模型 revision 与合同一致；
7. trace 大小、单条耗时和峰值显存允许后续 full Train 采集。

只有这些检查通过后，才能为第四阶段冻结 full Train 采集预算与服务器运行计划。

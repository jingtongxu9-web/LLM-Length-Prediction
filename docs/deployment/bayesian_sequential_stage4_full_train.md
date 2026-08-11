# Bayesian Sequential 第四阶段：完整 Train trace 采集

## 1. 本阶段做什么

第四阶段只让冻结 revision 的 Qwen2.5-7B-Instruct 生成完整 **Train** trace。它不会训练
ALPS prior、PLP、Bayesian scalar 或 Bayesian hidden-delta，也不会访问或创建 final holdout。
完成后，第五阶段的全部方法在同一批 trace 上离线进行 family-grouped OOF，不再为每个方法
重复生成回答。

冻结规模：

| 维度 | 数量 |
|---|---:|
| Train Prompt | 180 |
| Prompt family | 60 |
| task × intended-length cell | 9（每格 20 Prompt） |
| temperature | `0.3 / 0.7 / 1.0` |
| seed | `42 / 43 / 44` |
| 总 rollout | `180 × 3 × 3 = 1620` |

主开发 temperature 是 `0.7`；`0.3/1.0` 只用于冻结 robustness 分析，不能用于重新选择或
拟合方法。大体积原始 trace 不提交 Git。

## 2. 已冻结的预算

第三阶段 RTX 4090 pilot 的 9 条 trace 共耗时 `77,237.046 ms`、占用 `5,652,103 bytes`，
峰值 CUDA reserved memory 为 `15,758,000,128 bytes`。按相同比例外推：

- 理想 GPU 时间约 `3.86 h`；冻结的 2 倍运行预算约 `7.72 h`；
- 压缩 NPZ 预计约 `0.95 GiB`；
- 最坏未压缩 trace 上界约 `17.90 GiB`；
- 空目录开跑前要求至少 `75 GiB` 可用空间；
- 模型和数据合计建议使用至少 `100 GiB` 数据盘。

这些是容量预算，不是最终实际结果。若 preflight 报告可用空间不足，不要开始采集，应先在
AutoDL 无卡模式扩容数据盘。

## 3. 无卡模式准备

建议先在无卡模式完成 Git 同步和磁盘检查，避免为下载或解压付 GPU 费用：

```bash
cd /root/autodl-tmp/LLM-Length-Prediction
git pull --ff-only
git rev-parse --short HEAD
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

如果 AutoDL 到 GitHub 仍然超时，应从本地导出当前提交并上传，不能在服务器上手工复制几个
Python 文件。开始 GPU 采集前，服务器 HEAD 必须与本地冻结的第四阶段提交一致。

## 4. 启用 RTX 4090 后设置环境

```bash
cd /root/autodl-tmp/LLM-Length-Prediction
conda activate llm-length

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export OMP_NUM_THREADS=8

git rev-parse --short HEAD
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
du -sh "$MODEL_PATH"
```

`OMP_NUM_THREADS` 必须是正整数。不要写成空值、浮点数或带注释的字符串，否则 `libgomp`
会在 Python 启动时报警。

## 5. 先执行 full-Train preflight

```bash
python scripts/preflight_bayesian_full_train.py --model "$MODEL_PATH"
```

报告写入：

```text
artifacts/runs/bayesian_sequential_v1/full_train/environment/preflight.json
```

空目录首次运行必须看到：

- `ready: true`；
- `selected_prompt_count: 180`；
- `expected_rollout_count: 1620`；
- `collection_config_sha256: dd96edfc6ebca7800f372906db3de68df93785b62e313d54d3dd464ab220f9f7`；
- `stage3_pilot_status: pass`；
- `final_holdout_accessed: false`；
- Qwen model/tokenizer revision 完全匹配；
- CUDA、BF16、4090 标称 24 GB 和 Stage 3 峰值显存检查通过；
- `free_disk_gib >= required_free_disk_gib`。

任一 `failures` 非空都应先停止并处理。Preflight 会根据已经存在的 trace 数量降低续跑所需的
剩余磁盘预算，但 collector 仍会逐文件做完整合同校验。Full-Train 配置还固定了 collector、
preflight、统一 trace 和 Qwen instrumentation 源文件的 SHA-256；服务器代码与冻结实现只要有
一处不同，preflight 和 collector 都会在加载模型前停止。

## 6. 分块采集和断点续跑

第一次只新增 100 条：

```bash
python scripts/collect_bayesian_full_train.py \
  --model "$MODEL_PATH" \
  --max-new-jobs 100
```

检查进度报告：

```bash
python -m json.tool \
  artifacts/runs/bayesian_sequential_v1/full_train/collection_report.json
```

只要 `failures` 为空，就重复同一个 100 条命令。已存在的合法 NPZ 会被重新读取并严格验证，
然后跳过；缺失任务继续生成。若进程中断，已经原子写完的 trace 不会丢失，下次仍用相同命令
续跑。默认不写 `--max-new-jobs` 也等于 100 条。

不要删除错误 trace 后假装续跑。任何既有文件只要 hash 对应的配置、job、revision、shape、
stride、CUDA provenance 或 final-holdout 标记不匹配，collector 会停止且**不会覆盖原文件**。
应先保存错误信息和文件，再回到本地诊断。

如果在至少 30 条有效 trace 后 censoring rate 达到 `10%`，报告会给 warning；至少 90 条后
达到 `34%` 会给 failure 并停止下一批采集。此时不能通过修改阈值继续运行。

## 7. 完成验收

当预计已采集完时，先执行不加载 Qwen 的全量复验：

```bash
python scripts/collect_bayesian_full_train.py --report-only

python -m json.tool \
  artifacts/runs/bayesian_sequential_v1/full_train/collection_report.json

wc -l \
  artifacts/runs/bayesian_sequential_v1/full_train/collection_index.jsonl

find data/interim/bayesian_sequential_v1 \
  -type f -name '*.npz' | wc -l
```

只有以下条件同时成立才算第四阶段采集通过：

- `status: pass`；
- `valid_trace_count: 1620`，`missing_trace_count: 0`；
- 60 个 family、9 个 task-length cell、3 个 temperature、3 个 seed 全部覆盖且数量平衡；
- 每个 trace 的数组、first/stride/terminal schedule、EOS/censoring 语义和 provenance 合法；
- `full_train_collection_complete: true`；
- `final_holdout_accessed: false`；
- index 行数和 NPZ 数量均为 `1620`。

`status: incomplete` 只是安全的中间进度，不是失败，也不是阶段完成。

## 8. 归档并下载回本地

先生成文件级 hash 清单，再归档：

```bash
find data/interim/bayesian_sequential_v1 \
  artifacts/runs/bayesian_sequential_v1/full_train \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > artifacts/runs/bayesian_sequential_v1/full_train/file_sha256.txt

tar -czf /root/autodl-tmp/bayesian_stage4_full_train.tar.gz \
  data/interim/bayesian_sequential_v1 \
  artifacts/runs/bayesian_sequential_v1/full_train \
  configs/experiments/bayesian_sequential_full_train_v1.json \
  configs/reports/bayesian_sequential_full_train_report_schema.json

sha256sum /root/autodl-tmp/bayesian_stage4_full_train.tar.gz
```

下载 archive 后，保留服务器给出的 archive SHA-256。服务器不要在本地复验完成前释放；本地
需要再次检查 archive hash、1,620 个 NPZ、index、报告和逐 trace 合同。只有本地复验通过后，
才更新第四阶段结果摘要并进入第五阶段 OOF。

# ALPS v1 诊断：服务器操作清单

以下命令在服务器仓库根目录执行。它们不会重新生成 Qwen 输出，也不会读取 v1 final test
来选择超参数。

## 1. 更新代码并确认训练 trace

```bash
git pull --ff-only

test -d data/interim/alps_v1/train
find data/interim/alps_v1/train -name 'seed_*.jsonl' | wc -l
```

预期训练 trace 数量为 `432`。如果不是 432，先不要运行诊断。

## 2. 运行测试

```bash
python -m pytest -q
ruff check --no-cache .
```

## 3. Family-grouped CV 和基线

```bash
set -o pipefail

python scripts/evaluate_grouped_cv.py \
  2>&1 | tee artifacts/runs/alps_v1/logs/grouped_cv.log
```

输出：

```text
artifacts/runs/alps_v1/diagnostics/grouped_cv/results.json
artifacts/runs/alps_v1/diagnostics/grouped_cv/summary.csv
```

该命令比较：

- global mean；
- Prompt token count；
- task + intended length metadata；
- metadata + Prompt token count；
- Layer 14 ALPS hidden state。

并在 Train 内扫描 Ridge alpha：

```text
0.01, 0.1, 1, 10, 100, 1000
```

## 4. 学习曲线

先根据 grouped-CV 的结果选择一个 Train-OOF alpha，再执行：

```bash
set -o pipefail

python scripts/plot_learning_curve.py --alpha SELECTED_ALPHA \
  2>&1 | tee artifacts/runs/alps_v1/logs/learning_curve.log
```

`SELECTED_ALPHA` 只能根据 Train grouped-CV 确定，禁止根据 v1 Test 修改。

输出：

```text
artifacts/runs/alps_v1/diagnostics/learning_curve/results.json
artifacts/runs/alps_v1/diagnostics/learning_curve/learning_curve.csv
```

## 5. 封存 ALPS v1

```bash
python scripts/archive_alps_v1.py

sha256sum --check \
  artifacts/runs/alps_v1/archive/checksums.sha256
```

输出：

```text
artifacts/runs/alps_v1/archive/archive_metadata.json
artifacts/runs/alps_v1/archive/checksums.sha256
```

## 6. 返回分析文件

将以下目录压缩后复制回本地：

```bash
tar -czf alps_v1_diagnostics.tar.gz \
  artifacts/runs/alps_v1/diagnostics \
  artifacts/runs/alps_v1/archive \
  artifacts/runs/alps_v1/logs/grouped_cv.log \
  artifacts/runs/alps_v1/logs/learning_curve.log
```

在完成上述诊断之前，不需要采集 ALPS v2 trace，也不应打开任何新的 Test。

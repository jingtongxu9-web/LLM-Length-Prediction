# ALPS v2 本地诊断与服务器分工

ALPS v1 的最终测试集已经打开。它只能保留为历史结果，禁止继续用于选择 Ridge alpha、
特征层、PCA 维数或概率校准方法。

## 本地可完成

1. 检查并冻结 Prompt family 级切分规则。
2. 使用合成数据运行单元测试。
3. 维护 grouped-CV、baseline、学习曲线和封存脚本。
4. 设计 ALPS v2 Prompt schema、数量和质量检查。

## 需要训练 trace，但不需要重新运行 Qwen

从服务器复制 `data/interim/alps_v1/train/` 后，可在任何内存足够的机器执行：

```bash
python scripts/evaluate_grouped_cv.py
python scripts/plot_learning_curve.py
python scripts/archive_alps_v1.py
```

这些命令只读取 v1 train trace，不访问 v1 test 来选择超参数。

## 必须在 GPU 服务器完成

1. 使用冻结模型采集 ALPS v2 Train、Calibration、Test trace。
2. Train 先使用一个 seed，Calibration/Test 使用三个 seed。
3. 在打开 Test 前冻结模型选择和校准方法。
4. 只在最终配置确定后执行一次 Test。

## Gate

只有当 ALPS 在 family-grouped OOF 和新的 final Test 上都稳定超过
`metadata_prompt_tokens` baseline，且校准后的 95% coverage 接近目标时，才进入 PLP。

# Bayesian Sequential 第五阶段：Train-family OOF 与方法选择

## 1. 本阶段的科学边界

第五阶段第一次训练和验证真正的 Bayesian Sequential v1。它使用第四阶段已经冻结并在本地
复验通过的 1,620 条统一 trace，不重新加载 Qwen，不生成新回答，也不访问 final holdout。

主流程是：

```text
temperature=0.7 的外层 Train family
    -> 内层 4-fold family cross-fit ALPS prior/variance
    -> 拟合 baseline、Bayesian scalar、Bayesian hidden-delta
    -> 冻结该折全部参数
    -> 在外层 validation family 的 0.3/0.7/1.0 trace 上统一推断
```

`0.3/1.0` 只用于 robustness evaluation，严禁参与 scaler、prior、神经网络或阈值拟合。五折
OOF 按 `prompt_family_id` 分组；同一 family 的三种长度、三个 seed、三个 temperature 和全部
timestep 永远处于同一外层折。

Bayesian scalar 与 hidden-delta 按主开发温度 `0.7` 上的
`family_macro_sequence_balanced_posterior_nll` 比较。只有 hidden-delta 减 scalar 的 95% family
paired bootstrap CI 完全低于 0 才选择 hidden-delta，否则保留 scalar。OOF 选择完成前及完成后
均不允许 final holdout 参与选择。

## 2. 本地数据位置

第四阶段 archive 可以保留在仓库外。例如当前 Mac 路径：

```text
/Users/mininetfly/Desktop/LLM Length Prediction/实验结果/stage4_rsync/extracted
```

代码通过 `--dataset-root` 读取该目录，仓库不复制约 0.93 GiB NPZ，也不把大体积数据提交
Git。第五阶段配置冻结了 Stage-4 report、index、dataset 和 archive SHA-256。

## 3. 首次强校验

进入仓库的 Python 环境后运行：

```bash
export STAGE4_DATA_ROOT="/Users/mininetfly/Desktop/LLM Length Prediction/实验结果/stage4_rsync/extracted"

python scripts/preflight_bayesian_stage5_oof.py \
  --dataset-root "$STAGE4_DATA_ROOT" \
  --verify-trace-hashes
```

首次使用必须加 `--verify-trace-hashes`，逐个重算 1,620 个 NPZ 的 SHA。必须看到：

- `status: pass`、`ready: true`；
- `trace_count: 1620`、`family_count: 60`；
- `training_trace_count: 540`；
- 五折各 12 个 family；
- `final_holdout_accessed: false`；
- `failures: []`。在 Mac 没有 CUDA 时会出现一条“完整五折训练较慢”的 warning，这是硬件
  建议而不是数据失败；在服务器 RTX 4090 上应看到 CUDA 可用。

报告写入：

```text
artifacts/runs/bayesian_sequential_v1/stage5_oof/environment/preflight.json
```

## 4. 运行五个可恢复 fold

Mac 可以先做代码 smoke，但完整神经网络 OOF 建议使用 RTX 4090。服务器只需上传当前代码和
第四阶段 archive（或其解压目录），不需要 Qwen 模型权重。每个 fold 有独立目录和最终
`fold_report.json`；报告存在且数据 digest 匹配时，重复命令会安全跳过。

依次运行：

```bash
for FOLD in 0 1 2 3 4; do
  python scripts/run_bayesian_stage5_fold.py \
    --dataset-root "$STAGE4_DATA_ROOT" \
    --fold "$FOLD" \
    --device auto
done
```

每折固定：

- 训练 48 family × 9 prompt × 3 seed = 432 条 `temperature=0.7` trace；
- 验证 12 family × 9 prompt × 3 seed = 108 条/temperature，共 324 条；
- 内层 prior cross-fit 使用 4 folds；
- scalar/hidden-delta scorer 各自产生 checkpoint 和逐更新点 compact JSONL；
- ALPS countdown 同时保存完整概率评价；
- PLP/concat等判别式 baseline 保存逐点预测，不冒充 posterior。

如果进程中断，不删除已有目录。没有 `fold_report.json` 的折可以用同一命令重新执行；已经
完成的折会跳过。若存在不兼容报告，脚本会拒绝覆盖，应先保存错误产物并诊断。

## 5. 汇总与冻结选择

五折全部完成后执行：

```bash
python scripts/finalize_bayesian_stage5_oof.py \
  --dataset-root "$STAGE4_DATA_ROOT"
```

输出：

```text
artifacts/runs/bayesian_sequential_v1/stage5_oof/oof_report.json
artifacts/runs/bayesian_sequential_v1/stage5_oof/selection.json
```

最终报告必须满足：

- `status: pass`；
- 五折均 `status: pass`；
- 每个概率方法覆盖 1,620 条 trace；
- scalar 与 hidden-delta 的主选择只使用 `temperature=0.7`；
- robustness 没有 refit；
- `final_holdout_accessed: false`。

第五阶段完成只表示 Train-family OOF 方法选择完成，不是最终泛化 claim。后续先做第六阶段
不确定性、收敛与 serving，再做第七阶段 Train-only error feedback；所有内容冻结后才可以
创建并一次性打开新的 final holdout。

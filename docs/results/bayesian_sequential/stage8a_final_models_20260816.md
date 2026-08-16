# Bayesian Sequential Stage-8A 最终模型冻结

Stage-8A 已使用全部 60 个已打开 Train family 在主温度 `0.7` 下的 540 条 trace，完成七方法
最终拟合。运行环境与 Stage 5 冻结环境一致：Python 3.12.3、PyTorch 2.8.0+cu128、CUDA
12.8、RTX 4090 D，并支持 BF16。

本阶段没有方法选择或超参数搜索。Stage 5 已选择的
`bayesian_entropy_scalar_v1` 仍是唯一 primary method；hidden-delta 仅保留为冻结消融。所有七个
模型、训练报告和 checkpoint registry 均通过 SHA-256 复验，全部模型也在服务器 CPU 上完成
恢复测试。registry SHA-256 为
`c58a1d3d00b024da4c7db1a53ea6c3a19827992a2f0547c40a26cadb6ed0dd4a`。

训练报告中的 loss、NLL 和 residual variance 只描述全 Train 拟合过程，不是 final-holdout
泛化指标，不用于重新选择方法。完整脱敏数值和七模型哈希见
[`stage8a_final_models_20260816_summary.json`](stage8a_final_models_20260816_summary.json)。原始压缩包
外层 SHA-256 与内部逐文件 SHA-256 均已在 Mac 本地验证；截至冻结时，final holdout 没有被
采集、评价或用于调参。

# 项目文档

`docs/` 只存放方法解释、实验报告、研究计划、部署手册和参考资料。可执行命令位于 `scripts/`，机器
可读实验合同位于 `configs/`，大体积 trace 和模型输出不提交 Git。

注意：`configs/experiments/` 中的“experiments”指实验合同；`docs/results/` 才是已经完成
的人工可读实验报告，原始JSON/CSV结果则位于本地`artifacts/runs/`。

## 从哪里开始

- 想看 ALPS 与旧 baseline：从 [`results/v1/README.md`](results/v1/README.md) 开始；
- 想看 Hidden-State PLP v2：阅读 [`results/v2/plp_v2_results.md`](results/v2/plp_v2_results.md)；
- 想看 PLP-only v3 三消融和最终 Test：阅读
  [`results/v3/plp_terminal_zero_v3_results.md`](results/v3/plp_terminal_zero_v3_results.md)；
- 想综合比较 Baseline、ALPS 与 PLP v3：阅读
  [`results/baseline_alps_plp_v3_comparison.md`](results/baseline_alps_plp_v3_comparison.md)；
- 想从零理解 PLP-only：阅读 [`methods/plp_only_explained.md`](methods/plp_only_explained.md)；
- 想运行 Hidden-State PLP v2：阅读根目录 README 的“运行真正的 Hidden-State PLP v2”与
  [`planning/v2_experiment_plan.md`](planning/v2_experiment_plan.md)；
- 想运行最终 ALPS+PLP 确认性比较：先读
  [`planning/alps_plp_hybrid_v3.md`](planning/alps_plp_hybrid_v3.md)，再严格按
  [`deployment/alps_plp_hybrid_v3_direct_server.md`](deployment/alps_plp_hybrid_v3_direct_server.md)；
- 想看ALPS怎么修正：阅读 [`planning/alps_improvement_plan.md`](planning/alps_improvement_plan.md)；
- 想在服务器运行：进入 [`deployment/`](deployment/) 选择对应环境；
- 想理解整体研究路线：阅读 [`planning/research_plan.md`](planning/research_plan.md)。

## 目录结构

```text
docs/
├── README.md
├── methods/                       # 面向读者的方法原理解释
│   ├── README.md
│   └── plp_only_explained.md      # 导航类比、entropy、hidden state与20-bin预测
├── results/                       # 已完成实验的冻结报告
│   ├── README.md                  # 已完成版本索引
│   ├── v1/
│       ├── README.md              # v1 总体结果与方法对比
│       ├── alps.md                # ALPS 结果、五折与校准分析
│       └── dynamic_signal_mlp.md  # 项目版 PLP 方法与结果
│   └── v2/
│       └── README.md              # Hidden-State PLP v2 完整结果
├── planning/                      # 尚未执行或仍在设计的工作
│   ├── alps_improvement_plan.md   # ALPS概率校准的可执行计划
│   ├── hidden_state_plp_v2.md     # 真正PLP输入、论文边界与运行方法
│   ├── research_plan.md           # 整体研究路线
│   └── v2_experiment_plan.md      # 下一轮严格实验计划
├── deployment/                    # 不同机器环境的操作手册
│   ├── autodl_5090.md
│   ├── docker_4090.md
│   └── isolated_server.md
└── references/                    # 论文和背景资料
    └── 大模型输出长度预测.pdf
```

## 已完成结果

| 文档 | 内容 |
|---|---|
| [`results/baseline_alps_plp_v3_comparison.md`](results/baseline_alps_plp_v3_comparison.md) | 当前 Baseline、ALPS、PLP v3 的公平口径、能力边界与 Hybrid 动机 |
| [`results/v1/README.md`](results/v1/README.md) | ALPS、Prompt-token baseline 和 Dynamic-Signal MLP 的总体比较 |
| [`results/v1/alps.md`](results/v1/alps.md) | ALPS 冻结设置、Train/Test、九宫格、五折、泛化和区间校准 |
| [`results/v1/dynamic_signal_mlp.md`](results/v1/dynamic_signal_mlp.md) | 项目版 PLP 的特征、结构、冻结条件、分阶段结果和论文边界 |
| [`results/v2/plp_v2_results.md`](results/v2/plp_v2_results.md) | Hidden-State PLP v2 的 Train/Test、进度、九宫格、泛化和过拟合分析 |
| [`results/v3/plp_terminal_zero_v3_results.md`](results/v3/plp_terminal_zero_v3_results.md) | PLP v2 baseline、三个单因素消融、OOF 选择与 terminal-zero v3 最终 Test |

`results/` 只保存已经完成且数字固定的实验报告。Hidden-State PLP v2 已完成 GPU 采集、
训练和开发性 Test 评价；后续改进必须使用新方法 ID 和新 holdout，不覆盖 v2 数字。

PLP-only v3 已完成三个单因素消融和最终 Test。原 Hybrid 协议预留的 12-family holdout 已由
PLP-only 使用，因此不能继续作为 Hybrid 的未见 Test；ALPS+PLP 下一阶段必须新建 holdout。

## 方法解释

[`methods/plp_only_explained.md`](methods/plp_only_explained.md) 用导航剩余时间作为主类比，
从 token、hidden state 和 entropy 开始，解释一个 Prompt 怎样得到一个 `h_prompt`、生成
阶段的 `h'_t` 是什么、PLP 输入为何是7168维，以及20-bin soft-label prediction head怎样
输出具体的剩余token数。

## 部署方式

| 文档 | 用途 |
|---|---|
| [`deployment/autodl_5090.md`](deployment/autodl_5090.md) | AutoDL RTX 5090、CUDA 12.8 直接 Python 流程 |
| [`deployment/docker_4090.md`](deployment/docker_4090.md) | 学校 RTX 4090 的固定 Docker 流程 |
| [`deployment/isolated_server.md`](deployment/isolated_server.md) | 与硬件无关的隔离服务器检查清单 |

AutoDL 和学校 Docker 是两条并列部署路径。AutoDL 不替换根目录已有的 `Dockerfile`、
`docker-compose.yml`、`.env` 和 `requirements-docker.lock`。

## 计划与资料

- [`planning/research_plan.md`](planning/research_plan.md)：从静态 prior 到 serving benchmark
  的整体路线；
- [`planning/alps_improvement_plan.md`](planning/alps_improvement_plan.md)：保留点预测、使用
  OOF conformal修正预测区间的实施顺序；
- [`planning/hidden_state_plp_v2.md`](planning/hidden_state_plp_v2.md)：Hidden-State PLP v2
  的真实输入、论文实现边界、冻结条件与运行顺序；
- [`planning/v2_experiment_plan.md`](planning/v2_experiment_plan.md)：新 holdout、概率校准
  和动态方法公平比较；
- [`references/大模型输出长度预测.pdf`](references/大模型输出长度预测.pdf)：项目背景材料。

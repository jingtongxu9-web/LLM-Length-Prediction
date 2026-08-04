# Hidden-State PLP v2 实施说明

## 1. 方法身份

Hidden-State PLP v2 是本项目的 **PLP-only** 路线。它不读取 ALPS Layer-14 Ridge 的预测值，
也不把 Dynamic-Signal MLP v1 的五个标量当作主输入。

状态：采集、训练和评估代码已实现；尚未在 GPU 上完成 pilot、Train 或 Test，因此本文没有
结果数字。

参考来源：

- [Predicting LLM Output Length via Entropy-Guided Representations](https://arxiv.org/html/2602.11812v2)
- [作者公开仓库 LP_Bench](https://github.com/xiehuanyi/LP_Bench)
- [作者公开的 EGTP prediction head](https://github.com/xiehuanyi/LP_Bench/blob/main/EGTP/egtp/model.py)

## 2. 每个预测点的真实输入

论文先用 entropy-guided token pooling 把 Prompt 最终层隐藏状态聚合成一个向量 `h`，再在
第 `t` 个生成 token 后结合生成期隐藏状态，预测 `remaining_tokens = T - t`。

本项目固定为：

```text
prompt_feature = entropy_softmax_pool(final_layer_states_of_formatted_prompt)
decode_feature_t = final_layer_hidden_state_of_current_generated_token
z_t = concat(prompt_feature, decode_feature_t)
```

当前 token 的 causal hidden state 已经通过自回归注意力编码此前全部生成 token。Qwen 的
权重保持冻结，只有后面的 PLP head 训练。

论文写 `Aggregate(h, {h'_1,...,h'_t})` 为“simple concatenation”，但没有说明不断增长的
向量怎样进入固定维度的同一 prediction head；作者公开仓库当前也只有 EGTP 代码，没有
PLP 实现。因此这里是对论文输入语义的固定维度工程落实，不声称逐行 exact replication。
这一边界已经写入 `configs/experiments/plp_v2_manifest.json`。

作者当前公开的 EGTP feature extractor 默认只取前4个原始Prompt token；论文公式描述的是
输入序列上的entropy pooling。本项目冻结为“Qwen官方chat template格式化后的全部Prompt
token”，因此属于论文公式导向的实现，不是作者EGTP脚本逐行复现。Prediction head则与作者
公开代码一致。

## 3. 预测头与训练目标

目标长度不做 `log1p`。Train 中的剩余长度按 1%–99% 分位范围划分为 20 bins。真实值所在
bin 为 `i` 时，soft label 为：

```text
p_j ∝ exp(-abs(j-i))
```

预测头为：

```text
Linear(2d, d) -> LayerNorm -> ReLU -> Dropout(0.1) -> Linear(d, 20)
```

20-bin softmax 概率与各 bin center 的加权和就是剩余 token 点预测。联合损失为：

```text
0.95 * soft-target cross entropy + 0.05 * token-space MSE
```

训练固定 AdamW、learning rate `2e-5`、10 epochs、batch size 16、seed 42。每条 rollout
所有更新点的总权重相同，避免长回答因为点更多而支配训练。达到 `max_new_tokens` 的 rollout
属于右删失数据，默认不进入训练和评价。

Trace schema v2额外保存完整生成token ids，用于重建回答和核验stride点；训练报告明确区分
加载trace、有效trace和因`max_new_tokens`排除的trace。checkpoint使用原子写入并归档精确
method config，评价输出同时包含总体、解码进度、任务、长度条件、九宫格和seed分组。

### 3.1 RTX 5090 与内存预算

采集脚本只加载冻结的 Qwen，训练脚本只加载 PLP head，两者不会同时常驻显存。[Qwen 官方
模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)记录了 28 层、hidden size 3584、
7.61B 参数；BF16 权重本身约 14.2 GiB。结合本数据集最长
仅数百字符的 Prompt、KV cache 和临时 logits，31.36 GiB 的 RTX 5090 对单条 rollout 采集
有足够余量，实际峰值会写入每条 trace metadata 和 `collection_summary.json`。

PLP head 有 25,772,564 个 float32 可训练参数：权重约 98.31 MiB；连同梯度和 AdamW 的一阶、
二阶状态，核心训练状态约 393.26 MiB。训练时 Qwen 不加载，因此 GPU 训练不是瓶颈。

CPU 内存按极端情形估算：4096 token、stride 5 时每条 trace 最多 821 个预测点；432 条 Train
trace 构造一份 7168 维 float32 feature matrix 约 9.47 GiB，样本特征与训练矩阵短暂并存时约
18.94 GiB。加上 Python 对象和临时数组，按 25 GiB 峰值预算较稳妥，AutoDL 的 90 GB 内存
足够。全部 540 条 trace 的未压缩数组上界约 5.94 GiB；实际写入压缩 NPZ，但磁盘仍应按
preflight 给出的 recommended budget 预留，不应依赖压缩率。

## 4. 与 ALPS v1 共同冻结和方法特有的条件

共同条件：Qwen2.5-7B-Instruct 固定 revision、BF16、同一 Prompt manifest、family 80/20
Train/Test、temperature 0.7、top-p 0.95、max new tokens 4096、seeds 42/43/44、官方 chat
template、每 5 个生成 token 更新一次。

方法特有条件：ALPS 使用 zero-based Layer 14 最后一个 Prompt token；PLP v2 按论文使用
最终 Transformer 层、Prompt entropy pooling 和生成期 causal hidden state。这一差别是方法
本身的一部分，不是未经控制地更换实验环境。

## 5. 为什么必须重新运行 Qwen

`data/interim/alps_v1/` 只保存 Layer-14 Prompt 特征与 entropy/EOS 等标量，没有保存生成
token 的最终层隐藏向量。模型生成结束后无法仅从文本恢复这些内部激活，所以 v2 必须重新
生成并写入 `data/interim/plp_v2/*.npz`。

## 6. 运行顺序

```bash
# Train pilot
python scripts/preflight_server.py --plp-config configs/experiments/plp_v2_manifest.json
python scripts/collect_plp_dataset.py --splits train --limit 6

# 完整 Train，随后训练和评价
python scripts/collect_plp_dataset.py --splits train
python scripts/train_plp.py
python scripts/evaluate_plp.py --split train

# 最后采集和评价 Test
python scripts/collect_plp_dataset.py --splits test --confirm-final-test
python scripts/evaluate_plp.py --split test --confirm-final-test
```

现有 v1 Test 已经查看过，因此复用该 Test 的 v2 结果只能作为开发性对照；真正确认性结论
需要新 family-level holdout。

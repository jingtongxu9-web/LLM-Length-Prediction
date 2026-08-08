# 方法原理

这里存放与具体实验结果、服务器部署无关的方法解释。

| 文档 | 适合读者 | 内容 |
|---|---|---|
| [`plp_only_explained.md`](plp_only_explained.md) | 第一次接触隐藏状态、entropy 和 PLP 的读者 | 用导航类比解释 Hidden-State PLP-only 的完整工作过程 |
| [`alps_plp_hybrid_v1_v2.md`](alps_plp_hybrid_v1_v2.md) | 希望理解 ALPS 与 PLP 如何融合的读者 | 对比特征拼接 v1 与“ALPS 基线 + 残差修正”v2，说明训练、OOF 和输出 |

实验是否已经运行、使用什么配置以及如何执行命令，分别查看：

- [`../planning/hidden_state_plp_v2.md`](../planning/hidden_state_plp_v2.md)：PLP v2 实施边界与运行顺序；
- [`../../configs/experiments/plp_v2_manifest.json`](../../configs/experiments/plp_v2_manifest.json)：机器可读冻结合同；
- [`../results/`](../results/README.md)：已经完成并有数字的实验结果。

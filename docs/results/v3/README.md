# PLP v3 结果

PLP-only 第三阶段实验已经完成，包括 Train family 五折 OOF、三个单因素消融、最终模型冻结
和一次性新 family Test。

正式报告：

- [`plp_terminal_zero_v3_results.md`](plp_terminal_zero_v3_results.md)：PLP v2 baseline、三个
  消融、OOF 选择、最终 Test 和后续边界。

最终状态：`plp_terminal_zero_v3` 在 Test 上取得约 5.35% 的 MAE 点估计改善，但 family 配对
95% 置信区间略微跨 0，因此严格优越性声明未通过。PLP-only 阶段至此冻结结束，下一步进入
ALPS+PLP 结合实验。

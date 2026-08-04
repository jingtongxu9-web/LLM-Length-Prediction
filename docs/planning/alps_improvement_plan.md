# ALPS 改进计划

## 1. 当前诊断

ALPS v1 的主要结论已经冻结：

- Layer 14 Ridge 点预测能够泛化到未见 Prompt family：family-grouped OOF MAE 为60.87、
  Log R²为0.953，Final Test MAE为66.97、Log R²为0.929；
- Train接近插值，但五折与Test接近，因此当前首要问题不是点预测因严重过拟合而失效；
- 名义95%区间在五折只覆盖71.1%，在Final Test只覆盖63.9%，当前最明确的问题是概率
  区间使用Train in-sample residual后过窄、过度自信；
- 九宫格每格只有4个独立Prompt，无法稳定判断固定任务和长度条件内的细粒度排序能力。

## 2. v1阶段可以直接完成的改进

以下工作复用已有Train trace和family-grouped OOF prediction，在CPU上完成，不需要重新
加载Qwen或生成rollout：

1. 保存每条Train rollout的OOF log-space预测；
2. 计算未见family上的signed residual：

   \[
   r_i=\log(1+y_i)-\hat\mu_i^{OOF}
   \]

3. 使用OOF residual的2.5%和97.5%经验分位数构造不对称conformal区间；
4. 将校准参数保存为独立`calibration.json`，不覆盖Ridge点预测模型；
5. 报告OOF总体及任务、长度条件、预测长度分组的Coverage和区间宽度；
6. 与当前log-normal区间及global/metadata宽区间baseline并列比较。

v1 Final Test已经打开，只能用于记录现有缺陷，不能反复用于选择分位数、分组校准方案或
放大系数。v1上的新校准结果应标记为Train内部回顾性诊断。

## 3. v2确认性实验

v2使用新的family-level holdout，并按以下顺序执行：

```text
冻结模型、Prompt、split、seeds与评价指标
    -> Train内family-grouped OOF
    -> 冻结Ridge与conformal校准方法
    -> 全Train一次性拟合最终Ridge
    -> 一次性打开新Final Test
```

如果采用独立Calibration split，必须在采集前与Train/Test一起冻结。Prompt的Short、
Medium、Long变体及其全部seed必须跟随同一`prompt_family_id`进入同一split。

## 4. 不在当前改进中调整的条件

为保持v1结论可解释，当前不因为已查看的Test而修改：

- Qwen模型和revision；
- tokenizer与chat template；
- temperature、top-p、max new tokens和seeds；
- zero-based Layer 14；
- `Ridge(alpha=1.0)`；
- 80/20 v1历史split。

Layer或alpha的重新选择只能作为新版本、使用Train内部嵌套验证并由新holdout确认。

## 5. 完成标准

ALPS概率改进不能只追求Coverage，还必须同时报告区间宽度。完成时至少需要证明：

1. 点预测MAE、RMSE和R²没有因校准层发生变化；
2. OOF和新Test的实际Coverage接近预先声明的名义目标；
3. 区间没有退化成global-mean baseline那样几乎覆盖全部长度范围的超宽区间；
4. Code、Short和长输出等已知困难组得到单独报告；
5. 所有校准选择均未使用最终Test标签。

## 6. 当前状态

| 工作 | 状态 |
|---|---|
| Layer 14 Ridge点预测 | v1已完成 |
| Family-grouped OOF预测 | 已完成 |
| 区间欠校准诊断 | 已完成 |
| OOF conformal校准代码 | 尚未实现 |
| `calibration.json`与校准报告 | 尚未实现 |
| 新family-level v2 holdout | 尚未采集 |


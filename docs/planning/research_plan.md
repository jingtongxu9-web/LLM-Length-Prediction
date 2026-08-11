# Research execution plan

> **Authoritative direction (2026-08-10):**项目 proposed method 是
> [`Bayesian Sequential v1`](../methods/bayesian_sequential_inference.md)，即 ALPS 静态概率
> prior 经过非重叠 decode evidence 的递归 posterior update。Dynamic-Signal MLP、
> Hidden-State PLP 和 concat/residual/gated Hybrid 均为 baseline/消融。具体回归步骤见
> [`pdf_alignment_recovery_plan.md`](pdf_alignment_recovery_plan.md)。

## Stage 0: reproducible foundation

- Freeze the model, tokenizer, prompt template, decoding settings, and split policy.
- Build a pilot spanning QA, summarization, and code.
- Capture prompt/output lengths, candidate-layer `h0`, per-step entropy, EOS probability, and runtime metadata.
- Split by prompt before stochastic sampling to prevent leakage.

## Stage 1: offline prior

- Verify the heavy tail using length and log-length diagnostics.
- Compare a constant baseline, prompt-length baseline, and the paper-derived frozen Layer-14 Ridge
  probe with `alpha=1.0`.
- Before fitting the final prior on all Train traces, report five-fold family-grouped
  cross-validation for the frozen configuration. Use it as a generalisation check, not for
  hyperparameter selection.
- Fit Ridge on `log1p(output_tokens)`, estimate the train-residual MLE variance, and use the
  resulting shifted log-normal prior. Report MAE, RMSE, R-squared, NLL, interval coverage, and
  long-tail underestimation.
- Replace in-sample variance for the proposed method with family-grouped OOF log-residual
  calibration, then discretize the shifted-log-normal density onto integer total-token support.
- Gate: proceed after documenting whether the frozen hidden-state probe beats the prompt-length
  baseline; retain Dynamic-Signal MLP v1 as a required dynamic comparison.

## Stage 2: progressive correction

- Build sequence objects at steps `1, 5, 10, ...` plus terminal, with `R_t = L - t` and right-censoring
  for `max_new_tokens` stops.
- Use ALPS as `p_0(R_0)`; before each evidence update, shift the previous posterior by the observed
  decode delta and condition on survival.
- Build likelihood-ratio scores only from the non-overlapping token block since the previous update;
  do not recursively multiply full causal states that repeat the same history.
- Train the two predeclared Bayesian candidates: scalar entropy/EOS evidence and the same model plus
  a frozen projection of decode hidden-state delta.
- Retain Dynamic-Signal MLP v1, Hidden-State PLP v3 and concat v1 as comparators, not as the proposed
  Bayesian method.
- Report posterior NLL/CRPS, uncertainty cone, calibrated coverage/width, error versus progress,
  stable time-to-5%-accuracy and online update overhead.
- Gate: posterior uncertainty may shrink only if coverage remains valid; variance collapse alone is
  not success.

## Stage 3: end-to-end and serving benchmark

- Compare input-length, ALPS-only, Dynamic-Signal MLP v1, Hidden-State PLP v3, concat v1 and the
  selected Bayesian Sequential candidate on identical traces and timesteps.
- Stratify by task, temperature, and length quantile.
- Simulate prediction-aware batching or KV-cache reservation.
- Report average/p95/p99 latency, throughput, padding waste, GPU utilization, KV peak, and OOM count.
- Stress-test incorrect predictions and preserve fairness through aging.

## Stage 4: error feedback

- Review absolute errors above 100 tokens and the worst five percent.
- Label entropy rebound, oscillation, open-ended prompts, sampling divergence, repetition, hallucination, and early stop.
- Refine hazard/evidence features, uncertainty, tail weights, and soft labels only from Train-family
  OOF evidence. Every refinement creates a new method ID.
- Freeze the final model before a single final test-set run.

## Suggested ten-week schedule

| Week | Deliverable |
|---|---|
| 1 | literature matrix, environment lock, one-sample trace |
| 2 | validated pilot and immutable trace schema |
| 3-4 | heavy-tail report, layer sweep, static prior |
| 5-6 | dynamic model, uncertainty cone, overhead study |
| 7 | four-method prediction benchmark |
| 8 | serving simulation and robustness analysis |
| 9 | failure taxonomy and refinement ablations |
| 10 | frozen final evaluation and paper-ready figures |

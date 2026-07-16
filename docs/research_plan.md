# Research execution plan

## Stage 0: reproducible foundation

- Freeze the model, tokenizer, prompt template, decoding settings, and split policy.
- Build a pilot spanning QA, summarization, and code.
- Capture prompt/output lengths, candidate-layer `h0`, per-step entropy, EOS probability, and runtime metadata.
- Split by prompt before stochastic sampling to prevent leakage.

## Stage 1: offline prior

- Verify the heavy tail using length and log-length diagnostics.
- Compare a constant baseline, prompt-length baseline, and layer-wise ridge probes.
- Fit a log-normal prior and report MAE, RMSE, R-squared, NLL, interval coverage, and long-tail underestimation.
- Gate: proceed only after the hidden-state probe beats the prompt-length baseline.

## Stage 2: progressive correction

- Build samples every five decode tokens with label `R_t = L - t`.
- Train a small MLP from prior length, step, entropy, entropy trend, EOS probability, and optional hidden state.
- Report the uncertainty cone, error versus decode progress, time-to-target-accuracy, and prediction overhead.
- Gate: error and interval width should generally shrink as decoding progresses.

## Stage 3: end-to-end and serving benchmark

- Compare input-length, ALPS-only, PLP-only, and ALPS+PLP.
- Stratify by task, temperature, and length quantile.
- Simulate prediction-aware batching or KV-cache reservation.
- Report average/p95/p99 latency, throughput, padding waste, GPU utilization, KV peak, and OOM count.
- Stress-test incorrect predictions and preserve fairness through aging.

## Stage 4: error feedback

- Review absolute errors above 100 tokens and the worst five percent.
- Label entropy rebound, oscillation, open-ended prompts, sampling divergence, repetition, hallucination, and early stop.
- Refine hazard features, uncertainty, tail weights, and the dynamic fusion schedule.
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

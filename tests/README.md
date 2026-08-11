# Tests

The tests provide fast checks for data contracts and small mathematical components. They do not
run the 7B model or prove that a full GPU experiment will fit in memory.

Current coverage includes:

- frozen experiment revision, counts, and prompt-manifest checksum;
- cross-platform LF prompt-manifest generation and stale-trace rejection;
- model snapshot structure and missing weight-shard detection;
- deterministic prompt-family Train/Test grouping;
- local/`MODEL_PATH` model-source resolution;
- JSONL trace serialization and schema validation;
- `log1p` Ridge fitting, prediction, and saved-model round trips.
- five-fold family isolation and frozen-alpha grouped-CV output contracts;
- length/task/interaction/seed metric breakdowns and matched-family length contrasts;
- end-to-end grouped-CV, final all-Train fit, Train/Test prediction, and breakdown generation.
- input-token Ridge final fitting and guarded Test evaluation;
- Dynamic-Signal MLP v1 non-terminal sample construction, per-sequence weighting, JSON model round trip, and
  decode-progress metrics.
- Hidden-State PLP NPZ round trips, Prompt/decode concatenation, soft length bins, sequence-balanced
  metrics, progress/task/length/seed grouping, complete generated-token provenance, strict stride
  schedules, censored-trace accounting, atomic checkpoint writes, and the frozen v2 manifest
  contract.
- Hybrid v3 216-prompt manifest checksum and new-family holdout isolation;
- unified pickle-free trace provenance, family-stratified folds, shifted log-normal prior;
- exact terminal zero bin, censoring warning/abort policy, family-macro metrics and paired-family CI.
- Bayesian Sequential contract pinning, family-grouped OOF ALPS variance, integer prior/overflow,
  countdown transition, log-space update, hazard round trip, non-overlapping sequence construction,
  exact/right-censored loss, scalar/hidden-delta shared scorers, safe checkpoint round trip, CRPS,
  coverage and stable-time metrics.

The local suite only trains a tiny synthetic Bayesian scorer for two epochs as a wiring test. Real
training and Qwen hidden-state collection are exercised on the server runtime; local tests do not
claim model quality or GPU feasibility.

After installing the project, run from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Use `scripts/preflight_server.py` for real CUDA, BF16, model snapshot, disk, and output-directory
checks. Use the six-rollout pilot for the first end-to-end GPU validation.

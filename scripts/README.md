# Command-line scripts

Run commands from the repository root. Files in this directory are user-facing entry points; they
call reusable implementation under `src/llm_length_prediction/`.

## Current ALPS v1 workflow

```bash
# 1. Validate the machine, model snapshot, prompt hash, and output paths.
python scripts/preflight_server.py

# 2. Collect a small resumable training pilot.
python scripts/collect_dataset.py --splits train --limit 6

# 3. Resume and complete all 432 training rollouts.
python scripts/collect_dataset.py --splits train

# 4. Validate the frozen Layer-14 / alpha=1.0 configuration with family-grouped CV.
python scripts/evaluate_grouped_cv.py

# 5. Discard the temporary fold models, then fit one final Ridge on all Train traces.
python scripts/train_prior.py
python scripts/evaluate_prior.py --split train

# 6. After all choices are frozen, open the final test split once.
python scripts/collect_dataset.py --splits test --confirm-final-test
python scripts/evaluate_prior.py --split test --confirm-final-test

# 7. Analyze existing Train/Test predictions without loading Qwen again.
python scripts/analyze_prior.py --splits train test --confirm-final-test

# 8. Fit/evaluate the prompt-token Ridge comparator from the same traces.
python scripts/train_input_baseline.py
python scripts/evaluate_input_baseline.py --split test --confirm-final-test

# 9. Fit/evaluate Dynamic-Signal MLP v1 without regenerating Qwen outputs.
python scripts/train_dynamic.py
python scripts/evaluate_dynamic.py --split test --confirm-final-test

# 10. Collect, train, and evaluate Hidden-State PLP v2 (requires a new Qwen run).
python scripts/preflight_server.py --plp-config configs/experiments/plp_v2_manifest.json
python scripts/collect_plp_dataset.py --splits train --limit 6
python scripts/collect_plp_dataset.py --splits train
python scripts/train_plp.py
python scripts/evaluate_plp.py --split train
python scripts/collect_plp_dataset.py --splits test --confirm-final-test
python scripts/evaluate_plp.py --split test --confirm-final-test
```

The batch collector stores one atomic trace per `(prompt_id, seed)` and skips valid completed files,
so the same command safely resumes an interrupted run.

## Script status

| Script | Status | Use |
|---|---|---|
| `preflight_server.py` | Implemented | Validate CUDA/BF16, disk, model files/revision, prompt hash, and output paths |
| `collect_dataset.py` | Implemented, official v1 collector | Expand the frozen prompt manifest into resumable Train/Test rollouts |
| `evaluate_grouped_cv.py` | Implemented | Validate the frozen Layer-14 / alpha=1.0 probe and baselines with family-grouped Train folds; does not tune or save a final model |
| `train_prior.py` | Implemented | Fit StandardScaler + Ridge on Layer-14 features and `log1p(output_tokens)` |
| `evaluate_prior.py` | Implemented | Evaluate a saved prior with final-test access protection |
| `analyze_prior.py` | Implemented | Offline overall, length, task, interaction, seed, and matched-family analysis |
| `train_input_baseline.py` | Implemented | Fit final `prompt_tokens -> output_tokens` Ridge on all Train traces |
| `evaluate_input_baseline.py` | Implemented | Evaluate the input-token Ridge with final-test protection |
| `train_dynamic.py` | Implemented | Train Dynamic-Signal MLP v1 from non-terminal dynamic trace points |
| `evaluate_dynamic.py` | Implemented | Evaluate Dynamic-Signal MLP v1 overall and by decode progress |
| `collect_plp_dataset.py` | Implemented, not yet run | Re-run frozen prompts and save entropy-pooled Prompt plus final-layer decode hidden states as NPZ |
| `train_plp.py` | Implemented, not yet run | Train the PLP-only 20-bin soft-label head on Train hidden-state traces |
| `evaluate_plp.py` | Implemented, not yet run | Evaluate PLP overall, by decode progress, task, intended-length group, 3x3 task-length cell, and seed |
| `collect_traces.py` | Debug helper | Collect one manually supplied prompt; not the official 540-rollout experiment |
| `download_model.py` | Setup helper | Download the exact Qwen revision and write `.frozen_revision` |
| `build_prompt_manifest.py` | Maintenance helper | Deterministically rebuild the frozen 180-prompt manifest; do not run casually |
| `run_benchmark.py` | Placeholder | Future input-length/ALPS/PLP/hybrid serving comparison |

## Inputs and outputs

```text
configs/experiments/alps_v1_manifest.json
data/prompts/alps_v1_prompts.jsonl
models/Qwen2.5-7B-Instruct/ or MODEL_PATH
                    |
                    v
          scripts/collect_dataset.py
                    |
                    v
       data/interim/alps_v1/{train,test}/
                    |
                    v
      scripts/evaluate_grouped_cv.py
       (temporary fold models only)
                    |
                    v
            scripts/train_prior.py
                    |
                    v
        artifacts/runs/alps_v1/stage1/
```

Large outputs are ignored by Git. Copy or archive experiment artifacts before releasing a rented
instance. The frozen, human-readable v1 result summary is
[`docs/results/v1/README.md`](../docs/results/v1/README.md); raw JSON/CSV outputs remain under
`artifacts/runs/alps_v1/`.

`analyze_prior.py` consumes the existing `train_evaluation.csv` and `test_evaluation.csv`. It does
not load Qwen, regenerate answers, or refit Ridge. It writes `{split}_breakdown.json`,
`{split}_breakdown.csv`, `{split}_breakdown.md`, `{split}_prompt_mean_breakdown.csv`, and
`{split}_length_contrasts.csv` beside the saved prior. The report separates rollout-level
distribution metrics from three-seed prompt-mean point accuracy. Test analysis retains the explicit
`--confirm-final-test` guard.

`evaluate_grouped_cv.py` reads the frozen Ridge alpha from the experiment manifest. It writes
`validation.json`, `results.json`, and `summary.csv` under
`artifacts/runs/alps_v1/diagnostics/grouped_cv/`. The five fold-specific Ridge models are discarded;
the command neither selects hyperparameters nor creates or replaces `stage1/prior.json`.

The same grouped-CV run already includes the `prompt_tokens` baseline, so a second baseline-specific
five-fold run is unnecessary. `train_input_baseline.py` requires that validation report, then fits
one deployable Ridge on all Train traces.

`train_dynamic.py` reads `configs/experiments/plp_v1_manifest.json`. Dynamic-Signal MLP v1 uses only
`step`, entropy, rolling entropy mean/slope, and EOS probability. Terminal points are excluded and
each rollout contributes the same total training weight. It does not consume the ALPS prior or
prefill hidden state. The original PLP paper uses decode-time hidden states; because v1 traces do
not store those states, this implementation is an explicit project adaptation rather than an exact
paper reproduction. See [`docs/results/v1/dynamic_signal_mlp.md`](../docs/results/v1/dynamic_signal_mlp.md) for
the frozen architecture, 9089-parameter calculation, source boundary, and planned v2 scope.

`collect_plp_dataset.py`, `train_plp.py`, and `evaluate_plp.py` read
`configs/experiments/plp_v2_manifest.json`. This is a separate data route: existing ALPS JSONL
files do not contain decode hidden states and cannot train it. PLP v2 does not consume the ALPS
prediction. Its input concatenates one entropy-guided pooled final-layer Prompt vector with the
current generated token's final-layer causal vector. The prediction head produces a distribution
over 20 remaining-length bins and uses its expected value as the point prediction.

The PLP-aware preflight reports the frozen 7168-dimensional input, exact 25,772,564 trainable
parameters, estimated checkpoint size, worst-case trace storage, and minimum/recommended free-disk
budgets. A completed collection command resumes without loading Qwen when every selected trace is
already valid.

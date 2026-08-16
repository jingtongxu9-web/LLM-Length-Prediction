# Command-line scripts

Run commands from the repository root. Files in this directory are user-facing entry points; they
call reusable implementation under `src/llm_length_prediction/`.

## Bayesian Sequential stage-eight final freeze and one-time benchmark

Stage-8A first fits every frozen comparator on all 60 opened Train families at temperature 0.7;
it still does not author or read a final holdout:

```bash
python scripts/preflight_bayesian_stage8a.py \
  --dataset-root "$STAGE4_DATA_ROOT" --verify-trace-hashes \
  --verify-training-environment
python scripts/train_bayesian_final_models.py \
  --dataset-root "$STAGE4_DATA_ROOT" --device auto --verify-trace-hashes
python scripts/preflight_bayesian_final_models.py
python scripts/preflight_bayesian_final_holdout_gate.py
```

The last command remains blocked until the Stage-8B ready lock is merged. Candidate preparation is
deterministic and pins the merged commit, registry, all seven model hashes, new manifest, and
semantic-overlap review:

```bash
python scripts/audit_bayesian_stage8b_candidate.py
python scripts/finalize_bayesian_stage8b_lock.py
```

Before merge, collection still fails because the checkout is not clean current `origin/main`.
After merge, the 4090 server runs `preflight_bayesian_stage8b_ready.py --verify-model-loading`;
only then may
`collect_bayesian_final_holdout.py` and `run_bayesian_final_benchmark.py` run. See
[`docs/deployment/bayesian_sequential_stage8_final_benchmark.md`](../docs/deployment/bayesian_sequential_stage8_final_benchmark.md).

The one-time run completed on 2026-08-16. After downloading the immutable result archive, verify
both the outer digest and every internal file, then generate only the small committable evidence:

```bash
python scripts/archive_bayesian_stage8_final.py \
  --archive /path/to/bayesian_stage8b_final_results.tar.gz \
  --expected-archive-sha256 \
  7da438422268c7471e572215f6ac6008cc2a12625f50075be8d40a0bc537853d
```

The archiver fails closed on any archive, nested benchmark-manifest, trace-count, preflight, or
no-selection boundary mismatch. It commits no raw trace or per-update prediction data.

## Bayesian Sequential stage-seven OOF error feedback

Stage seven audits the frozen selected Stage-5 OOF predictions; it does not need a GPU or refit:

```bash
python scripts/preflight_bayesian_stage7_error_feedback.py \
  --stage4-root "$STAGE4_ROOT" --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files

python scripts/run_bayesian_stage7_error_feedback.py \
  --stage4-root "$STAGE4_ROOT" --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files --verify-trace-hashes
```

The automatic labels remain trace-level review cues. Open-endedness and hallucination are emitted
as unresolved manual-review items, not automatic negatives. See
[`docs/deployment/bayesian_sequential_stage7_error_feedback.md`](../docs/deployment/bayesian_sequential_stage7_error_feedback.md).

## Bayesian Sequential stage-six analysis

Stage six consumes the verified Stage-4 traces and Stage-5 OOF archive without refitting:

```bash
python scripts/preflight_bayesian_stage6.py \
  --stage4-root "$STAGE4_ROOT" --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files

python scripts/run_bayesian_stage6_analysis.py \
  --stage4-root "$STAGE4_ROOT" --stage5-root "$STAGE5_ROOT" \
  --verify-stage5-files
```

It generates sequence-balanced uncertainty curves, an exact-checkpoint uncertainty cone, strict
stable-5% convergence, long-tail underestimation, recorded update overhead, and deterministic
batch/KV replay. It never accesses a final holdout. See
[`docs/deployment/bayesian_sequential_stage6_analysis.md`](../docs/deployment/bayesian_sequential_stage6_analysis.md).

## Bayesian Sequential stage-five family OOF

Stage four completed 1,620/1,620 traces and the downloaded archive passed both archive-level and
per-file SHA-256 verification. Stage five reads that extracted directory outside Git, fits only on
temperature 0.7, and evaluates each frozen fold model at 0.3/0.7/1.0 without robustness refitting:

```bash
python scripts/preflight_bayesian_stage5_oof.py \
  --dataset-root "$STAGE4_DATA_ROOT" \
  --verify-trace-hashes

for FOLD in 0 1 2 3 4; do
  python scripts/run_bayesian_stage5_fold.py \
    --dataset-root "$STAGE4_DATA_ROOT" --fold "$FOLD" --device auto
done

python scripts/finalize_bayesian_stage5_oof.py \
  --dataset-root "$STAGE4_DATA_ROOT"
```

The finalizer applies the frozen paired-family NLL rule to scalar versus hidden-delta. These
commands never access or author a final holdout. See
[`docs/deployment/bayesian_sequential_stage5_oof.md`](../docs/deployment/bayesian_sequential_stage5_oof.md).

## Bayesian Sequential stage-four full-Train collection

The real-Qwen nine-rollout stage-three pilot passed. Stage four expands only the opened Train
manifest to 1,620 frozen jobs and still does not train a scorer or access a final holdout:

```bash
python scripts/preflight_bayesian_full_train.py --model "$MODEL_PATH"
python scripts/collect_bayesian_full_train.py --model "$MODEL_PATH" --max-new-jobs 100
python scripts/collect_bayesian_full_train.py --report-only
```

Repeat the 100-job command until the report reaches 1,620 valid traces. Existing files are skipped
only after complete contract validation; invalid files abort without overwrite. See
[`docs/deployment/bayesian_sequential_stage4_full_train.md`](../docs/deployment/bayesian_sequential_stage4_full_train.md).

## Bayesian Sequential stage-three pilot

The Bayesian CPU core is complete. The next commands validate a GPU server and collect only the
frozen nine-rollout Train-family pilot; they do not train a scorer or access a final holdout:

```bash
python scripts/preflight_bayesian_pilot.py
python scripts/collect_bayesian_pilot.py --limit 1
python scripts/collect_bayesian_pilot.py --limit 3
python scripts/collect_bayesian_pilot.py
```

The collector stores every token ID, entropy, and EOS probability, while hidden states follow
`1,5,10,...,+terminal`. Valid existing NPZ files are contract-checked and resumed. See
[`docs/deployment/bayesian_sequential_stage3_pilot.md`](../docs/deployment/bayesian_sequential_stage3_pilot.md).

## Explicit Hybrid v1/v2 development workflow

The historical ten-method Hybrid v3 protocol remains frozen. The following commands compare the
two explicit fusion algorithms on the existing Train traces without regenerating Qwen outputs and
without reading the already-consumed Test:

```bash
python scripts/evaluate_hybrid_versions_oof.py --device auto
python scripts/train_hybrid_versions.py --device auto
```

The first command produces leakage-safe five-fold family-grouped OOF evidence. The second fits
deployable models on all design-Train families only after that report exists. Technical details are
in [`docs/methods/hybrid_concat_residual_gated.md`](../docs/methods/hybrid_concat_residual_gated.md).

After concat v1 selection, add the missing minimal input-length comparator without retraining Qwen
or any neural head:

```bash
python scripts/evaluate_main_comparison_oof.py
python scripts/train_main_baseline.py
```

The first command fits only five tiny Prompt-token Ridge fold models and reuses the frozen ALPS,
PLP and concat-v1 OOF predictions. The second fits one full-Train Ridge for the future holdout.

Direct residual v2 underperformed concat v1 in all five outer folds. Its evidence remains frozen.
The conservative gated residual v2.1 follow-up reuses the verified control predictions and exact
family folds, so it trains only the new candidate:

```bash
python scripts/evaluate_gated_residual_v2_1_oof.py --device auto
```

The report separates learned gate confidence from the progress-limited effective gate, then reports
their quantiles/usage thresholds, correction direction and saturation,
terminal precision/recall, decode-progress bands, gate bands, task, intended length, the 3x3
task-length grid, and outer-fold stability. Positive per-band MAE improvement means v2.1 reduced
error relative to the named control; a high gate value by itself is not evidence of usefulness.

Only if the frozen familywise selection rule passes may the final Train model be fitted:

```bash
python scripts/train_gated_residual_v2_1.py --device auto
```

## Selected PLP-only terminal-zero v3 workflow

The completed five-fold family-grouped OOF ablation selects `plp_terminal_zero_v3`. This
standalone route freezes only the old PLP v2 control and the selected terminal-zero candidate; it
does not train the other eight Hybrid-suite methods.

```bash
# If train_hybrid_v3_models.py already completed, reuse its byte-identical two checkpoints.
python scripts/train_plp_v3_models.py --reuse-hybrid-models

# Otherwise train only the two PLP-only heads on the existing 540 Train traces.
python scripts/train_plp_v3_models.py --device auto
```

Both commands recompute and freeze the OOF selection report before writing the two model hashes.
No Qwen generation is repeated. See
[`docs/results/plp/README.md`](../docs/results/plp/README.md) before opening final Test.

The following commands irreversibly assign the currently unopened 12-family holdout to PLP-only:

```bash
python scripts/open_plp_v3_test_gate.py --confirm-final-test
python scripts/collect_hybrid_v3_dataset.py \
  --splits test \
  --test-owner plp-terminal-v3 \
  --confirm-final-test
python scripts/evaluate_plp_v3_final.py
```

After this, the same families cannot be described as an untouched Hybrid final Test. A later
confirmatory Hybrid experiment must author a new holdout.

## Frozen ALPS+PLP Hybrid v3 workflow

The complete direct-server command sequence and the meaning of every step are in
[`docs/deployment/alps_plp_hybrid_v3_direct_server.md`](../docs/deployment/alps_plp_hybrid_v3_direct_server.md).
The strict order is:

```bash
python scripts/build_hybrid_v3_manifest.py --check
python scripts/collect_hybrid_v3_dataset.py --splits train --limit 6
python scripts/collect_hybrid_v3_dataset.py --splits train
python scripts/evaluate_hybrid_v3_oof.py --device auto
python scripts/train_hybrid_v3_models.py --device auto
python scripts/open_hybrid_v3_test_gate.py --confirm-final-test
python scripts/collect_hybrid_v3_dataset.py --splits test --confirm-final-test
python scripts/evaluate_hybrid_v3_final.py
python scripts/run_hybrid_v3_serving_benchmark.py
```

Do not move the Test commands earlier. The gate rejects Test traces created before OOF/model freeze.

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
| `build_hybrid_v3_manifest.py` | Implemented, frozen input builder/checker | Reproduce or verify the 60-family Train plus 12-new-family Test manifest |
| `collect_hybrid_v3_dataset.py` | Implemented, resumable | Capture one unified trace for all ten v3 methods; Test requires the one-way gate |
| `evaluate_hybrid_v3_oof.py` | Implemented | Nested family-grouped OOF, ten methods, censoring and family bootstrap |
| `train_hybrid_v3_models.py` | Implemented | Fit all final models on Train and freeze every artifact SHA-256 |
| `open_hybrid_v3_test_gate.py` | Implemented | Re-run tests/lint, validate hashes, and irreversibly open final Test |
| `evaluate_hybrid_v3_final.py` | Implemented | One-time holdout metrics and Bonferroni paired-family claim |
| `run_hybrid_v3_serving_benchmark.py` | Implemented | Frozen deterministic offline serving replay after final evaluation |
| `train_plp_v3_models.py` | Implemented | Reproduce the OOF choice and freeze only PLP v2 control + terminal-zero v3; can safely import matching Hybrid checkpoints |
| `open_plp_v3_test_gate.py` | Implemented | Irreversibly assign the shared unopened holdout to PLP-only v3 after tests, hashes, and model freeze |
| `evaluate_plp_v3_final.py` | Implemented | Final PLP-only comparison with overall, task, length, 3x3, seed, progress, and terminal breakdowns |
| `preflight_bayesian_pilot.py` | Implemented, server run passed | Validate stage-three contract, model revision, CUDA/BF16, disk and output paths |
| `collect_bayesian_pilot.py` | Implemented, 9/9 server pilot passed | Resumable nine-rollout Train-only unified Bayesian trace pilot |
| `preflight_bayesian_full_train.py` | Implemented, server run passed | Validate Stage 3 gate, 1,620 jobs, model/revision, CUDA/BF16, memory and dynamic disk budget |
| `collect_bayesian_full_train.py` | Implemented, 1,620/1,620 passed | Deterministic 100-job chunks, atomic trace writes, strict resume and full-Train acceptance report |
| `preflight_bayesian_stage5_oof.py` | Implemented, local real-data preflight passed | Pin Stage-4 report/index/dataset SHA, verify the grid and optionally rehash all 1,620 NPZ files |
| `run_bayesian_stage5_fold.py` | Implemented, awaiting full training | One resumable outer fold with nested ALPS prior, frozen baselines and two Bayesian candidates |
| `finalize_bayesian_stage5_oof.py` | Implemented, awaiting five folds | Combine OOF predictions, report robustness, and apply the paired-family NLL selection rule |

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
[`docs/results/comparisons/stage1_alps_baselines_dynamic.md`](../docs/results/comparisons/stage1_alps_baselines_dynamic.md); raw JSON/CSV outputs remain under
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
paper reproduction. See [`docs/results/plp/dynamic_signal_mlp_v1_results.md`](../docs/results/plp/dynamic_signal_mlp_v1_results.md) for
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

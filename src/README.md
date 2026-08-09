# Python package

`src/llm_length_prediction/` contains reusable implementation. Users normally run files under
`scripts/` rather than executing modules in `src/` directly.

## Module map

| Package | Responsibility | Used by |
|---|---|---|
| `experiment.py` | Load and validate the frozen manifest, prompt hash/counts, rollout jobs, paths, and trace provenance | preflight, collection, training, evaluation |
| `runtime/` | Resolve the Qwen source from an explicit path, `MODEL_PATH`, local `models/`, or Hub ID | preflight and collectors |
| `instrumentation/` | Run Qwen and capture either ALPS Layer-14 signals or PLP Prompt/decode final-layer states | trace and dataset collectors |
| `data/` | Validate ALPS JSONL traces and compressed, pickle-free PLP NPZ traces | collectors, training, evaluation |
| `comparison.py` | Load and validate shared Train/Test traces for frozen comparator methods | baseline/PLP scripts |
| `plp_experiment.py` | Validate the PLP v2 manifest, trace provenance, completeness, and digests | PLP v2 scripts |
| `models/` | Ridge models, Dynamic-Signal MLP v1, and the Hidden-State PLP soft-label head | training/evaluation |
| `evaluation/` | Prediction, tail-risk, grouped-CV, and v1/v2 decode-progress metrics | training/evaluation scripts |
| `serving/` | Early scheduling/bucketing simulation structures | future benchmark |

## Current call flow

```text
scripts/collect_dataset.py
  -> runtime/model_paths.py
  -> instrumentation/huggingface.py
  -> data/schema.py + data/io.py
  -> data/interim/alps_v1/

scripts/train_prior.py
  -> data/io.py
  -> models/prior.py
  -> evaluation/metrics.py
  -> artifacts/runs/alps_v1/stage1/

scripts/evaluate_prior.py
  -> data/io.py
  -> models/prior.py
  -> evaluation/metrics.py

scripts/train_input_baseline.py
  -> comparison.py
  -> models/prior.py
  -> artifacts/runs/alps_v1/comparisons/input_token_ridge/

scripts/train_dynamic.py
  -> comparison.py
  -> models/dynamic.py
  -> evaluation/progressive.py
  -> artifacts/runs/alps_v1/comparisons/plp_only/

scripts/collect_plp_dataset.py
  -> instrumentation/plp.py
  -> data/plp.py
  -> data/interim/plp_v2/

scripts/train_plp.py + scripts/evaluate_plp.py
  -> plp_experiment.py
  -> models/plp.py
  -> evaluation/plp.py
  -> artifacts/runs/plp_v2/

scripts/evaluate_main_comparison_oof.py
  -> hybrid_experiment.py
  -> models/prompt_token_baseline.py
  -> evaluation/hybrid.py
  -> artifacts/runs/alps_plp_main_comparison/oof/

scripts/train_main_baseline.py
  -> models/prompt_token_baseline.py
  -> artifacts/runs/alps_plp_main_comparison/models/
```

The Qwen weights and tokenizer are never trained here. Stage 1 fits small Ridge models; project
Dynamic-Signal MLP v1 fits a small project-defined MLP on already saved decode signals. Hidden-State
PLP v2 trains a separate soft-label head while Qwen remains frozen; it does not consume ALPS output.
`serving/simulator.py` remains a foundation for the later serving benchmark.

PLP v2 Trace schema 2 stores one 3584-dimensional Prompt feature, sampled 3584-dimensional decode
states, their step/target metadata, and the complete generated token-ID sequence. It does not store
the Qwen weights or a duplicated 7168-dimensional feature matrix per Trace. Training constructs the
7168-dimensional concatenation in host memory, then fits only the PLP head. Task and intended-length
labels are carried solely for subgroup reporting and never enter the model feature vector.

The four-method comparison adds a deliberately small Prompt-token baseline. It fits Ridge on
`prompt_token_count -> log1p(total_output_tokens)` within the same family-grouped OOF folds used by
ALPS, PLP, and Hybrid, then converts the total-length estimate into a remaining-length countdown at
each saved decode step. This keeps the target, samples, folds, and evaluation points aligned across
all four methods; it does not rerun Qwen or use task/family metadata as model input.

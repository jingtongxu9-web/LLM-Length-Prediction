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

# 4. Fit and inspect the train-only Ridge prior.
python scripts/train_prior.py
python scripts/evaluate_prior.py --split train

# 5. After all choices are frozen, open the final test split once.
python scripts/collect_dataset.py --splits test --confirm-final-test
python scripts/evaluate_prior.py --split test --confirm-final-test
```

The batch collector stores one atomic trace per `(prompt_id, seed)` and skips valid completed files,
so the same command safely resumes an interrupted run.

## Script status

| Script | Status | Use |
|---|---|---|
| `preflight_server.py` | Implemented | Validate CUDA/BF16, disk, model files/revision, prompt hash, and output paths |
| `collect_dataset.py` | Implemented, official v1 collector | Expand the frozen prompt manifest into resumable Train/Test rollouts |
| `train_prior.py` | Implemented | Fit StandardScaler + Ridge on Layer-14 features and `log1p(output_tokens)` |
| `evaluate_prior.py` | Implemented | Evaluate a saved prior with final-test access protection |
| `collect_traces.py` | Debug helper | Collect one manually supplied prompt; not the official 540-rollout experiment |
| `download_model.py` | Setup helper | Download the exact Qwen revision and write `.frozen_revision` |
| `build_prompt_manifest.py` | Maintenance helper | Deterministically rebuild the frozen 180-prompt manifest; do not run casually |
| `train_dynamic.py` | Placeholder | Future PLP remaining-length model |
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
            scripts/train_prior.py
                    |
                    v
        artifacts/runs/alps_v1/stage1/
```

Large outputs are ignored by Git. Copy or archive experiment artifacts before releasing a rented
instance.

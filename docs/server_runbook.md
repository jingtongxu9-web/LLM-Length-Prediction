# Isolated server runbook

## Hardware and storage

- One CUDA GPU with BF16 support. A 24 GB GPU is the practical minimum; 32 GB or more leaves
  safer headroom for the 4096-token pilot.
- At least 32 GB system RAM and 40 GB free local disk after the repository is copied.
- Stable power and enough wall-clock time for 540 rollouts. Use `tmux`, `screen`, or the cluster
  scheduler rather than an interactive SSH shell.

## Transfer and environment

1. Check out the exact experiment commit or release tag on the server.
2. Create Python 3.10+ environment and install `pip install -e '.[dev,hf]'`.
3. Download `Qwen/Qwen2.5-7B-Instruct` at revision
   `a09a35458c702b33eeacc393d103063234e8bc28`; do not use `main`.
4. Place the model at `models/Qwen2.5-7B-Instruct`, or set `MODEL_PATH`.
5. Add `.frozen_revision` containing the exact SHA after verifying the download.
6. Run `python scripts/preflight_server.py` and save its JSON report with the experiment artifacts.

Do not transfer credentials, `.env` files, Hugging Face tokens, or final-test outputs into Git.

## Execution order

1. Run `python scripts/collect_dataset.py --splits train --limit 6`.
2. Inspect stop reasons, output lengths, entropy, Layer-14 feature dimension, runtime, and GPU memory.
3. Delete only invalid pilot files; valid files are resumable inputs to the full run.
4. Run `python scripts/collect_dataset.py --splits train`.
5. Confirm `collection_summary.json` reports exactly 432 training rollouts.
6. Run `python scripts/train_prior.py` and train-only evaluation.
7. Freeze the prior JSON, its checksum, analysis code, and decision criteria.
8. Only then collect the 108 test rollouts with
   `python scripts/collect_dataset.py --splits test --confirm-final-test`.
9. Run final evaluation once with
   `python scripts/evaluate_prior.py --split test --confirm-final-test`.

## Artifact return

Return the complete `artifacts/runs/alps_v1` directory and the trace-root checksum index. Keep the
large `data/interim/alps_v1` tree outside Git. Archive both directories together so every reported
row can be traced to an immutable `(prompt_id, seed)` file and SHA-256 checksum.

Before interpreting results, confirm that no trace used a different model/tokenizer SHA, precision,
Layer index, temperature, top-p, max token limit, prompt-manifest hash, or chat template.

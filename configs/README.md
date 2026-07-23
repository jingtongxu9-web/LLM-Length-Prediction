# Configurations

This directory records scientific experiment choices. It does **not** choose the physical GPU or
install CUDA. Hardware and environment settings are documented separately in the root README.

## Which file is authoritative today?

The current ALPS v1 command-line pipeline reads:

```text
configs/experiments/alps_v1_manifest.json
```

`preflight_server.py`, `collect_dataset.py`, `train_prior.py`, and `evaluate_prior.py` use this JSON
manifest directly. It is the machine-readable experiment contract for:

- model and tokenizer revisions;
- BF16 precision and zero-based feature Layer 14;
- prompt-manifest path and SHA-256;
- temperature, top-p, token limit, seeds, trace stride, and entropy window;
- train-only standardization and frozen Ridge `alpha = 1.0`;
- Train/Test rollout counts;
- trace, model, metric, and prediction output locations.

Changing a frozen value creates a different experiment and should use a new manifest or experiment
ID. Do not edit the v1 manifest after viewing final-test results.

## Other files

| File | Current role | Read directly by the ALPS v1 commands? |
|---|---|---:|
| `base.yaml` | Human-readable shared design settings and future config foundation | No |
| `experiments/stage1_prior.yaml` | Stage 1 Ridge/prior design notes | No |
| `experiments/stage2_dynamic.yaml` | Planned PLP dynamic-correction design | No |
| `experiments/stage3_benchmark.yaml` | Planned serving benchmark design | No |
| `experiments/alps_v1_manifest.json` | Frozen executable ALPS v1 contract | Yes |

Some values currently appear in both YAML documentation and Python defaults. When they disagree,
do not guess: the executable manifest and the command being run determine actual behavior. A future
cleanup should make all frozen command defaults derive from one manifest.

Environment-specific values belong elsewhere:

- Python dependencies: `pyproject.toml`;
- container PyTorch/CUDA base: `Dockerfile`;
- Docker GPU ID and host mounts: `.env` plus `docker-compose.yml`;
- model location outside Docker: `MODEL_PATH`;
- actual GPU model: the local machine, server, or rented instance.

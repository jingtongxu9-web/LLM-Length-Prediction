# Configurations

This directory records scientific experiment choices. It does **not** choose the physical GPU or
install CUDA. Hardware and environment settings are documented separately in the root README.

The name `configs/experiments/` means **experiment definitions and frozen contracts**, not result
files. Generated machine outputs live under `artifacts/runs/`; committed human-readable reports
live under [`docs/results/`](../docs/results/README.md).

## Which file is authoritative today?

The current ALPS v1 command-line pipeline reads:

```text
configs/experiments/alps_v1_manifest.json
```

`preflight_server.py`, `collect_dataset.py`, `evaluate_grouped_cv.py`, `train_prior.py`,
`evaluate_prior.py`, and the input-token baseline use this JSON manifest directly. It is the
machine-readable experiment
contract for:

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
| `experiments/stage2_dynamic.yaml` | Earlier human-readable PLP/hybrid design notes | No |
| `experiments/stage3_benchmark.yaml` | Planned serving benchmark design | No |
| `experiments/alps_v1_manifest.json` | Frozen executable ALPS v1 contract | Yes |
| `experiments/plp_v1_manifest.json` | Frozen Dynamic-Signal MLP v1 contract; project adaptation, not paper PLP reproduction | Yes |
| `experiments/plp_v2_manifest.json` | Hidden-State PLP v2 contract; PLP-only, requires a new hidden-state collection | Yes |
| `experiments/alps_plp_hybrid_v3_base.json` | Shared Qwen, prompt, split and generation contract for Hybrid v3 | Yes |
| `experiments/alps_plp_hybrid_v3.json` | Isolated unified trace, ALPS stacking and progressive-head contract | Yes |
| `experiments/alps_plp_hybrid_v3_protocol.json` | Ten methods, three single-factor PLP ablations, nested grouped OOF, bootstrap, gate and serving replay | Yes |
| `experiments/alps_plp_hybrid_versions.json` | Development-only four-method comparison of concat v1 and residual-correction v2; forbids reuse of the old Test | Yes |
| `experiments/alps_plp_gated_residual_v2_1.json` | Supplemental OOF contract for conservative progress-gated residual v2.1; reuses verified controls and identical folds | Yes |
| `reports/alps_plp_hybrid_v3_report_schema.json` | Minimum machine-readable final-report contract | Yes |

For Hybrid v3, `alps_plp_hybrid_v3.json.progressive_head` defines the shared head settings and the
primary Hybrid defaults. The per-method differences that make the three PLP-only ablations valid
are frozen in `alps_plp_hybrid_v3_protocol.json`; training reads those fields directly.

The method definition and paper boundary are documented in
[`docs/results/v1/dynamic_signal_mlp.md`](../docs/results/v1/dynamic_signal_mlp.md). Its internal compatibility
label may still appear as `project_plp_only`, but user-facing reports should call it
**Dynamic-Signal MLP v1（项目版 PLP）**.

The explicit Hybrid method versions are implemented separately from the historical ten-method v3
protocol. `alps_plp_concat_v1` reproduces feature concatenation; `alps_plp_residual_v2` adds a
learned PLP correction to the ALPS countdown. See
[`docs/methods/alps_plp_hybrid_v1_v2.md`](../docs/methods/alps_plp_hybrid_v1_v2.md).
The same document records why direct residual v2 failed and freezes the bounded, progress-gated
v2.1 follow-up without overwriting the original evidence.
Its frozen diagnostics additionally test gate usage by decode progress, gate band, task, intended
length, 3x3 task-length cell, and fold, together with correction saturation and terminal confusion
metrics. These fields describe behavior only and are not added to the model input.

`plp_v2_manifest.json` is the executable contract for the paper-aligned, non-exact hidden-state
route. It keeps the
base Qwen revision, Prompt split, sampling settings, seeds, and five-token update frequency, while
using the method-specific final Transformer layer, entropy-guided Prompt pooling, current causal
decode hidden state, and the paper's 20-bin soft-label head. The public paper repository does not
currently expose PLP source code, so the fixed-dimension aggregation decision and non-exact
replication status are recorded directly in that manifest.

The v2 contract also freezes the representation dimensions (`3584 + 3584 = 7168`), Trace schema
version, 20-bin head architecture, optimizer, learning rate, epoch count, and censoring rule. Task,
intended-length group, Prompt family, and seed remain provenance/evaluation fields only; they are
not fed to the PLP head. Any change to these frozen values must use a new method ID or manifest.

Some values currently appear in both YAML documentation and Python defaults. When they disagree,
do not guess: the executable manifest and the command being run determine actual behavior. A future
cleanup should make all frozen command defaults derive from one manifest.

Environment-specific values belong elsewhere:

- Python dependencies: `pyproject.toml`;
- container PyTorch/CUDA base: `Dockerfile`;
- Docker GPU ID and host mounts: `.env` plus `docker-compose.yml`;
- model location outside Docker: `MODEL_PATH`;
- actual GPU model: the local machine, server, or rented instance.

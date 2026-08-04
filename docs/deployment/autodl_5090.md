# AutoDL RTX 5090 runbook

This is the direct-Python deployment path for one AutoDL RTX 5090. It does not use this
repository's RTX 4090 Docker image, `docker-compose.yml`, or `.env`.

Environment guidance was checked on 2026-07-23. AutoDL currently lists PyTorch 2.7/2.8 images
with Python 3.12 and CUDA 12.8. NVIDIA lists CUDA 12.8 as the first toolkit support for Blackwell
compute capabilities 10.0/12.0:

- <https://www.autodl.com/docs/base_config/>
- <https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html>

Re-check the image label when creating the instance because platform images can change.

## 1. Create the instance

Select:

- one RTX 5090 32 GB;
- PyTorch 2.8, Python 3.12, CUDA 12.8 (PyTorch 2.7 + CUDA 12.8 is also acceptable);
- at least 100 GB data-disk capacity for the repository, 15.2 GB model snapshot, traces, cache,
  and returned artifacts.

Store the project and model on `/root/autodl-tmp`, not only on the small system disk.

## 2. Clone and verify the base runtime

```bash
cd /root/autodl-tmp
git clone https://github.com/jingtongxu9-web/LLM-Length-Prediction.git
cd LLM-Length-Prediction

nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
    print("BF16:", torch.cuda.is_bf16_supported())
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

Expected: one visible RTX 5090, CUDA available, CUDA runtime 12.8 or newer, and BF16 support.
The collector automatically selects `cuda`; no source-code GPU setting is required.

## 3. Install the pinned non-PyTorch runtime

Keep the CUDA-enabled PyTorch supplied by the AutoDL image. Install the repository's pinned
Transformers/runtime packages, then install this project without dependency resolution:

```bash
python -m pip install --requirement requirements-autodl.lock
python -m pip install --no-deps --editable .
python -m pytest
python -m ruff check .
```

Do not install the RTX 4090 `pytorch/pytorch:2.6.0-cuda12.4` Docker image on the 5090.

## 4. Download the frozen model to the data disk

```bash
mkdir -p /root/autodl-tmp/models
python scripts/download_model.py \
  --output /root/autodl-tmp/models/Qwen2.5-7B-Instruct

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

The command downloads the exact revision declared by the repository and writes
`.frozen_revision` after completion. Run the `export` again after opening a new shell, or add the
same environment variable through the instance's normal shell configuration.

## 5. Run and retain preflight

```bash
python scripts/preflight_server.py
```

The command prints a report and writes
`artifacts/runs/alps_v1/environment/preflight.json`. Do not continue unless `ready` is `true`.

## 6. Run the six-rollout pilot

```bash
python scripts/collect_dataset.py --splits train --limit 6 \
  2>&1 | tee artifacts/runs/alps_v1/pilot.log
```

Check GPU memory, runtime per rollout, stop reasons, output lengths, and the generated trace files.
The six valid files are resumable and will be skipped during full collection.

## 7. Complete Train, validate the frozen probe, and fit the prior

```bash
python scripts/collect_dataset.py --splits train \
  2>&1 | tee artifacts/runs/alps_v1/train_collection.log

python scripts/evaluate_grouped_cv.py \
  2>&1 | tee artifacts/runs/alps_v1/grouped_cv.log

python scripts/train_prior.py
python scripts/evaluate_prior.py --split train
```

Training is blocked until all 432 frozen Train rollout files exist and satisfy the experiment
contract. Confirm `collection_summary.json` reports 432 Train rollouts. The grouped five-fold step
uses the manifest's fixed Layer 14 and `alpha=1.0`; it does not tune settings or save a final model.
`train_prior.py` then discards the temporary-fold context and fits one Ridge on all Train traces.

## 8. Open the final Test once

After the prior, metrics, and analysis decisions are frozen:

```bash
python scripts/collect_dataset.py --splits test --confirm-final-test \
  2>&1 | tee artifacts/runs/alps_v1/test_collection.log

python scripts/evaluate_prior.py --split test --confirm-final-test
```

Test evaluation is blocked until all 108 frozen Test rollout files exist and validate.

## 9. Run the frozen comparator methods from existing traces

These commands do not load Qwen or regenerate answers:

```bash
# One final prompt-input-token Ridge after the shared grouped-CV report exists.
python scripts/train_input_baseline.py
python scripts/evaluate_input_baseline.py \
  --split test \
  --confirm-final-test

# Dynamic-Signal MLP v1 using saved per-step entropy/EOS signals.
python scripts/train_dynamic.py
python scripts/evaluate_dynamic.py \
  --split test \
  --confirm-final-test
```

Dynamic-Signal MLP v1 is a project adaptation, not the paper PLP architecture. Qwen generated the
expensive traces earlier. PyTorch from the AutoDL image is still required to fit the MLP, but the
task can run on CPU or GPU according to the frozen `configs/experiments/plp_v1_manifest.json`.
See [`results/v1/dynamic_signal_mlp.md`](../results/v1/dynamic_signal_mlp.md) for the exact
version boundary.

Inspect:

```bash
cat artifacts/runs/alps_v1/comparisons/input_token_ridge/test_evaluation.json
cat artifacts/runs/alps_v1/comparisons/plp_only/test_evaluation.json
```

The v1 Test was already viewed during ALPS analysis. Treat these new Test comparisons as post-hoc
diagnostics and do not tune their settings against the same Test.

## 10. Run Hidden-State PLP v2

PLP v2 is a separate paper-aligned, non-exact route. Existing ALPS JSONL files do not contain its
Prompt/decode final-layer states, so collection must run Qwen again. No network accelerator is
needed after the model and repository are already local.

```bash
# Resource and frozen-contract check.
python scripts/preflight_server.py \
  --plp-config configs/experiments/plp_v2_manifest.json

# Six-rollout pilot. The files are resumable and the full run will reuse them.
python scripts/collect_plp_dataset.py --splits train --limit 6 \
  2>&1 | tee artifacts/runs/plp_v2/train_pilot.log
cat artifacts/runs/plp_v2/collection_summary.json

# Complete Train, fit the PLP head, and evaluate Train.
python scripts/collect_plp_dataset.py --splits train \
  2>&1 | tee artifacts/runs/plp_v2/train_collection.log
python scripts/train_plp.py 2>&1 | tee artifacts/runs/plp_v2/training.log
python scripts/evaluate_plp.py --split train

# Development-only Test comparison: the shared v1 Test has already been opened.
python scripts/collect_plp_dataset.py --splits test --confirm-final-test \
  2>&1 | tee artifacts/runs/plp_v2/test_collection.log
python scripts/evaluate_plp.py --split test --confirm-final-test
```

During collection, monitor another terminal with `watch -n 2 nvidia-smi`. Collection writes peak
CUDA allocation/reservation, stop-reason counts, point counts, and trace size into
`artifacts/runs/plp_v2/collection_summary.json`. Train/Test evaluation additionally reports task,
intended-length, 3x3 task-length, seed, and decode-progress breakdowns. The task and length fields
are evaluation labels only and never enter the PLP feature vector.

## 11. Back up before releasing the instance

Copy at least:

```text
artifacts/runs/alps_v1/
data/interim/alps_v1/
artifacts/runs/plp_v2/
data/interim/plp_v2/
```

The first directory contains the preflight report, collection index, Ridge model, predictions, and
metrics. The second contains the raw evidence needed to reproduce those results. AutoDL local data
disks are not a substitute for durable experiment storage.

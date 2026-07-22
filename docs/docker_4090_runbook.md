# RTX 4090 Docker runbook

This runbook targets the inspected server: Ubuntu 22.04, Docker 28, NVIDIA driver 580, eight
24 GiB RTX 4090 GPUs, 1 TiB RAM, and persistent storage under `/data`. The host CUDA compiler is
not used; the image supplies PyTorch 2.6.0 with CUDA 12.4 and cuDNN 9.

The current collector is intentionally single-GPU. Select one idle GPU. Do not start multiple
collectors against the same trace directory: they would claim the same jobs and race while
rewriting the collection index.

## 1. Verify the host

```bash
nvidia-smi
docker version
docker compose version
df -h /data
id -u
id -g
```

Re-check GPU utilization immediately before the experiment. GPUs 4-7 were idle in the supplied
snapshot, but that is not a reservation. Coordinate ownership with other server users.

Verify that NVIDIA Container Toolkit exposes the selected GPU (replace `4` if needed):

```bash
docker run --rm --gpus 'device=4' \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi
```

If this fails, install/configure NVIDIA Container Toolkit on the host before continuing. Do not
try to solve a missing container runtime by installing CUDA inside the project image.

## 2. Clone into the empty directory

```bash
mkdir -p /data/home/mulei/Summer_Camp
cd /data/home/mulei/Summer_Camp
git clone https://github.com/jingtongxu9-web/LLM-Length-Prediction.git .
git status
git rev-parse HEAD
```

The final `.` is intentional: it clones the repository directly into the existing empty
`/data/home/mulei/Summer_Camp` directory. If the directory is no longer empty, stop and inspect it
instead of deleting or overwriting files.

Save the exact Git commit and never pull new code in the middle of the run:

```bash
mkdir -p artifacts/runs/alps_v1/environment
git rev-parse HEAD > artifacts/runs/alps_v1/environment/git_commit.txt
```

## 3. Configure persistent paths and identity

```bash
cp .env.example .env
sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/" .env
sed -i "s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env
```

Edit `.env` and set `GPU_ID` to the GPU reserved for this experiment:

```bash
nano .env
```

Expected layout:

```text
/data/home/mulei/Summer_Camp/
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.lock
├── .env
├── models/                 # persistent, not in the image
├── cache/huggingface/      # persistent download cache
├── data/interim/           # persistent generation traces
└── artifacts/runs/         # persistent indexes, models, metrics, logs
```

Create writable directories before building:

```bash
mkdir -p \
  models/Qwen2.5-7B-Instruct \
  cache/huggingface \
  data/interim \
  artifacts/runs/alps_v1/environment \
  artifacts/runs/alps_v1/logs
```

`.env`, models, traces, caches, and run artifacts are ignored by Git and excluded from the Docker
build context.

## 4. Build the image

```bash
docker compose build --pull alps \
  2>&1 | tee artifacts/runs/alps_v1/logs/docker_build.log
```

The image contains:

- PyTorch 2.6.0, CUDA 12.4 runtime, cuDNN 9;
- pinned NumPy, Transformers, Hugging Face Hub, tokenizers, safetensors, pytest, and ruff;
- the repository source code and frozen prompt/experiment manifests.

It deliberately does not contain model weights, Hugging Face credentials, traces, or results.

Confirm the resolved Compose configuration and selected GPU:

```bash
docker compose config > artifacts/runs/alps_v1/environment/compose_resolved.yaml
docker compose run --rm alps python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

Expected: one visible RTX 4090, CUDA available, and BF16 supported.

## 5. Validate the repository in the image

```bash
docker compose run --rm alps pytest
docker compose run --rm alps ruff check .
```

Save installed versions:

```bash
docker compose run --rm alps python -m pip freeze \
  > artifacts/runs/alps_v1/environment/pip_freeze.txt
```

## 6. Download the frozen Qwen snapshot

Download the model and tokenizer at the full immutable revision:

```bash
docker compose run --rm alps \
  hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir /models/Qwen2.5-7B-Instruct \
  2>&1 | tee artifacts/runs/alps_v1/logs/model_download.log
```

Qwen2.5-7B-Instruct is public. If Hugging Face authentication is ever required, authenticate on
the host or pass a short-lived token only to the download command. Never write it to `.env`, the
Dockerfile, logs, or Git.

Create the revision marker required by preflight:

```bash
docker compose run --rm alps sh -lc \
  "printf '%s\n' a09a35458c702b33eeacc393d103063234e8bc28 \
  > /models/Qwen2.5-7B-Instruct/.frozen_revision"
```

Inspect the mounted model:

```bash
ls -lh models/Qwen2.5-7B-Instruct
cat models/Qwen2.5-7B-Instruct/.frozen_revision
```

The directory must include `config.json`, `tokenizer_config.json`, safetensors weight shards, and
the `.frozen_revision` marker.

## 7. Run preflight

```bash
docker compose run --rm alps python scripts/preflight_server.py \
  | tee artifacts/runs/alps_v1/environment/preflight.json
```

Do not continue unless the JSON ends with:

```json
"failures": [],
"ready": true
```

## 8. Run the six-rollout pilot

Use a persistent terminal session such as `tmux`:

```bash
tmux new -s alps-v1
cd /data/home/mulei/Summer_Camp
```

Run the pilot:

```bash
docker compose run --rm alps \
  python scripts/collect_dataset.py --splits train --limit 6 \
  2>&1 | tee artifacts/runs/alps_v1/logs/pilot.log
```

Validate the count:

```bash
jq . artifacts/runs/alps_v1/collection_summary.json
find data/interim/alps_v1/train -name 'seed_*.jsonl' -type f | wc -l
```

Both should report six train rollouts. Inspect at least one trace and confirm:

- model and tokenizer revisions equal the frozen Qwen SHA;
- Layer 14 prefill hidden state exists and contains finite values;
- entropy and EOS probability are finite;
- remaining length equals output tokens minus step;
- stop reason, output tokens, and duration are plausible;
- no CUDA OOM or NaN occurred.

Re-run the same command to test resume behavior. It should report `new=0 resumed=6`.

Monitor the selected GPU from another terminal:

```bash
watch -n 2 nvidia-smi
```

## 9. Collect all 432 training rollouts

```bash
docker compose run --rm alps \
  python scripts/collect_dataset.py --splits train \
  2>&1 | tee artifacts/runs/alps_v1/logs/train_collection.log
```

This command is resumable. If SSH, the container, or the server stops, run the same command again.
Completed valid `(prompt_id, seed)` files are skipped.

After completion:

```bash
jq . artifacts/runs/alps_v1/collection_summary.json
find data/interim/alps_v1/train -name 'seed_*.jsonl' -type f | wc -l
```

Required result: train count 432, test count 0.

## 10. Fit and inspect the train-only prior

```bash
docker compose run --rm alps python scripts/train_prior.py \
  2>&1 | tee artifacts/runs/alps_v1/logs/train_prior.log

docker compose run --rm alps python scripts/evaluate_prior.py --split train \
  2>&1 | tee artifacts/runs/alps_v1/logs/train_evaluation.log
```

Inspect:

```bash
jq . artifacts/runs/alps_v1/stage1/prior.json
jq . artifacts/runs/alps_v1/stage1/metrics.json
```

Confirm `target=log1p_output_tokens`, `fit_split=train`, Layer 14, the exact model/tokenizer SHA,
and a finite non-negative residual variance.

Freeze train artifacts before accessing test:

```bash
sha256sum \
  artifacts/runs/alps_v1/stage1/prior.json \
  artifacts/runs/alps_v1/stage1/metrics.json \
  artifacts/runs/alps_v1/stage1/predictions.csv \
  artifacts/runs/alps_v1/collection_index.jsonl \
  > artifacts/runs/alps_v1/frozen_train_artifacts.sha256
```

Do not change features, layer, alpha, prompts, generation settings, or evaluation rules after this
point. If the Stage 1 gate fails, document the failure before designing a new experiment version.

## 11. Collect and evaluate the final test once

Only after the prior and decision rules are frozen:

```bash
docker compose run --rm alps \
  python scripts/collect_dataset.py --splits test --confirm-final-test \
  2>&1 | tee artifacts/runs/alps_v1/logs/test_collection.log

docker compose run --rm alps \
  python scripts/evaluate_prior.py --split test --confirm-final-test \
  2>&1 | tee artifacts/runs/alps_v1/logs/final_test_evaluation.log
```

The final summary must report 432 train and 108 test rollouts, total 540.

## 12. Archive and verify

```bash
find data/interim/alps_v1 artifacts/runs/alps_v1 \
  -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > alps_v1_all_files.sha256

tar -czf alps_v1_results.tar.gz \
  data/interim/alps_v1 \
  artifacts/runs/alps_v1 \
  configs/experiments/alps_v1_manifest.json \
  data/prompts/alps_v1_prompts.jsonl \
  requirements-docker.lock \
  alps_v1_all_files.sha256
```

Do not add the model, trace tree, `.env`, cache, credentials, or generated artifacts to Git.

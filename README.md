# LLM Length Prediction

Research codebase for **serving-aware LLM output-length prediction**. The project combines an ALPS-style prefill prior with PLP-style progressive correction and evaluates whether better length estimates improve batching, latency, and KV-cache planning.

## Research questions

1. Does output length follow a heavy-tailed distribution?
2. How much length information is linearly recoverable from prefill hidden states?
3. Do entropy and hidden-state trajectories reduce remaining-length uncertainty during decoding?
4. Does `ALPS + PLP` converge earlier and reduce severe underestimation on long outputs?
5. Do the predictions improve serving metrics rather than only MAE?

## Frozen ALPS experiment (v1)

The first implementation milestone is a corrected, reproducible ALPS baseline. The following
conditions are frozen for the primary experiment; layer, temperature, and decoding-policy sweeps
belong to later ablations and must not be mixed into the first reported result.

| Condition | Frozen value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model precision | BF16; 4-bit is permitted only for pipeline debugging |
| Temperature | `0.7` |
| Top-p | `0.95` |
| Max new tokens | `4096` |
| ALPS layer | Transformer block index `14` (zero-based) |
| PLP update frequency | Every `5` generated tokens |
| Chat template | Official Qwen tokenizer chat template |
| Output-length definition | Number of newly generated tokens, including EOS |
| Sampling seeds | Exactly `[42, 43, 44]`; three rollouts per prompt |
| Ridge preprocessing | `StandardScaler` fitted on the training split only |
| Ridge regularization | `alpha = 1.0` |
| Data split | `80%` train / `20%` test, grouped by prompt family |
| LLM weights | Completely frozen |

The ALPS feature is the last prompt token's hidden state after transformer block 14, captured in a
standalone prefill pass before any response token is sampled. The primary Ridge model predicts
output length from that feature. Prompt-token count is retained only as a baseline.

ALPS v1 deliberately has no validation split because its layer, preprocessing, Ridge alpha,
decoding policy, and sampling seeds are fixed before data collection. The test split is final: it
must not be used to change those choices. If any frozen choice is changed after inspecting test
results, the run becomes exploratory and requires a new untouched test split for a confirmatory
result.

### Frozen prompt pilot

The pilot uses only two prompt dimensions: task and intended response length.

| Task | Short | Medium | Long |
|---|---|---|---|
| QA / explanation | One or two sentences | Explanation with an example | Structured tutorial-style answer |
| Summarization | One-sentence summary | Five to eight key points | Detailed sectioned summary |
| Code | Core function only | Function with documentation and edge cases | Implementation, tests, usage, and analysis |

Create 20 independent prompt families for each task. Every family has matched short, medium, and
long variants, giving 60 families and 180 unique prompts. Assign complete families to splits:

| Split | Families | Unique prompts | Rollouts with seeds 42/43/44 |
|---|---:|---:|---:|
| Train | 48 | 144 | 432 |
| Test | 12 | 36 | 108 |
| Total | 60 | 180 | 540 |

Use 16 train families and 4 test families from each task so QA, summarization, and code remain
balanced. All three length variants and all three seeded rollouts from one family must stay in the
same split. The split is created before generation and is never changed based on observed output
length.

The frozen prompt manifest is stored at `data/prompts/alps_v1_prompts.jsonl`. It is generated from
the curated family definitions in `scripts/build_prompt_manifest.py` and validated by the test
suite. Rebuild it with `python scripts/build_prompt_manifest.py`; the command must reproduce the
same 180 prompt records and 80/20 family split.

Only BF16 runs count toward the primary result. A 4-bit run may verify the pipeline on smaller
hardware, but it must be labeled as a debug run and must not be compared directly with BF16 metrics.

## Repository layout

```text
.
├── artifacts/                  # Generated metrics, tables, and checkpoints (not raw data)
├── configs/
│   ├── base.yaml               # Shared model/data/runtime settings
│   └── experiments/            # One config per experimental stage
├── data/                       # Local-only datasets and trace manifests
├── docs/research_plan.md       # Four-stage research plan and milestones
├── models/                     # Local Qwen snapshot mount point; model files stay out of Git
├── notebooks/                  # Exploration only; production logic stays in src/
├── scripts/                    # Reproducible command-line entry points
├── src/llm_length_prediction/
│   ├── data/                   # Trace schemas and dataset contracts
│   ├── evaluation/             # Prediction and tail-risk metrics
│   ├── instrumentation/        # Prefill/decode signal capture interfaces
│   ├── models/                 # Static prior and dynamic correction logic
│   └── serving/                # Prediction-aware scheduling simulation
└── tests/                      # Fast unit and smoke tests
```

## Local model layout

The frozen model is `Qwen/Qwen2.5-7B-Instruct`. Its project-local snapshot belongs at:

```text
models/Qwen2.5-7B-Instruct/
```

Model weights, tokenizer files, and downloaded snapshot metadata are machine-local and ignored by
Git. The repository tracks only `models/README.md`, which documents the expected contents and
download command. The runtime resolves the model source in this order:

1. an explicit `--model` argument;
2. the `MODEL_PATH` environment variable;
3. the project-local directory when it contains `config.json`;
4. the frozen Hugging Face Hub ID.

For a future server or container, mount the server model directory at the same project-relative
location or set `MODEL_PATH=/models/Qwen2.5-7B-Instruct`. No source-code changes are required when
moving from the local layout to the server layout.

## Four experimental stages

| Stage | Goal | Core outputs |
|---|---|---|
| 1. Offline prior | Validate heavy tails and `h0 -> log(L)` signal | distribution fit, layer sweep, ALPS-style probe |
| 2. Dynamic correction | Predict `R_t = L - t` from decode evidence | convergence curves, uncertainty cone, overhead |
| 3. End-to-end benchmark | Compare input-length, ALPS-only, PLP-only, hybrid | MAE/RMSE, time-to-accuracy, long-tail underestimation |
| 4. Error feedback | Explain outliers and refine the model | failure taxonomy, hazard/fusion/loss ablations |

The prediction study is followed by a serving evaluation covering average and tail latency, throughput, padding waste, GPU utilization, and KV-cache peak usage.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The scripts are intentionally lightweight scaffolds. Connect model-specific Hugging Face or vLLM instrumentation behind the interfaces in `src/llm_length_prediction/instrumentation/`.

## Collect the first Hugging Face trace

Install the lightweight collector dependencies and run a real-model smoke trace:

```bash
pip install -e '.[dev,hf]'
python scripts/collect_traces.py \
  --model sshleifer/tiny-gpt2 \
  --dtype auto \
  --layers 1 \
  --max-new-tokens 16 \
  --prompt "Explain why output length prediction helps LLM serving in one sentence." \
  --output artifacts/examples/first_trace.jsonl
```

The command captures candidate-layer prefill hidden states, decode entropy, EOS probability,
token counts, stop reason, timing, and runtime versions. It then reads the JSONL record back
through the schema validator before reporting success. The default `sshleifer/tiny-gpt2` model
is for pipeline validation only; research runs should use the frozen model in `configs/base.yaml`.

## Reproducibility rules

- For ALPS v1, split by prompt family before sampling; all matched prompt variants and all seeded
  trajectories from one family stay in the same split.
- Keep the final test set untouched until models and metrics are frozen.
- Record model/tokenizer revision, prompt template, decoding settings, seed, hardware, and stop reason.
- Treat raw traces and model checkpoints as external artifacts; do not commit secrets or large generated files.

## Primary comparison

1. Input-length baseline
2. ALPS-only static prediction
3. PLP-only progressive prediction
4. ALPS + PLP hybrid

An oracle-length scheduler may be added as an upper bound. Do not claim superiority over EGTP unless it is independently reproduced.

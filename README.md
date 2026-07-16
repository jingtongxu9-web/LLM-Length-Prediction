# LLM Length Prediction

Research codebase for **serving-aware LLM output-length prediction**. The project combines an ALPS-style prefill prior with PLP-style progressive correction and evaluates whether better length estimates improve batching, latency, and KV-cache planning.

## Research questions

1. Does output length follow a heavy-tailed distribution?
2. How much length information is linearly recoverable from prefill hidden states?
3. Do entropy and hidden-state trajectories reduce remaining-length uncertainty during decoding?
4. Does `ALPS + PLP` converge earlier and reduce severe underestimation on long outputs?
5. Do the predictions improve serving metrics rather than only MAE?

## Repository layout

```text
.
├── artifacts/                  # Generated metrics, tables, and checkpoints (not raw data)
├── configs/
│   ├── base.yaml               # Shared model/data/runtime settings
│   └── experiments/            # One config per experimental stage
├── data/                       # Local-only datasets and trace manifests
├── docs/research_plan.md       # Four-stage research plan and milestones
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

## Reproducibility rules

- Split by `prompt_id` before sampling; all trajectories from one prompt stay in the same split.
- Keep the final test set untouched until models and metrics are frozen.
- Record model/tokenizer revision, prompt template, decoding settings, seed, hardware, and stop reason.
- Treat raw traces and model checkpoints as external artifacts; do not commit secrets or large generated files.

## Primary comparison

1. Input-length baseline
2. ALPS-only static prediction
3. PLP-only progressive prediction
4. ALPS + PLP hybrid

An oracle-length scheduler may be added as an upper bound. Do not claim superiority over EGTP unless it is independently reproduced.

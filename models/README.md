# Local models

This directory is the local mount point for model files. Model weights are intentionally excluded
from Git; only this README is versioned.

Expected local layout:

```text
models/
├── README.md
└── Qwen2.5-7B-Instruct/
    ├── config.json
    ├── tokenizer_config.json
    ├── model-00001-of-00004.safetensors
    └── ...
```

Download the frozen model when storage and network access are ready:

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision main \
  --local-dir models/Qwen2.5-7B-Instruct
```

Do not pass access tokens on the command line. If authentication is required, use `hf auth login`
or the `HF_TOKEN` environment variable.

The runtime resolves the model in this order:

1. an explicit `--model` argument;
2. the `MODEL_PATH` environment variable;
3. `models/Qwen2.5-7B-Instruct` when it contains `config.json`;
4. the Hub ID `Qwen/Qwen2.5-7B-Instruct`.

For the future server container, mount this directory at `/models` and set:

```bash
MODEL_PATH=/models/Qwen2.5-7B-Instruct
```

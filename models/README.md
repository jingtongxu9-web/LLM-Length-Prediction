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
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir models/Qwen2.5-7B-Instruct
```

After download, record the verified revision for the server preflight:

```bash
echo a09a35458c702b33eeacc393d103063234e8bc28 > models/Qwen2.5-7B-Instruct/.frozen_revision
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

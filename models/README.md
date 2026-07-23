# Local models

This directory is the local mount point for model files. Model weights are intentionally excluded
from Git; only this README is versioned.

Expected local layout:

```text
models/
|-- README.md
`-- Qwen2.5-7B-Instruct/
    |-- .frozen_revision
    |-- config.json
    |-- tokenizer_config.json
    |-- model-00001-of-00004.safetensors
    `-- ...
```

This directory supplies model files to the collectors; it contains neither training prompts nor
Ridge outputs. Qwen weights and tokenizer files are frozen inputs and are never updated by the
ALPS Ridge training step.

After installing the runtime dependencies, download the frozen model with the project-owned
command. It pins the revision and writes `.frozen_revision` only after the snapshot download
completes:

```bash
python scripts/download_model.py --output models/Qwen2.5-7B-Instruct
```

Do not pass access tokens on the command line. The frozen Qwen snapshot is public; if Hub
authentication is nevertheless required, use the `HF_TOKEN` environment variable rather than
writing a token into this repository.

The runtime resolves the model in this order:

1. an explicit `--model` argument;
2. the `MODEL_PATH` environment variable;
3. `models/Qwen2.5-7B-Instruct` when it contains `config.json`;
4. the Hub ID `Qwen/Qwen2.5-7B-Instruct`.

For the future server container, mount this directory at `/models` and set:

```bash
MODEL_PATH=/models/Qwen2.5-7B-Instruct
```

On AutoDL without Docker, the model may instead live on the data disk, for example
`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`, with `MODEL_PATH` set to that absolute path.

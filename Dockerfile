ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface \
    MODEL_PATH=/models/Qwen2.5-7B-Instruct

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        jq \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${HOST_GID}" experiment \
    && useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home experiment

WORKDIR /workspace/repo

COPY requirements-docker.lock /tmp/requirements-docker.lock
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --requirement /tmp/requirements-docker.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps --editable .

COPY --chown=${HOST_UID}:${HOST_GID} . .

USER experiment

CMD ["bash"]

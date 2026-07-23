# Documentation index

This directory contains research and deployment documentation. It does not contain executable
entry points; commands referenced here live under `scripts/`.

| Document | Purpose |
|---|---|
| `research_plan.md` | Research questions, stages, comparisons, and milestones |
| `server_runbook.md` | Hardware-neutral isolated-server execution checklist |
| `docker_4090_runbook.md` | Pinned PyTorch 2.6/CUDA 12.4 Docker procedure for the supplied RTX 4090 server |
| `autodl_5090_runbook.md` | Direct-Python AutoDL RTX 5090/CUDA 12.8 setup and full ALPS v1 run |
| `大模型输出长度预测.pdf` | Project background/reference document |

The two deployment paths are intentionally independent. The school RTX 4090 Docker baseline keeps
its original `Dockerfile`, `docker-compose.yml`, `.env`, and `requirements-docker.lock`. A rented
Blackwell/RTX 5090 AutoDL instance uses a separate direct-Python procedure with
`requirements-autodl.lock`; it does not replace or modify the Docker baseline.

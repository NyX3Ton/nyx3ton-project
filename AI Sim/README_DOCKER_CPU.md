# AI Prompt Simulator RAG + Markdown - CPU Docker

This bundle runs the existing `ai-prompt-sim-rag.py` script in Docker without modifying the Python script.

## Why Python 3.11?

This image uses `python:3.11-slim-bookworm` intentionally. OpenVINO itself supports newer Python versions, but `optimum-intel` 1.27.0 is safest on Python 3.11 for this stack.

## Main pinned versions

```text
Python: 3.11
torch: 2.12.0 CPU wheel
transformers: 4.57.6
openvino: 2026.2.0
optimum-intel[openvino]: 1.27.0
```

`transformers` is pinned to 4.57.6 because this script was built around Transformers 4.x APIs. Newer Transformers 5.x exists, but is more risky with this existing OpenVINO/Optimum path.

## Start

```powershell
.\run_cpu.ps1
```

Or manually:

```powershell
docker compose -f docker-compose.cpu.yml down --remove-orphans
docker compose -f docker-compose.cpu.yml up --build
```

## URLs

```text
Gradio: http://127.0.0.1:7860
MLflow: http://127.0.0.1:5002
```

MLflow runs inside the container on 5001 but is published to host port 5002 to avoid local conflicts.

## First CPU test

For first run, consider changing the compose model to:

```yaml
LOCAL_MODEL_NAME: "Qwen/Qwen3-0.6B"
```

After the pipeline works, switch back to:

```yaml
LOCAL_MODEL_NAME: "Qwen/Qwen3-4B-Instruct-2507"
```

The first OpenVINO export can take time. It will be cached in `ov_models/`.

## Persistent folders

```text
hf_cache/
ov_models/
ov_cache/
mlflow_runs/
rag_uploads/
markdown_outputs/
outputs/
```

RAG uploads should appear in `rag_uploads/`.

## Stop

```powershell
.\stop_cpu.ps1
```

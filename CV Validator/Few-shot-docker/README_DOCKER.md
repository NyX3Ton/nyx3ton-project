# Few-shot CV Validator - Docker version

This Docker setup runs the Gradio Few-shot CV Validator with local Hugging Face models, optional GPU acceleration and MLflow tracking.

## What changed

- The app can bind to `0.0.0.0` inside Docker via environment variables.
- MLflow uses local SQLite under `./mlflow_runs` next to the app, not the main Git root.
- Hugging Face cache is mounted to `./hf_cache` so models are not downloaded again after every container rebuild.
- The Docker build installs Transformers directly from the Hugging Face GitHub repository to support newer model architectures.
- The PDF loader supports both modern `pymupdf` and legacy `fitz` imports.
- The DOCX loader uses `python-docx` via `from docx import Document`.

## GPU run

Requirements on the host:

- Docker Desktop or Docker Engine
- NVIDIA driver
- NVIDIA Container Toolkit / Docker GPU support

Build and start:

```powershell
docker compose up --build
```

Open:

```text
Gradio:  http://127.0.0.1:7860
MLflow:  http://127.0.0.1:5001
```

## CPU-only run

```powershell
docker compose -f docker-compose.cpu.yml up --build
```

CPU mode is useful for testing the UI, document parsing and MLflow tracking, but Qwen 3.5 4B will be slow on CPU.

## Model configuration

Edit `docker-compose.yml`:

```yaml
LLM_MODEL_ID: "unsloth/Qwen3.5-4B"
FALLBACK_LLM_MODEL_ID: "Qwen/Qwen2.5-3B-Instruct"
AUX_LLM_MODEL_ID: "Qwen/Qwen2.5-3B-Instruct"
```

If the Qwen 3.5 4B model ID is private/gated, set `HF_TOKEN` in `.env` or directly in compose.

## Volumes created next to this app

```text
hf_cache/      Hugging Face model cache
mlflow_runs/   MLflow SQLite DB and artifacts
outputs/       optional local outputs
```

These folders are intentionally mounted locally and should normally stay out of Git.


## Docker networking note

Inside the container, Gradio and MLflow must listen on `0.0.0.0`.
Do **not** change the container-side binding to `localhost` or `127.0.0.1`, because Docker would not be able to publish the service correctly.

For local-only access from the host machine, the compose files publish ports like this:

```yaml
ports:
  - "127.0.0.1:7860:7860"
  - "127.0.0.1:5001:5001"
```

So the rule is:

```text
Inside container: 0.0.0.0
In browser on host: http://127.0.0.1:7860 and http://127.0.0.1:5001
```

## Useful commands

Show logs:

```powershell
docker logs -f few-shot-validator
```

Stop:

```powershell
docker compose down
```

Rebuild from scratch:

```powershell
docker compose build --no-cache
```

Check GPU from inside container:

```powershell
docker exec -it few-shot-validator python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

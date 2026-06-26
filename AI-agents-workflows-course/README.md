# Agentic AI Weather Pipeline

An explainable, multi-agent demo that turns a plain-language request into a formatted
Excel weather report and (optionally) emails it. Four cooperating agents each perform one
job, and every step is recorded in an auditable trace so you can see exactly what happened
and why.

The pipeline runs **fully deterministically without any GPU or language model**. An optional
local LLM can assist with parsing, and the app automatically selects the best available
inference backend: **CUDA → OpenVINO → rule-based**.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Reasoning backends and device selection](#reasoning-backends-and-device-selection)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Using the interface](#using-the-interface)
- [Output files](#output-files)
- [How explainability works](#how-explainability-works)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [License and disclaimer](#license-and-disclaimer)

---

## What it does

1. You type a request such as *"Please prepare a 3-day weather report for Bratislava and send it as Excel."*
2. The **Prompt Agent** extracts the target location and the number of forecast days.
3. The **Weather Agent** geocodes the location and fetches the forecast from the free
   [Open-Meteo](https://open-meteo.com/) API (no API key required).
4. The **Excel Report Agent** builds a styled `.xlsx` workbook with a summary, a daily
   forecast table, and the full agent trace.
5. The **Email Agent** optionally emails the workbook as an attachment to a recipient you
   choose in the interface.

Everything is exposed through a [Gradio](https://www.gradio.app/) web interface, including a
live table of every agent action with timings and status codes.

---

## Architecture

```
            ┌──────────────┐     ┌──────────────┐     ┌────────────────────┐     ┌─────────────┐
  prompt →  │ Prompt Agent │ →   │ Weather Agent│ →   │ Excel Report Agent │ →   │ Email Agent │ → email
            └──────┬───────┘     └──────┬───────┘     └─────────┬──────────┘     └──────┬──────┘
                   │                    │                       │                       │
                   └────────────────────┴───────── Trace ───────┴───────────────────────┘
                                     (every step logged with status + elapsed time)
```

| Agent | Responsibility | Key dependencies |
|-------|----------------|------------------|
| **Prompt Agent** | Parse the request into a location + forecast-day count. Uses regex rules, optionally refined by the local LLM. | `re`, optional reasoner |
| **Weather Agent** | Geocode the location and fetch current + daily forecast data. | Open-Meteo Geocoding & Forecast APIs |
| **Excel Report Agent** | Produce a styled multi-sheet `.xlsx` (Summary, Daily forecast, Agent trace). | `openpyxl`, `pandas` |
| **Email Agent** | Validate the recipient address and send the workbook over SMTP with STARTTLS. | `smtplib`, `ssl`, `email` |

The workflow never depends on LLM output. If no model is available, deterministic parsing
handles everything end-to-end.

---

## Reasoning backends and device selection

The optional LLM helper improves location extraction from free-form prompts. On startup the
app picks a backend in this order (in `auto` mode):

1. **CUDA (PyTorch + Transformers)** — used when an NVIDIA GPU is detected *and* a model id
   is configured. Loads a Hugging Face causal-LM on the GPU. Quantization is automatic: if
   `bitsandbytes` is installed the model loads in 4-bit (so larger models fit in less VRAM),
   otherwise it falls back to fp16.
2. **OpenVINO GenAI** — Intel CPU / GPU / NPU runtime, used when an OpenVINO model directory
   is configured.
3. **Rule-based** — deterministic regex parsing. Always available; guarantees the pipeline
   runs even with no GPU, no model, and none of the optional packages installed.

**Recommended model:** `unsloth/Qwen3-4B-Instruct-2507` (instruction-tuned, no `<think>`
output) runs on both backends. On CUDA, set it as `CUDA_MODEL_ID`. For OpenVINO, export the
same model once and point `OPENVINO_MODEL_DIR` at the result:

```bash
pip install "optimum[openvino]"
optimum-cli export openvino --model unsloth/Qwen3-4B-Instruct-2507 \
    --weight-format int4 ./ov_qwen3_4b_instruct
```

The model is downloaded to the Hugging Face cache on first use and loaded from cache on later
runs; pre-fetch it without launching the UI via `python app.py --download`. Pick a size that
fits your hardware: a 4B model needs ~8 GB VRAM in fp16 (~3 GB in 4-bit). Use an **instruct**
model — base models (e.g. `*-Base`) don't follow the extraction instruction, and pure
"reasoning" models (e.g. DeepSeek-R1 distills) emit `<think>` text that breaks the JSON
parser. The LLM only assists prompt parsing; the pipeline works fully without it.

> **Note:** CUDA (NVIDIA) and OpenVINO (Intel) are distinct runtimes. OpenVINO does **not**
> run on CUDA. The detection routine tries CUDA first and cleanly falls back to OpenVINO,
> then to rule-based, logging the reason for each step in the runtime status.

You can force a specific backend with the `REASONER_BACKEND` environment variable
(`auto`, `cuda`, `openvino`, or `rule`). The active backend and its status are shown in the
interface and printed in the pipeline result after every run.

---

## Features

- **Four-agent, single-responsibility design** that is easy to read and extend.
- **Automatic CUDA detection** with graceful OpenVINO and rule-based fallback.
- **Editable recipient email** in the UI, defaulting to an environment variable and
  validated before any message is sent.
- **Styled Excel output** with frozen headers, auto-sized columns, and three sheets.
- **Full execution trace** — every agent step is timestamped with a status and detail.
- **No API keys for weather data** — Open-Meteo is free and open.
- **Dry-run by default** — email is only sent when you explicitly enable the checkbox.
- **Configuration via environment variables** / `.env` file, nothing hard-coded.

---

## Requirements

- Python 3.10 or newer (the code uses modern typing syntax).
- Required Python packages (see `requirements.txt`):
  - `gradio`, `pandas`, `requests`, `python-dotenv`, `openpyxl`
- Optional, only for local LLM assistance:
  - CUDA backend: `torch`, `transformers` (plus an NVIDIA GPU and CUDA drivers)
  - OpenVINO backend: `openvino-genai`

---

## Installation

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your configuration
copy .env.example .env           # macOS/Linux: cp .env.example .env

# 5. (optional) install an LLM backend
# pip install torch transformers          # NVIDIA CUDA
# pip install openvino-genai              # Intel OpenVINO
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values you need. All variables are optional
unless you want the Email Agent to actually send mail.

```dotenv
# --- Reasoning backend ---
REASONER_BACKEND=auto            # auto | cuda | openvino | rule

# CUDA (PyTorch + Transformers) — only used when an NVIDIA GPU is present
CUDA_MODEL_ID=                   # e.g. Qwen/Qwen2.5-0.5B-Instruct  (empty = skip CUDA)
CUDA_DEVICE=cuda:0

# OpenVINO GenAI — Intel CPU/GPU/NPU
OPENVINO_MODEL_DIR=              # path to an exported OpenVINO model (empty = skip)
OPENVINO_DEVICE=CPU              # CPU | GPU | NPU

# --- Email (SMTP) ---
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURITY=starttls           # starttls | ssl | none
SMTP_USER=you@example.com        # optional; login only happens if user+password set
SMTP_PASSWORD=your_app_password  # optional
SMTP_FROM=you@example.com
PREDEFINED_RECIPIENT=default-recipient@example.com

# --- Gradio server ---
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7861
```

| Variable | Purpose | Default |
|----------|---------|---------|
| `REASONER_BACKEND` | Force a backend or let the app choose | `auto` |
| `CUDA_MODEL_ID` | Hugging Face model id / local path for the CUDA backend | empty (disabled) |
| `CUDA_DEVICE` | Torch device string | `cuda:0` |
| `CUDA_QUANTIZATION` | `auto` (4-bit if bitsandbytes present, else fp16), `4bit`, or `fp16` | `auto` |
| `OPENVINO_MODEL_DIR` | Path to an OpenVINO model directory | empty (disabled) |
| `OPENVINO_DEVICE` | OpenVINO target device | `CPU` |
| `SMTP_HOST` / `SMTP_PORT` | Mail server and port | — / `587` |
| `SMTP_SECURITY` | Transport security: `starttls`, `ssl` (implicit, port 465), or `none` | `starttls` |
| `SMTP_USER` / `SMTP_PASSWORD` | SMTP login credentials (optional; login only if both set) | — |
| `SMTP_FROM` | From address | falls back to `SMTP_USER` |
| `PREDEFINED_RECIPIENT` | Default recipient pre-filled in the UI | empty |
| `GRADIO_SERVER_NAME` | Bind address | `127.0.0.1` |
| `GRADIO_SERVER_PORT` | Bind port | `7861` |

> **Gmail:** use an [App Password](https://support.google.com/accounts/answer/185833), not
> your normal account password, and keep STARTTLS on port `587`.

### OpenVINO mode example

```powershell
$env:OPENVINO_MODEL_DIR="D:\models\openvino\Qwen2.5-1.5B-Instruct-int4"
$env:OPENVINO_DEVICE="CPU"
python app.py
```

---

## Running the app

```bash
python app.py
```

The Gradio interface opens automatically at `http://127.0.0.1:7861`. The console and the UI
both report which reasoning backend was selected.

---

## Run with Docker

A `Dockerfile` and `docker-compose.yml` are included. By default Compose runs just the weather
app; the MLflow dashboard is an **optional service behind a `metrics` profile**. The app runs
the full pipeline (Prompt → Weather → Excel → Email) in deterministic **rule-based** mode — the
optional CUDA/OpenVINO LLM backends are not bundled (they need a GPU or extra runtimes).

```bash
# 1. Create your .env (kept on the host, never baked into the image)
cp .env.example .env        # then fill in SMTP_* for email

# 2a. App only
docker compose up --build

# 2b. App + MLflow dashboard
docker compose --profile metrics up --build
```

- App UI: `http://localhost:7861`
- MLflow dashboard (only with the `metrics` profile): `http://localhost:5000`

How it fits together:

- Your existing **`.env` is used unchanged** via Compose `env_file` — no secrets are copied
  into the image.
- `app.py` is **unchanged**. Compose sets `GRADIO_SERVER_NAME=0.0.0.0` (so the app is
  reachable from the host) and `GRADIO_SERVER_PORT=7861`; `app.py` already reads both from the
  environment.
- Generated reports are written to **`./Outputs`** on the host via a volume mount.
- The image includes `mlflow`. With `--profile metrics`, a **SQLite-backed MLflow tracking
  server** (`mlflow` service, port 5000) starts; the app logs to it over HTTP
  (`MLFLOW_TRACKING_URI=http://mlflow:5000`), and the DB + artifacts persist in the
  `mlflow-data` named volume. Without the profile, the server is absent and logging no-ops.
- Healthchecks poll the running services.

Stop with `docker compose down` (add `--profile metrics` if you started the dashboard).

**Enabling the LLM backends in Docker (advanced):** install `transformers`/`torch` (and use
an NVIDIA CUDA base image plus `nvidia-container-toolkit` and a `deploy.resources` GPU
reservation for CUDA), or install `openvino-genai` and mount an exported model directory for
OpenVINO. Add a Hugging Face cache volume (e.g. `~/.cache/huggingface`) so models are not
re-downloaded on each run.

---

## Using the interface

- **User prompt** — free-form request, e.g. *"3-day forecast for Vienna, email it."*
- **Location override** — optional; pins the location for reliable demos and skips parsing.
- **Forecast days** — 1 to 7.
- **Recipient email** — pre-filled from `PREDEFINED_RECIPIENT`; edit it to send elsewhere.
  The address is validated before anything is sent.
- **Send email to the recipient above** — leave unchecked for a dry run (report only).
- **Run agent pipeline** — executes all four agents.

The results panel shows the pipeline summary, the generated Excel file for download, the
agent trace, and the daily forecast table.

---

## Output files

Reports are written to an `Outputs/` folder next to `app.py`:

```
weather_report_<Location>_<YYYYMMDD_HHMMSS>.xlsx
```

Each workbook contains three sheets:

- **Summary** — requested vs. matched location, coordinates, timezone, current conditions, source, timestamp.
- **Daily forecast** — date, condition, min/max temperature, precipitation, max wind.
- **Agent trace** — every agent step with status and elapsed time.

---

## How explainability works

The pipeline is "explainable AI" in the sense that **nothing is a black box**:

- Each agent records a `Trace` event for every action (parse, geocode, fetch, build, send),
  including a human-readable detail string and the time it took.
- The trace is shown live in the UI and embedded as its own sheet in the Excel report.
- The reasoning backend's status (CUDA / OpenVINO / rule-based, with the reason it was
  chosen) is reported on every run.
- Weather data comes from a single, citable public source (Open-Meteo), and the location the
  API actually matched is shown next to the location you requested.

This makes it easy to audit how a given report was produced and to reproduce it.

---

## Metrics & evaluation (MLflow)

The app can log detailed performance metrics to [MLflow](https://mlflow.org/). It is
**optional and graceful** — if `mlflow` is not installed (or `MLFLOW_ENABLED=off`), the
pipeline runs unchanged and nothing is logged.

```bash
pip install mlflow
```

**Per-run telemetry (background).** Every pipeline run logs, in a non-blocking daemon thread:

- **Params:** reasoning backend, model id, quantization mode, SMTP security, location, forecast days.
- **Metrics:** total latency, per-agent/step latency (derived from the `Trace`), step counts
  (OK / ERROR / SKIPPED), a `success` flag, and current temperature / coordinates.
- **Tags & artifacts:** matched location, condition, email status, and the generated Excel file.

**Evaluation suite (on demand).** Score location/day extraction over a fixed prompt set and
log accuracy + parse latency as a separate experiment:

```bash
python app.py --eval
```

**Storage backends.** Set `MLFLOW_TRACKING_URI` in `.env`:

| Value | Backend | Notes |
|-------|---------|-------|
| *(empty)* | `./mlruns` file store | Simplest; deprecated in MLflow 3.x — needs `MLFLOW_ALLOW_FILE_STORE=true` |
| `sqlite:///mlflow.db` | local SQLite DB | **Recommended for durable, long-term validation history** |
| `http://host:5000` | tracking server | Used by Docker (`http://mlflow:5000`) |

**View the results (host).** For a SQLite backend, point the UI at the same DB:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # open http://127.0.0.1:5000
# Or for the legacy file store (needs the opt-out):
#   MLFLOW_ALLOW_FILE_STORE=true mlflow ui
```

**Long-term validations with Docker.** `docker compose --profile metrics up` starts a
**SQLite-backed MLflow tracking server** (`mlflow` service) at `http://localhost:5000`. The
app logs to it over HTTP, and the database + artifacts are kept in a persistent named volume
(`mlflow-data`), so validation history survives rebuilds and restarts. Run the eval suite
repeatedly (e.g. on a schedule) to accumulate accuracy/latency trends over time.

---

## Project structure

```
.
├── app.py                # the full application (agents, backends, Gradio UI)
├── requirements.txt      # Python dependencies
├── .env.example          # configuration template (copy to .env)
├── Dockerfile            # container image for the pipeline
├── docker-compose.yml    # one-command run (uses your .env, maps port 7861)
├── .dockerignore         # keeps secrets/noise out of the build context
├── .gitignore            # keeps .env and generated files out of git
├── README.md             # this file
├── Outputs/              # generated Excel reports (created at runtime)
└── mlruns/               # MLflow run logs, if enabled (created at runtime)
```

---

## Troubleshooting

- **"Location was not detected"** — add an explicit location in the prompt or use the
  Location override field.
- **"Invalid recipient email address"** — the Email Agent validates the address; check for
  typos. Sending is skipped until it is valid.
- **"Missing email configuration"** — only `SMTP_HOST` and `SMTP_FROM` are required. The
  email step is skipped (not failed) when they are absent. Authentication is optional.
- **CUDA not used** — confirm `torch.cuda.is_available()` is `True`, that `CUDA_MODEL_ID` is
  set, and that `torch`/`transformers` are installed. The runtime status explains the
  fallback reason.
- **Port already in use** — change `GRADIO_SERVER_PORT`.
- **No runs in MLflow / `Invalid Host header` 403** — MLflow 3.5+ has DNS-rebinding
  protection that only accepts allow-listed `Host` headers. The `mlflow` service passes
  `--allowed-hosts` (including `mlflow:5000`, the host the app connects as). If you change the
  service name or port, update that list. The app prints `[MetricsLogger] logging runs to …`
  on success or `[MetricsLogger] MLflow logging failed … <reason>` on failure — check
  `docker compose logs weather-pipeline`. Also confirm you're viewing the
  `agentic-weather-pipeline` experiment in the UI, not `Default`.

---

## Security notes

- Credentials are read from environment variables / `.env`; do **not** commit `.env`.
  Add it to `.gitignore`.
- Email transport security defaults to STARTTLS. Set `SMTP_SECURITY=ssl` for implicit-SSL
  servers (port `465`), or `SMTP_SECURITY=none` for unencrypted relays (not recommended on
  public networks).
- The recipient field is validated, but you are responsible for who you send reports to.

---

## License and disclaimer

This is a demonstration project. Add a license of your choice (for example, MIT) before
publishing. Weather data is provided by Open-Meteo under their terms of use. This software is
provided "as is", without warranty of any kind.

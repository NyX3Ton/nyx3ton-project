# Agentic AI Weather Pipeline Demo

A small Gradio demo showing a simple multi-agent workflow:

1. **Prompt / Weather Agent** parses the request and fetches weather data from Open-Meteo.
2. **Excel Report Agent** writes a clean Excel workbook.
3. **Email Agent** sends the workbook to a predefined recipient through SMTP.

The core workflow is deterministic and auditable. Optional local LLM assistance can run through **OpenVINO GenAI** when `OPENVINO_MODEL_DIR` is configured.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python app.py
```

## OpenVINO mode

Set a local OpenVINO GenAI model directory:

```powershell
$env:OPENVINO_MODEL_DIR="D:\\models\\openvino\\Qwen2.5-1.5B-Instruct-int4"
$env:OPENVINO_DEVICE="CPU"
python app.py
```

If no OpenVINO model is configured, the pipeline still works in rule-based mode. This keeps the live demo stable even without a local LLM.

## Email setup

Email sending is disabled by default unless SMTP variables are configured and the Gradio checkbox is enabled.

```text
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=user@example.com
SMTP_PASSWORD=your-password
SMTP_FROM=user@example.com
PREDEFINED_RECIPIENT=recipient@example.com
```

The UI does not accept arbitrary recipient addresses. This keeps the demo safer for live presentations.

## Weather source

The app uses Open-Meteo Geocoding API and Forecast API. No API key is required for basic usage.

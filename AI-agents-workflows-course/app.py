from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

OPENVINO_MODEL_DIR = os.getenv("OPENVINO_MODEL_DIR", "").strip()
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU").strip() or "CPU"

# CUDA (PyTorch + Transformers) backend config.
# CUDA_MODEL_ID is a Hugging Face model id or local path used only when an
# NVIDIA GPU is detected. Left empty by default so the app never downloads a
# model unexpectedly; when empty, the app falls back to OpenVINO / rule-based.
CUDA_MODEL_ID = os.getenv("CUDA_MODEL_ID", "").strip()
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0").strip() or "cuda:0"
# Backend preference: "auto" (CUDA -> OpenVINO -> rule-based), or force one of
# "cuda", "openvino", "rule".
REASONER_BACKEND = (os.getenv("REASONER_BACKEND", "auto").strip().lower() or "auto")

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or 587)
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()
PREDEFINED_RECIPIENT = os.getenv("PREDEFINED_RECIPIENT", "").strip()

CUSTOM_CSS = """
.gradio-container { max-width: 1500px !important; }
.agent-box textarea { font-size: 1rem !important; line-height: 1.45 !important; }
"""

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


@dataclass
class AgentEvent:
    agent: str
    step: str
    status: str
    detail: str
    elapsed_sec: float


class Trace:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def add(self, agent: str, step: str, status: str, detail: str, start: float) -> None:
        self.events.append(
            AgentEvent(
                agent=agent,
                step=step,
                status=status,
                detail=detail,
                elapsed_sec=round(time.perf_counter() - start, 3),
            )
        )

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(event) for event in self.events])


def detect_cuda() -> tuple[bool, str]:
    """Detect an available NVIDIA CUDA device via PyTorch.

    Returns (available, detail). Never raises: if torch is not installed or the
    CUDA runtime is unavailable, this reports False and the app falls back to
    OpenVINO and then to deterministic rule-based parsing.
    """
    try:
        import torch
    except Exception as exc:  # torch not installed
        return False, f"PyTorch not available: {exc}"

    try:
        if not torch.cuda.is_available():
            return False, "PyTorch is installed but no CUDA device is available."
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        return True, f"CUDA available: {count} device(s), primary GPU '{name}'."
    except Exception as exc:
        return False, f"CUDA detection failed: {exc}"


class Reasoner:
    """Common interface for optional local LLM helpers.

    The workflow never depends on LLM output. Any reasoner may be disabled, in
    which case the agents use deterministic parsing and still run end-to-end.
    """

    backend = "none"

    def __init__(self) -> None:
        self.status = "Rule-based mode."

    def is_enabled(self) -> bool:
        return False

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        return ""


class RuleBasedReasoner(Reasoner):
    """No-op reasoner: deterministic parsing only."""

    backend = "rule-based"

    def __init__(self, status: str = "Rule-based mode: no LLM backend enabled.") -> None:
        self.status = status


class CudaReasoner(Reasoner):
    """LLM helper backed by PyTorch + Transformers on an NVIDIA CUDA device."""

    backend = "cuda"

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = CUDA_DEVICE

        available, detail = detect_cuda()
        if not available:
            self.status = f"CUDA backend unavailable: {detail}"
            return

        if not CUDA_MODEL_ID:
            self.status = f"CUDA detected ({detail}) but CUDA_MODEL_ID is not configured."
            return

        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            tokenizer: Any = AutoTokenizer.from_pretrained(CUDA_MODEL_ID)
            model: Any = AutoModelForCausalLM.from_pretrained(
                CUDA_MODEL_ID,
                torch_dtype=torch.float16,
            )
            model = model.to(self.device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.status = f"CUDA (PyTorch/Transformers) loaded: {CUDA_MODEL_ID} on {self.device}. {detail}"
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.status = f"CUDA backend load failed: {exc}"

    def is_enabled(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        if not self.is_enabled():
            return ""

        try:
            import torch  # type: ignore[import-not-found]

            text = prompt
            if getattr(self.tokenizer, "chat_template", None):
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception:
            return ""


class OpenVINOReasoner(Reasoner):
    """Optional local LLM helper backed by OpenVINO GenAI (Intel CPU/GPU/NPU)."""

    backend = "openvino"

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.status = "OpenVINO backend unavailable: OPENVINO_MODEL_DIR is not configured."

        if not OPENVINO_MODEL_DIR:
            return

        model_dir = Path(OPENVINO_MODEL_DIR)
        if not model_dir.exists():
            self.status = f"OpenVINO backend unavailable: model path not found: {model_dir}"
            return

        try:
            import openvino_genai as ov_genai

            self.pipeline = ov_genai.LLMPipeline(str(model_dir), OPENVINO_DEVICE)
            self.status = f"OpenVINO GenAI loaded: {model_dir.name} on {OPENVINO_DEVICE}"
        except Exception as exc:
            self.pipeline = None
            self.status = f"OpenVINO backend load failed: {exc}"

    def is_enabled(self) -> bool:
        return self.pipeline is not None

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        if not self.pipeline:
            return ""

        try:
            return str(self.pipeline.generate(prompt, max_new_tokens=max_new_tokens))
        except Exception:
            return ""


def build_reasoner() -> Reasoner:
    """Select a reasoning backend.

    Order in "auto" mode: try CUDA (PyTorch/Transformers); if it cannot be
    enabled, fall back to OpenVINO; if that also fails, use deterministic
    rule-based parsing so the pipeline always runs end-to-end.
    """
    notes: list[str] = []

    def try_cuda() -> Reasoner | None:
        reasoner = CudaReasoner()
        notes.append(reasoner.status)
        return reasoner if reasoner.is_enabled() else None

    def try_openvino() -> Reasoner | None:
        reasoner = OpenVINOReasoner()
        notes.append(reasoner.status)
        return reasoner if reasoner.is_enabled() else None

    if REASONER_BACKEND == "cuda":
        selected = try_cuda()
    elif REASONER_BACKEND == "openvino":
        selected = try_openvino()
    elif REASONER_BACKEND == "rule":
        selected = None
    else:  # auto
        selected = try_cuda() or try_openvino()

    if selected is not None:
        return selected

    fallback_status = "Rule-based mode (deterministic parsing). " + " | ".join(notes)
    return RuleBasedReasoner(fallback_status.strip())


class PromptAgent:
    def __init__(self, reasoner: Reasoner) -> None:
        self.reasoner = reasoner

    def run(self, prompt: str, location_override: str, forecast_days: int, trace: Trace) -> dict[str, Any]:
        start = time.perf_counter()
        prompt = prompt.strip()
        location = location_override.strip() or self._extract_location(prompt)
        days = max(1, min(int(forecast_days), 7))

        if self.reasoner.is_enabled() and not location_override.strip():
            llm_location = self._try_llm_extract_location(prompt)
            if llm_location:
                location = llm_location

        if not location:
            raise gr.Error("Location was not detected. Add a location in the prompt or use the Location field.")

        result = {"prompt": prompt, "location": location, "forecast_days": days}
        trace.add("Prompt Agent", "Parse request", "OK", json.dumps(result, ensure_ascii=False), start)
        return result

    def _extract_location(self, prompt: str) -> str:
        patterns = [
            r"(?:weather|forecast)\s+(?:in|for|at)\s+([A-Za-zÁ-ž .'-]{2,60})",
            r"(?:pocasie|predpoved)\s+(?:v|pre)\s+([A-Za-zÁ-ž .'-]{2,60})",
            r"(?:in|for|at|v|pre)\s+([A-Za-zÁ-ž .'-]{2,60})",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .,!?:;")
                value = re.split(r"\b(today|tomorrow|next|for|please|pros[ií]m)\b", value, flags=re.I)[0]
                return value.strip(" .,!?:;")
        return ""

    def _try_llm_extract_location(self, prompt: str) -> str:
        llm_prompt = (
            "Extract only the location name from the user request. "
            "Return JSON only: {\"location\": \"...\"}.\n"
            f"User request: {prompt}"
        )
        text = self.reasoner.generate(llm_prompt, max_new_tokens=80)
        try:
            data = json.loads(text[text.find("{") : text.rfind("}") + 1])
            location = str(data.get("location", "")).strip()
            if 2 <= len(location) <= 60:
                return location
        except Exception:
            return ""
        return ""


class WeatherAgent:
    def run(self, request_data: dict[str, Any], trace: Trace) -> dict[str, Any]:
        location = request_data["location"]
        days = request_data["forecast_days"]

        start = time.perf_counter()
        geo = self._geocode(location)
        trace.add("Weather Agent", "Geocode location", "OK", f"{geo['name']}, {geo.get('country', '')}", start)

        start = time.perf_counter()
        forecast = self._forecast(geo, days)
        trace.add("Weather Agent", "Fetch forecast", "OK", f"Fetched {days} day(s) from Open-Meteo", start)

        return {"request": request_data, "geocode": geo, "forecast": forecast}

    def _geocode(self, location: str) -> dict[str, Any]:
        response = requests.get(
            OPEN_METEO_GEOCODING,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not results:
            raise gr.Error(f"Location was not found by Open-Meteo geocoding: {location}")
        return results[0]

    def _forecast(self, geo: dict[str, Any], days: int) -> dict[str, Any]:
        response = requests.get(
            OPEN_METEO_FORECAST,
            params={
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "forecast_days": days,
                "timezone": "auto",
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


class ExcelReportAgent:
    def run(self, weather_payload: dict[str, Any], trace: Trace) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
        start = time.perf_counter()
        geo = weather_payload["geocode"]
        forecast = weather_payload["forecast"]
        request_data = weather_payload["request"]

        daily = forecast.get("daily", {})
        daily_df = pd.DataFrame(
            {
                "date": daily.get("time", []),
                "condition": [WEATHER_CODES.get(code, f"Code {code}") for code in daily.get("weather_code", [])],
                "temp_min_c": daily.get("temperature_2m_min", []),
                "temp_max_c": daily.get("temperature_2m_max", []),
                "precipitation_mm": daily.get("precipitation_sum", []),
                "max_wind_kmh": daily.get("wind_speed_10m_max", []),
            }
        )

        current = forecast.get("current", {})
        summary = {
            "requested_location": request_data["location"],
            "matched_location": f"{geo.get('name')}, {geo.get('country', '')}",
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "timezone": forecast.get("timezone"),
            "current_temperature_c": current.get("temperature_2m"),
            "current_apparent_temperature_c": current.get("apparent_temperature"),
            "current_humidity_percent": current.get("relative_humidity_2m"),
            "current_condition": WEATHER_CODES.get(current.get("weather_code"), current.get("weather_code")),
            "source": "Open-Meteo Forecast API",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        safe_location = re.sub(r"[^A-Za-z0-9_-]+", "_", str(geo.get("name", "weather"))).strip("_")
        file_path = OUTPUT_DIR / f"weather_report_{safe_location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        self._write_workbook(file_path, summary, daily_df, trace)

        trace.add("Excel Report Agent", "Create workbook", "OK", str(file_path.name), start)
        return file_path, daily_df, summary

    def _write_workbook(self, file_path: Path, summary: dict[str, Any], daily_df: pd.DataFrame, trace: Trace) -> None:
        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"
        ws.append(["Field", "Value"])
        for key, value in summary.items():
            ws.append([key, value])

        ws_daily = wb.create_sheet("Daily forecast")
        ws_daily.append(list(daily_df.columns))
        for _, row in daily_df.iterrows():
            ws_daily.append(list(row))

        ws_trace = wb.create_sheet("Agent trace")
        trace_df = trace.dataframe()
        ws_trace.append(list(trace_df.columns))
        for _, row in trace_df.iterrows():
            ws_trace.append(list(row))

        self._style_sheet(ws)
        self._style_sheet(ws_daily)
        self._style_sheet(ws_trace)
        wb.save(file_path)

    def _style_sheet(self, ws) -> None:
        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)
        thin = Side(style="thin", color="D9E2F3")

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = Border(bottom=thin)

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(bottom=thin)

        for col in range(1, ws.max_column + 1):
            column_letter = get_column_letter(col)
            max_len = max(len(str(ws.cell(row=row, column=col).value or "")) for row in range(1, ws.max_row + 1))
            ws.column_dimensions[column_letter].width = min(max(max_len + 3, 14), 42)

        ws.freeze_panes = "A2"


class EmailAgent:
    def run(self, file_path: Path, summary: dict[str, Any], should_send: bool, trace: Trace) -> str:
        start = time.perf_counter()

        if not should_send:
            trace.add("Email Agent", "Send email", "SKIPPED", "Dry run: send checkbox not enabled", start)
            return "Email Agent skipped: dry run mode."

        missing = [
            name
            for name, value in {
                "SMTP_HOST": SMTP_HOST,
                "SMTP_USER": SMTP_USER,
                "SMTP_PASSWORD": SMTP_PASSWORD,
                "SMTP_FROM": SMTP_FROM,
                "PREDEFINED_RECIPIENT": PREDEFINED_RECIPIENT,
            }.items()
            if not value
        ]
        if missing:
            detail = "Missing email configuration: " + ", ".join(missing)
            trace.add("Email Agent", "Send email", "SKIPPED", detail, start)
            return detail

        msg = EmailMessage()
        msg["Subject"] = f"Weather report - {summary.get('matched_location', 'location')}"
        msg["From"] = SMTP_FROM
        msg["To"] = PREDEFINED_RECIPIENT
        msg.set_content(
            "Hello,\n\n"
            "Attached is the weather report generated by the Agentic AI demo pipeline.\n\n"
            f"Location: {summary.get('matched_location')}\n"
            f"Current condition: {summary.get('current_condition')}\n"
            f"Current temperature: {summary.get('current_temperature_c')} °C\n\n"
            "Regards,\nAgentic AI Weather Demo"
        )

        with file_path.open("rb") as fh:
            msg.add_attachment(
                fh.read(),
                maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=file_path.name,
            )

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        trace.add("Email Agent", "Send email", "OK", f"Sent to predefined recipient: {PREDEFINED_RECIPIENT}", start)
        return f"Email sent to predefined recipient: {PREDEFINED_RECIPIENT}"


REASONER = build_reasoner()
PROMPT_AGENT = PromptAgent(REASONER)
WEATHER_AGENT = WeatherAgent()
EXCEL_AGENT = ExcelReportAgent()
EMAIL_AGENT = EmailAgent()


def run_pipeline(prompt: str, location_override: str, forecast_days: int, send_email: bool):
    trace = Trace()
    started = time.perf_counter()

    request_data = PROMPT_AGENT.run(prompt, location_override, forecast_days, trace)
    weather_payload = WEATHER_AGENT.run(request_data, trace)
    file_path, daily_df, summary = EXCEL_AGENT.run(weather_payload, trace)
    email_status = EMAIL_AGENT.run(file_path, summary, bool(send_email), trace)

    total_sec = round(time.perf_counter() - started, 3)
    status = (
        f"Pipeline finished in {total_sec}s\n\n"
        f"Reasoning backend: {REASONER.backend}\n"
        f"Runtime: {REASONER.status}\n"
        f"Weather source: Open-Meteo\n"
        f"Excel file: {file_path.name}\n"
        f"Email status: {email_status}\n\n"
        f"Current weather in {summary.get('matched_location')}: "
        f"{summary.get('current_condition')}, {summary.get('current_temperature_c')} °C"
    )

    return status, trace.dataframe(), daily_df, str(file_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic AI Weather Pipeline", css=CUSTOM_CSS) as demo:
        gr.Markdown(
            """
            # Agentic AI Weather Pipeline

            **Prompt Agent -> Weather Agent -> Excel Agent -> Email Agent**  
            Local OpenVINO reasoning is optional. The workflow remains deterministic and auditable.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="User prompt",
                    lines=4,
                    value="Please prepare a 3-day weather report for Bratislava and send it as Excel.",
                    elem_classes=["agent-box"],
                )
                location = gr.Textbox(
                    label="Location override - optional, useful for live demo reliability",
                    value="",
                    placeholder="e.g. Bratislava, Vienna, Prague",
                )
                days = gr.Slider(1, 7, value=3, step=1, label="Forecast days")
                send = gr.Checkbox(False, label="Send email to predefined recipient")
                run_btn = gr.Button("Run agent pipeline", variant="primary")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Pipeline result", lines=12, elem_classes=["agent-box"])
                excel_file = gr.File(label="Generated Excel report")

        with gr.Row():
            trace = gr.Dataframe(label="Agent trace", interactive=False)
            daily = gr.Dataframe(label="Daily forecast", interactive=False)

        gr.Markdown(
            f"""
            ### Runtime status

            `{REASONER.status}`

            Email recipient is fixed by environment variable `PREDEFINED_RECIPIENT`; the UI does not accept arbitrary email addresses.
            """
        )

        run_btn.click(run_pipeline, [prompt, location, days, send], [status, trace, daily, excel_file])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
        inbrowser=True,
        share=False,
    )

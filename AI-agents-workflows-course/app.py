from __future__ import annotations

import json, os, re, smtplib, ssl, threading, time, requests
from dataclasses import asdict, dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
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

CUDA_MODEL_ID = os.getenv("CUDA_MODEL_ID", "").strip()
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0").strip() or "cuda:0"
CUDA_QUANTIZATION = (os.getenv("CUDA_QUANTIZATION", "auto").strip().lower() or "auto")

REASONER_BACKEND = (os.getenv("REASONER_BACKEND", "auto").strip().lower() or "auto")

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or 587)
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()

SMTP_SECURITY = (os.getenv("SMTP_SECURITY", "starttls").strip().lower() or "starttls")
PREDEFINED_RECIPIENT = os.getenv("PREDEFINED_RECIPIENT", "").strip()

# Metrics / evaluation (optional, MLflow). Disabled gracefully if not installed.
MLFLOW_ENABLED = (os.getenv("MLFLOW_ENABLED", "on").strip().lower() or "auto")  # auto | on | off
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()                # empty = local ./mlruns
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "agentic-weather-pipeline").strip()

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
        self.events.append(AgentEvent(agent=agent,step=step,status=status,detail=detail,elapsed_sec=round(time.perf_counter() - start, 3)))

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(event) for event in self.events])

def _metric_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./: -]", "_", name)[:250]
class MetricsLogger:

    def __init__(self, experiment: str = MLFLOW_EXPERIMENT) -> None:
        self.enabled = False
        self.experiment = experiment
        self.uri = MLFLOW_TRACKING_URI or (BASE_DIR / "mlruns").as_uri()
        self.status = "MLflow disabled."

        if MLFLOW_ENABLED in ("off", "false", "0", "no"):
            return
        try:
            import mlflow  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:
            self.status = f"MLflow not installed: {exc}"
            return
        self.enabled = True
        self.status = f"MLflow -> {self.uri} (experiment: {self.experiment})"

    def log_run(self, *, params: dict[str, Any], metrics: dict[str, float],
                tags: dict[str, Any] | None = None, artifacts: list[str] | None = None,
                run_name: str | None = None, blocking: bool = False) -> None:
        if not self.enabled:
            return
        args = (params, metrics, tags, artifacts, run_name)
        if blocking:
            self._log(*args)
        else:
            threading.Thread(target=self._log, args=args, daemon=True).start()

    def _log(self, params: dict[str, Any], metrics: dict[str, float],
                tags: dict[str, Any] | None, artifacts: list[str] | None,
                run_name: str | None) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]

            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(self.experiment)
            with mlflow.start_run(run_name=run_name):
                if tags:
                    mlflow.set_tags(tags)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                for path in (artifacts or []):
                    try:
                        mlflow.log_artifact(path)
                    except Exception:
                        pass
        except Exception:
            pass  # never let logging break/raise


def detect_cuda() -> tuple[bool, str]:

    try:
        import torch  # type: ignore[import-not-found]
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

def resolve_model_path(model_id: str) -> tuple[str, str]:
    candidate = Path(model_id)
    if candidate.exists() and candidate.is_dir():
        return str(candidate), "local directory"

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_id, "huggingface (auto-cache)"

    try:
        local = snapshot_download(model_id, local_files_only=True)
        return local, "local cache"
    except Exception:
        local = snapshot_download(model_id)
        return local, "downloaded"

def download_model(model_id: str) -> None:
    if not model_id:
        print("No model id configured (set CUDA_MODEL_ID). Nothing to download.")
        return
    path, source = resolve_model_path(model_id)
    print(f"Model '{model_id}' ready ({source}): {path}")
class Reasoner:
    backend = "none"

    def __init__(self) -> None:
        self.status = "Rule-based mode."

    def is_enabled(self) -> bool:
        return False

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        return ""
class RuleBasedReasoner(Reasoner):
    backend = "rule-based"

    def __init__(self, status: str = "Rule-based mode: no LLM backend enabled.") -> None:
        self.status = status
class CudaReasoner(Reasoner):
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
            import torch
            from transformers import (AutoModelForCausalLM,AutoTokenizer)
            model_source, source_label = resolve_model_path(CUDA_MODEL_ID)
            tokenizer: Any = AutoTokenizer.from_pretrained(model_source)
            quant_config: Any = None
            quant_mode = "fp16"
            if CUDA_QUANTIZATION not in ("fp16", "none", "off"):
                try:
                    import bitsandbytes
                    from transformers import BitsAndBytesConfig

                    quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
                    quant_mode = "4-bit nf4"
                except Exception:
                    quant_config = None  # bitsandbytes unavailable -> fp16

            if quant_config is not None:
                # 4-bit models are placed via device_map and must not be .to()'d.
                model: Any = AutoModelForCausalLM.from_pretrained(model_source,quantization_config=quant_config,device_map={"": self.device})
            else:
                model = AutoModelForCausalLM.from_pretrained(model_source,torch_dtype=torch.float16)
                model = model.to(self.device)

            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.status = (
                            f"CUDA (PyTorch/Transformers, {quant_mode}, {source_label}) loaded: "
                            f"{CUDA_MODEL_ID} on {self.device}. {detail}"
                        )
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
                text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],tokenize=False,add_generation_prompt=True)

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
            generated = output[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception:
            return ""
class OpenVINOReasoner(Reasoner):
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
            import openvino_genai as ov_genai  # type: ignore[import-not-found]

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

    def run(self, prompt: str, location_override: str, trace: Trace) -> dict[str, Any]:
        start = time.perf_counter()
        prompt = prompt.strip()
        location = location_override.strip() or self._extract_location(prompt)
        days = self._extract_days(prompt)

        if self.reasoner.is_enabled() and not location_override.strip():
            llm_location = self._try_llm_extract_location(prompt)
            if llm_location:
                location = llm_location

        if not location:
            raise gr.Error("Location was not detected. Add a location in the prompt or use the Location field.")

        result = {"prompt": prompt, "location": location, "forecast_days": days}
        trace.add("Prompt Agent", "Parse request", "OK", json.dumps(result, ensure_ascii=False), start)
        return result

    def _extract_days(self, prompt: str, default: int = 3) -> int:
        match = re.search(r"(\d+)\s*-?\s*days?\b", prompt, re.IGNORECASE)
        if not match:
            match = re.search(r"\bfor\s+(\d+)\b", prompt, re.IGNORECASE)
        if match:
            try:
                return max(1, min(int(match.group(1)), 7))
            except ValueError:
                pass
        return default

    def _extract_location(self, prompt: str) -> str:
        patterns = [
                    r"(?:weather|forecast)\s+(?:in|for|at)\s+([A-Za-zÁ-ž .'-]{2,60})",
                    r"(?:pocasie|predpoved)\s+(?:v|pre)\s+([A-Za-zÁ-ž .'-]{2,60})",
                    r"(?:in|for|at|v|pre)\s+([A-Za-zÁ-ž .'-]{2,60})",
                    ]
        # Words that signal the location name has ended (the rest is intent,
        # timing, or delivery instructions, not part of the place name).
        stop_words = (
                        r"\b(?:and|then|today|tomorrow|tonight|next|for|please|send|sending|"
                        r"email|e-?mail|mail|as|with|by|pros[ií]m)\b"
                    )
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .,!?:;")
                value = re.split(stop_words, value, flags=re.I)[0]
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
        response = requests.get(OPEN_METEO_GEOCODING,params={"name": location, "count": 1, "language": "en", "format": "json"},timeout=20)
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
        daily_df = pd.DataFrame({
                                "date": daily.get("time", []),
                                "condition": [WEATHER_CODES.get(code, f"Code {code}") for code in daily.get("weather_code", [])],
                                "temp_min_c": daily.get("temperature_2m_min", []),
                                "temp_max_c": daily.get("temperature_2m_max", []),
                                "precipitation_mm": daily.get("precipitation_sum", []),
                                "max_wind_kmh": daily.get("wind_speed_10m_max", []),
                                })

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
        if ws is None:
            ws = wb.create_sheet("Summary")
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
    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def _parse_recipients(self, raw: str) -> tuple[list[str], list[str]]:
        parts = [p.strip() for p in re.split(r"[,;\n]+", raw or "")]
        seen: set[str] = set()
        valid: list[str] = []
        invalid: list[str] = []
        for part in parts:
            if not part or part.lower() in seen:
                continue
            seen.add(part.lower())
            (valid if self.EMAIL_RE.match(part) else invalid).append(part)
        return valid, invalid

    def run(self,file_path: Path,summary: dict[str, Any],should_send: bool,recipient: str,trace: Trace) -> str:
        start = time.perf_counter()
        recipient = (recipient or "").strip() or PREDEFINED_RECIPIENT

        if not should_send:
            trace.add("Email Agent", "Send email", "SKIPPED", "Dry run: send checkbox not enabled", start)
            return "Email Agent skipped: dry run mode."

        recipients, invalid = self._parse_recipients(recipient)

        if invalid:
            detail = "Invalid recipient email address(es): " + ", ".join(invalid)
            trace.add("Email Agent", "Send email", "SKIPPED", detail, start)
            return detail

        if not recipients:
            detail = "No recipient email provided. Enter one or more addresses (comma-separated) or set PREDEFINED_RECIPIENT."
            trace.add("Email Agent", "Send email", "SKIPPED", detail, start)
            return detail

        missing = [
                    name
                    for name, value in {"SMTP_HOST": SMTP_HOST,"SMTP_FROM": SMTP_FROM}.items()
                    if not value
                    ]
        if missing:
            detail = "Missing email configuration: " + ", ".join(missing)
            trace.add("Email Agent", "Send email", "SKIPPED", detail, start)
            return detail

        msg = EmailMessage()
        msg["Subject"] = f"Weather report - {summary.get('matched_location', 'location')}"
        msg["From"] = SMTP_FROM
        msg["To"] = ", ".join(recipients)
        msg.set_content(
                        "Hello,\n\n"
                        "I have attached the weather report generated by the Agentic AI demo pipeline.\n\n"
                        f"Location: {summary.get('matched_location')}\n"
                        f"Current condition: {summary.get('current_condition')}\n"
                        f"Current temperature: {summary.get('current_temperature_c')} °C\n\n"
                        "Regards, \n Agentic AI Weather Demo"
                        )

        with file_path.open("rb") as fh:
                msg.add_attachment(
                                    fh.read(),
                                    maintype="application",
                                    subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    filename=file_path.name,
                                    )

        context = ssl.create_default_context()
        use_implicit_ssl = SMTP_SECURITY in ("ssl", "smtps", "tls")
        use_starttls = SMTP_SECURITY not in ("ssl", "smtps", "tls", "none", "plain", "off")

        try:
            server: smtplib.SMTP
            if use_implicit_ssl:
                server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30)
            else:
                server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)

            with server:
                if use_starttls:
                    server.starttls(context=context)
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        except (OSError, smtplib.SMTPException) as exc:
            detail = (
                        f"Email send failed via {SMTP_HOST}:{SMTP_PORT} ({SMTP_SECURITY}): {exc}. "
                        "Check SMTP_HOST/SMTP_PORT, network access, and credentials."
                    )
            trace.add("Email Agent", "Send email", "ERROR", detail, start)
            return detail

        security_label = "implicit SSL" if use_implicit_ssl else ("STARTTLS" if use_starttls else "no encryption")
        trace.add("Email Agent", "Send email", "OK", f"Sent to: {recipient} ({security_label})", start)
        return f"Email sent to: {recipient}"

REASONER = build_reasoner()
PROMPT_AGENT = PromptAgent(REASONER)
WEATHER_AGENT = WeatherAgent()
EXCEL_AGENT = ExcelReportAgent()
EMAIL_AGENT = EmailAgent()
METRICS = MetricsLogger()

def run_pipeline(prompt: str,location_override: str,send_email: bool,recipient_email: str):
    trace = Trace()
    started = time.perf_counter()

    request_data = PROMPT_AGENT.run(prompt, location_override, trace)
    weather_payload = WEATHER_AGENT.run(request_data, trace)
    file_path, daily_df, summary = EXCEL_AGENT.run(weather_payload, trace)
    email_status = EMAIL_AGENT.run(file_path, summary, bool(send_email), recipient_email, trace)

    total_sec = round(time.perf_counter() - started, 3)
    status = (
                f"Pipeline finished in {total_sec}s\n\n"
                f"Reasoning backend: {REASONER.backend}\n"
                f"Runtime: {REASONER.status}\n"
                f"Weather source: Open-Meteo\n"
                f"Forecast days (from prompt): {request_data['forecast_days']}\n"
                f"Excel file: {file_path.name}\n"
                f"Email status: {email_status}\n\n"
                f"Current weather in {summary.get('matched_location')}: "
                f"{summary.get('current_condition')}, {summary.get('current_temperature_c')} °C"
                )

    try:
        events = trace.events
        status_counts: dict[str, int] = {}
        for e in events:
            status_counts[e.status] = status_counts.get(e.status, 0) + 1

        metrics: dict[str, float] = {
                                    "total_latency_sec": total_sec,
                                    "n_steps": len(events),
                                    "n_ok": status_counts.get("OK", 0),
                                    "n_error": status_counts.get("ERROR", 0),
                                    "n_skipped": status_counts.get("SKIPPED", 0),
                                    "forecast_days": float(request_data["forecast_days"]),
                                    "success": 1.0 if status_counts.get("ERROR", 0) == 0 else 0.0,
                                    }
        for e in events:  # per-agent/step latency
            metrics[_metric_key(f"latency.{e.agent}.{e.step}")] = e.elapsed_sec
        for k in ("current_temperature_c", "latitude", "longitude"):
            v = summary.get(k)
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

        params = {
                    "reasoning_backend": REASONER.backend,
                    "model_id": CUDA_MODEL_ID or "(none)",
                    "quantization": CUDA_QUANTIZATION,
                    "smtp_security": SMTP_SECURITY,
                    "location": request_data["location"],
                    "forecast_days": request_data["forecast_days"],
                }
        tags = {
                    "matched_location": str(summary.get("matched_location")),
                    "condition": str(summary.get("current_condition")),
                    "email_status": email_status[:250],
                }
        METRICS.log_run(params=params, metrics=metrics, tags=tags,
                        artifacts=[str(file_path)], run_name=request_data["location"])
    except Exception:
        pass

    return status, trace.dataframe(), daily_df, str(file_path)


EVAL_CASES = [
                {"prompt": "Please prepare a 3-day weather report for Bratislava and send it as Excel.", "location": "Bratislava", "days": 3},
                {"prompt": "5 day forecast for Vienna", "location": "Vienna", "days": 5},
                {"prompt": "weather for Prague for 7 days", "location": "Prague", "days": 7},
                {"prompt": "give me a 10-day outlook for London", "location": "London", "days": 7},
                {"prompt": "weather in Paris", "location": "Paris", "days": 3},
                {"prompt": "2-day report for Madrid please", "location": "Madrid", "days": 2},
                #{"prompt": "10-day report for Berlin please", "location": "Berlin", "days": 10},
                ]

def run_evaluation() -> None:
    agent = PromptAgent(REASONER)
    loc_hits = day_hits = 0
    latencies: list[float] = []
    for case in EVAL_CASES:
        t = time.perf_counter()
        try:
            rd = agent.run(case["prompt"], "", Trace())
            loc_ok = str(rd["location"]).strip().lower() == case["location"].lower()
            day_ok = rd["forecast_days"] == case["days"]
        except Exception:
            loc_ok = day_ok = False
        latencies.append(time.perf_counter() - t)
        loc_hits += int(loc_ok)
        day_hits += int(day_ok)
        print(f"[{'OK ' if loc_ok and day_ok else 'XX '}] {case['prompt']!r}")

    n = len(EVAL_CASES)
    metrics = {
                "location_accuracy": loc_hits / n,
                "days_accuracy": day_hits / n,
                "n_cases": float(n),
                "avg_parse_latency_sec": sum(latencies) / n,
                }
    print(
            f"\nlocation_accuracy={metrics['location_accuracy']:.2f} "
            f"days_accuracy={metrics['days_accuracy']:.2f} "
            f"avg_latency={metrics['avg_parse_latency_sec'] * 1000:.1f}ms"
        )
    MetricsLogger(experiment=f"{MLFLOW_EXPERIMENT}-eval").log_run(params={"reasoning_backend": REASONER.backend, "model_id": CUDA_MODEL_ID or "(none)"},metrics=metrics, run_name="extraction-eval", blocking=True)


def _launch_supports_css() -> bool:
    try:
        import inspect

        return "css" in inspect.signature(gr.Blocks.launch).parameters
    except Exception:
        return False

def build_ui() -> gr.Blocks:
    blocks_kwargs: dict[str, Any] = {"title": "Agentic AI Weather Pipeline"}
    if not _launch_supports_css():
        blocks_kwargs["css"] = CUSTOM_CSS
    with gr.Blocks(**blocks_kwargs) as demo:
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
                                    lines=5,
                                    value="Please prepare a 3-day weather report for Bratislava and send it as Excel.",
                                    info="Forecast length is read from the text, e.g. '3-day' or 'for 5 days' (1-7, default 3).",
                                    elem_classes=["agent-box"],
                                    )
                location = gr.Textbox(
                                        label="Location override (optional)",
                                        value="",
                                        placeholder="e.g. Bratislava, Vienna, Prague",
                                    )
                recipient = gr.Textbox(
                                        label="Recipient email",
                                        value=PREDEFINED_RECIPIENT,
                                        lines=3,
                                        placeholder="name@example.com",
                                        info="Defaults to PREDEFINED_RECIPIENT; edit to send elsewhere. Validated before sending.",
                                        )
                send = gr.Checkbox(False, label="Send email to the recipient above")
                run_btn = gr.Button("Run agent pipeline", variant="primary")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Pipeline result", lines=15, elem_classes=["agent-box"])
                excel_file = gr.File(label="Generated Excel report")

        with gr.Row():
            trace = gr.Dataframe(label="Agent trace", interactive=False)
            daily = gr.Dataframe(label="Daily forecast", interactive=False)

        gr.Markdown(
                    f"""
                    ### Runtime status

                    `{REASONER.status}`

                    `{METRICS.status}`

                    The recipient email defaults to `PREDEFINED_RECIPIENT` but can be edited in the interface. The address is validated before sending, and SMTP credentials still come from environment variables.
                    """
                    )

        run_btn.click(run_pipeline,[prompt, location, send, recipient],[status, trace, daily, excel_file])

    return demo

if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_model(CUDA_MODEL_ID)
        sys.exit(0)

    if "--eval" in sys.argv:
        run_evaluation()
        sys.exit(0)

    app = build_ui()
    launch_kwargs: dict[str, Any] = dict(
                                        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                                        #server_port=int(os.getenv("GRADIO_SERVER_PORT", "7861")),
                                        inbrowser=True,
                                        share=False,
                                        allowed_paths=[str(OUTPUT_DIR)],
                                        )
    if _launch_supports_css():
        launch_kwargs["css"] = CUSTOM_CSS
    app.launch(**launch_kwargs)
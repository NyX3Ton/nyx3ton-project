# 0. Mandatory library installs

#!python -m pip install --upgrade pip setuptools wheel
#!python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
#!python -m pip install --upgrade ipykernel jupyter
#!python -m pip install --upgrade mlflow
#!python -m pip install --upgrade openvino optimum-intel transformers accelerate safetensors sentencepiece huggingface_hub requests python-dotenv gradio bs4
#!python -m pip install --upgrade hf_xet nncf
#!python -m pip install --upgrade sentence-transformers faiss-cpu numpy python-docx PyMuPDF striprtf

# 1. Imports
import os, platform, traceback, torch, transformers, re, requests, time, json, hashlib, uuid, shutil, mlflow
import sys, socket, atexit, subprocess, webbrowser

from pathlib import Path
from importlib import import_module
from datetime import datetime
SCRIPT_DIR = Path(__file__).resolve().parent

from typing import Optional, Any, cast
import gradio as gr

from bs4 import BeautifulSoup
from urllib.parse import urlparse

import openvino as ov
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM

# 2. Environment configurations

#MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
#MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "google/gemma-4-E4B")

HF_TOKEN = os.getenv("HF_TOKEN") or None

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SCRIPT_DIR / "hf_cache"))
OV_MODELS_DIR = os.getenv("OV_MODELS_DIR", str(SCRIPT_DIR / "ov_models"))
OV_CACHE_DIR = os.getenv("OV_CACHE_DIR", str(SCRIPT_DIR / "ov_cache"))

OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "320"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

OFFLINE_MODE = os.getenv("LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"} # for first run set it to 0, to download seledcted model from HF
FORCE_HF_FALLBACK = os.getenv("FORCE_HF_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}

cache_path = Path(HF_CACHE_DIR).expanduser().resolve()
ov_root = Path(OV_MODELS_DIR).expanduser().resolve()
ov_cache = Path(OV_CACHE_DIR).expanduser().resolve()

cache_path.mkdir(parents=True, exist_ok=True)
ov_root.mkdir(parents=True, exist_ok=True)
ov_cache.mkdir(parents=True, exist_ok=True)

# RAG + Markdown output configuration
RAG_UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", str(SCRIPT_DIR / "rag_uploads"))).expanduser().resolve()
RAG_EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cpu")
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

MARKDOWN_OUTPUT_DIR = Path(os.getenv("MARKDOWN_OUTPUT_DIR", str(SCRIPT_DIR / "markdown_outputs"))).expanduser().resolve()

RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_CACHE"] = str(cache_path)
os.environ["TRANSFORMERS_CACHE"] = str(cache_path)

if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

def safe_model_dir_name(model_name: str) -> str:
    return (model_name.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_"))

ov_model_path = ov_root / safe_model_dir_name(MODEL_NAME)

OV_CONFIG = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": str(ov_cache),
            }
# a. MLFlow ENV setup
MLFLOW_ROOT_DIR = Path(os.getenv("MLFLOW_ROOT_DIR", str(SCRIPT_DIR / "mlflow_runs"))).expanduser().resolve()

MLFLOW_DB_PATH = MLFLOW_ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = MLFLOW_ROOT_DIR / "artifacts"

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME","local-prompt-simulator")

MLFLOW_ROOT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

START_MLFLOW_UI = os.getenv("START_MLFLOW_UI", "1").strip().lower() in {"1", "true", "yes", "y"}
OPEN_MLFLOW_UI_BROWSER = os.getenv("OPEN_MLFLOW_UI_BROWSER", "1").strip().lower() in {"1", "true", "yes", "y"}

MLFLOW_UI_HOST = os.getenv("MLFLOW_UI_HOST", "127.0.0.1")
MLFLOW_UI_PORT = int(os.getenv("MLFLOW_UI_PORT", "5000"))

MLFLOW_UI_PROCESS = None

PUBLIC_GRADIO_URL = os.getenv("PUBLIC_GRADIO_URL", "http://localhost:7860/")
PUBLIC_MLFLOW_URL = os.getenv("PUBLIC_MLFLOW_URL", "http://127.0.0.1:5002/")

def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False

def start_mlflow_ui(open_browser: bool = False):
    global MLFLOW_UI_PROCESS

    if not START_MLFLOW_UI:
        print("MLflow UI autostart is disabled.")
        return None

    if is_port_open(MLFLOW_UI_HOST, MLFLOW_UI_PORT):
        print(f"MLflow UI already running at http://{MLFLOW_UI_HOST}:{MLFLOW_UI_PORT}")
        if open_browser:
            webbrowser.open(f"http://{MLFLOW_UI_HOST}:{MLFLOW_UI_PORT}")
        return None

    backend_store_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

    cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            backend_store_uri,
            "--host",
            MLFLOW_UI_HOST,
            "--port",
            str(MLFLOW_UI_PORT),
            ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    MLFLOW_UI_PROCESS = subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,creationflags=creationflags)

    print(f"MLflow UI started at http://{MLFLOW_UI_HOST}:{MLFLOW_UI_PORT}")

    if open_browser:
        time.sleep(2)
        webbrowser.open(f"http://{MLFLOW_UI_HOST}:{MLFLOW_UI_PORT}")

    return MLFLOW_UI_PROCESS

def stop_mlflow_ui():
    global MLFLOW_UI_PROCESS

    if MLFLOW_UI_PROCESS is not None and MLFLOW_UI_PROCESS.poll() is None:
        try:
            MLFLOW_UI_PROCESS.terminate()
        except Exception:
            pass

atexit.register(stop_mlflow_ui)

def setup_mlflow() -> bool:
    try:
        MLFLOW_ROOT_DIR.mkdir(parents=True, exist_ok=True)
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)

        existing_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if existing_experiment is None:
            mlflow.create_experiment(
                name=MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri(),
            )

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        print("MLflow tracking initialized.")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow artifacts: {MLFLOW_ARTIFACTS_DIR}")

        return True

    except Exception as exc:
        print("MLflow initialization failed. Experiment tracking is disabled.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return False

MLFLOW_READY = setup_mlflow()
if MLFLOW_READY:
    start_mlflow_ui(open_browser=OPEN_MLFLOW_UI_BROWSER)

print("Runtime:", platform.platform())
print("Python backend libraries:")
print("Transformers:", transformers.__version__)
print("OpenVINO:", ov.__version__)
print("Available OpenVINO devices:", ov.Core().available_devices)
print(f"Selected model: {MODEL_NAME}")
print(f"OpenVINO device: {OPENVINO_DEVICE}")
print(f"HF cache: {cache_path}")
print(f"OpenVINO model path: {ov_model_path}")
print(f"OpenVINO runtime cache: {ov_cache}")
print(f"Offline/cache-only mode: {OFFLINE_MODE}")
print(f"Force Hugging Face fallback: {FORCE_HF_FALLBACK}")

# 3. Model loader local\offline or remote\online from HuggingFace (initial load)

def load_tokenizer():
    tokenizer_kwargs = {"trust_remote_code": True,"cache_dir": str(cache_path),"local_files_only": OFFLINE_MODE, "fix_mistral_regex": True}

    if HF_TOKEN and not OFFLINE_MODE:
        tokenizer_kwargs["token"] = HF_TOKEN

    if (ov_model_path / "tokenizer_config.json").exists():
        print("\nLoading tokenizer from exported OpenVINO model directory.")
        tok = AutoTokenizer.from_pretrained(str(ov_model_path),trust_remote_code=True,local_files_only=True, fix_mistral_regex=True)
    else:
        print("\nLoading tokenizer from Hugging Face cache / Hub.")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return tok

tokenizer = load_tokenizer()

def load_openvino_model():
    if (ov_model_path / "openvino_model.xml").exists():
        print("\nLoading already exported OpenVINO model from disk.")
        ov_model: Any = OVModelForCausalLM.from_pretrained(str(ov_model_path),device=OPENVINO_DEVICE,compile=False,ov_config=OV_CONFIG)
    else:
        print("\nExporting Hugging Face model to OpenVINO IR.")
        print("First export can take several minutes. After export, the model will be reused from ./ov_models.")

        model_kwargs = {
                        "export": True,
                        "trust_remote_code": True,
                        "cache_dir": str(cache_path),
                        "local_files_only": OFFLINE_MODE,
                        "device": OPENVINO_DEVICE,
                        "compile": False,
                        "ov_config": OV_CONFIG,
                        }

        if HF_TOKEN and not OFFLINE_MODE:
            model_kwargs["token"] = HF_TOKEN

        ov_model: Any = OVModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        ov_model_path.mkdir(parents=True, exist_ok=True)
        ov_model.save_pretrained(str(ov_model_path))
        tokenizer.save_pretrained(str(ov_model_path))

        print(f"OpenVINO model saved to: {ov_model_path}")

    print("\nCompiling OpenVINO model once...")
    ov_model.compile()
    print("OpenVINO model is loaded and ready.")
    return ov_model


def load_huggingface_fallback_model():
    print("\nLoading Hugging Face Transformers fallback model.")
    print("This keeps the notebook usable if OpenVINO export/load fails, but OpenVINO is still the preferred path.")

    hf_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(cache_path),
                "local_files_only": OFFLINE_MODE,
                "low_cpu_mem_usage": True,
                }

    if HF_TOKEN and not OFFLINE_MODE:
        hf_kwargs["token"] = HF_TOKEN

    # CPU fallback is safest in lab/no-GPU environments.
    # If CUDA exists, device_map='auto' lets Transformers place the model automatically.
    if torch.cuda.is_available():
        hf_kwargs["device_map"] = "auto"
        hf_kwargs["dtype"] = "auto"
    else:
        hf_kwargs["dtype"] = "auto"

    hf_model: Any = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **hf_kwargs)

    if not torch.cuda.is_available():
        hf_model.to("cpu")

    hf_model.eval()
    print("Hugging Face fallback model is loaded and ready.")
    return hf_model

BACKEND = None

if FORCE_HF_FALLBACK:
    model = load_huggingface_fallback_model()
    BACKEND = "huggingface"
else:
    try:
        model = load_openvino_model()
        BACKEND = "openvino"
    except Exception as exc:
        print("\nOpenVINO load/export failed. Falling back to Hugging Face Transformers.")
        print("Error:", repr(exc))
        traceback.print_exc(limit=2)
        model = load_huggingface_fallback_model()
        BACKEND = "huggingface"

print(f"\nActive backend: {BACKEND}")


def build_chat_input(prompt: str, system_prompt: Optional[str] = None) -> str:
    prompt = str(prompt).strip()
    messages = []

    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})

    messages.append({"role": "user", "content": prompt})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    if system_prompt and str(system_prompt).strip():
        return f"System:\n{system_prompt.strip()}\n\nUser:\n{prompt}\n\nAssistant:\n"

    return prompt

# 4. Model run function + manual conmfiguration

def run_local_llm(prompt: str,system_prompt: Optional[str] = None,max_new_tokens: Optional[int] = None,temperature: Optional[float] = None,top_p: Optional[float] = None) -> str:
    prompt = str(prompt).strip()

    if not prompt:
        return ""

    input_text = build_chat_input(prompt, system_prompt=system_prompt)

    inputs = tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_INPUT_TOKENS,
                        )

    gen_temperature = TEMPERATURE if temperature is None else float(temperature)
    gen_top_p = TOP_P if top_p is None else float(top_p)
    gen_max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)

    generation_kwargs = {
                        "max_new_tokens": gen_max_new_tokens,
                        "do_sample": gen_temperature > 0,
                        "pad_token_id": tokenizer.pad_token_id,
                        "use_cache": True,
                        }

    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if gen_temperature > 0:
        generation_kwargs["temperature"] = gen_temperature
        generation_kwargs["top_p"] = gen_top_p

    active_model: Any = model

    if BACKEND == "huggingface":
        model_device = next(active_model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = active_model.generate(**inputs, **generation_kwargs)
    else:
        output_ids = active_model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print("\nHelper function ready: run_local_llm(prompt, system_prompt=None)")

# a. Token counter function
def count_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0

    try:
        return len(tokenizer(text, return_tensors=None)["input_ids"])
    except Exception:
        return 0

def short_text(value: Any, max_len: int = 500) -> str:
    """Shorten long values before saving them as MLflow params/tags."""
    value = "" if value is None else str(value)
    value = value.replace("\x00", "")
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"... [truncated, original_length={len(value)}]"


def sha256_text(value: str, length: int = 16) -> str:
    value = str(value or "")
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def safe_log_params(params: dict, max_len: int = 500) -> None:
#Log MLflow params defensively. Large values are shortened.
    safe_params = {}
    for key, value in params.items():
        if value is None:
            safe_params[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe_params[key] = short_text(value, max_len=max_len)
        else:
            safe_params[key] = short_text(json.dumps(value, ensure_ascii=False, default=str), max_len=max_len)

    if safe_params:
        mlflow.log_params(safe_params)


def safe_log_metrics(metrics: dict) -> None:
#Log only numeric, finite MLflow metrics."""
    safe_metrics = {}
    for key, value in metrics.items():
        try:
            if value is None:
                continue
            value = float(value)
            if value != value or value in {float("inf"), float("-inf")}:
                continue
            safe_metrics[key] = value
        except Exception:
            continue

    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def safe_log_text(text_value: str, artifact_file: str) -> None:
#Log text artifact without breaking the user flow if MLflow fails."""
    try:
        mlflow.log_text(str(text_value or ""), artifact_file)
    except Exception as exc:
        print(f"MLflow text artifact failed for {artifact_file}: {type(exc).__name__}: {exc}")


def text_profile(text_value: str, prefix: str) -> dict:
#Return numeric profile for prompt/output text."""
    text_value = str(text_value or "")
    words = re.findall(r"\S+", text_value)
    lines = text_value.splitlines()

    return {
            f"{prefix}_chars": len(text_value),
            f"{prefix}_words": len(words),
            f"{prefix}_lines": len(lines),
            f"{prefix}_tokens": count_tokens(text_value),
            }


def collect_runtime_metadata() -> dict:
#Collect runtime/library/hardware metadata for MLflow artifacts."""
    cuda_devices = []

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0

        for index in range(cuda_device_count):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                                {
                                "index": index,
                                "name": props.name,
                                "total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
                                "major": props.major,
                                "minor": props.minor,
                                "multi_processor_count": props.multi_processor_count,
                                }
                                )
    except Exception as exc:
        cuda_available = False
        cuda_device_count = 0
        cuda_devices.append({"error": f"{type(exc).__name__}: {exc}"})

    try:
        openvino_devices = ov.Core().available_devices
    except Exception as exc:
        openvino_devices = [f"OpenVINO device check failed: {type(exc).__name__}: {exc}"]

    selected_env = {
        key: os.getenv(key)
        for key in [
                    "LOCAL_MODEL_NAME",
                    "OPENVINO_DEVICE",
                    "LOCAL_FILES_ONLY",
                    "FORCE_HF_FALLBACK",
                    "MAX_INPUT_TOKENS",
                    "MAX_NEW_TOKENS",
                    "TEMPERATURE",
                    "TOP_P",
                    "HF_CACHE_DIR",
                    "OV_MODELS_DIR",
                    "OV_CACHE_DIR",
                    "RAG_UPLOAD_DIR",
                    "RAG_EMBED_MODEL_NAME",
                    "RAG_EMBED_DEVICE",
                    "RAG_MAX_CONTEXT_CHARS",
                    "MARKDOWN_OUTPUT_DIR",
                    "MLFLOW_EXPERIMENT_NAME",
                    "MLFLOW_TRACKING_URI",
                    "START_MLFLOW_UI",
                    "GRADIO_SERVER_NAME",
                    "GRADIO_SERVER_PORT",
                    ]
                    if os.getenv(key) is not None
                    }

    return {
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "python_version": sys.version,
            "library_versions": {
                                "torch": getattr(torch, "__version__", "unknown"),
                                "transformers": getattr(transformers, "__version__", "unknown"),
                                "openvino": getattr(ov, "__version__", "unknown"),
                                "gradio": getattr(gr, "__version__", "unknown"),
                                "mlflow": getattr(mlflow, "__version__", "unknown"),
                                },
                    "cuda": {
                            "available": cuda_available,
                            "torch_cuda_version": getattr(torch.version, "cuda", None),
                            "device_count": cuda_device_count,
                            "devices": cuda_devices,
                            },
                    "openvino": {
                                "selected_device": OPENVINO_DEVICE,
                                "available_devices": openvino_devices,
                                "ov_config": OV_CONFIG,
                                },
                        "rag": {
                                "embed_model": RAG_EMBED_MODEL_NAME,
                                "embed_device": RAG_EMBED_DEVICE,
                                "upload_dir": str(RAG_UPLOAD_DIR),
                                "max_context_chars": RAG_MAX_CONTEXT_CHARS,
                                },
                    "paths": {
                                "script_dir": str(SCRIPT_DIR),
                                "hf_cache": str(cache_path),
                                "ov_model_path": str(ov_model_path),
                                "ov_cache": str(ov_cache),
                                "rag_upload_dir": str(RAG_UPLOAD_DIR),
                                "markdown_output_dir": str(MARKDOWN_OUTPUT_DIR),
                                "mlflow_root_dir": str(MLFLOW_ROOT_DIR),
                                "mlflow_db_path": str(MLFLOW_DB_PATH),
                                "mlflow_artifacts_dir": str(MLFLOW_ARTIFACTS_DIR),
                            },
        "selected_environment": selected_env,
                            }


def log_prompt_error_to_mlflow(
                                system_prompt: str,
                                user_prompt: str,
                                topic: str,
                                final_prompt: str,
                                few_shot_enabled: bool,
                                few_shot_examples: str,
                                web_scraping_enabled: bool,
                                web_urls: str,
                                rag_enabled: bool,
                                rag_metadata: dict,
                                markdown_force_enabled: bool,
                                markdown_save_enabled: bool,
                                max_new_tokens: int,
                                temperature: float,
                                top_p: float,
                                error: Exception,
                                traceback_text: str,
                                ) -> str:
    #Log failed generation attempts to MLflow as failed runs."""
    request_id = str(uuid.uuid4())
    prompt_hash = sha256_text(final_prompt, length=16)

    with mlflow.start_run(run_name=f"failed_prompt_run_{prompt_hash}") as run:
        mlflow.set_tags(
                        {
                            "app": "local-openvino-prompt-simulator",
                            "task_type": "prompt_generation",
                            "run_status": "failed",
                            "backend": str(BACKEND),
                            "model_name": short_text(MODEL_NAME, 200),
                            "request_id": request_id,
                            "error_type": type(error).__name__,
                            }
                            )

        safe_log_params(
                        {
                        "model_name": MODEL_NAME,
                        "backend": BACKEND,
                        "openvino_device": OPENVINO_DEVICE if BACKEND == "openvino" else "hf_fallback",
                        "max_input_tokens": MAX_INPUT_TOKENS,
                        "max_new_tokens": int(max_new_tokens),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "few_shot_enabled": bool(few_shot_enabled),
                        "web_scraping_enabled": bool(web_scraping_enabled),
                        "rag_enabled": bool(rag_enabled),
                        "markdown_force_enabled": bool(markdown_force_enabled),
                        "markdown_save_enabled": bool(markdown_save_enabled),
                        "prompt_hash": prompt_hash,
                        "request_id": request_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        }
                        )

        metrics = {}
        metrics.update(text_profile(system_prompt, "system_prompt"))
        metrics.update(text_profile(user_prompt, "user_prompt_template"))
        metrics.update(text_profile(topic, "topic"))
        metrics.update(text_profile(final_prompt, "final_prompt"))

        for key, value in (rag_metadata or {}).items():
            if isinstance(value, (int, float, bool)):
                metrics[f"rag_{key}"] = value

        safe_log_metrics(metrics)

        safe_log_text(system_prompt, "prompts/system_prompt.txt")
        safe_log_text(user_prompt, "prompts/user_prompt_template.txt")
        safe_log_text(topic, "prompts/topic.txt")
        safe_log_text(final_prompt, "prompts/final_prompt.txt")
        safe_log_text(few_shot_examples, "prompts/few_shot_examples.txt")
        safe_log_text(web_urls, "sources/web_urls.txt")
        safe_log_text(json.dumps(rag_metadata or {}, indent=2, ensure_ascii=False, default=str), "rag/rag_metadata.json")
        safe_log_text(str(error), "errors/error_message.txt")
        safe_log_text(traceback_text, "errors/traceback.txt")
        safe_log_text(json.dumps(collect_runtime_metadata(), indent=2, ensure_ascii=False, default=str),"metadata/runtime_metadata.json")

        return run.info.run_id


def log_prompt_run_to_mlflow(
                            system_prompt: str,
                            user_prompt: str,
                            topic: str,
                            base_prompt: str,
                            web_context: str,
                            prompt_after_web: str,
                            rag_enabled: bool,
                            rag_context: str,
                            rag_metadata: dict,
                            prompt_after_rag: str,
                            markdown_force_enabled: bool,
                            markdown_save_enabled: bool,
                            markdown_file_path: str,
                            prompt_after_markdown_instruction: str,
                            final_prompt: str,
                            chat_input: str,
                            output: str,
                            few_shot_enabled: bool,
                            few_shot_examples: str,
                            web_scraping_enabled: bool,
                            web_urls: str,
                            max_new_tokens: int,
                            temperature: float,
                            top_p: float,
                            generation_time_sec: float,
                            ) -> str:
    request_id = str(uuid.uuid4())

    hashes = {
                "system_prompt": sha256_text(system_prompt),
                "user_prompt_template": sha256_text(user_prompt),
                "topic": sha256_text(topic),
                "base_prompt": sha256_text(base_prompt),
                "web_context": sha256_text(web_context),
                "rag_context": sha256_text(rag_context),
                "prompt_after_rag": sha256_text(prompt_after_rag),
                "prompt_after_markdown_instruction": sha256_text(prompt_after_markdown_instruction),
                "final_prompt": sha256_text(final_prompt),
                "chat_input": sha256_text(chat_input),
                "output": sha256_text(output),
                }

    input_tokens = count_tokens(final_prompt)
    chat_input_tokens = count_tokens(chat_input)
    output_tokens = count_tokens(output)
    total_tokens = chat_input_tokens + output_tokens

    output_tokens_per_sec = output_tokens / generation_time_sec if generation_time_sec > 0 else 0
    input_context_utilization_pct = (chat_input_tokens / MAX_INPUT_TOKENS) * 100 if MAX_INPUT_TOKENS else 0

    urls = extract_urls_from_text(web_urls)
    url_count = len(urls)
    failed_scrape_count = str(web_context or "").count("Status: Failed to scrape")

    run_name = f"prompt_run_{hashes['final_prompt']}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
                        {
                        "app": "local-openvino-prompt-simulator",
                        "task_type": "prompt_generation",
                        "run_status": "success",
                        "backend": str(BACKEND),
                        "model_name": short_text(MODEL_NAME, 200),
                        "request_id": request_id,
                        "prompt_hash": hashes["final_prompt"],
                        "output_hash": hashes["output"],
                        "few_shot_enabled": str(bool(few_shot_enabled)),
                        "web_scraping_enabled": str(bool(web_scraping_enabled)),
                        "rag_enabled": str(bool(rag_enabled)),
                        "markdown_force_enabled": str(bool(markdown_force_enabled)),
                        "markdown_save_enabled": str(bool(markdown_save_enabled)),
                        }
                        )

        runtime_metadata = collect_runtime_metadata()

        safe_log_params(
                        {
                            "request_id": request_id,
                            "model_name": MODEL_NAME,
                            "backend": BACKEND,
                            "openvino_device": OPENVINO_DEVICE if BACKEND == "openvino" else "hf_fallback",
                            "openvino_available_devices": ",".join(map(str, runtime_metadata["openvino"]["available_devices"])),
                            "force_hf_fallback": FORCE_HF_FALLBACK,
                            "offline_mode": OFFLINE_MODE,
                            "max_input_tokens": MAX_INPUT_TOKENS,
                            "max_new_tokens": int(max_new_tokens),
                            "temperature": float(temperature),
                            "top_p": float(top_p),
                            "do_sample": float(temperature) > 0,
                            "few_shot_enabled": bool(few_shot_enabled),
                            "web_scraping_enabled": bool(web_scraping_enabled),
                            "rag_enabled": bool(rag_enabled),
                            "rag_embed_model": RAG_EMBED_MODEL_NAME,
                            "rag_embed_device": RAG_EMBED_DEVICE,
                            "rag_top_k": (rag_metadata or {}).get("top_k", ""),
                            "rag_chunk_size": (rag_metadata or {}).get("chunk_size", ""),
                            "rag_chunk_overlap": (rag_metadata or {}).get("chunk_overlap", ""),
                            "markdown_force_enabled": bool(markdown_force_enabled),
                            "markdown_save_enabled": bool(markdown_save_enabled),
                            "markdown_file_path": markdown_file_path,
                            "url_count": url_count,
                            "failed_scrape_count": failed_scrape_count,
                            "topic_preview": short_text(topic, 250),
                            "system_prompt_preview": short_text(system_prompt, 250),
                            "user_prompt_preview": short_text(user_prompt, 250),
                            "output_preview": short_text(output, 300),
                            "hf_cache": str(cache_path),
                            "ov_model_path": str(ov_model_path),
                            "ov_cache": str(ov_cache),
                            "rag_upload_dir": str(RAG_UPLOAD_DIR),
                            "markdown_output_dir": str(MARKDOWN_OUTPUT_DIR),
                            "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
                            **{f"{key}_hash": value for key, value in hashes.items()},
                            },
                            max_len=500,
                            )

        metrics = {
                    "generation_time_sec": float(generation_time_sec),
                    "input_tokens": input_tokens,
                    "chat_input_tokens": chat_input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "output_tokens_per_sec": output_tokens_per_sec,
                    "input_context_utilization_pct": input_context_utilization_pct,
                    "url_count": url_count,
                    "failed_scrape_count": failed_scrape_count,
                    "few_shot_example_chars": len(str(few_shot_examples or "")),
                    "rag_context_tokens": count_tokens(rag_context),
                    "rag_context_chars": len(str(rag_context or "")),
                    "markdown_saved": 1 if markdown_file_path else 0,
                    }

        for key, value in (rag_metadata or {}).items():
            if isinstance(value, (int, float, bool)):
                metrics[f"rag_{key}"] = value

        metrics.update(text_profile(system_prompt, "system_prompt"))
        metrics.update(text_profile(user_prompt, "user_prompt_template"))
        metrics.update(text_profile(topic, "topic"))
        metrics.update(text_profile(base_prompt, "base_prompt"))
        metrics.update(text_profile(web_context, "web_context"))
        metrics.update(text_profile(prompt_after_web, "prompt_after_web"))
        metrics.update(text_profile(rag_context, "rag_context"))
        metrics.update(text_profile(prompt_after_rag, "prompt_after_rag"))
        metrics.update(text_profile(prompt_after_markdown_instruction, "prompt_after_markdown_instruction"))
        metrics.update(text_profile(final_prompt, "final_prompt"))
        metrics.update(text_profile(chat_input, "chat_input"))
        metrics.update(text_profile(output, "output"))

        safe_log_metrics(metrics)

        # Full prompt/output artifacts.
        safe_log_text(system_prompt, "prompts/system_prompt.txt")
        safe_log_text(user_prompt, "prompts/user_prompt_template.txt")
        safe_log_text(topic, "prompts/topic.txt")
        safe_log_text(base_prompt, "prompts/base_prompt.txt")
        safe_log_text(prompt_after_web, "prompts/prompt_after_web.txt")
        safe_log_text(prompt_after_rag, "prompts/prompt_after_rag.txt")
        safe_log_text(prompt_after_markdown_instruction, "prompts/prompt_after_markdown_instruction.txt")
        safe_log_text(final_prompt, "prompts/final_prompt.txt")
        safe_log_text(chat_input, "prompts/chat_input_actual_model_input.txt")
        safe_log_text(output, "outputs/model_output.txt")

        if few_shot_examples: safe_log_text(few_shot_examples, "prompts/few_shot_examples.txt")

        if web_urls: safe_log_text(web_urls, "sources/web_urls.txt")

        if web_context: safe_log_text(web_context, "sources/web_context.txt")

        if rag_context: safe_log_text(rag_context, "rag/rag_context.txt")

        safe_log_text(json.dumps(rag_metadata or {}, indent=2, ensure_ascii=False, default=str),"rag/rag_metadata.json")

        if markdown_file_path:
            safe_log_text(markdown_file_path, "markdown/markdown_file_path.txt")
            try:
                md_path = Path(markdown_file_path)
                if md_path.exists():
                    mlflow.log_artifact(str(md_path), artifact_path="markdown")
            except Exception as exc:
                print(f"MLflow markdown artifact failed: {type(exc).__name__}: {exc}")

        generation_config = {
                                "max_input_tokens": MAX_INPUT_TOKENS,
                                "max_new_tokens": int(max_new_tokens),
                                "temperature": float(temperature),
                                "top_p": float(top_p),
                                "do_sample": float(temperature) > 0,
                                "pad_token_id": tokenizer.pad_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                                }

        run_metadata = {
                        "request_id": request_id,
                        "mlflow_run_id": run.info.run_id,
                        "model_name": MODEL_NAME,
                        "backend": BACKEND,
                        "openvino_device": OPENVINO_DEVICE,
                        "generation_config": generation_config,
                        "flags": {
                                    "few_shot_enabled": bool(few_shot_enabled),
                                    "web_scraping_enabled": bool(web_scraping_enabled),
                                    "rag_enabled": bool(rag_enabled),
                                    "markdown_force_enabled": bool(markdown_force_enabled),
                                    "markdown_save_enabled": bool(markdown_save_enabled),
                                    "offline_mode": bool(OFFLINE_MODE),
                                    "force_hf_fallback": bool(FORCE_HF_FALLBACK),
                                    },
                        "rag_metadata": rag_metadata or {},
                        "markdown_file_path": markdown_file_path,
                        "hashes": hashes,
                        "metrics": metrics,
                        }

        safe_log_text(json.dumps(run_metadata, indent=2, ensure_ascii=False, default=str),"metadata/run_metadata.json")
        safe_log_text(json.dumps(generation_config, indent=2, ensure_ascii=False, default=str),"metadata/generation_config.json")
        safe_log_text(json.dumps(runtime_metadata, indent=2, ensure_ascii=False, default=str),"metadata/runtime_metadata.json")

        return run.info.run_id

# 5. Web Scrapper BeatifulSoup 4 loader
def extract_urls_from_text(urls_text: str) -> list[str]:
    urls_text = str(urls_text or "").strip()

    if not urls_text:
        return []

    urls = []

    for line in urls_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if not line.startswith(("http://", "https://")):
            line = "https://" + line

        parsed = urlparse(line)
        if parsed.netloc:
            urls.append(line)

    return urls

def clean_scraped_text(text: str, max_chars: int = 6000) -> str:
    text = str(text or "")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Text was truncated due to character limit.]"

    return text

def scrape_single_url(url: str, timeout: int = 15, max_chars: int = 6000) -> dict:
    headers = {
                "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0 Safari/537.36"
                                )
                }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url

        main = soup.find("main")
        if main:
            raw_text = main.get_text(" ", strip=True)
        else:
            raw_text = soup.get_text(" ", strip=True)

        cleaned_text = clean_scraped_text(raw_text, max_chars=max_chars)

        return {
                    "url": url,
                    "title": title,
                    "text": cleaned_text,
                    "error": None,
                }

    except Exception as e:
        return {
                "url": url,
                "title": url,
                "text": "",
                "error": str(e),
                }

def scrape_urls_to_context(urls_text: str,max_urls: int = 3,max_chars_per_url: int = 12000) -> str:
    urls = extract_urls_from_text(urls_text)

    if not urls:
        return ""

    urls = urls[:max_urls]
    scraped_blocks = []

    for index, url in enumerate(urls, start=1):
        result = scrape_single_url(url, max_chars=max_chars_per_url)

        if result["error"]:
            scraped_blocks.append(
                f"""[Source {index}]
URL: {result["url"]}
Status: Failed to scrape
Error: {result["error"]}"""
            )
            continue

        scraped_blocks.append(
            f"""[Source {index}]
Title: {result["title"]}
URL: {result["url"]}

Content:
{result["text"]}"""
        )

    return "\n\n---\n\n".join(scraped_blocks)

def apply_web_context_to_prompt(final_prompt: str,web_scraping_enabled: bool,urls_text: str) -> str:

    final_prompt = str(final_prompt or "").strip()

    if not web_scraping_enabled:
        return final_prompt

    web_context = scrape_urls_to_context(urls_text)

    if not web_context.strip():
        return final_prompt

    return f"""Use the following scraped web context as supporting source material.
Answer based on the web context when it is relevant. If the context is incomplete, say so clearly.

<web_context>
{web_context}
</web_context>

User task:
{final_prompt}"""


# 5b. RAG document context loader and retriever

_RAG_EMBEDDER = None


def get_file_path_from_gradio(file_item: Any) -> Optional[Path]:
#Normalize Gradio file object/string into Path."""
    if file_item is None:
        return None

    if isinstance(file_item, (str, Path)):
        return Path(file_item)

    # Gradio may return tempfile-like objects with .name
    name = getattr(file_item, "name", None)
    if name:
        return Path(name)

    return None


def safe_uploaded_filename(filename: str) -> str:
#Create a safe file name while preserving extension."""
    filename = str(filename or "uploaded_file").strip()
    stem = Path(filename).stem or "uploaded_file"
    suffix = Path(filename).suffix

    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
    if not stem:
        stem = "uploaded_file"

    suffix = re.sub(r"[^a-zA-Z0-9.]+", "", suffix)

    return f"{stem}{suffix}"


def persist_uploaded_rag_file(source_path: Path) -> Path:
#Copy Gradio temp upload into RAG_UPLOAD_DIR so the file is visibly persisted."""
    source_path = Path(source_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded RAG file does not exist: {source_path}")

    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # If the file is already in the target folder, do not copy it again.
    try:
        if source_path.parent.resolve() == RAG_UPLOAD_DIR.resolve():
            return source_path
    except Exception:
        pass

    safe_name = safe_uploaded_filename(source_path.name)
    target_path = RAG_UPLOAD_DIR / safe_name

    if target_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = sha256_text(str(source_path), length=8)
        target_path = RAG_UPLOAD_DIR / f"{target_path.stem}_{stamp}_{short_hash}{target_path.suffix}"

    shutil.copy2(str(source_path), str(target_path))
    return target_path


def format_rag_status(rag_metadata: dict) -> str:
#Create a short human-readable RAG status block for Gradio output."""
    rag_metadata = rag_metadata or {}

    if not rag_metadata.get("enabled"):
        return ""

    lines = [
                "RAG status:",
                f"- source files detected: {rag_metadata.get('source_file_count', 0)}",
                f"- files loaded: {rag_metadata.get('loaded_file_count', 0)}",
                f"- files failed: {rag_metadata.get('failed_file_count', 0)}",
                f"- chunks created: {rag_metadata.get('chunk_count', 0)}",
                f"- chunks retrieved: {rag_metadata.get('retrieved_chunk_count', 0)}",
                f"- persisted upload folder: {RAG_UPLOAD_DIR}",
                ]

    if rag_metadata.get("errors"):
        lines.append("- errors:")
        for err in rag_metadata.get("errors", [])[:5]:
            lines.append(f"  - {err.get('file_name', 'unknown')}: {err.get('error_type', 'Error')}: {err.get('error', '')}")

    if rag_metadata.get("chunk_count", 0) == 0:
        lines.append(
                    "- warning: no chunks were created. SentenceTransformer will not load until at least one readable text chunk exists."
                    )

    return "\n".join(lines)


def load_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_for_rag(path: Path) -> str:
    try:
        fitz = import_module("pymupdf")
    except ModuleNotFoundError:
        fitz = import_module("fitz")

    pages = []
    with fitz.open(str(path)) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(f"[Page {index}]\n{text}")
    return "\n\n".join(pages)


def load_docx_for_rag(path: Path) -> str:
    try:
        docx_module = import_module("docx")
    except ModuleNotFoundError as exc:
        raise RuntimeError("DOCX support is missing. Install it with: pip install python-docx") from exc

    document = docx_module.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def load_rtf_for_rag(path: Path) -> str:
    try:
        striprtf_module = import_module("striprtf.striprtf")
    except ModuleNotFoundError as exc:
        raise RuntimeError("RTF support is missing. Install it with: pip install striprtf") from exc

    raw = load_text_file(path)
    return striprtf_module.rtf_to_text(raw)


def load_rag_document(path: Path) -> str:
#Load supported RAG documents into text."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}:
        return load_text_file(path)

    if suffix == ".pdf":
        return load_pdf_for_rag(path)

    if suffix == ".docx":
        return load_docx_for_rag(path)

    if suffix == ".rtf":
        return load_rtf_for_rag(path)

    # Last-resort text loader
    return load_text_file(path)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
#Simple character-based chunking with overlap."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    if not text:
        return []

    chunk_size = max(int(chunk_size), 200)
    overlap = max(min(int(overlap), chunk_size - 1), 0)

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def get_rag_embedder():
#Lazy-load CPU sentence-transformers embedder."""
    global _RAG_EMBEDDER

    if _RAG_EMBEDDER is not None:
        return _RAG_EMBEDDER

    try:
        sentence_transformers_module = import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
                            "RAG dependencies are missing. Install them with: "
                            "pip install sentence-transformers faiss-cpu numpy"
                            ) from exc

    print(f"Loading RAG embedding model: {RAG_EMBED_MODEL_NAME} on {RAG_EMBED_DEVICE}")
    _RAG_EMBEDDER = sentence_transformers_module.SentenceTransformer(RAG_EMBED_MODEL_NAME,device=RAG_EMBED_DEVICE)
    return _RAG_EMBEDDER

def build_rag_context(
                        query: str,
                        rag_files: Any,
                        top_k: int = 4,
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200,
                        max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
                    ) -> tuple[str, dict]:
#Build a temporary in-memory FAISS index from uploaded files and retrieve relevant chunks."""
    metadata = {
                "enabled": True,
                "embed_model": RAG_EMBED_MODEL_NAME,
                "embed_device": RAG_EMBED_DEVICE,
                "top_k": int(top_k),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "source_file_count": 0,
                "loaded_file_count": 0,
                "failed_file_count": 0,
                "chunk_count": 0,
                "retrieved_chunk_count": 0,
                "files": [],
                "errors": [],
                }

    if rag_files is None:
        return "", metadata

    if not isinstance(rag_files, list):
        rag_files = [rag_files]

    docs = []

    for file_item in rag_files:
        original_path = get_file_path_from_gradio(file_item)
        if not original_path:
            metadata["errors"].append(
                {"file_name": "unknown","error_type": "MissingPath","error": "Gradio did not provide a usable file path."})
            continue

        metadata["source_file_count"] += 1

        try:
            path = persist_uploaded_rag_file(original_path)

            text = load_rag_document(path)
            text = str(text or "").strip()

            file_record = {
                            "file_name": path.name,
                            "original_path": str(original_path),
                            "persisted_path": str(path),
                            "suffix": path.suffix.lower(),
                            "chars": len(text),
                            "tokens_estimate": count_tokens(text[:MAX_INPUT_TOKENS * 4]),
                            }

            if not text:
                file_record["status"] = "empty"
                metadata["files"].append(file_record)
                continue

            docs.append((path.name, text))
            file_record["status"] = "loaded"
            metadata["loaded_file_count"] += 1
            metadata["files"].append(file_record)

        except Exception as exc:
            metadata["failed_file_count"] += 1
            metadata["errors"].append({"file_name": path.name,"error_type": type(exc).__name__,"error": str(exc)})

    chunks = []
    chunk_sources = []

    for file_name, doc_text in docs:
        for chunk in chunk_text(doc_text, chunk_size=chunk_size, overlap=chunk_overlap):
            chunks.append(chunk)
            chunk_sources.append(file_name)

    metadata["chunk_count"] = len(chunks)

    print(
            f"RAG: source_files={metadata['source_file_count']}, "
            f"loaded_files={metadata['loaded_file_count']}, "
            f"failed_files={metadata['failed_file_count']}, "
            f"chunks={metadata['chunk_count']}, "
            f"upload_dir={RAG_UPLOAD_DIR}"
            )

    if not chunks:
        return "", metadata

    print(f"RAG: loading embedding model if needed: {RAG_EMBED_MODEL_NAME} on {RAG_EMBED_DEVICE}")

    try:
        faiss = import_module("faiss")
        np = import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("RAG dependency is missing. Install it with: pip install faiss-cpu numpy") from exc

    embedder = get_rag_embedder()

    embeddings = embedder.encode(chunks,normalize_embeddings=True,show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = embedder.encode([str(query or "")],normalize_embeddings=True,show_progress_bar=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    top_k = max(1, min(int(top_k), len(chunks)))
    scores, ids = index.search(query_embedding, top_k)

    selected_blocks = []

    for rank, chunk_id in enumerate(ids[0], start=1):
        if chunk_id < 0:
            continue

        score = float(scores[0][rank - 1])
        source_name = chunk_sources[int(chunk_id)]
        chunk_text_value = chunks[int(chunk_id)]

        selected_blocks.append(
            f"""[RAG chunk {rank}]
Source: {source_name}
Similarity score: {score:.4f}

{chunk_text_value}"""
        )

    rag_context = "\n\n---\n\n".join(selected_blocks)

    if len(rag_context) > int(max_context_chars):
        rag_context = rag_context[: int(max_context_chars)] + "\n\n[RAG context truncated due to character limit.]"

    metadata["retrieved_chunk_count"] = len(selected_blocks)
    metadata["rag_context_chars"] = len(rag_context)
    metadata["rag_context_tokens"] = count_tokens(rag_context)

    return rag_context, metadata


def apply_rag_context_to_prompt(final_prompt: str, rag_context: str) -> str:
    final_prompt = str(final_prompt or "").strip()
    rag_context = str(rag_context or "").strip()

    if not rag_context:
        return final_prompt

    return f"""Use the following retrieved document context as source material.
If the answer is not supported by the retrieved context, say that the information is missing.

<rag_context>
{rag_context}
</rag_context>

User task:
{final_prompt}"""


# 5c. Markdown output / memory writer

def safe_markdown_filename(filename: str) -> str:
    filename = str(filename or "").strip()

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_output_{timestamp}.md"

    filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)

    if not filename.lower().endswith(".md"):
        filename += ".md"

    return filename


def apply_markdown_output_instruction(final_prompt: str, markdown_force_enabled: bool) -> str:
    final_prompt = str(final_prompt or "").strip()

    if not markdown_force_enabled:
        return final_prompt

    return f"""Return the answer as valid Markdown.

Rules:
- Use clear headings and short sections.
- Use bullet points when useful.
- Do not wrap the entire answer in a code block.
- Keep the content directly reusable in a .md file.

Task:
{final_prompt}"""


def save_output_to_markdown(
                            output: str,
                            filename: str,
                            system_prompt: str = "",
                            final_prompt: str = "",
                            include_prompt: bool = True,
                            metadata: Optional[dict] = None,
                            ) -> str:
    safe_name = safe_markdown_filename(filename)
    path = MARKDOWN_OUTPUT_DIR / safe_name

    metadata = metadata or {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if include_prompt:
        content = f"""# LLM Markdown Output

Generated at: {timestamp}

## Metadata

```json
{json.dumps(metadata, indent=2, ensure_ascii=False, default=str)}
```

## System Prompt

```text
{system_prompt or ""}
```

## Final Prompt

```text
{final_prompt or ""}
```

## Model Output

{output or ""}
"""
    else:
        content = str(output or "")

    path.write_text(content, encoding="utf-8")
    return str(path)


if "run_local_llm" not in globals():
    raise RuntimeError("Run model initialization cell first.")

DEFAULT_TOPIC = "New business laptop model"
DEFAULT_WEB_URLS = """https://www.dell.com/en-us/shop/dell-laptops/dell-16-plus-laptop/spd/dell-db16250-laptop"""

DEFAULT_SYSTEM_PROMPT = """You are an expert social media manager."""

DEFAULT_USER_PROMPT = """You write concise, readable and engaging posts for X social network. Create a post based on the topic below.

                        <topic>
                        {topic}
                        </topic>

                        Rules:
                        - concise and focused
                        - no hashtags
                        - maximum one emoji
                        - clean formatting with line breaks
                        - return only the final post"""

DEFAULT_FEW_SHOT_EXAMPLES = """Example 1
Input topic: compact electric city car
Output:
Small footprint. Big city energy.
A compact electric car built for tight streets, quick charging and everyday comfort.

Example 2
Input topic: premium family SUV
Output:
Room for the family. Presence for the road.
A premium SUV designed around calm driving, smart space and confident long-distance travel."""

# 6. CSS styling of the Geradio interface

APP_CSS = r"""
                /* Local Assistant UI*/
                :root {
                    --bg-0: #070b14;
                    --bg-1: #0b1220;
                    --panel: rgba(15, 23, 42, 0.88);
                    --panel-soft: rgba(30, 41, 59, 0.72);
                    --border: rgba(148, 163, 184, 0.20);
                    --border-strong: rgba(96, 165, 250, 0.45);
                    --text: #e5e7eb;
                    --muted: #94a3b8;
                    --accent: #60a5fa;
                    --accent-2: #a78bfa;
                    --accent-3: #22d3ee;
                    --success: #34d399;
                    --warning: #fbbf24;
                    }

                body,
                .gradio-container {
                    background:
                        radial-gradient(circle at 18% 8%, rgba(96, 165, 250, 0.22), transparent 34%),
                        radial-gradient(circle at 88% 0%, rgba(167, 139, 250, 0.22), transparent 32%),
                        linear-gradient(135deg, var(--bg-0), var(--bg-1) 48%, #08111f) !important;
                    color: var(--text) !important;
                    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
                }

                #app-shell {
                    max-width: 1280px;
                    margin: 0 auto;
                    padding: 22px 18px 28px 18px;
                }

                #hero {
                    border: 1px solid var(--border);
                    border-radius: 28px;
                    padding: 22px 24px;
                    margin-bottom: 18px;
                    background:
                        linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.72)),
                        linear-gradient(135deg, rgba(96, 165, 250, 0.24), rgba(167, 139, 250, 0.18));
                    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
                }

                #hero h1 {
                    margin: 0 0 8px 0;
                    font-size: 2.15rem;
                    line-height: 1.1;
                    letter-spacing: -0.04em;
                }

                #hero p {
                    margin: 0;
                    color: var(--muted);
                    font-size: 0.98rem;
                }

                .status-row {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                    margin-top: 16px;
                }

                .status-pill {
                    display: inline-flex;
                    align-items: center;
                    gap: 7px;
                    border: 1px solid var(--border);
                    border-radius: 999px;
                    padding: 7px 11px;
                    background: rgba(15, 23, 42, 0.72);
                        color: #cbd5e1;
                    font-size: 0.86rem;
                }

                .status-dot {
                    width: 8px; 
                    height: 8px;
                    border-radius: 50%;
                    background: var(--success);
                    box-shadow: 0 0 18px rgba(52, 211, 153, 0.85);
                }

                #main-grid {
                    gap: 18px !important;
                }

                #left-panel,
                #right-panel {
                    border: 1px solid var(--border);
                    border-radius: 28px;
                    padding: 18px;
                    background: var(--panel);
                    box-shadow: 0 22px 55px rgba(0, 0, 0, 0.28);
                    backdrop-filter: blur(18px);
                }

                .section-title {
                    margin: 0 0 12px 0;
                    color: #f8fafc;
                    font-weight: 720;
                    letter-spacing: -0.02em;
                }

                .section-subtitle {
                    margin: -6px 0 14px 0;
                    color: var(--muted);
                    font-size: 0.90rem;
                }

                /* Text inputs */
                #system-prompt textarea,
                #topic-box textarea,
                #user-prompt textarea,
                #few-shot-examples textarea,
                #output-box textarea {
                    background: rgba(2, 6, 23, 0.76) !important;
                    color: #f8fafc !important;
                    border: 1px solid rgba(148, 163, 184, 0.22) !important;
                    border-radius: 18px !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03) !important;
                    line-height: 1.55 !important;
                }

                #system-prompt textarea:focus,
                #topic-box textarea:focus,
                #user-prompt textarea:focus,
                #few-shot-examples textarea:focus,
                #output-box textarea:focus {
                    border-color: var(--border-strong) !important;
                    box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.16) !important;
                }

                #few-shot-examples textarea {
                    min-height: 260px !important;
                }

                #output-box textarea {
                    min-height: 430px !important;
                    font-size: 1.02rem !important;
                }

                label,
                .block label,
                span[data-testid="block-info"] {
                    color: #cbd5e1 !important;
                    font-weight: 650 !important;
                }

                #generate-btn {
                    border-radius: 999px !important;
                    border: 0 !important;
                    min-height: 46px !important;
                    font-weight: 750 !important;
                    letter-spacing: -0.01em;
                    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
                    box-shadow: 0 16px 35px rgba(96, 165, 250, 0.28) !important;
                }

                #clear-btn {
                    border-radius: 999px !important;
                    min-height: 46px !important;
                    border: 1px solid var(--border) !important;
                    background: rgba(15, 23, 42, 0.74) !important;
                    color: #dbeafe !important;
                }

                #settings-panel,
                #few-shot-panel,
                #rag-panel,
                #markdown-panel {
                    border: 1px solid var(--border) !important;
                    border-radius: 20px !important;
                    background: rgba(15, 23, 42, 0.54) !important;
                    margin-top: 12px !important;
                }
                #web-scraping-panel {
                border: 1px solid var(--border) !important;
                border-radius: 20px !important;
                background: rgba(15, 23, 42, 0.54) !important;
                margin-top: 12px !important;
                }
                #mlflow-panel {
                border: 1px solid var(--border) !important;
                border-radius: 20px !important;
                background: rgba(15, 23, 42, 0.54) !important;
                margin-top: 12px !important;
                }
                #mlflow-toggle {
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                border-radius: 16px !important;
                padding: 8px 10px !important;
                background: rgba(2, 6, 23, 0.34) !important;
                }
                #web-urls textarea {
                min-height: 150px !important;
                }
                #web-scraping-toggle {
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                border-radius: 16px !important;
                padding: 8px 10px !important;
                background: rgba(2, 6, 23, 0.34) !important;
                }

                #settings-panel .wrap,
                #settings-panel .block,
                #few-shot-panel .wrap,
                #few-shot-panel .block {
                    background: transparent !important;
                }

                #few-shot-toggle {
                    border: 1px solid rgba(148, 163, 184, 0.18) !important;
                    border-radius: 16px !important;
                    padding: 8px 10px !important;
                    background: rgba(2, 6, 23, 0.34) !important;
                }

                .few-shot-note {
                    color: var(--muted);
                    font-size: 0.84rem;
                    margin: 6px 0 10px 0;
                    line-height: 1.45;
                }


                #rag-files,
                #markdown-toggle,
                #markdown-force-toggle,
                #markdown-include-prompt-toggle {
                    border: 1px solid rgba(148, 163, 184, 0.18) !important;
                    border-radius: 16px !important;
                    padding: 8px 10px !important;
                    background: rgba(2, 6, 23, 0.34) !important;
                }

                .footer-note {
                    margin-top: 14px;
                    color: var(--muted);
                    font-size: 0.84rem;
                }

                @media (max-width: 900px) {
                    #hero h1 { font-size: 1.65rem; }
                    #app-shell { padding: 14px 10px 22px 10px; }
                    #left-panel, #right-panel { padding: 14px; border-radius: 22px; }
                }
                """
# 7. Gradio frontend and prepwork for run

def prepare_user_prompt(user_prompt: str, topic: str) -> str:
    user_prompt = str(user_prompt or "").strip()
    topic = str(topic or "").strip()

    if not user_prompt and topic:
        return topic

    if "{topic}" in user_prompt:
        return user_prompt.replace("{topic}", topic)

    if topic:
        return f"{user_prompt}\n\n<topic>\n{topic}\n</topic>"

    return user_prompt

def apply_few_shot_prompting(final_prompt: str,few_shot_enabled: bool,few_shot_examples: str) -> str:
    final_prompt = str(final_prompt or "").strip()
    examples = str(few_shot_examples or "").strip()

    if not few_shot_enabled or not examples:
        return final_prompt

    return f"""Use the following examples as few-shot guidance.
                Match the structure, quality level and style pattern, but do not copy the examples verbatim.

                <few_shot_examples>
                {examples}
                </few_shot_examples>

                Now complete the actual task below.

                <actual_task>
                {final_prompt}
                </actual_task>
            """

def generate_from_ui(
                    system_prompt: str,
                    user_prompt: str,
                    topic: str,
                    few_shot_enabled: bool,
                    few_shot_examples: str,
                    web_scraping_enabled: bool,
                    web_urls: str,
                    rag_enabled: bool,
                    rag_files: Any,
                    rag_top_k: int,
                    rag_chunk_size: int,
                    rag_chunk_overlap: int,
                    markdown_force_enabled: bool,
                    markdown_save_enabled: bool,
                    markdown_filename: str,
                    markdown_include_prompt: bool,
                    mlflow_enabled: bool,
                    max_new_tokens: int,
                    temperature: float,
                    top_p: float,
                    ) -> str:
    base_prompt = prepare_user_prompt(user_prompt, topic)

    web_context = ""
    prompt_after_web = base_prompt

    if web_scraping_enabled:
        web_context = scrape_urls_to_context(web_urls)
        if web_context.strip():
            prompt_after_web = f"""Use the following scraped web context as supporting source material.
Answer based on the web context when it is relevant. If the context is incomplete, say so clearly.

<web_context>
{web_context}
</web_context>

User task:
{base_prompt}"""

    rag_context = ""
    rag_metadata = {"enabled": bool(rag_enabled)}
    rag_status_text = ""
    prompt_after_rag = prompt_after_web

    if rag_enabled:
        rag_context, rag_metadata = build_rag_context(
                                                        query=prompt_after_web,
                                                        rag_files=rag_files,
                                                        top_k=int(rag_top_k),
                                                        chunk_size=int(rag_chunk_size),
                                                        chunk_overlap=int(rag_chunk_overlap),
                                                        max_context_chars=RAG_MAX_CONTEXT_CHARS,
                                                    )
        rag_status_text = format_rag_status(rag_metadata)
        prompt_after_rag = apply_rag_context_to_prompt(prompt_after_web, rag_context)

    prompt_after_markdown_instruction = apply_markdown_output_instruction(final_prompt=prompt_after_rag,markdown_force_enabled=markdown_force_enabled)

    final_prompt = apply_few_shot_prompting(final_prompt=prompt_after_markdown_instruction,few_shot_enabled=few_shot_enabled,few_shot_examples=few_shot_examples)

    if not final_prompt.strip():
        return "Add user prompt or main topic."

    markdown_file_path = ""

    try:
        started_at = time.perf_counter()

        output = run_local_llm(
                                final_prompt,
                                system_prompt=system_prompt,
                                max_new_tokens=int(max_new_tokens),
                                temperature=float(temperature),
                                top_p=float(top_p),
                                )

        generation_time_sec = time.perf_counter() - started_at
        chat_input = build_chat_input(final_prompt, system_prompt=system_prompt)

        if markdown_save_enabled:
            markdown_metadata = {
                                "model_name": MODEL_NAME,
                                "backend": BACKEND,
                                "openvino_device": OPENVINO_DEVICE,
                                "few_shot_enabled": bool(few_shot_enabled),
                                "web_scraping_enabled": bool(web_scraping_enabled),
                                "rag_enabled": bool(rag_enabled),
                                "markdown_force_enabled": bool(markdown_force_enabled),
                                "generation_time_sec": generation_time_sec,
                                "chat_input_tokens": count_tokens(chat_input),
                                "output_tokens": count_tokens(output),
                                "rag_metadata": rag_metadata,
                                }

            markdown_file_path = save_output_to_markdown(
                                                        output=output,
                                                        filename=markdown_filename,
                                                        system_prompt=system_prompt,
                                                        final_prompt=final_prompt,
                                                        include_prompt=markdown_include_prompt,
                                                        metadata=markdown_metadata,
                                                        )

        if mlflow_enabled and MLFLOW_READY:
            run_id = log_prompt_run_to_mlflow(
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                topic=topic,
                                                base_prompt=base_prompt,
                                                web_context=web_context,
                                                prompt_after_web=prompt_after_web,
                                                rag_enabled=rag_enabled,
                                                rag_context=rag_context,
                                                rag_metadata=rag_metadata,
                                                prompt_after_rag=prompt_after_rag,
                                                markdown_force_enabled=markdown_force_enabled,
                                                markdown_save_enabled=markdown_save_enabled,
                                                markdown_file_path=markdown_file_path,
                                                prompt_after_markdown_instruction=prompt_after_markdown_instruction,
                                                final_prompt=final_prompt,
                                                chat_input=chat_input,
                                                output=output,
                                                few_shot_enabled=few_shot_enabled,
                                                few_shot_examples=few_shot_examples,
                                                web_scraping_enabled=web_scraping_enabled,
                                                web_urls=web_urls,
                                                max_new_tokens=max_new_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                generation_time_sec=generation_time_sec,
                                            )

            footer = (
                        f"\n\n---\n"
                        f"MLflow run logged: {run_id}\n"
                        f"Generation time: {generation_time_sec:.2f} sec\n"
                        f"Input tokens: {count_tokens(chat_input)} | Output tokens: {count_tokens(output)}"
                    )

            if rag_enabled:
                footer += (
                            f"\nRAG chunks: {rag_metadata.get('retrieved_chunk_count', 0)} / "
                            f"{rag_metadata.get('chunk_count', 0)}"
                            )
                if rag_status_text:
                    footer += f"\n\n{rag_status_text}"

            if markdown_file_path:
                footer += f"\nMarkdown saved to: {markdown_file_path}"

            return f"{output}{footer}"

        if mlflow_enabled and not MLFLOW_READY:
            output += (
                        "\n\n---\n"
                        "MLflow logging was enabled, but MLflow is not ready. Output was not logged."
                        )

        if rag_enabled and rag_status_text:
            output += f"\n\n---\n{rag_status_text}"

        if markdown_file_path:
            output += f"\n\n---\nMarkdown saved to: {markdown_file_path}"

        return output

    except Exception as exc:
        trace = traceback.format_exc()

        if mlflow_enabled and MLFLOW_READY:
            try:
                error_run_id = log_prompt_error_to_mlflow(
                                                            system_prompt=system_prompt,
                                                            user_prompt=user_prompt,
                                                            topic=topic,
                                                            final_prompt=final_prompt,
                                                            few_shot_enabled=few_shot_enabled,
                                                            few_shot_examples=few_shot_examples,
                                                            web_scraping_enabled=web_scraping_enabled,
                                                            web_urls=web_urls,
                                                            rag_enabled=rag_enabled,
                                                            rag_metadata=rag_metadata,
                                                            markdown_force_enabled=markdown_force_enabled,
                                                            markdown_save_enabled=markdown_save_enabled,
                                                            max_new_tokens=max_new_tokens,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            error=exc,
                                                            traceback_text=trace,
                                                            )

                return (
                        f"Generation failed: {type(exc).__name__}: {exc}\n\n"
                        f"---\n"
                        f"MLflow failed run logged: {error_run_id}"
                        )
            except Exception as mlflow_exc:
                return (
                        f"Generation failed: {type(exc).__name__}: {exc}\n\n"
                        f"Additionally, MLflow error logging failed: "
                        f"{type(mlflow_exc).__name__}: {mlflow_exc}"
                        )

        return f"Generation failed: {type(exc).__name__}: {exc}"


def clear_output() -> str:
    return ""

with gr.Blocks(title="Local OpenVINO Assistant",css=APP_CSS) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            f"""
            <div id="hero">
                <h1>Local Prompting Assistant</h1>
                <div class="status-row">
                    <span class="status-pill"><span class="status-dot"></span>Backend: <strong>{BACKEND}</strong></span>
                    <span class="status-pill">Model: <strong>{MODEL_NAME}</strong></span>
                    <span class="status-pill">Device: <strong>{OPENVINO_DEVICE if BACKEND == 'openvino' else 'HF fallback'}</strong></span>
                    <span class="status-pill">Local URL: <strong>127.0.0.1:7860</strong></span>
                    <span class="status-pill">Tracking: <strong>MLflow optional</strong></span>
                </div>
            </div>
            """
        )

        with gr.Row(elem_id="main-grid"):
            with gr.Column(scale=5, elem_id="left-panel"):
                gr.HTML(
                        """
                            <h2 class="section-title">Prompt workspace</h2>
                        """
                        )

                system_prompt_box = gr.Textbox(
                                                label="System prompt / Persona",
                                                value=DEFAULT_SYSTEM_PROMPT,
                                                lines=10,
                                                placeholder="Example: You are a helpful assistant.",
                                                elem_id="system-prompt")

                topic_box = gr.Textbox(
                                        label="Topic",
                                        value=DEFAULT_TOPIC,
                                        lines=5,
                                        placeholder="Example: new car model",
                                        elem_id="topic-box",
                                        )

                user_prompt_box = gr.Textbox(
                                            label="User prompt / instruction",
                                            value=DEFAULT_USER_PROMPT,
                                            lines=15,
                                            placeholder="Insert model instruction. You can use {topic}.",
                                            elem_id="user-prompt",
                                            )

                with gr.Accordion("Few-shot prompting", open=False, elem_id="few-shot-panel"):
                    gr.HTML(
                            """
                            <p class="few-shot-note">
                                Enable few-shot prompting when you want to show the model examples of the expected structure, tone or output style.
                                Examples can be inserted as plain text in Input / Output format.
                            </p>
                            """
                            )

                    few_shot_enabled_box = gr.Checkbox(
                                                        label="Enable few-shot prompting",
                                                        value=False,
                                                        elem_id="few-shot-toggle",
                                                        )

                    few_shot_examples_box = gr.Textbox(
                                                        label="Few-shot examples - plain text",
                                                        value=DEFAULT_FEW_SHOT_EXAMPLES,
                                                        lines=20,
                                                        placeholder="Example 1\nInput: ...\nOutput: ...\n\nExample 2\nInput: ...\nOutput: ...",
                                                        elem_id="few-shot-examples",
                                                        )

                with gr.Accordion("Web scraping", open=False, elem_id="web-scraping-panel"):
                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Optional URL context. The app will try to download and clean text from static web pages.
                                This works best for non-JavaScript websites.
                                </p>
                            """
                            )

                    web_scraping_enabled_box = gr.Checkbox(
                                                            label="Enable web scraping",
                                                            value=False,
                                                            elem_id="web-scraping-toggle",
                                                            )

                    web_urls_box = gr.Textbox(
                                                label="URLs - one web location per line",
                                                value=DEFAULT_WEB_URLS,
                                                lines=10,
                                                placeholder="https://example.com\nhttps://wikipedia.org",
                                                elem_id="web-urls",
                                            )

                with gr.Accordion("RAG document context", open=False, elem_id="rag-panel"):
                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Optional local RAG context. Upload .txt, .md, .pdf, .docx or .rtf files.
                                The app copies uploads into the local rag_uploads folder, builds a temporary CPU FAISS index and retrieves the most relevant chunks for the current prompt.
                                </p>
                            """
                            )

                    rag_enabled_box = gr.Checkbox(
                                                    label="Enable RAG document context",
                                                    value=False,
                                                    elem_id="rag-toggle",
                                                    )

                    rag_files_box = gr.File(
                                            label="RAG source files",
                                            file_count="multiple",
                                            type="filepath",
                                            elem_id="rag-files",
                                            )

                    rag_top_k_slider = gr.Slider(
                                                minimum=1,
                                                maximum=10,
                                                value=4,
                                                step=1,
                                                label="RAG Top-K - number of retrieved chunks",
                                                )

                    rag_chunk_size_slider = gr.Slider(
                                                    minimum=300,
                                                    maximum=2500,
                                                    value=900,
                                                    step=100,
                                                    label="RAG chunk size - larger = broader context",
                                                    )

                    rag_chunk_overlap_slider = gr.Slider(
                                                        minimum=0,
                                                        maximum=500,
                                                        value=150,
                                                        step=50,
                                                        label="RAG chunk overlap - helps preserve context continuity",
                                                        )

                with gr.Accordion("Markdown output / memory", open=False, elem_id="markdown-panel"):
                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Optional Markdown mode. The model can be instructed to answer in Markdown and the generated output can be saved
                                as a reusable .md file.
                                </p>
                            """
                            )

                    markdown_force_enabled_box = gr.Checkbox(
                                                            label="Force Markdown-formatted answer",
                                                            value=False,
                                                            elem_id="markdown-force-toggle",
                                                            )

                    markdown_save_enabled_box = gr.Checkbox(
                                                            label="Save output to .md file",
                                                            value=False,
                                                            elem_id="markdown-toggle",
                                                            )

                    markdown_filename_box = gr.Textbox(
                                                        label="Markdown filename",
                                                        value="llm_output.md",
                                                        lines=1,
                                                        placeholder="example: project_memory.md",
                                                        elem_id="markdown-filename",
                                                        )

                    markdown_include_prompt_box = gr.Checkbox(
                                                            label="Include prompt and metadata in saved .md",
                                                            value=True,
                                                            elem_id="markdown-include-prompt-toggle",
                                                            )

                with gr.Accordion("Experiment tracking", open=False, elem_id="mlflow-panel"):
                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Optional MLflow logging. When enabled, the app saves the prompt, output, model settings,
                                token counts and generation time into a local MLflow experiment.
                                </p>
                            """
                            )

                    mlflow_enabled_box = gr.Checkbox(
                        label="Enable MLflow logging",value=False,elem_id="mlflow-toggle")

                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Local MLflow UI can be found under URL:<br>
                                <code>http://127.0.0.1:5000</code> or the value configured by MLFLOW_UI_PORT.
                                </p>
                            """
                            )

                with gr.Accordion("Generation settings", open=False, elem_id="settings-panel"):
                    max_new_tokens_slider = gr.Slider(
                                                        minimum=32,
                                                        maximum=2048,
                                                        value=MAX_NEW_TOKENS,
                                                        step=32,
                                                        label="Max new tokens - limit of answer length",
                                                    )

                    temperature_slider = gr.Slider(
                                                    minimum=0.0,
                                                    maximum=2.0,
                                                    value=TEMPERATURE,
                                                    step=0.1,
                                                    label="Temperature - lower = precise, higher = creative",
                                                    )

                    top_p_slider = gr.Slider(
                                            minimum=0.1,
                                            maximum=2.0,
                                            value=TOP_P,
                                            step=0.1,
                                            label="Top-p - lower = conservative, higher = varied",
                                            )

            with gr.Column(scale=4, elem_id="right-panel"):
                gr.HTML(
                        """
                        <h2 class="section-title">Assistant output</h2>
                        """
                        )

                output_box = gr.Textbox(
                                        label="Output",
                                        lines=20,
                                        placeholder="Model output:",
                                        elem_id="output-box",
                                        )

                with gr.Row():
                    generate_button = gr.Button("Generate",variant="primary",elem_id="generate-btn")
                    clear_button = gr.Button("Clear output",elem_id="clear-btn")

        generate_button.click(
                                fn=generate_from_ui,
                                inputs=[
                                        system_prompt_box,
                                        user_prompt_box,
                                        topic_box,
                                        few_shot_enabled_box,
                                        few_shot_examples_box,
                                        web_scraping_enabled_box,
                                        web_urls_box,
                                        rag_enabled_box,
                                        rag_files_box,
                                        rag_top_k_slider,
                                        rag_chunk_size_slider,
                                        rag_chunk_overlap_slider,
                                        markdown_force_enabled_box,
                                        markdown_save_enabled_box,
                                        markdown_filename_box,
                                        markdown_include_prompt_box,
                                        mlflow_enabled_box,
                                        max_new_tokens_slider,
                                        temperature_slider,
                                        top_p_slider,
                                        ],
                                outputs=output_box
                            )

        clear_button.click(fn=clear_output,inputs=None,outputs=output_box)

demo.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            share=False,
            inbrowser=os.getenv("GRADIO_INBROWSER", "1").strip().lower() in {"1", "true", "yes", "y"},
            )
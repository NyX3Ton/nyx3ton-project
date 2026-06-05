
#!python -m pip install --upgrade pip setuptools wheel
#!python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
#!python -m pip install --upgrade ipykernel jupyter
#!python -m pip install --upgrade openvino optimum-intel transformers accelerate safetensors sentencepiece huggingface_hub requests python-dotenv gradio bs4

import os, platform, traceback, torch, transformers, re, requests
from pathlib import Path
from typing import Optional
import gradio as gr

from bs4 import BeautifulSoup
from urllib.parse import urlparse

import openvino as ov
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM

#MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-0.6B")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

HF_TOKEN = os.getenv("HF_TOKEN") or None

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")
OV_MODELS_DIR = os.getenv("OV_MODELS_DIR", "./ov_models")
OV_CACHE_DIR = os.getenv("OV_CACHE_DIR", "./ov_cache")

# Pre Intel CPU/NPU/GPU mozes skusit napr. CPU, GPU, NPU alebo AUTO.
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

OFFLINE_MODE = os.getenv("LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}
FORCE_HF_FALLBACK = os.getenv("FORCE_HF_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "y"}

cache_path = Path(HF_CACHE_DIR).expanduser().resolve()
ov_root = Path(OV_MODELS_DIR).expanduser().resolve()
ov_cache = Path(OV_CACHE_DIR).expanduser().resolve()

cache_path.mkdir(parents=True, exist_ok=True)
ov_root.mkdir(parents=True, exist_ok=True)
ov_cache.mkdir(parents=True, exist_ok=True)

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
        ov_model = OVModelForCausalLM.from_pretrained(str(ov_model_path),device=OPENVINO_DEVICE,compile=False,ov_config=OV_CONFIG)
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

        ov_model = OVModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

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
        hf_kwargs["torch_dtype"] = "auto"
    else:
        hf_kwargs["torch_dtype"] = "auto"

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **hf_kwargs)

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

    if BACKEND == "huggingface":
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
    else:
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

print("\nHelper function ready: run_local_llm(prompt, system_prompt=None)")

def extract_urls_from_text(urls_text: str) -> list[str]:
    urls_text = str(urls_text or "").strip()

    if not urls_text:
        return []

    urls = []

    for line in urls_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Ak user zada URL bez https://
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
        text = text[:max_chars] + "\n\n[Text bol skrateny kvoli limitu dlzky.]"

    return text

def scrape_single_url(url: str, timeout: int = 15, max_chars: int = 6000) -> dict:
    """Stiahne a vycisti text z jednej webovej stranky."""
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

        # Odstranit technicke / navigacne casti
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
                #few-shot-panel {
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

def generate_from_ui(system_prompt: str,user_prompt: str,topic: str,few_shot_enabled: bool,few_shot_examples: str,web_scraping_enabled: bool,web_urls: str,max_new_tokens: int,temperature: float,top_p: float) -> str:
    final_prompt = prepare_user_prompt(user_prompt, topic)

    final_prompt = apply_web_context_to_prompt(final_prompt=final_prompt,web_scraping_enabled=web_scraping_enabled,urls_text=web_urls)

    final_prompt = apply_few_shot_prompting(final_prompt=final_prompt,few_shot_enabled=few_shot_enabled,few_shot_examples=few_shot_examples)

    if not final_prompt.strip():
        return "Add user prompt or main topic."

    try:
        return run_local_llm(
                            final_prompt,
                            system_prompt=system_prompt,
                            max_new_tokens=int(max_new_tokens),
                            temperature=float(temperature),
                            top_p=float(top_p),
                            )
    except Exception as exc:
        return f"Generation failed: {type(exc).__name__}: {exc}"

def clear_output() -> str:
    return ""

with gr.Blocks(title="Local OpenVINO Assistant",css=APP_CSS,) as demo:
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
                                                label="System prompt/Persona",
                                                value=DEFAULT_SYSTEM_PROMPT,
                                                lines=10,
                                                placeholder="Example: You are a helpful assistant",
                                                elem_id="system-prompt"
                                                )

                topic_box = gr.Textbox(
                                        label="Topic",
                                        value=DEFAULT_TOPIC,
                                        lines=5,
                                        placeholder="Example: new car model",
                                        elem_id="topic-box"
                                        )

                user_prompt_box = gr.Textbox(
                                            label="User prompt / instruction",
                                            value=DEFAULT_USER_PROMPT,
                                            lines=15,
                                            placeholder="Inset model instruction. Please use {topic}.",
                                            elem_id="user-prompt",
                                            )

                with gr.Accordion("Few-shot prompting", open=False, elem_id="few-shot-panel"):
                    gr.HTML(
                            """
                                <p class="few-shot-note">
                                Enable few-shot prompting.
                                Examples maybe inserted in plan text format as Input / Output.
                                </p>
                            """
                    )

                    few_shot_enabled_box = gr.Checkbox(label="Enable few-shot prompting",value=False,elem_id="few-shot-toggle")

                    few_shot_examples_box = gr.Textbox(
                                                        label="Few-shot examples - plain text",
                                                        value=DEFAULT_FEW_SHOT_EXAMPLES,
                                                        lines=20,
                                                        placeholder="Example 1\nInput: ...\nOutput: ...\n\nExample 2\nInput: ...\nOutput: ...",
                                                        elem_id="few-shot-examples",
                                                        )
                with gr.Accordion("Web scraping", open=False, elem_id="web-scraping-panel"):
                    gr.HTML("""<p class="few-shot-note">Toggle for optional URL Web scrapping (Non-JavaScript).</p>""")

                    web_scraping_enabled_box = gr.Checkbox(label="Enable web scraping",value=False,elem_id="web-scraping-toggle")

                    web_urls_box = gr.Textbox(label="URLs - one web location per line",value=DEFAULT_WEB_URLS,lines=10,placeholder="https://example.com\nhttps://wikipedia.org",elem_id="web-urls")

                with gr.Accordion("Generation settings", open=False, elem_id="settings-panel"):
                    max_new_tokens_slider = gr.Slider(minimum=32,maximum=2048,value=MAX_NEW_TOKENS,step=16,label="Max new tokens - Output lenght ")

                    temperature_slider = gr.Slider(minimum=0.0,maximum=2.0,value=TEMPERATURE,step=0.1,label="Temperature - Creativity/Randomnes")

                    top_p_slider = gr.Slider(minimum=0.1,maximum=1.0,value=TOP_P,step=0.1,label="Top-P - Output text colorfulness")

            with gr.Column(scale=4, elem_id="right-panel"):
                gr.HTML(
                            """
                            <h2 class="section-title">Assistant output</h2>
                            """
                        )

                output_box = gr.Textbox(label="Output",lines=20,placeholder="Model output: ",elem_id="output-box")

                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary", elem_id="generate-btn")
                    clear_button = gr.Button("Clear output", elem_id="clear-btn")

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
                                    max_new_tokens_slider,
                                    temperature_slider,
                                    top_p_slider,
                                    ],
                            outputs=output_box,
                            )

        clear_button.click(fn=clear_output,inputs=None,outputs=output_box)

demo.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            #server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            share=False,
            inbrowser=True,
            )

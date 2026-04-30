# helper script: validator_llm.py of nyx3ton-project\CV Validator 
# https://huggingface.co/Qwen/Qwen3.5-4B
# Thinking mode for general tasks: 
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0

from __future__ import annotations

import gc, os, torch
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from validator_utils import normalize_space, strip_thinking

# -----------------------------------------------------------------------------
# 1. LOCAL HUGGING FACE LLM
# -----------------------------------------------------------------------------

def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

DEFAULT_LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "unsloth/Qwen3.5-4B")
DEFAULT_FALLBACK_LLM_MODEL_ID = os.getenv("FALLBACK_LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
LLM_LOAD_MODE = os.getenv("LLM_LOAD_MODE", "fp16_gpu")
MAX_GPU_MEMORY = os.getenv("MAX_GPU_MEMORY", "10.5GiB")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "900"))
P_SETTING = float(os.getenv("DEF_P_SETTING", "0.95"))
GEN_TOP_K_SETTING = int(os.getenv("DEF_TOP_K_SETTING", "20"))
REPETITION_PEN = float(os.getenv("DEF_REPETITION_PEN", "1.0"))

_TOKENIZER = None
_MODEL = None
_MODEL_INFO = "Model este nie je nacitany."


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA nie je dostupna, pouzije sa CPU alebo fallback."
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    allocated_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
    reserved_gb = torch.cuda.memory_reserved(0) / 1024 ** 3
    return f"CUDA OK: {name}, VRAM total={total_gb:.1f} GB, allocated={allocated_gb:.2f} GB, reserved={reserved_gb:.2f} GB"


def unload_llm() -> str:
    global _TOKENIZER, _MODEL, _MODEL_INFO
    _TOKENIZER = None
    _MODEL = None
    _MODEL_INFO = "Model bol uvolneny."
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _MODEL_INFO + "\n" + cuda_summary()


def load_llm(model_id: str = DEFAULT_LLM_MODEL_ID,load_mode: str = LLM_LOAD_MODE,fallback_model_id: str = DEFAULT_FALLBACK_LLM_MODEL_ID,
):
    global _TOKENIZER, _MODEL, _MODEL_INFO

    desired_signature = f"{model_id}|{load_mode}|fallback={fallback_model_id}"
    if _MODEL is not None and _TOKENIZER is not None and desired_signature in _MODEL_INFO:
        return _TOKENIZER, _MODEL, _MODEL_INFO

    unload_llm()

    errors = []
    has_cuda = torch.cuda.is_available()

    def _load_tokenizer(mid: str):
        return AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    def _try_4bit(mid: str):
        if not has_cuda:
            raise RuntimeError("CUDA nie je dostupna pre 4-bit GPU load.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tok = _load_tokenizer(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: MAX_GPU_MEMORY, "cpu": "48GiB"},
            trust_remote_code=True,
        )
        return tok, mdl, f"Nacitany model: {mid} | mode=bnb_4bit | {desired_signature}"

    def _try_fp16_gpu(mid: str):
        if not has_cuda:
            raise RuntimeError("CUDA nie je dostupna pre fp16_gpu load.")
        tok = _load_tokenizer(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: MAX_GPU_MEMORY, "cpu": "48GiB"},
            trust_remote_code=True,
        )
        return tok, mdl, f"Nacitany model: {mid} | mode=fp16_gpu | {desired_signature}"

    def _try_cpu(mid: str):
        tok = _load_tokenizer(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        return tok, mdl, f"Nacitany model: {mid} | mode=cpu | {desired_signature}"

    attempts = []
    mode = (load_mode or "auto").lower().strip()

    if mode == "bnb_4bit":
        attempts = [lambda: _try_4bit(model_id)]
    elif mode == "fp16_gpu":
        attempts = [lambda: _try_fp16_gpu(model_id)]
    elif mode == "cpu":
        attempts = [lambda: _try_cpu(model_id)]
    else:
        attempts = [
            lambda: _try_4bit(model_id),
            lambda: _try_fp16_gpu(model_id),
            lambda: _try_fp16_gpu(fallback_model_id),
            lambda: _try_cpu(fallback_model_id),
        ]

    for attempt in attempts:
        try:
            _TOKENIZER, _MODEL, _MODEL_INFO = attempt()
            _MODEL.eval()
            return _TOKENIZER, _MODEL, _MODEL_INFO + "\n" + cuda_summary()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        "Nepodarilo sa nacitat lokalny LLM.\n\nPokusy:\n- " + "\n- ".join(errors)
    )


def model_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"


def lc_messages_to_hf_messages(messages: List[Any]) -> List[Dict[str, str]]:
    role_map = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
                }

    converted = []
    for msg in messages:
        msg_type = getattr(msg, "type", "human")
        role = role_map.get(msg_type, "user")
        content = normalize_space(str(getattr(msg, "content", "")))
        if content:
            converted.append({"role": role, "content": content})
    return converted


def chat_generate_messages(
                            messages: List[Dict[str, str]],
                            model_id: str,
                            load_mode: str,
                            fallback_model_id: str,
                            max_new_tokens: int = MAX_NEW_TOKENS,
                            do_sample: bool = False,
                            temperature: float = 0.2,
                            ) -> str:
    tok, mdl, _ = load_llm(model_id, load_mode, fallback_model_id)

    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}:\n{m['content']}")
        prompt = "\n\n".join(parts) + "\n\nAssistant:\n"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    dev = model_device(mdl)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.eos_token_id,
        "repetition_penalty": REPETITION_PEN,
        "do_sample": do_sample,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = P_SETTING
        generation_kwargs["top_k"] = GEN_TOP_K_SETTING

    with torch.no_grad():
        out = mdl.generate(**generation_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    text = tok.decode(generated, skip_special_tokens=True)
    return strip_thinking(text)


SYSTEM_JSON = """
                Si lokalny AI asistent pre validaciu zivotopisov voci pracovnemu inzeratu.

                Tvoje pravidla:
                - odpovedaj iba validnym JSON objektom alebo validnym JSON polom
                - nepouzivaj markdown
                - nepouzivaj text mimo JSON
                - nehadaj informacie, ktore nie su v dodanom texte
                - pri hodnoteni kandidata pouzivaj iba dodane odkazy z CV
                - nevykonavaj finalne rozhodnutie o prijati kandidata
                - vystup sluzi iba ako odporucanie pre cloveka
                - nehodnot citlive atributy ako vek, pohlavie, narodnost, zdravotny stav, rodinny stav, fotografia alebo adresa
                - pis slovensky bez diakritiky
                """.strip()

SYSTEM_REQUIREMENT_UTILS_JSON = """
                Si lokalny AI asistent pre mikro-ulohy pri spracovani poziadaviek z pracovnych inzeratov.

                Tvoje pravidla:
                - odpovedaj iba validnym JSON objektom
                - nepouzivaj markdown
                - nepouzivaj text mimo JSON
                - bud konzistentny a kratky
                - canonical_key pis malymi pismenami bez zbytocnych slov
                - pis slovensky bez diakritiky
                """.strip()

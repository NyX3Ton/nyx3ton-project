# helper script: validator_llm.py of nyx3ton-project\CV Validator
# https://huggingface.co/Qwen/Qwen3.5-4B
# Thinking mode for general tasks:
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
#
# Backend volba:
#   - ak je dostupna CUDA  -> GPU (bnb_4bit / fp16)
#   - ak CUDA nie je / zlyha -> konverzia a beh cez OpenVINO (CPU / Intel iGPU)
#   - posledna zachrana      -> cisty torch CPU (fp32)

from __future__ import annotations

import gc, os, torch
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from validator_utils import normalize_space, strip_thinking

# -----------------------------------------------------------------------------
# 1. KONFIGURACIA
# -----------------------------------------------------------------------------
def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

DEFAULT_LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "unsloth/Qwen3.5-4B")
DEFAULT_FALLBACK_LLM_MODEL_ID = os.getenv("FALLBACK_LLM_MODEL_ID", "unsloth/DeepSeek-R1-Distill-Qwen-7B")
LLM_LOAD_MODE = os.getenv("LLM_LOAD_MODE", "fp16_gpu")  # auto | bnb_4bit | fp16_gpu | openvino | cpu
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "CPU")   # CPU | GPU | AUTO
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
        return "CUDA nie je dostupna, pouzije sa OpenVINO (CPU/iGPU) alebo CPU fallback."
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
    return f"CUDA OK: {name}, VRAM total={total:.1f} GB, allocated={alloc:.2f} GB, reserved={reserved:.2f} GB"

def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def unload_llm() -> str:
    global _TOKENIZER, _MODEL, _MODEL_INFO
    _TOKENIZER = _MODEL = None
    _MODEL_INFO = "Model bol uvolneny."
    _free_cuda()
    return _MODEL_INFO + "\n" + cuda_summary()
    
# -----------------------------------------------------------------------------
# 2. NACITANIE MODELU (CUDA -> OpenVINO -> CPU)
# -----------------------------------------------------------------------------
def load_llm(model_id: str = DEFAULT_LLM_MODEL_ID, load_mode: str = LLM_LOAD_MODE, fallback_model_id: str = DEFAULT_FALLBACK_LLM_MODEL_ID):
    global _TOKENIZER, _MODEL, _MODEL_INFO

    signature = f"{model_id}|{load_mode}|fallback={fallback_model_id}"
    if _MODEL is not None and _TOKENIZER is not None and signature in _MODEL_INFO:
        return _TOKENIZER, _MODEL, _MODEL_INFO

    unload_llm()
    has_cuda = torch.cuda.is_available()

    def _tok(mid: str):
        return AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    def _try_4bit(mid: str):
        if not has_cuda:
            raise RuntimeError("CUDA nie je dostupna pre bnb_4bit load.")
        bnb = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True,
                                )
        mdl = AutoModelForCausalLM.from_pretrained(
                                                    mid, 
                                                    quantization_config=bnb, 
                                                    device_map="auto",
                                                    max_memory={0: MAX_GPU_MEMORY, "cpu": "32GiB"}, 
                                                    trust_remote_code=True,
                                                    )
        return _tok(mid), mdl, f"Nacitany model: {mid} | mode=bnb_4bit | {signature}"

    def _try_fp16_gpu(mid: str):
        if not has_cuda:
            raise RuntimeError("CUDA nie je dostupna pre fp16_gpu load.")
        mdl = AutoModelForCausalLM.from_pretrained(
                                                    mid, 
                                                    dtype=torch.float16, 
                                                    device_map="auto",
                                                    max_memory={0: MAX_GPU_MEMORY, "cpu": "32GiB"}, 
                                                    trust_remote_code=True,
                                                    )
        return _tok(mid), mdl, f"Nacitany model: {mid} | mode=fp16_gpu | {signature}"

    def _try_openvino(mid: str):
        # Konverzia HF modelu do OpenVINO IR za behu (export=True) + inference cez OV runtime.
        try:
            from optimum.intel import OVModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                                "OpenVINO backend nie je dostupny. Nainstaluj: pip install \"optimum[openvino]\""
                                ) from exc
        kwargs = dict(export=True, trust_remote_code=True)
        try:
            mdl = OVModelForCausalLM.from_pretrained(mid, **kwargs)
        except TypeError:
            kwargs.pop("trust_remote_code", None)
            mdl = OVModelForCausalLM.from_pretrained(mid, **kwargs)
        try:
            mdl.to(OPENVINO_DEVICE)
        except Exception:
            pass
        return _tok(mid), mdl, f"Nacitany model: {mid} | mode=openvino:{OPENVINO_DEVICE} | {signature}"

    def _try_cpu(mid: str):
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True,
        )
        return _tok(mid), mdl, f"Nacitany model: {mid} | mode=cpu | {signature}"

    single = {
                "bnb_4bit": [(_try_4bit, model_id)],
                "fp16_gpu": [(_try_fp16_gpu, model_id)],
                "openvino": [(_try_openvino, model_id)],
                "cpu": [(_try_cpu, model_id)],
                }
    mode = (load_mode or "auto").lower().strip()
    if mode in single:
        attempts = single[mode]
    elif has_cuda:
        # CUDA dostupna -> skus GPU, pri zlyhani prejdi na OpenVINO, nakoniec CPU.
        attempts = [
                    (_try_4bit, model_id),
                    (_try_fp16_gpu, model_id),
                    (_try_openvino, model_id),
                    (_try_openvino, fallback_model_id),
                    (_try_cpu, fallback_model_id),
                    ]
    else:
        # CUDA nedostupna -> rovno OpenVINO, nakoniec CPU.
        attempts = [
                    (_try_openvino, model_id),
                    (_try_openvino, fallback_model_id),
                    (_try_cpu, fallback_model_id),
                    ]

    errors: List[str] = []
    for fn, mid in attempts:
        try:
            _TOKENIZER, _MODEL, _MODEL_INFO = fn(mid)
            try:
                _MODEL.eval()
            except Exception:
                pass
            return _TOKENIZER, _MODEL, _MODEL_INFO + "\n" + cuda_summary()
        except Exception as exc:
            errors.append(f"{fn.__name__}({mid}): {type(exc).__name__}: {exc}")
            _free_cuda()

    raise RuntimeError("Nepodarilo sa nacitat lokalny LLM.\n\nPokusy:\n- " + "\n- ".join(errors))
# -----------------------------------------------------------------------------
# 3. GENEROVANIE
# -----------------------------------------------------------------------------
def model_device(model: Any):
    try:
        return next(model.parameters()).device  # torch backend
    except Exception:
        return "cuda" if torch.cuda.is_available() else "cpu"  # OpenVINO/CPU


def lc_messages_to_hf_messages(messages: List[Any]) -> List[Dict[str, str]]:
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    converted = []
    for msg in messages:
        role = role_map.get(getattr(msg, "type", "human"), "user")
        content = normalize_space(str(getattr(msg, "content", "")))
        if content:
            converted.append({"role": role, "content": content})
    return converted


def chat_generate_messages(
    messages: List[Dict[str, str]],model_id: str,load_mode: str,fallback_model_id: str,max_new_tokens: int = MAX_NEW_TOKENS,do_sample: bool = False,temperature: float = 0.2) -> str:
    tok, mdl, _ = load_llm(model_id, load_mode, fallback_model_id)

    if getattr(tok, "chat_template", None):
        # Qwen/DeepSeek thinking modely mozu emitovat <think> bloky a rozbit JSON
        # parsing. Pre validaciu CV potrebujeme deterministicky JSON, takze
        # thinking explicitne vypneme, ak to tokenizer podporuje.
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"{m['role'].capitalize()}:\n{m['content']}" for m in messages) + "\n\nAssistant:\n"

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    inputs = {k: v.to(model_device(mdl)) for k, v in inputs.items()}

    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": pad,
                    "eos_token_id": eos,
                    "repetition_penalty": REPETITION_PEN,
                    "do_sample": do_sample,
                    }
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=P_SETTING, top_k=GEN_TOP_K_SETTING)

    with torch.no_grad():
        out = mdl.generate(**gen_kwargs)

    generated = out[0][inputs["input_ids"].shape[-1]:]
    raw_text = tok.decode(generated, skip_special_tokens=True)
    text = strip_thinking(raw_text).strip()

    if env_bool("DEBUG_LLM_RAW", False):
        print("\n--- RAW LLM OUTPUT START ---")
        print(raw_text[:4000])
        print("--- RAW LLM OUTPUT END ---\n")

    return text
# -----------------------------------------------------------------------------
# 4. SYSTEM PROMPTY
# -----------------------------------------------------------------------------

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
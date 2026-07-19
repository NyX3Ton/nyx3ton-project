# agent.py — local Gemma tool-calling and optional writer--critic refinement

from __future__ import annotations

import ast, json, logging, re, torch
from typing import Any, Callable, Optional, Protocol

from . import config, semantic_match
from .govee_client import Device, GoveeAPIError
from .memory_store import MemoryStore
from .news_client import NewsClient, NewsError
from .weather_client import WeatherClient, WeatherError

logger = logging.getLogger("agent")

THINKING_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)

# <tool_call>{...}</tool_call>  (Qwen / Hermes)
_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Backwards-compatible public name used by older integrations and offline tests.
TOOL_CALL_RE = _XML_RE
# Qwen 3.5 XML function form, e.g. <function=set_power><parameter=on>true
# </parameter></function>. Keep it supported because Qwen is the documented
# fallback model even though Gemma's compact call format is the primary path.
_QWEN_FUNCTION_RE = re.compile(r"<function=([\w-]+)>\s*(.*?)\s*</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# Fenced code block with an optional language tag we recognize
_FENCE_RE = re.compile(r"```(?:tool_call|tool_code|json|python)?\s*\n?(.*?)```", re.DOTALL)
# Gemma 4 compact format:  call:name{key:value, key:value}   or   call:name{}
# Keys and values are UNQUOTED; values may contain spaces (e.g. "table lamp 1").
_COMPACT_RE = re.compile(r"call\s*:\s*(\w+)\s*\{([^{}]*)\}", re.IGNORECASE)

def _coerce(v: str) -> Any:
    v = v.strip().strip('"').strip("'").strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+", v):
        return float(v)
    return v

def _compact_to_calls(text: str) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for name, body in _COMPACT_RE.findall(text):
        args: dict[str, Any] = {}
        body = body.strip()
        if body:
            for pair in body.split(","):
                if ":" in pair:
                    k, _, val = pair.partition(":")
                elif "=" in pair:
                    k, _, val = pair.partition("=")
                else:
                    continue
                k = k.strip().strip('"').strip("'").strip()
                if k:
                    args[k] = _coerce(val)
        calls.append((name, args))
    return calls
def _find_json_objects(text: str) -> list[str]:
    objs, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(text[start:i + 1])
    return objs

def _json_to_call(blob: str) -> Optional[tuple[str, dict]]:
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return (obj["name"], args if isinstance(args, dict) else {})

def _python_to_calls(code: str) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for line in code.strip().splitlines():
        line = line.strip().strip(",").strip("[]").strip()
        if not line or "(" not in line:
            continue
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        args: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue
        calls.append((node.func.id, args))
    return calls

def _qwen_xml_to_calls(text: str) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for name, body in _QWEN_FUNCTION_RE.findall(text):
        arguments = {key: _coerce(value) for key, value in _QWEN_PARAMETER_RE.findall(body)}
        calls.append((name, arguments))
    return calls

def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    # 0. Gemma 4 compact format: call:name{k:v,k:v}
    calls = _compact_to_calls(text)
    if calls:
        return calls

    # 1. <tool_call> XML wrapping JSON
    calls = [c for blob in _XML_RE.findall(text) if (c := _json_to_call(blob))]
    if calls:
        return calls

    # 2. Qwen's XML function form (also used by the configured fallback).
    calls = _qwen_xml_to_calls(text)
    if calls:
        return calls

    # 3. Fenced code blocks (```tool_code / ```json / ```python / ```)
    for block in _FENCE_RE.findall(text):
        block = block.strip()
        whole = _json_to_call(block)
        if whole:
            calls.append(whole)
            continue
        inline = [c for m in _find_json_objects(block) if (c := _json_to_call(m))]
        if inline:
            calls.extend(inline)
            continue
        calls.extend(_python_to_calls(block))  # Gemma python-style
    if calls:
        return calls

    # 4. Bare JSON object anywhere in the text (last resort)
    return [c for m in _find_json_objects(text) if (c := _json_to_call(m))]

SYSTEM_PROMPT = (
                    "You are a home assistant that controls Govee smart home devices through "
                    "tool calls. Always call list_devices or get_device_state first if you're "
                    "not sure a device exists or what it supports, rather than guessing a "
                    "device name or capability. "
                    "To turn several devices on or off at once (e.g. 'turn off everything', "
                    "'turn off all the lights', 'turn off the bedroom') make a SINGLE "
                    "set_power_all call instead of calling set_power once per device. "
                    "You can also check the weather (get_weather), fetch news headlines "
                    "(get_news), read the full text of a specific article (get_article_extract, "
                    "using the 'link' from a get_news result - use this when the user wants "
                    "more than the headline/summary, e.g. 'tell me more about that' or 'what "
                    "does the article say'), and recall things from earlier conversations or "
                    "previous weather/news lookups (recall_memories). Every chat turn and "
                    "every weather/news lookup is remembered automatically - there is no "
                    "separate 'save' step. Call recall_memories whenever the user asks you "
                    "to remember, recall, or refers to something discussed earlier; pass a "
                    "query describing what to look for, or leave it empty for the most "
                    "recent memories. "
                    "Keep replies short and state what changed. "
                    "The user may write in any language - always reply in the same language "
                    "they used, but pass device/scene names to tools as the user wrote them "
                    "(untranslated); device resolution handles matching across languages.\n\n"
                    "TOOL CALL FORMAT — emit each call on its own line as:\n"
                    "  call:tool_name{arg:value, arg:value}\n"
                    "Use true/false for booleans; leave the braces empty when there are no "
                    "arguments. Do not wrap calls in quotes, code fences, or JSON. "
                    "After a tool result is returned to you, reply to the user in plain "
                    "language describing what changed - do NOT emit another call unless more "
                    "actions are still needed.\n\n"
                    "Examples:\n"
                    "User: turn off the bedroom light\n"
                    "call:set_power{device_name:Bedroom Light, on:false}\n"
                    "(after the result) Turned off the Bedroom Light.\n\n"
                    "User: what devices do I have?\n"
                    "call:list_devices{}\n"
                    "(after the result) You have a Bedroom Light and a Table Lamp 1.\n\n"
                    "User: set the kitchen to 40%\n"
                    "call:set_brightness{device_name:Kitchen, percent:40}\n"
                    "(after the result) Set the Kitchen to 40% brightness.\n\n"
                    "User: what did I ask you about the weather earlier?\n"
                    "call:recall_memories{query:weather}\n"
                    "(after the result) Earlier you asked about the weather in Bratislava - "
                    "it was 24C and partly cloudy."
                    )

class DeviceNotFoundError(Exception):
    pass

# ---------------------------------------------------------------------------
# Backend loading: CUDA (+ optional bitsandbytes quantization) -> CPU
# ---------------------------------------------------------------------------
class ModelBackend:
    def __init__(
        self,
        model_id: str = config.GOVEE_LLM_MODEL,
        fallback_model_id: str = config.GOVEE_FALLBACK_MODEL,
    ):
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.backend_name, self.model, self.processor = self._load()
        # Compatibility alias so any code still referencing backend.tokenizer keeps working.
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # self.model_id reflects whatever actually loaded (may be the fallback).
        logger.info("Loaded %s on backend=%s", self.model_id, self.backend_name)

    def _load(self) -> tuple[str, Any, Any]:
        # Try the primary model first; if it can't be loaded at all (e.g. too
        # large for available VRAM and even CPU load fails, or the repo is
        # unavailable), fall back to the configured secondary model.
        try:
            return self._load_model(self.model_id)
        except Exception:
            logger.exception("Failed to load primary model %s", self.model_id)
            if self.fallback_model_id and self.fallback_model_id != self.model_id:
                logger.warning("Falling back to secondary model %s", self.fallback_model_id)
                result = self._load_model(self.fallback_model_id)
                self.model_id = self.fallback_model_id  # report what actually loaded
                return result
            raise

    def _load_model(self, model_id: str) -> tuple[str, Any, Any]:
        from transformers import AutoProcessor, AutoModelForCausalLM

        # Gemma 4 processor handles text (+ image/audio in multimodal paths).
        # For text-only tool-calling we only use the text pathway.
        processor = AutoProcessor.from_pretrained(model_id)

        quant_bits = config.QUANTIZE_BITS
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}

        if quant_bits in (4, 8):
            try:
                from transformers import BitsAndBytesConfig
                if quant_bits == 4:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using %d-bit quantization via bitsandbytes for %s", quant_bits, model_id)
            except ImportError:
                logger.warning("bitsandbytes not installed — loading in full precision")

        # 1. CUDA (device_map="auto" handles multi-GPU if present)
        if torch.cuda.is_available():
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                return "cuda", model, processor
            except Exception:
                logger.exception("CUDA load failed for %s, trying CPU", model_id)

        # 2. Plain CPU — quantization not supported without CUDA
        if quant_bits:
            logger.warning(
                            "Quantization requires CUDA — loading %s in float32. "
                            "Expect slow inference and high RAM usage.", model_id,
                            )
        cpu_kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
        model = AutoModelForCausalLM.from_pretrained(model_id, **cpu_kwargs)
        return "cpu", model, processor

    def generate(self, chat_text: str, max_new_tokens: int = 512) -> str:
        # processor(text=...) returns a BatchEncoding; .to() moves it to device in one call.
        inputs = self.processor(text=chat_text, return_tensors="pt")
        if self.backend_name == "cuda":
            inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs,max_new_tokens=max_new_tokens,temperature=1.0,top_p=0.95,top_k=64,do_sample=True,repetition_penalty=1.05)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        decoded = self.processor.decode(new_tokens, skip_special_tokens=True)

        # Strip residual thinking blocks (defensive; shouldn't appear with enable_thinking=False)
        decoded = THINKING_RE.sub("", decoded).strip()
        logger.info("Raw model output (first 400 chars): %s", decoded[:400])
        return decoded
# ---------------------------------------------------------------------------
# Govee tool implementations (unchanged from Qwen3 version)
# ---------------------------------------------------------------------------
class GoveeClientLike(Protocol):
    def list_devices(self, force_refresh: bool = False) -> list[Device]: ...
    def get_state(self, sku: str, device_id: str) -> dict: ...
    def control(self, sku: str, device_id: str, cap_type: str, instance: str, value: Any) -> dict: ...
    def set_power(self, device: Device, on: bool) -> dict: ...
    def set_brightness(self, device: Device, percent: int) -> dict: ...
    def set_color_rgb(self, device: Device, r: int, g: int, b: int) -> dict: ...
    def set_color_temp(self, device: Device, kelvin: int) -> dict: ...
    def set_scene(self, device: Device, scene_value: int, instance: str = "lightScene") -> dict: ...

class GoveeTools:
    def __init__(self, client: GoveeClientLike):
        self.client = client

    def _find_device(self, name: str) -> Device:
        devices = self.client.list_devices()
        name_l = name.strip().lower()

        for d in devices:
            if d.device_name.lower() == name_l:
                return d

        candidates = [d for d in devices if name_l in d.device_name.lower()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(d.device_name for d in candidates)
            raise DeviceNotFoundError(f"'{name}' matches multiple devices: {names}. Be more specific.")

        sem = semantic_match.best_match(name, [d.device_name for d in devices])
        if sem:
            matched_name, score = sem
            logger.info("Semantic device match: '%s' -> '%s' (score=%.2f)", name, matched_name, score)
            return next(d for d in devices if d.device_name == matched_name)

        names = ", ".join(d.device_name for d in devices)
        raise DeviceNotFoundError(f"No device named '{name}'. Known devices: {names}")

    def list_devices(self) -> dict:
        devices = self.client.list_devices()
        out = []
        for d in devices:
            controls = sorted({c.instance for c in d.capabilities})
            out.append({"name": d.device_name, "type": d.device_type.split(".")[-1], "controls": controls})
        return {"devices": out}

    def get_device_state(self, device_name: str) -> dict:
        d = self._find_device(device_name)
        state = self.client.get_state(d.sku, d.device_id)
        return {"device": d.device_name, "state": state}

    def set_power(self, device_name: str, on: bool) -> dict:
        d = self._find_device(device_name)
        if not d.has("devices.capabilities.on_off", "powerSwitch"):
            return {"error": f"{d.device_name} doesn't support power control"}
        self.client.set_power(d, on)
        return {"ok": True, "device": d.device_name, "power": "on" if on else "off"}

    def set_power_all(
        self,
        on: bool,
        device_type: Optional[str] = None,
        name_contains: Optional[str] = None,
    ) -> dict:
        """Set power independently on every matching power-capable device."""
        type_filter = (device_type or "").strip().lower()
        name_filter = (name_contains or "").strip().lower()
        matched = [
            device for device in self.client.list_devices()
            if (not type_filter or device.device_type.rsplit(".", 1)[-1].lower() == type_filter)
            and (not name_filter or name_filter in device.device_name.lower())
        ]
        if not matched:
            filters = ", ".join(part for part in (
                f"type={device_type!r}" if type_filter else "",
                f"name_contains={name_contains!r}" if name_filter else "",
            ) if part)
            return {"error": f"No devices matched {filters or 'the requested filters'}"}

        changed: list[str] = []
        skipped: list[str] = []
        errors: list[dict[str, str]] = []
        for device in matched:
            if not device.has("devices.capabilities.on_off", "powerSwitch"):
                skipped.append(device.device_name)
                continue
            try:
                self.client.set_power(device, on)
                changed.append(device.device_name)
            except Exception as exc:  # noqa: BLE001 - one device must not abort the batch
                logger.exception("Bulk power change failed for %s", device.device_name)
                errors.append({"device": device.device_name, "error": str(exc)})
        return {"ok": bool(changed) and not errors, "power": "on" if on else "off", "changed": changed, "skipped": skipped, "errors": errors}

    def set_brightness(self, device_name: str, percent: int) -> dict:
        d = self._find_device(device_name)
        cap = d.capability("devices.capabilities.range", "brightness")
        if not cap:
            return {"error": f"{d.device_name} doesn't support brightness control"}
        if cap.value_range:
            lo, hi = cap.value_range["min"], cap.value_range["max"]
            percent = max(lo, min(hi, percent))
        self.client.set_brightness(d, percent)
        return {"ok": True, "device": d.device_name, "brightness": percent}

    def set_color_rgb(self, device_name: str, r: int, g: int, b: int) -> dict:
        d = self._find_device(device_name)
        if not d.has("devices.capabilities.color_setting", "colorRgb"):
            return {"error": f"{d.device_name} doesn't support RGB color control"}
        self.client.set_color_rgb(d, r, g, b)
        return {"ok": True, "device": d.device_name, "color": [r, g, b]}

    def set_color_temp(self, device_name: str, kelvin: int) -> dict:
        d = self._find_device(device_name)
        cap = d.capability("devices.capabilities.color_setting", "colorTemperatureK")
        if not cap:
            return {"error": f"{d.device_name} doesn't support color temperature control"}
        if cap.value_range:
            lo, hi = cap.value_range["min"], cap.value_range["max"]
            kelvin = max(lo, min(hi, kelvin))
        self.client.set_color_temp(d, kelvin)
        return {"ok": True, "device": d.device_name, "color_temp_k": kelvin}

    def set_scene(self, device_name: str, scene_name: str) -> dict:
        d = self._find_device(device_name)
        cap = d.capability("devices.capabilities.dynamic_scene", "lightScene")
        if not cap:
            return {"error": f"{d.device_name} doesn't support scenes"}
        scene_l = scene_name.strip().lower()
        match = next((o for o in cap.options if o["name"].lower() == scene_l), None)
        if not match:
            sem = semantic_match.best_match(scene_name, [o["name"] for o in cap.options], threshold=0.35)
            if sem:
                matched_name, score = sem
                logger.info("Semantic scene match: '%s' -> '%s' (score=%.2f)", scene_name, matched_name, score)
                match = next(o for o in cap.options if o["name"] == matched_name)
        if not match:
            available = ", ".join(o["name"] for o in cap.options)
            return {"error": f"Unknown scene '{scene_name}' for {d.device_name}. Available: {available}"}
        self.client.set_scene(d, match["value"])
        return {"ok": True, "device": d.device_name, "scene": match["name"]}

    def set_toggle(self, device_name: str, toggle_name: str, on: bool) -> dict:
        d = self._find_device(device_name)
        cap = d.capability("devices.capabilities.toggle", toggle_name)
        if not cap:
            available = sorted({c.instance for c in d.capabilities if c.type == "devices.capabilities.toggle"})
            return {"error": f"{d.device_name} has no toggle '{toggle_name}'. Available: {available}"}
        self.client.control(d.sku, d.device_id, "devices.capabilities.toggle", toggle_name, 1 if on else 0)
        return {"ok": True, "device": d.device_name, toggle_name: on}

    def set_fan_speed(self, device_name: str, speed: str) -> dict:
        d = self._find_device(device_name)
        cap = d.capability("devices.capabilities.work_mode", "workMode")
        if not cap:
            return {"error": f"{d.device_name} doesn't support fan speed control"}

        fields = cap.parameters.get("fields", [])
        work_mode_field = next((f for f in fields if f.get("fieldName") == "workMode"), None)
        mode_value_field = next((f for f in fields if f.get("fieldName") == "modeValue"), None)
        if not work_mode_field or not mode_value_field:
            return {"error": f"Unexpected work_mode schema for {d.device_name}"}

        wm_options = work_mode_field.get("options", [])
        gear_option = next(
            (o for o in wm_options if o.get("name", "").lower() in ("gearmode", "manual", "custom", "normal")),
            None,
        ) or next((o for o in wm_options if o.get("name", "").lower() != "auto"), None)
        if not gear_option:
            names = [o.get("name") for o in wm_options]
            return {"error": f"{d.device_name} has no manual speed mode to target (only: {names})"}

        speed_options: list[dict] = []
        if mode_value_field.get("dataType") == "INTEGER" and mode_value_field.get("range"):
            lo, hi = mode_value_field["range"]["min"], mode_value_field["range"]["max"]
            speed_options = [{"value": v} for v in range(lo, hi + 1)]
        else:
            raw_options = mode_value_field.get("options", [])
            nested = next(
                (o for o in raw_options if o.get("name", "").lower() == gear_option.get("name", "").lower()),
                None,
            )
            if nested and "options" in nested:
                speed_options = nested["options"]
            elif raw_options and all("value" in o for o in raw_options):
                speed_options = raw_options

        if not speed_options:
            return {"error": f"Couldn't determine speed levels for {d.device_name} from its capabilities"}

        speed_l = speed.strip().lower()
        match = next((o for o in speed_options if o.get("name", "").lower() == speed_l), None)
        if not match:
            by_value = sorted(speed_options, key=lambda o: o["value"])
            if speed_l in ("low", "lowest", "min", "minimum"):
                match = by_value[0]
            elif speed_l in ("high", "highest", "max", "maximum"):
                match = by_value[-1]
            elif speed_l in ("medium", "mid", "middle"):
                match = by_value[len(by_value) // 2]

        if not match:
            available = ", ".join(o.get("name", str(o["value"])) for o in speed_options)
            return {"error": f"Unknown speed '{speed}' for {d.device_name}. Available: {available}"}

        self.client.control(
            d.sku, d.device_id, "devices.capabilities.work_mode", "workMode",
            {"workMode": gear_option["value"], "modeValue": match["value"]},
        )
        return {"ok": True, "device": d.device_name, "speed": match.get("name", match["value"])}
# ---------------------------------------------------------------------------
# Weather / news / long-term memory tools
# ---------------------------------------------------------------------------
class InfoTools:
    def __init__(self,weather_client: Optional[WeatherClient] = None,news_client: Optional[NewsClient] = None,memory_store: Optional[MemoryStore] = None):
        self.weather_client = weather_client or WeatherClient()
        self.news_client = news_client or NewsClient()
        self.memory_store = memory_store or MemoryStore()

    def get_weather(self, location: Optional[str] = None) -> dict:
        try:
            forecast = self.weather_client.get_forecast(location)
        except WeatherError as e:
            return {"error": str(e)}

        summary = (
                    f"Weather in {forecast['location']}: {forecast['temperature_c']}C, "
                    f"{forecast['condition']}, humidity {forecast['humidity_pct']}%."
                )
        self.memory_store.add("weather", summary)
        return forecast

    def get_news(self, topic: Optional[str] = None, limit: int = 5) -> dict:
        try:
            headlines = self.news_client.get_headlines(topic, limit)
        except NewsError as e:
            return {"error": str(e)}

        if not headlines:
            return {"headlines": [], "topic": topic}

        label = f" (topic: {topic})" if topic else ""
        summary = f"News{label}: " + "; ".join(h["title"] for h in headlines)
        self.memory_store.add("news", summary)
        return {"headlines": headlines, "topic": topic}

    def get_article_extract(self, link: str) -> dict:
        try:
            extract = self.news_client.get_article_extract(link)
        except NewsError as e:
            return {"error": str(e)}

        self.memory_store.add("news", f"Article extract ({link}): {extract}")
        return {"link": link, "extract": extract}

    def recall_memories(self, query: Optional[str] = None, limit: int = 5) -> dict:
        if query and query.strip():
            memories = self.memory_store.search(query.strip(), top_k=limit)
        else:
            memories = self.memory_store.recent(limit=limit)
        return {"memories": memories}

# ---------------------------------------------------------------------------
# Tool schema (unchanged — OpenAI function-calling format, understood by Gemma 4)
# ---------------------------------------------------------------------------
def build_tool_schema() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "list_devices",
            "description": (
                "List all Govee devices and what each one can be controlled for "
                "(power, brightness, color, scenes, toggles, etc). Call this first "
                "if you don't know the exact device name or its capabilities."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }},
        {"type": "function", "function": {
            "name": "get_device_state",
            "description": "Get the current state (power, brightness, color, online status, etc) of one device.",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string", "description": "Device name, e.g. 'Bedroom Light'"},
            }, "required": ["device_name"]},
        }},
        {"type": "function", "function": {
            "name": "set_power",
            "description": "Turn a single device on or off.",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "on": {"type": "boolean"},
            }, "required": ["device_name", "on"]},
        }},
        {"type": "function", "function": {
            "name": "set_power_all",
            "description": "Turn every matching power-capable device on or off in one call. Use for requests about all devices, all lights, or a room.",
            "parameters": {"type": "object", "properties": {
                "on": {"type": "boolean"},
                "device_type": {"type": "string", "description": "Optional type filter, e.g. light, fan, socket."},
                "name_contains": {"type": "string", "description": "Optional case-insensitive name/room filter, e.g. bedroom."},
            }, "required": ["on"]},
        }},
        {"type": "function", "function": {
            "name": "set_brightness",
            "description": "Set a light's brightness as a percentage (1-100).",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "percent": {"type": "integer", "minimum": 1, "maximum": 100},
            }, "required": ["device_name", "percent"]},
        }},
        {"type": "function", "function": {
            "name": "set_color_rgb",
            "description": "Set a light's color using RGB values (0-255 each).",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "r": {"type": "integer", "minimum": 0, "maximum": 255},
                "g": {"type": "integer", "minimum": 0, "maximum": 255},
                "b": {"type": "integer", "minimum": 0, "maximum": 255},
            }, "required": ["device_name", "r", "g", "b"]},
        }},
        {"type": "function", "function": {
            "name": "set_color_temp",
            "description": "Set a light's white color temperature in Kelvin (roughly 2000=warm to 9000=cool).",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "kelvin": {"type": "integer", "minimum": 3000, "maximum": 10000},
            }, "required": ["device_name", "kelvin"]},
        }},
        {"type": "function", "function": {
            "name": "set_scene",
            "description": (
                "Activate a preset light scene by name (e.g. 'Christmas', 'Party', 'Sunrise'). "
                "Call get_device_state first if unsure which scenes a device supports."
            ),
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "scene_name": {"type": "string"},
            }, "required": ["device_name", "scene_name"]},
        }},
        {"type": "function", "function": {
            "name": "set_toggle",
            "description": "Turn a named toggle feature on/off, e.g. 'gradientToggle' or 'oscillationToggle'.",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "toggle_name": {"type": "string"},
                "on": {"type": "boolean"},
            }, "required": ["device_name", "toggle_name", "on"]},
        }},
        {"type": "function", "function": {
            "name": "set_fan_speed",
            "description": "Set a fan's speed gear: 'low', 'medium', or 'high'.",
            "parameters": {"type": "object", "properties": {
                "device_name": {"type": "string"},
                "speed": {"type": "string", "enum": ["low", "medium", "high"]},
            }, "required": ["device_name", "speed"]},
        }},
        {"type": "function", "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather and a short forecast for a location. "
                "If the user doesn't name a location, omit it to use the "
                "configured default location."
            ),
            "parameters": {"type": "object", "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Bratislava' or 'Paris, France'"},
            }, "required": []},
        }},
        {"type": "function", "function": {
            "name": "get_news",
            "description": (
                "Get recent news headlines, optionally filtered by topic. "
                "Each headline includes a short RSS summary and a 'link'. "
                "Omit topic for general top headlines. Call get_article_extract "
                "with a headline's link if the user wants more than the summary."
            ),
            "parameters": {"type": "object", "properties": {
                "topic": {"type": "string", "description": "Topic or search query, e.g. 'technology' or 'climate change'"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max number of headlines (default 5)"},
            }, "required": []},
        }},
        {"type": "function", "function": {
            "name": "get_article_extract",
            "description": (
                "Fetch and read the main body text of a specific news article, "
                "using the 'link' field from a get_news result. Use this when "
                "the user wants details, not just the headline/summary."
            ),
            "parameters": {"type": "object", "properties": {
                "link": {"type": "string", "description": "The article's link, taken from a get_news result"},
            }, "required": ["link"]},
        }},
        {"type": "function", "function": {
            "name": "recall_memories",
            "description": (
                "Recall things said in earlier conversations or previous "
                "weather/news lookups. Pass a query describing what to look "
                "for (semantic search), or omit it to get the most recent "
                "memories."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "What to search for, e.g. 'weather in Paris' or 'news about elections'"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max number of memories to return (default 5)"},
            }, "required": []},
        }},
    ]

# ---------------------------------------------------------------------------
# Agent: chat loop with tool-call parsing
# ---------------------------------------------------------------------------
class GoveeAgent:
    def __init__(self,client: GoveeClientLike,backend: Optional[ModelBackend] = None,max_tool_iters: int = 5,info_tools: Optional[InfoTools] = None):
        self.tools_impl = GoveeTools(client)
        self.info_tools = info_tools or InfoTools()
        self.backend = backend or ModelBackend()
        self.tool_schema = build_tool_schema()
        self.max_tool_iters = max_tool_iters
        self._dispatch: dict[str, Callable[..., dict]] = {
                                                            "list_devices":    self.tools_impl.list_devices,
                                                            "get_device_state": self.tools_impl.get_device_state,
                                                            "set_power":       self.tools_impl.set_power,
                                                            "set_power_all":   self.tools_impl.set_power_all,
                                                            "set_brightness":  self.tools_impl.set_brightness,
                                                            "set_color_rgb":   self.tools_impl.set_color_rgb,
                                                            "set_color_temp":  self.tools_impl.set_color_temp,
                                                            "set_scene":       self.tools_impl.set_scene,
                                                            "set_toggle":      self.tools_impl.set_toggle,
                                                            "set_fan_speed":   self.tools_impl.set_fan_speed,
                                                            "get_weather":     self.info_tools.get_weather,
                                                            "get_news":        self.info_tools.get_news,
                                                            "get_article_extract": self.info_tools.get_article_extract,
                                                            "recall_memories": self.info_tools.recall_memories,
                                                            }

    def _call_tool(self, name: str, arguments: dict) -> dict:
        fn = self._dispatch.get(name)
        if not fn:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return fn(**arguments)
        except (DeviceNotFoundError, GoveeAPIError) as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool '%s' raised an unexpected error", name)
            return {"error": f"{name} failed unexpectedly: {e}"}

    def chat(self, user_message: str, history: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
        history = history or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]

        final_reply: Optional[str] = None
        for _ in range(self.max_tool_iters):
            chat_text = self._render(messages)
            reply = self.backend.generate(chat_text)
            calls = parse_tool_calls(reply)

            if not calls:
                final_reply = reply
                break

            logger.info("Parsed %d tool call(s): %s", len(calls), [c[0] for c in calls])
            messages.append({"role": "assistant", "content": reply})

            result_lines = []
            for name, args in calls:
                result = self._call_tool(name, args) if isinstance(args, dict) \
                    else {"error": "Malformed tool call arguments from model"}
                logger.info("Tool %s(%s) -> %s", name, args, result)
                result_lines.append(f"{name} result: {json.dumps(result)}")
            messages.append({
                                "role": "user",
                                "content": (
                                            "Tool results:\n" + "\n".join(result_lines) +
                                            "\n\nIf the task is now complete, reply to the user in plain "
                                            "language describing what changed. If more actions are still "
                                            "needed (e.g. turning on each device from a list), emit the "
                                            "next call now."
                                            ),
                            })

        if final_reply is None:
            final_reply = "I wasn't able to finish that - could you rephrase or be more specific?"

        try:
            self.info_tools.memory_store.add("chat", f"User: {user_message}\nAssistant: {final_reply}")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write chat turn to long-term memory")
        new_history = history + [{"role": "user", "content": user_message},{"role": "assistant", "content": final_reply}]
        return final_reply, new_history

    def _render(self, messages: list[dict]) -> str:
        proc = self.backend.processor
        try:
            return proc.apply_chat_template(messages, tools=self.tool_schema,add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return proc.apply_chat_template(messages, tools=self.tool_schema,add_generation_prompt=True, tokenize=False)


# ---------------------------------------------------------------------------
# Optional writer--critic loop
# ---------------------------------------------------------------------------
class CritiqueAgent:
    """Tool-free final-answer reviewer backed by the already-loaded local LLM.

    It deliberately runs *after* the writer has finished all tool calls.  A
    critic must not be allowed to repeat commands with external side effects.
    """

    REVIEW_SYSTEM_PROMPT = (
        "You are CriticAgent in a bounded writer-critic loop for a local smart-home "
        "assistant. Review the draft answer for accuracy, completeness, clarity, and "
        "safety. Never assume a device action or a fact happened unless the draft says "
        "so. Never suggest tool calls. If there is no material issue, reply with exactly "
        "APPROVE. Otherwise give concise, actionable feedback only."
    )
    REVISE_SYSTEM_PROMPT = (
        "You are WriterAgent revising a completed smart-home assistant response. Return "
        "only the final answer for the user. Apply useful critic feedback, while preserving "
        "all facts from the original draft. Do not invent tool results, device actions, or "
        "new information, and do not mention the critic or this revision process."
    )

    def __init__(self, backend: ModelBackend, max_passes: int = 1):
        self.backend = backend
        self.max_passes = max(0, max_passes)

    def _render(self, messages: list[dict]) -> str:
        try:
            return self.backend.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return self.backend.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # A small compatibility fallback for processors with an unusual
            # chat template. ModelBackend still handles generation normally.
            return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"

    def refine(self, user_message: str, draft: str) -> str:
        current = draft
        for _ in range(self.max_passes):
            feedback = self.backend.generate(self._render([
                {"role": "system", "content": self.REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"User request:\n{user_message}\n\nDraft answer:\n{current}"},
            ]), max_new_tokens=256).strip()
            if feedback.upper().startswith("APPROVE"):
                break

            revised = self.backend.generate(self._render([
                {"role": "system", "content": self.REVISE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"User request:\n{user_message}\n\nOriginal draft:\n{current}"
                    f"\n\nCritic feedback:\n{feedback}"
                )},
            ]), max_new_tokens=512).strip()
            if not revised:
                logger.warning("Critic revision was empty; retaining the original draft")
                break
            current = revised
        return current


class WriterCriticAgent:
    """Drop-in wrapper that adds a tool-free critique pass to any agent mode."""

    def __init__(self, writer: Any, critic: Optional[CritiqueAgent] = None):
        self.writer = writer
        self.backend = writer.backend
        self.critic = critic or CritiqueAgent(self.backend, config.GOVEE_CRITIQUE_MAX_PASSES)

    def chat(self, user_message: str, history: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
        draft, new_history = self.writer.chat(user_message, history)
        try:
            reply = self.critic.refine(user_message, draft)
        except Exception:  # noqa: BLE001
            logger.exception("Critique pass failed; returning the writer draft")
            return draft, new_history

        if reply != draft and new_history and new_history[-1].get("role") == "assistant":
            new_history = [*new_history[:-1], {"role": "assistant", "content": reply}]
        return reply, new_history

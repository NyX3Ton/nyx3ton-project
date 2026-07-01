#agent.py

from __future__ import annotations

import json, logging, re, torch
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import semantic_match
from govee_client import Device, GoveeAPIError

logger = logging.getLogger("agent")

MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507"
OV_CACHE_DIR = Path("ov_cache")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SYSTEM_PROMPT = (
                    "You are a home assistant that controls Govee smart home devices through "
                    "tool calls. Always call list_devices or get_device_state first if you're "
                    "not sure a device exists or what it supports, rather than guessing a "
                    "device name or capability. Keep replies short and state what changed."
                )

class DeviceNotFoundError(Exception):
    pass

# ---------------------------------------------------------------------------
# Backend loading: CUDA -> OpenVINO -> CPU
# ---------------------------------------------------------------------------

class ModelBackend:
    def __init__(self, model_id: str = MODEL_ID, int8: bool = False):
        self.model_id = model_id
        self.int8 = int8
        self.backend_name, self.model, self.tokenizer = self._load()
        logger.info("Loaded %s on backend=%s", model_id, self.backend_name)

    def _load(self) -> tuple[str, Any, Any]:
        # 1. CUDA
        try:
            if torch.cuda.is_available():
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto", device_map="cuda")
                return "cuda", model, tokenizer
        except Exception:
            logger.exception("CUDA load failed, falling back to OpenVINO")

        # 2. OpenVINO
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer

            OV_CACHE_DIR.mkdir(exist_ok=True)
            model_cache = OV_CACHE_DIR / self.model_id.replace("/", "__")

            ov_kwargs: dict[str, Any] = {}
            if self.int8:
                ov_kwargs["quantization_config"] = {"bits": 8}

            if model_cache.exists():
                tokenizer = AutoTokenizer.from_pretrained(model_cache)
                model = OVModelForCausalLM.from_pretrained(model_cache, **ov_kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = OVModelForCausalLM.from_pretrained(self.model_id, export=True, **ov_kwargs)
                model.save_pretrained(model_cache)
                tokenizer.save_pretrained(model_cache)
            return "openvino", model, tokenizer
        except Exception:
            logger.exception("OpenVINO load failed, falling back to CPU")

        # 3. Plain CPU
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="float32")
        return "cpu", model, tokenizer

    def generate(self, chat_text: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(chat_text, return_tensors="pt")
        if self.backend_name == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs,max_new_tokens=max_new_tokens,temperature=0.7,top_p=0.8,top_k=20,do_sample=True,pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Govee tool implementations (bridges GoveeClient <-> LLM-callable functions)
# ---------------------------------------------------------------------------

class GoveeClientLike(Protocol):
    """Structural type for what GoveeTools/GoveeAgent actually call on the client,
    so test doubles (FakeGoveeClient) type-check without inheriting from the real
    (network-backed) GoveeClient."""
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

        # Resolve the list of selectable speed levels for that gear mode.
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
# Tool schema exposed to the LLM (OpenAI-style function schema)
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
                                            "description": "Turn a device on or off.",
                                            "parameters": {"type": "object", "properties": {
                                                                                            "device_name": {"type": "string"},
                                                                                            "on": {"type": "boolean"},
                                                                                            }, "required": ["device_name", "on"]
                                                        },
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
            ]

# ---------------------------------------------------------------------------
# Agent: chat loop with tool-call parsing
# ---------------------------------------------------------------------------

class GoveeAgent:
    def __init__(self, client: GoveeClientLike, backend: Optional[ModelBackend] = None, max_tool_iters: int = 5):
        self.tools_impl = GoveeTools(client)
        self.backend = backend or ModelBackend()
        self.tool_schema = build_tool_schema()
        self.max_tool_iters = max_tool_iters
        self._dispatch: dict[str, Callable[..., dict]] = {
                                                            "list_devices": self.tools_impl.list_devices,
                                                            "get_device_state": self.tools_impl.get_device_state,
                                                            "set_power": self.tools_impl.set_power,
                                                            "set_brightness": self.tools_impl.set_brightness,
                                                            "set_color_rgb": self.tools_impl.set_color_rgb,
                                                            "set_color_temp": self.tools_impl.set_color_temp,
                                                            "set_scene": self.tools_impl.set_scene,
                                                            "set_toggle": self.tools_impl.set_toggle,
                                                            "set_fan_speed": self.tools_impl.set_fan_speed,
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
        except Exception as e:  # noqa: BLE001 - a tool bug must not crash the chat loop
            logger.exception("Tool '%s' raised an unexpected error", name)
            return {"error": f"{name} failed unexpectedly: {e}"}

    def chat(self, user_message: str, history: Optional[list[dict]] = None) -> tuple[str, list[dict]]:

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or []) + [{"role": "user", "content": user_message}]

        for _ in range(self.max_tool_iters):
            chat_text = self.backend.tokenizer.apply_chat_template(messages, tools=self.tool_schema, add_generation_prompt=True, tokenize=False)
            reply = self.backend.generate(chat_text)
            calls = TOOL_CALL_RE.findall(reply)

            if not calls:
                messages.append({"role": "assistant", "content": reply})
                return reply, messages[1:]

            messages.append({"role": "assistant", "content": reply})
            for raw_call in calls:
                try:
                    call = json.loads(raw_call)
                    name, args = call["name"], call.get("arguments", {})
                except (json.JSONDecodeError, KeyError):
                    name, result = "unknown", {"error": "Malformed tool call from model"}
                else:
                    result = self._call_tool(name, args)
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

        return ("I ran out of tool-call steps trying to complete that - could you simplify the request?",messages[1:])

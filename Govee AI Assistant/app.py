#app.py

from __future__ import annotations

import logging, os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from dotenv import load_dotenv
from typing import Protocol

from agent import GoveeAgent
from govee_client import Device, GoveeAPIError, GoveeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

load_dotenv()

DEVICE_ICONS = {
                "light": "\U0001F4A1",
                "fan": "\U0001F300",
                "thermometer": "\U0001F321\uFE0F",
                "socket": "\U0001F50C",
                "humidifier": "\U0001F4A7",
                "air_purifier": "\U0001FAE7",
                "heater": "\U0001F525",
                }


def _icon_for(device_type: str) -> str:
    key = device_type.split(".")[-1]
    return DEVICE_ICONS.get(key, "\U0001F3E0")


def _format_state(state: dict) -> str:
    if not state.get("online", True):
        return "\u26AB offline (last known state may be stale)"

    parts = []
    if "powerSwitch" in state:
        parts.append("\U0001F7E2 on" if state["powerSwitch"] else "\u26AA off")
    if state.get("brightness") not in (None, ""):
        parts.append(f"brightness {state['brightness']}%")
    if state.get("colorTemperatureK") not in (None, ""):
        parts.append(f"{state['colorTemperatureK']}K")
    if "sensorTemperature" in state:
        parts.append(f"{state['sensorTemperature']}\u00b0C")
    if "sensorHumidity" in state:
        parts.append(f"{state['sensorHumidity']}% humidity")
    if "oscillationToggle" in state:
        parts.append("oscillating" if state["oscillationToggle"] else "still")
    return " \u00b7 ".join(parts) if parts else "no readable state"

# structural types for what build_ui actually calls, so test doubles (FakeGoveeClient,
# DummyAgent) type-check without inheriting from the real (network/model-backed) classes
class _AgentBackend(Protocol):
    @property
    def backend_name(self) -> str: ...

class AgentLike(Protocol):
    @property
    def backend(self) -> _AgentBackend: ...
    def chat(self, user_message: str, history: list[dict] | None, /) -> tuple[str, list[dict]]: ...

class ClientLike(Protocol):
    def list_devices(self, force_refresh: bool = False) -> list[Device]: ...
    def get_state(self, sku: str, device_id: str) -> dict: ...
    def set_power(self, device: Device, on: bool) -> dict: ...
    def set_brightness(self, device: Device, percent: int) -> dict: ...

def build_ui(client: ClientLike, agent: AgentLike) -> gr.Blocks:
    devices: list[Device] = client.list_devices()

    with gr.Blocks(title="Govee Home Dashboard") as demo:
        gr.Markdown("# \U0001F3E0 Govee Home Dashboard")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Devices")
                refresh_btn = gr.Button("\U0001F504 Refresh")

                state_boxes: dict[str, gr.Markdown] = {}
                power_buttons: dict[str, gr.Button] = {}
                brightness_sliders: dict[str, gr.Slider] = {}

                for d in devices:
                    with gr.Group():
                        gr.Markdown(f"**{_icon_for(d.device_type)} {d.device_name}**")
                        state_boxes[d.device_id] = gr.Markdown("Loading")

                        with gr.Row():
                            if d.has("devices.capabilities.on_off", "powerSwitch"):
                                power_buttons[d.device_id] = gr.Button("Toggle power", size="sm")
                            if d.has("devices.capabilities.range", "brightness"):
                                cap = d.capability("devices.capabilities.range", "brightness")
                                if cap is not None:
                                    lo = cap.value_range["min"] if cap.value_range else 1
                                    hi = cap.value_range["max"] if cap.value_range else 100
                                    brightness_sliders[d.device_id] = gr.Slider(lo, hi, value=lo, step=1, label="Brightness")

            with gr.Column(scale=1):
                gr.Markdown(f"### Assistant (backend: `{agent.backend.backend_name}`)")
                chatbot = gr.Chatbot(height=500)
                msg_box = gr.Textbox(placeholder="Ask me to check or control a device...", show_label=False)
                agent_history = gr.State([])  # tool-call-aware history fed back into agent.chat

        def refresh_all():
            results = []
            for d in devices:
                try:
                    state = client.get_state(d.sku, d.device_id)
                    results.append(_format_state(state))
                except GoveeAPIError as e:
                    results.append(f"\u26A0\uFE0F {e}")
            return results

        def toggle_power(device_id: str):
            d = next(dd for dd in devices if dd.device_id == device_id)
            try:
                state = client.get_state(d.sku, d.device_id)
                client.set_power(d, not bool(state.get("powerSwitch")))
            except GoveeAPIError as e:
                logger.warning("toggle_power failed for %s: %s", d.device_name, e)
            return refresh_all()

        def set_brightness_direct(device_id: str, value: float):
            d = next(dd for dd in devices if dd.device_id == device_id)
            try:
                client.set_brightness(d, int(value))
            except GoveeAPIError as e:
                logger.warning("set_brightness failed for %s: %s", d.device_name, e)
            return refresh_all()

        def respond(user_message: str, chat_display: list, history: list):
            if not user_message.strip():
                return chat_display, history, ""
            reply, history = agent.chat(user_message, history)
            chat_display = chat_display + [{"role": "user", "content": user_message},{"role": "assistant", "content": reply}]
            return chat_display, history, ""

        all_state_outputs = [state_boxes[d.device_id] for d in devices]

        refresh_btn.click(refresh_all, outputs=all_state_outputs)
        demo.load(refresh_all, outputs=all_state_outputs)

        for device_id, btn in power_buttons.items():
            btn.click(lambda did=device_id: toggle_power(did), outputs=all_state_outputs)

        for device_id, slider in brightness_sliders.items():
            slider.release(
                            lambda value, did=device_id: set_brightness_direct(did, value),
                            inputs=slider,
                            outputs=all_state_outputs,
                            )

        msg_box.submit(respond,inputs=[msg_box, chatbot, agent_history],outputs=[chatbot, agent_history, msg_box]).then(refresh_all, outputs=all_state_outputs)

    return demo

def main():
    client = GoveeClient()
    logger.info("Loading local LLM agent (first run can take a while)")
    agent = GoveeAgent(client)
    demo = build_ui(client, agent)
    demo.launch()

if __name__ == "__main__":
    main()

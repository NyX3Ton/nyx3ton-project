#app.py

from __future__ import annotations

import html, logging
from typing import Protocol
from govee_assistant import config

import gradio as gr
from gradio.themes import Soft as SoftTheme, GoogleFont
from gradio.themes.utils import colors as theme_colors, sizes as theme_sizes

from govee_assistant import speech_to_text
from govee_assistant.agent import GoveeAgent
from govee_assistant.govee_client import Device, GoveeAPIError, GoveeClient

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet per-request HTTP noise
logger = logging.getLogger("app")

DEVICE_ICONS = {
                "light": "\U0001F4A1",
                "fan": "\U0001F300",
                "thermometer": "\U0001F321️",
                "socket": "\U0001F50C",
                "humidifier": "\U0001F4A7",
                "air_purifier": "\U0001FAE7",
                "heater": "\U0001F525",
                }

def _icon_for(device_type: str) -> str:
    key = device_type.split(".")[-1]
    return DEVICE_ICONS.get(key, "\U0001F3E0")

def _pill(text: str, kind: str) -> str:
    return f'<span class="gv-pill {kind}">{html.escape(text)}</span>'

def _chip(text: str) -> str:
    return f'<span class="gv-chip">{html.escape(text)}</span>'

def _device_status_html(state: dict) -> str:
    if not state.get("online", True):
        return _pill("Offline", "offline") + '<span class="gv-note">last known state may be stale</span>'

    parts: list[str] = []
    if "powerSwitch" in state:
        parts.append(_pill("On", "on") if state["powerSwitch"] else _pill("Off", "off"))

    chips: list[str] = []
    if state.get("brightness") not in (None, ""):
        chips.append(f"{state['brightness']}% brightness")
    if state.get("colorTemperatureK") not in (None, ""):
        chips.append(f"{state['colorTemperatureK']}K")
    if "sensorTemperature" in state:
        chips.append(f"{state['sensorTemperature']}°C")
    if "sensorHumidity" in state:
        chips.append(f"{state['sensorHumidity']}% humidity")
    if "oscillationToggle" in state:
        chips.append("oscillating" if state["oscillationToggle"] else "still")

    body = "".join(parts) + "".join(_chip(c) for c in chips)
    return body or '<span class="gv-note">no readable state</span>'

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

THEME = SoftTheme(
                    primary_hue=theme_colors.indigo,
                    neutral_hue=theme_colors.slate,
                    font=[GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
                    radius_size=theme_sizes.radius_lg,
                )

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

#gv-header {
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    padding:6px 4px 16px; margin-bottom:10px;
    border-bottom:1px solid var(--border-color-primary);
}
.gv-brand { display:flex; align-items:center; gap:13px; }
.gv-logo {
    font-size:1.5rem; width:46px; height:46px; flex:0 0 auto;
    display:flex; align-items:center; justify-content:center;
    background:var(--background-fill-secondary);
    border-radius: var(--radius-lg, 12px);
}
.gv-title { font-size:1.25rem; font-weight:600; margin:0; }
.gv-subtitle { font-size:0.85rem; color:var(--body-text-color-subdued); margin:0; }

.gv-backend-badge {
    font-size:0.8rem; padding:4px 10px; border-radius:999px;
    background:var(--background-fill-secondary);
    border:1px solid var(--border-color-primary);
    white-space:nowrap;
}

#gv-devices-header { display:flex; align-items:center; gap:8px; margin-bottom:2px; }
.gv-device-count { font-size:0.8rem; color:var(--body-text-color-subdued); }

.gv-device-card {
    border:1px solid var(--border-color-primary);
    border-radius: var(--radius-lg, 12px);
    padding:12px 14px;
    margin-bottom:10px;
    background:var(--background-fill-primary);
    transition: box-shadow 0.15s ease;
}
.gv-device-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.gv-device-name { font-weight:600; display:flex; align-items:center; gap:8px; margin-bottom:6px; }

.gv-pill {
    display:inline-block; font-size:0.75rem; font-weight:600;
    padding:2px 10px; border-radius:999px; margin-right:6px;
}
.gv-pill.on { background:rgba(22,163,74,0.15); color:#16a34a; }
.gv-pill.off { background:var(--background-fill-secondary); color:var(--body-text-color-subdued); }
.gv-pill.offline { background:rgba(220,38,38,0.15); color:#dc2626; }

.gv-chip {
    display:inline-block; font-size:0.75rem;
    padding:2px 8px; border-radius:999px; margin-right:6px;
    background:var(--background-fill-secondary);
    color:var(--body-text-color-subdued);
}
.gv-note { font-size:0.75rem; color:var(--body-text-color-subdued); margin-left:4px; }
"""

def build_ui(client: ClientLike, agent: AgentLike) -> gr.Blocks:
    devices: list[Device] = client.list_devices()

    with gr.Blocks(title="Govee Home Dashboard") as demo:
        gr.HTML(
            '<div id="gv-header">'
            '<div class="gv-brand">'
            '<div class="gv-logo">\U0001F3E0</div>'
            '<div>'
            '<p class="gv-title">Govee Home Dashboard</p>'
            '<p class="gv-subtitle">Devices &amp; AI assistant</p>'
            '</div></div>'
            f'<span class="gv-backend-badge">backend &middot; {html.escape(agent.backend.backend_name)}</span>'
            '</div>'
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_id="gv-devices-header"):
                    gr.Markdown(f"### Devices <span class='gv-device-count'>({len(devices)})</span>")
                    refresh_btn = gr.Button("\U0001F504 Refresh", size="sm", scale=0, min_width=110)
                show_offline = gr.Checkbox(label="Show offline devices", value=True)

                state_boxes: dict[str, gr.HTML] = {}
                power_buttons: dict[str, gr.Button] = {}
                brightness_sliders: dict[str, gr.Slider] = {}
                device_cards: dict[str, gr.Group] = {}

                for d in devices:
                    with gr.Group(elem_classes=["gv-device-card"]) as card:
                        device_cards[d.device_id] = card
                        gr.HTML(f'<div class="gv-device-name">{_icon_for(d.device_type)} {html.escape(d.device_name)}</div>')
                        state_boxes[d.device_id] = gr.HTML("Loading&hellip;")

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
                with gr.Row():
                    gr.Markdown("### Assistant")
                    expand_btn = gr.Button("⤢ Expand", size="sm", scale=0, min_width=110)
                chatbot = gr.Chatbot(height=500)
                with gr.Row():
                    msg_box = gr.Textbox(placeholder="Ask me to check or control a device, or use the mic ->", show_label=False, scale=4)
                    mic = gr.Audio(sources=["microphone"], type="filepath", show_label=False, scale=1)
                agent_history = gr.State([])  # tool-call-aware history fed back into agent.chat
                chat_expanded = gr.State(False)  # tracks expand/collapse for the chat window
                online_status = gr.State({})  # device_id -> bool, refreshed alongside device state

        def _card_visibility(show_offline_value: bool, online_map: dict):
            return [gr.update(visible=(show_offline_value or online_map.get(d.device_id, True))) for d in devices]

        def refresh_all(show_offline_value: bool):
            html_updates = []
            online_map: dict[str, bool] = {}
            for d in devices:
                try:
                    state = client.get_state(d.sku, d.device_id)
                    online_map[d.device_id] = bool(state.get("online", True))
                    html_updates.append(_device_status_html(state))
                except GoveeAPIError as e:
                    online_map[d.device_id] = True  # unknown status on a transient error - don't hide it
                    html_updates.append(_pill("Error", "offline") + f'<span class="gv-note">{html.escape(str(e))}</span>')
            return html_updates + _card_visibility(show_offline_value, online_map) + [online_map]

        def apply_offline_filter(show_offline_value: bool, online_map: dict):
            return _card_visibility(show_offline_value, online_map or {})

        def toggle_power(device_id: str, show_offline_value: bool):
            d = next(dd for dd in devices if dd.device_id == device_id)
            try:
                state = client.get_state(d.sku, d.device_id)
                client.set_power(d, not bool(state.get("powerSwitch")))
            except GoveeAPIError as e:
                logger.warning("toggle_power failed for %s: %s", d.device_name, e)
            return refresh_all(show_offline_value)

        def set_brightness_direct(device_id: str, value: float, show_offline_value: bool):
            d = next(dd for dd in devices if dd.device_id == device_id)
            try:
                client.set_brightness(d, int(value))
            except GoveeAPIError as e:
                logger.warning("set_brightness failed for %s: %s", d.device_name, e)
            return refresh_all(show_offline_value)

        def respond(user_message: str, chat_display: list, history: list):
            if not user_message.strip():
                return chat_display, history, ""
            reply, history = agent.chat(user_message, history)
            chat_display = chat_display + [{"role": "user", "content": user_message},{"role": "assistant", "content": reply}]
            return chat_display, history, ""

        def toggle_chat_size(expanded: bool):
            new_expanded = not expanded
            height = 820 if new_expanded else 500
            label = "⤡ Collapse" if new_expanded else "⤢ Expand"
            return gr.update(height=height), new_expanded, gr.update(value=label)

        def transcribe_audio(audio_path: str | None):
            text = speech_to_text.transcribe(audio_path)
            if not text:
                logger.warning("Speech-to-text produced no usable transcription")
            return text

        all_state_outputs = [state_boxes[d.device_id] for d in devices]
        all_card_outputs = [device_cards[d.device_id] for d in devices]
        refresh_outputs = all_state_outputs + all_card_outputs + [online_status]

        refresh_btn.click(refresh_all, inputs=[show_offline], outputs=refresh_outputs)
        demo.load(refresh_all, inputs=[show_offline], outputs=refresh_outputs)
        show_offline.change(apply_offline_filter, inputs=[show_offline, online_status], outputs=all_card_outputs)

        for device_id, btn in power_buttons.items():
            btn.click(
                        lambda so, did=device_id: toggle_power(did, so),
                        inputs=[show_offline],
                        outputs=refresh_outputs,
                    )

        for device_id, slider in brightness_sliders.items():
            slider.release(lambda value, so, did=device_id: set_brightness_direct(did, value, so),inputs=[slider, show_offline],outputs=refresh_outputs)

        msg_box.submit(respond,inputs=[msg_box, chatbot, agent_history],outputs=[chatbot, agent_history, msg_box]).then(refresh_all, inputs=[show_offline], outputs=refresh_outputs)

        expand_btn.click(toggle_chat_size, inputs=[chat_expanded], outputs=[chatbot, chat_expanded, expand_btn])

        mic.stop_recording(transcribe_audio, inputs=mic, outputs=msg_box).then(lambda: None, outputs=mic)

    return demo

def _build_agent(client):
    # GOVEE_AGENT_MODE=workflow opts into the LlamaIndex multi-agent orchestrator;
    # anything else (default "single") uses the built-in tool-calling loop.
    if config.GOVEE_AGENT_MODE == "workflow":
        from govee_assistant.orchestrator import OrchestratedAgent
        logger.info("Agent mode: workflow (LlamaIndex multi-agent orchestration)")
        agent = OrchestratedAgent(client)
    else:
        logger.info("Agent mode: single (built-in tool-calling loop)")
        agent = GoveeAgent(client)
    if config.GOVEE_CRITIQUE_ENABLED:
        from govee_assistant.agent import WriterCriticAgent
        logger.info("Writer-critic refinement: enabled (%d pass(es))", config.GOVEE_CRITIQUE_MAX_PASSES)
        agent = WriterCriticAgent(agent)
    return agent

def main():
    client = GoveeClient()
    logger.info("Loading local LLM agent (first run can take a while)")
    agent = _build_agent(client)
    demo = build_ui(client, agent)
    demo.launch(theme=THEME,css=CSS,server_name=config.GRADIO_SERVER_NAME,server_port=config.GRADIO_SERVER_PORT)

if __name__ == "__main__":
    main()

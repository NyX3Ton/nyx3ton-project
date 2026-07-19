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
.gradio-container { max-width: 1440px !important; padding: 22px 28px 36px !important; }
#gv-shell { min-height: calc(100vh - 96px); }

#gv-header { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:0 0 22px; border-bottom:1px solid var(--border-color-primary); }
.gv-brand { display:flex; align-items:center; gap:12px; min-width:0; }
.gv-logo { width:40px; height:40px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:#172554; color:#dbeafe; font-size:1.2rem; font-weight:700; box-shadow:inset 0 0 0 1px rgba(255,255,255,.12); }
.gv-title { font-size:1.05rem; line-height:1.25; font-weight:700; letter-spacing:-.015em; margin:0; }
.gv-subtitle { font-size:.78rem; color:var(--body-text-color-subdued); margin:2px 0 0; }
.gv-header-meta { display:flex; align-items:center; justify-content:flex-end; gap:8px; flex-wrap:wrap; }
.gv-backend-badge, .gv-header-chip { display:inline-flex; align-items:center; gap:6px; font-size:.74rem; line-height:1; padding:7px 9px; border-radius:6px; background:var(--background-fill-secondary); border:1px solid var(--border-color-primary); white-space:nowrap; }
.gv-live-dot { width:6px; height:6px; border-radius:99px; background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.12); }

#gv-main { gap:20px; margin-top:22px; align-items:stretch; }
.gv-panel { border:1px solid var(--border-color-primary); border-radius:12px; background:var(--background-fill-primary); box-shadow:0 1px 2px rgba(15,23,42,.04); overflow:hidden; }
.gv-panel-inner { padding:16px; }
.gv-panel-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:14px; }
.gv-panel-title { font-size:.88rem; font-weight:700; letter-spacing:.01em; margin:0; }
.gv-panel-copy { font-size:.76rem; color:var(--body-text-color-subdued); margin:3px 0 0; }

#gv-device-toolbar { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 0 12px; border-bottom:1px solid var(--border-color-primary); margin-bottom:10px; }
#gv-device-toolbar .wrap { display:flex; align-items:center; gap:8px; }
#gv-device-controls { flex-wrap:nowrap !important; gap:7px !important; }
#gv-device-controls .gr-button { white-space:nowrap; }
.gv-device-count { color:var(--body-text-color-subdued); font-size:.76rem; font-weight:500; }
.gv-device-card { border-bottom:1px solid var(--border-color-primary); padding:13px 0; background:transparent; }
.gv-device-card:last-child { border-bottom:0; }
.gv-device-card:hover { background:var(--background-fill-secondary); margin-left:-8px; margin-right:-8px; padding-left:8px; padding-right:8px; border-radius:8px; }
.gv-device-name { font-weight:650; font-size:.86rem; display:flex; align-items:center; gap:8px; margin:0 0 7px; }
.gv-device-name .gv-icon { width:21px; text-align:center; opacity:.9; }

.gv-pill { display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.01em; padding:3px 7px; border-radius:5px; margin-right:5px; }
.gv-pill.on { background:rgba(22,163,74,.13); color:#15803d; }
.gv-pill.off { background:var(--background-fill-secondary); color:var(--body-text-color-subdued); }
.gv-pill.offline { background:rgba(220,38,38,.12); color:#b91c1c; }
.gv-chip { display:inline-block; font-size:.7rem; padding:3px 7px; border-radius:5px; margin-right:5px; background:var(--background-fill-secondary); color:var(--body-text-color-subdued); }
.gv-note { font-size:.7rem; color:var(--body-text-color-subdued); margin-left:3px; }

.gv-device-card .gr-button, .gv-actions .gr-button { border-radius:7px !important; font-size:.74rem !important; min-height:31px !important; }
.gv-device-card .gr-slider { margin:1px 0 0 !important; }

#gv-chat-panel { display:flex; flex-direction:column; min-height:680px; }
#gv-chat-panel .gv-panel-inner { display:flex; flex-direction:column; height:100%; }
#gv-chat { flex:1; min-height:0; border:1px solid var(--border-color-primary); border-radius:8px; overflow:hidden; }
#gv-chat .wrap { min-height:420px; }
#gv-chat ::-webkit-scrollbar, #gv-overlay-chat ::-webkit-scrollbar { width:8px; height:8px; }
#gv-chat ::-webkit-scrollbar-track, #gv-overlay-chat ::-webkit-scrollbar-track { background:#0f172a; }
#gv-chat ::-webkit-scrollbar-thumb, #gv-overlay-chat ::-webkit-scrollbar-thumb { background:#475569; border-radius:8px; }
.gv-chat-footer { margin-top:12px; }
.gv-chat-hint { color:var(--body-text-color-subdued); font-size:.7rem; margin:8px 2px 0; }
.gv-icon-button { min-width:34px !important; width:34px !important; padding:0 !important; }
#gv-voice-input { margin-top:8px; border:0 !important; background:transparent !important; }
.gv-primary-button button { background:#1d4ed8 !important; border-color:#1d4ed8 !important; color:#fff !important; }
.gv-primary-button button:hover { background:#1e40af !important; border-color:#1e40af !important; }
body.gv-overlay-active { overflow:hidden; }

#gv-chat-overlay { display:none !important; position:fixed !important; z-index:1000 !important; inset:0 !important; width:100vw !important; max-width:none !important; padding:28px !important; overflow:auto; background:rgba(15,23,42,.62); backdrop-filter:blur(4px); }
#gv-chat-overlay.gv-overlay-open { display:block !important; }
#gv-overlay-shell { width:min(1080px, 100%); height:calc(100vh - 56px); min-height:650px; margin:0 auto; padding:20px; border:1px solid rgba(148,163,184,.45); border-radius:14px; background:var(--body-background-fill, #fff); box-shadow:0 28px 70px rgba(15,23,42,.35); display:flex; flex-direction:column; }
#gv-overlay-chat { flex:1; min-height:0; border:1px solid var(--border-color-primary); border-radius:9px; overflow:hidden; }
#gv-overlay-chat .wrap { min-height:520px; }
.gv-overlay-header { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:14px; }

@media (max-width: 760px) { .gradio-container { padding:16px !important; } #gv-header { align-items:flex-start; flex-direction:column; } .gv-header-meta { justify-content:flex-start; } #gv-main { margin-top:16px; } #gv-chat-panel { min-height:580px; } #gv-chat .wrap { min-height:350px; } #gv-chat-overlay { padding:10px !important; } #gv-overlay-shell { min-height:calc(100vh - 20px); height:calc(100vh - 20px); padding:14px; border-radius:10px; } #gv-overlay-chat .wrap { min-height:420px; } }
"""

def build_ui(client: ClientLike, agent: AgentLike) -> gr.Blocks:
    devices: list[Device] = client.list_devices()

    with gr.Blocks(title="Govee Operations", elem_id="gv-shell") as demo:
        gr.HTML(
            '<div id="gv-header">'
            '<div class="gv-brand">'
            '<div class="gv-logo">G</div>'
            '<div>'
            '<p class="gv-title">Govee Operations</p>'
            '<p class="gv-subtitle">Local device control and AI-assisted operations</p>'
            '</div></div>'
            '<div class="gv-header-meta">'
            f'<span class="gv-header-chip"><span class="gv-live-dot"></span>{len(devices)} managed devices</span>'
            f'<span class="gv-backend-badge">Local model · {html.escape(agent.backend.backend_name)}</span>'
            '</div>'
            '</div>'
        )

        with gr.Row(elem_id="gv-main"):
            with gr.Column(scale=7, min_width=410):
                with gr.Group(elem_classes=["gv-panel"]):
                    with gr.Column(elem_classes=["gv-panel-inner"]):
                        gr.HTML(
                            '<div class="gv-panel-heading"><div><p class="gv-panel-title">DEVICE ESTATE</p>'
                            '<p class="gv-panel-copy">Live state, direct controls, and capability-aware actions.</p></div></div>'
                        )
                        with gr.Row(elem_id="gv-device-toolbar"):
                            gr.HTML(f'<div class="wrap"><strong>Devices</strong><span class="gv-device-count">{len(devices)} enrolled</span></div>')
                            with gr.Row(scale=0, elem_id="gv-device-controls"):
                                show_offline = gr.Checkbox(label="Offline", value=True, scale=1, min_width=85)
                                refresh_btn = gr.Button("Refresh", size="sm", scale=0, min_width=84)

                        state_boxes: dict[str, gr.HTML] = {}
                        power_buttons: dict[str, gr.Button] = {}
                        brightness_sliders: dict[str, gr.Slider] = {}
                        device_cards: dict[str, gr.Group] = {}

                        for d in devices:
                            with gr.Group(elem_classes=["gv-device-card"]) as card:
                                device_cards[d.device_id] = card
                                gr.HTML(
                                    f'<div class="gv-device-name"><span class="gv-icon">{_icon_for(d.device_type)}</span>'
                                    f'{html.escape(d.device_name)}</div>'
                                )
                                state_boxes[d.device_id] = gr.HTML("Loading&hellip;")

                                with gr.Row(elem_classes=["gv-actions"]):
                                    if d.has("devices.capabilities.on_off", "powerSwitch"):
                                        power_buttons[d.device_id] = gr.Button("Toggle power", size="sm", scale=0, min_width=108)
                                    if d.has("devices.capabilities.range", "brightness"):
                                        cap = d.capability("devices.capabilities.range", "brightness")
                                        if cap is not None:
                                            lo = cap.value_range["min"] if cap.value_range else 1
                                            hi = cap.value_range["max"] if cap.value_range else 100
                                            brightness_sliders[d.device_id] = gr.Slider(lo, hi, value=lo, step=1, label="Brightness")

            with gr.Column(scale=5, min_width=390):
                with gr.Group(elem_id="gv-chat-panel", elem_classes=["gv-panel"]):
                    with gr.Column(elem_classes=["gv-panel-inner"]):
                        with gr.Row(elem_classes=["gv-panel-heading"]):
                            gr.HTML('<div><p class="gv-panel-title">OPERATIONS ASSISTANT</p><p class="gv-panel-copy">Ask, verify, and act with a complete audit-friendly conversation.</p></div>')
                            with gr.Row(scale=0):
                                zoom_btn = gr.Button("Zoom chat", size="sm", scale=0, min_width=92)
                                expand_btn = gr.Button("Open overlay", size="sm", scale=0, min_width=108, elem_classes=["gv-primary-button"])
                        chatbot = gr.Chatbot(height=440, elem_id="gv-chat", show_label=False)
                        with gr.Row(elem_classes=["gv-chat-footer"]):
                            msg_box = gr.Textbox(placeholder="Ask about a device, a room, or an operational task…", show_label=False, scale=8)
                            clear_btn = gr.Button("Clear", size="sm", scale=0, min_width=58)
                        with gr.Accordion("Voice input", open=False, elem_id="gv-voice-input"):
                            mic = gr.Audio(sources=["microphone"], type="filepath", show_label=False)
                        gr.HTML('<p class="gv-chat-hint">Enter to send · Device tools run only when the assistant determines they are needed.</p>')

            agent_history = gr.State([])  # tool-call-aware history fed back into agent.chat
            online_status = gr.State({})  # device_id -> bool, refreshed alongside device state
            chat_zoomed = gr.State(False)

        # A separate, synchronised viewport gives the assistant room for longer
        # operational threads without changing the device-control layout.
        with gr.Group(elem_id="gv-chat-overlay"):
            with gr.Column(elem_id="gv-overlay-shell"):
                with gr.Row(elem_classes=["gv-overlay-header"]):
                    gr.HTML('<div><p class="gv-panel-title">OPERATIONS ASSISTANT</p><p class="gv-panel-copy">Expanded conversation workspace</p></div>')
                    close_overlay_btn = gr.Button("Close overlay", size="sm", scale=0, min_width=108)
                overlay_chatbot = gr.Chatbot(elem_id="gv-overlay-chat", show_label=False)
                with gr.Row(elem_classes=["gv-chat-footer"]):
                    overlay_msg_box = gr.Textbox(placeholder="Continue the conversation…", show_label=False, scale=8)
                    overlay_clear_btn = gr.Button("Clear", size="sm", scale=0, min_width=58)

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

        def respond(user_message: str, chat_display: list, overlay_display: list, history: list):
            if not user_message.strip():
                return chat_display, overlay_display, history, "", ""
            reply, history = agent.chat(user_message, history)
            messages = list(chat_display or []) + [{"role": "user", "content": user_message},{"role": "assistant", "content": reply}]
            return messages, messages, history, "", ""

        def open_chat_overlay(chat_display: list):
            return list(chat_display or [])

        def close_chat_overlay(chat_display: list):
            # Keep the hidden overlay's copy of the transcript synchronized.
            # Supplying an input/output also ensures Gradio runs the client-side
            # close hook on versions that skip JavaScript-only click handlers.
            return list(chat_display or [])

        def toggle_chat_zoom(zoomed: bool):
            expanded = not zoomed
            return (
                gr.update(height=680 if expanded else 440),
                expanded,
                gr.update(value="Zoom out" if expanded else "Zoom chat"),
            )

        def clear_chat():
            return [], [], [], "", ""

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

        chat_outputs = [chatbot, overlay_chatbot, agent_history, msg_box, overlay_msg_box]
        msg_box.submit(respond,inputs=[msg_box, chatbot, overlay_chatbot, agent_history],outputs=chat_outputs).then(refresh_all, inputs=[show_offline], outputs=refresh_outputs)
        overlay_msg_box.submit(respond,inputs=[overlay_msg_box, chatbot, overlay_chatbot, agent_history],outputs=chat_outputs).then(refresh_all, inputs=[show_offline], outputs=refresh_outputs)
        clear_btn.click(clear_chat, outputs=chat_outputs)
        overlay_clear_btn.click(clear_chat, outputs=chat_outputs)
        zoom_btn.click(toggle_chat_zoom, inputs=[chat_zoomed], outputs=[chatbot, chat_zoomed, zoom_btn])
        expand_btn.click(
            open_chat_overlay,
            inputs=[chatbot],
            outputs=[overlay_chatbot],
            js="(chat) => { document.querySelectorAll('#gv-chat-overlay').forEach((element) => element.classList.add('gv-overlay-open')); document.body.classList.add('gv-overlay-active'); return chat; }",
        )
        close_overlay_btn.click(
            close_chat_overlay,
            inputs=[chatbot],
            outputs=[overlay_chatbot],
            js="(chat) => { document.querySelectorAll('#gv-chat-overlay').forEach((element) => element.classList.remove('gv-overlay-open')); document.body.classList.remove('gv-overlay-active'); return chat; }",
        )

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

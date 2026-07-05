#app.py

from __future__ import annotations

import html, logging, os, warnings
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
warnings.filterwarnings(
                    "ignore",
                    message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*",
                    category=Warning,
                    )

import gradio as gr
from gradio.themes import Soft as SoftTheme, GoogleFont
from gradio.themes.utils import colors as theme_colors, sizes as theme_sizes
from dotenv import load_dotenv
from typing import Protocol

_GMAJOR = int(gr.__version__.split(".")[0])

import speech_to_text
from agent import GoveeAgent
from govee_client import Device, GoveeAPIError, GoveeClient

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # quiet per-request HTTP noise
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

def _pill(text: str, kind: str) -> str:
    return f'<span class="gv-pill {kind}">{html.escape(text)}</span>'

def _chip(text: str) -> str:
    return f'<span class="gv-chip">{html.escape(text)}</span>'

def _format_state(state: dict) -> str:
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
        chips.append(f"{state['sensorTemperature']}\u00b0C")
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
    display:flex; align-items:center; justify-content:space-between; gap:16px;padding:6px 4px 16px; margin-bottom:10px;border-bottom:1px solid var(--border-color-primary)}
.gv-brand { display:flex; align-items:center; gap:13px; }

.gv-logo {
            font-size:1.5rem; width:46px; height:46px; flex:0 0 auto;
            display:flex; align-items:center; justify-content:center;
            background:var(--background-fill-secondary);
            border:1px solid var(--border-color-primary); border-radius:13px;}
.gv-title { font-size:1.4rem; font-weight:750; line-height:1.15; }
.gv-sub { font-size:.85rem; color:var(--body-text-color-subdued); }
.gv-badge {
            font-size:.76rem; font-weight:600; white-space:nowrap; padding:6px 13px; border-radius:999px;
            background:var(--background-fill-secondary); border:1px solid var(--border-color-primary);
            color:var(--body-text-color-subdued);
            }
.gv-badge b { color:var(--body-text-color); font-weight:700; }

.gv-sec {
            font-size:1.05rem; font-weight:700; margin:2px 2px 8px;
            display:flex; align-items:center; gap:9px;
        }
.gv-count {
            font-size:.72rem; font-weight:700; padding:2px 9px; border-radius:999px;
            background:var(--background-fill-secondary); border:1px solid var(--border-color-primary);
            color:var(--body-text-color-subdued);
            }

.gv-card { border-radius:14px !important; }
.gv-devhead { display:flex; align-items:center; gap:11px; margin-bottom:2px; }
.gv-ic {
        font-size:1.1rem; width:36px; height:36px; flex:0 0 auto;
        display:flex; align-items:center; justify-content:center;
        background:var(--background-fill-secondary);
        border:1px solid var(--border-color-primary); border-radius:11px;
        }
.gv-devname { font-weight:650; font-size:.98rem; }
.gv-state { min-height:28px; display:flex; align-items:center; flex-wrap:wrap; gap:7px; padding:2px 0; }

.gv-pill {
            display:inline-flex; align-items:center; gap:7px; padding:3px 12px; border-radius:999px;
            font-size:.77rem; font-weight:700;
        }
.gv-pill::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }
.gv-pill.on { background:rgba(22,163,74,.14); color:#16a34a; }
.gv-pill.off { background:rgba(100,116,139,.16); color:#64748b; }
.gv-pill.offline { background:rgba(220,38,38,.14); color:#dc2626; }

.gv-chip {
            display:inline-block; padding:3px 10px; border-radius:8px; font-size:.77rem;
            background:var(--background-fill-secondary); border:1px solid var(--border-color-primary);
            color:var(--body-text-color-subdued);
        }
.gv-note { font-size:.8rem; color:var(--body-text-color-subdued); }

/* When the chat is toggled fullscreen, fill the screen and give it a backdrop
   (the element is transparent by default, which looks broken in fullscreen). */
#gv-chat:fullscreen { height:100vh !important; background:var(--body-background-fill); padding:18px; }
#gv-chat:-webkit-full-screen { height:100vh !important; background:var(--body-background-fill); padding:18px; }
"""

EXAMPLE_PROMPTS = [
                    ["What devices do I have?"],
                    ["Turn off everything"],
                    ["Turn on all the lights"],
                    ["Set the bedroom light to 30% warm white"],
                    ["Turn on all the lights"],
                    ["Zapni vsetky svetla"],
                    ["Ktore zariadenia su k dispozicii"],
                    ["Nastav teplu bielu"],
                    ["Nastav svetelnost na 30 percent"]
                ]

def _blocks_kwargs() -> dict:
    kw: dict = {"title": "Govee Home"}
    if _GMAJOR < 6:  # Gradio 5 takes theme/css on Blocks; Gradio 6 on launch()
        kw["theme"] = THEME
        kw["css"] = CSS
    return kw

def _launch_kwargs() -> dict:
    kw: dict = {"theme": THEME, "css": CSS} if _GMAJOR >= 6 else {}
    server_name = os.getenv("GRADIO_SERVER_NAME")
    server_port = os.getenv("GRADIO_SERVER_PORT")
    if server_name:
        kw["server_name"] = server_name
    if server_port:
        kw["server_port"] = int(server_port)
    return kw

def _chatbot_kwargs() -> dict:
    kw: dict = {"elem_id": "gv-chat","height": 800,"show_label": False,"placeholder": "Ask me to check or control your devices — in any language."}
    if _GMAJOR < 6:
                    kw["type"] = "messages"
                    kw["show_copy_button"] = True
    return kw

def build_ui(client: ClientLike, agent: AgentLike) -> gr.Blocks:
    devices: list[Device] = client.list_devices()
    backend_name = html.escape(str(agent.backend.backend_name))

    with gr.Blocks(**_blocks_kwargs()) as demo:
        gr.HTML(
                '<div id="gv-header">'
                '  <div class="gv-brand">'
                '    <span class="gv-logo">\U0001F3E0</span>'
                '    <div>'
                '      <div class="gv-title">Govee Home</div>'
                '      <div class="gv-sub">Local smart-home dashboard &amp; assistant</div>'
                '    </div>'
                '  </div>'
                f'  <span class="gv-badge">backend&nbsp;·&nbsp;<b>{backend_name}</b></span>'
                '</div>'
                )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5) as devices_col:
                with gr.Row():
                    gr.HTML(f'<div class="gv-sec">Devices <span class="gv-count">{len(devices)}</span></div>')
                    refresh_btn = gr.Button("\u21BB  Refresh", size="sm", scale=0, min_width=120)

                state_boxes: dict[str, gr.HTML] = {}
                power_buttons: dict[str, gr.Button] = {}
                brightness_sliders: dict[str, gr.Slider] = {}

                for d in devices:
                    with gr.Group(elem_classes="gv-card"):
                        gr.HTML(
                                '<div class="gv-devhead">'
                                f'<span class="gv-ic">{_icon_for(d.device_type)}</span>'
                                f'<span class="gv-devname">{html.escape(d.device_name)}</span>'
                                '</div>'
                                )
                        state_boxes[d.device_id] = gr.HTML('<div class="gv-state"><span class="gv-pill off">Loading…</span></div>')

                        has_power = d.has("devices.capabilities.on_off", "powerSwitch")
                        has_bright = d.has("devices.capabilities.range", "brightness")
                        if has_power or has_bright:
                            with gr.Row():
                                if has_power:
                                    power_buttons[d.device_id] = gr.Button("Toggle power", size="sm", variant="secondary")
                                if has_bright:
                                    cap = d.capability("devices.capabilities.range", "brightness")
                                    if cap is not None:
                                        lo = cap.value_range["min"] if cap.value_range else 1
                                        hi = cap.value_range["max"] if cap.value_range else 100
                                        brightness_sliders[d.device_id] = gr.Slider(
                                            lo, hi, value=lo, step=1, label="Brightness"
                                        )

            with gr.Column(scale=6):
                with gr.Row():
                    gr.HTML('<div class="gv-sec">Assistant</div>')
                    expand_btn = gr.Button("\u2921  Expand chat", size="sm", scale=0, min_width=150)
                chatbot = gr.Chatbot(**_chatbot_kwargs())
                with gr.Row():
                    msg_box = gr.Textbox(
                                            placeholder="e.g. \u201cturn off the christmas lights and dim the bedroom lamp to 30%\u201d",
                                            show_label=False, scale=7, autofocus=True, container=False,
                                        )
                    send_btn = gr.Button("Send", variant="primary", scale=0, min_width=90)
                with gr.Row():
                    mic = gr.Audio(sources=["microphone"], type="filepath", show_label=False, scale=1)
                voice_status = gr.Markdown("", elem_classes="gv-note")
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=msg_box)

                agent_history = gr.State([])  # tool-call-aware history fed back into agent.chat
                zoom_state = gr.State(False)   # tracks whether the chat is expanded

        def refresh_all():
            results = []
            for d in devices:
                try:
                    state = client.get_state(d.sku, d.device_id)
                    results.append(f'<div class="gv-state">{_format_state(state)}</div>')
                except GoveeAPIError as e:
                    results.append(
                                    f'<div class="gv-state">{_pill("Error", "offline")}'
                                    f'<span class="gv-note">{html.escape(str(e))}</span></div>'
                                    )
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

        def respond_voice(audio_path, chat_display: list, history: list):

            logger.info("Voice recording received: %r", audio_path)
            text = speech_to_text.transcribe(audio_path)
            if audio_path and not text:
                logger.warning("Speech-to-text produced no usable transcription")
                if not speech_to_text.ffmpeg_available():
                    gr.Warning("Couldn't transcribe the audio. ffmpeg wasn't found on PATH - install it and restart (see the README), or record in WAV format.")
                else:
                    gr.Warning("The recording was silent or too quiet. Check the selected microphone and mute switch, then try again.")
                return chat_display, history, "", "Recording was silent or too quiet."
            if not text:
                return chat_display, history, "", "No recording received."

            status = f"Heard: {html.escape(text)}"
            chat_display, history, msg_value = respond(text, chat_display, history)
            return chat_display, history, msg_value, status

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

        chat_inputs = [msg_box, chatbot, agent_history]
        chat_outputs = [chatbot, agent_history, msg_box]
        msg_box.submit(respond, inputs=chat_inputs, outputs=chat_outputs).then(refresh_all, outputs=all_state_outputs)
        send_btn.click(respond, inputs=chat_inputs, outputs=chat_outputs).then(refresh_all, outputs=all_state_outputs)

        voice_outputs = [chatbot, agent_history, msg_box, voice_status]
        mic.stop_recording(respond_voice, inputs=[mic, chatbot, agent_history], outputs=voice_outputs).then(refresh_all, outputs=all_state_outputs).then(lambda: None, outputs=mic)
        def toggle_zoom(zoomed: bool):
            zoomed = not zoomed
            return (
                    zoomed,
                    gr.update(visible=not zoomed),                       # devices_col
                    gr.update(height=860 if zoomed else 620),           # chatbot
                    gr.update(value="\u2921  Minimize" if zoomed else "\u2921  Expand chat"),  # button
                    )

        expand_btn.click(toggle_zoom,inputs=zoom_state,outputs=[zoom_state, devices_col, chatbot, expand_btn])

    return demo

def main():
    client = GoveeClient()
    logger.info("Loading local LLM agent (first run can take a while)")
    agent = GoveeAgent(client)
    demo = build_ui(client, agent)
    host_port = os.getenv("GRADIO_HOST_PORT")
    if host_port:
        logger.info("Open the app at http://127.0.0.1:%s (Docker host port)", host_port)
    demo.launch(**_launch_kwargs())

if __name__ == "__main__":
    main()

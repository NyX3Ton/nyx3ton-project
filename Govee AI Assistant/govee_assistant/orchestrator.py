#orchestrator.py

from __future__ import annotations

import asyncio, logging
from typing import Any, Optional, Sequence

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.base.llms.types import (
                                                ChatMessage,
                                                ChatResponse,
                                                ChatResponseAsyncGen,
                                                ChatResponseGen,
                                                CompletionResponse,
                                                CompletionResponseAsyncGen,
                                                CompletionResponseGen,
                                                MessageRole,
                                            )
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.tools import FunctionTool

from . import config
from .agent import GoveeClientLike, GoveeTools, InfoTools, ModelBackend

logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# LlamaIndex LLM adapter around the local Gemma ModelBackend
# ---------------------------------------------------------------------------
class GemmaLocalLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 512
    model_name: str = "gemma-local"

    _backend: Any = PrivateAttr()

    def __init__(self, backend: ModelBackend, num_output: int = 512, **kwargs: Any):
        super().__init__(num_output=num_output, model_name=backend.model_id, **kwargs)
        self._backend = backend

    # -- metadata --
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
                            context_window=self.context_window,
                            num_output=self.num_output,
                            is_chat_model=True,
                            is_function_calling_model=False,
                            model_name=self.model_name,
                            )

    # -- prompt rendering --
    def _render(self, messages: Sequence[ChatMessage]) -> str:
        proc = self._backend.processor
        dict_msgs = [{"role": m.role.value, "content": (m.content or "")} for m in messages]
        try:
            return proc.apply_chat_template(dict_msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            # some chat-template signatures don't accept enable_thinking
            return proc.apply_chat_template(dict_msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            # Defensive: if the template rejects a role layout (e.g. a standalone
            # system role), flatten everything into a single user turn.
            logger.warning("apply_chat_template failed; flattening messages to a plain prompt")
            flat = "\n\n".join(f"{m.role.value.upper()}: {(m.content or '')}" for m in messages)
            return flat + "\n\nASSISTANT:"

    # -- chat (primary path for ReActAgent) --
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        text = self._backend.generate(self._render(messages), max_new_tokens=self.num_output)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        text = self._backend.generate(self._render(messages), max_new_tokens=self.num_output)

        def gen() -> ChatResponseGen:
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=text
            )

        return gen()

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        resp = await asyncio.to_thread(self.chat, messages, **kwargs)

        async def gen() -> ChatResponseAsyncGen:
            yield ChatResponse(message=resp.message, delta=resp.message.content or "")

        return gen()

    # -- complete --
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text = self._backend.generate(prompt, max_new_tokens=self.num_output)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = self._backend.generate(prompt, max_new_tokens=self.num_output)

        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text=text, delta=text)

        return gen()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return await asyncio.to_thread(self.complete, prompt, formatted, **kwargs)

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        resp = await asyncio.to_thread(self.complete, prompt, formatted, **kwargs)

        async def gen() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text=resp.text, delta=resp.text)

        return gen()


# ---------------------------------------------------------------------------
# Tool wrapping: reuse the existing GoveeTools / InfoTools methods verbatim
# ---------------------------------------------------------------------------
DEVICE_SYSTEM_PROMPT = (
                        "You are the device-control specialist for Govee smart-home devices. Use "
                        "your tools to inspect and control lights, fans, plugs and sensors. Call "
                        "list_devices or get_device_state first if you're unsure a device exists or "
                        "what it supports rather than guessing. Pass device/scene names to tools as "
                        "the user wrote them. Keep replies short and state what changed. If the user "
                        "asks about weather, news, or recalling past conversation, hand off to the "
                        "Information agent."
                        )

INFO_SYSTEM_PROMPT = (
                        "You are the information specialist. Use your tools to report the weather "
                        "(get_weather), fetch news headlines (get_news), read a specific article "
                        "(get_article_extract, using a 'link' from get_news), and recall earlier "
                        "conversation or past weather/news lookups (recall_memories). Keep replies "
                        "short. If the user wants to control a smart-home device, hand off to the "
                        "DeviceControl agent."
                    )


def _device_tools(tools: GoveeTools) -> list[FunctionTool]:
    specs = [
            (tools.list_devices, "list_devices", "List all Govee devices and what each can be controlled for (power, brightness, color, scenes, toggles). Call first if unsure of a device name or capability."),
            (tools.get_device_state, "get_device_state", "Get the current state (power, brightness, color, online status) of one device by name."),
            (tools.set_power, "set_power", "Turn a single device on or off."),
            (tools.set_power_all, "set_power_all", "Turn every matching power-capable device on or off in one call. Use for all devices, all lights, or a room; optional device_type and name_contains filters combine with AND."),
            (tools.set_brightness, "set_brightness", "Set a light's brightness as a percentage (1-100)."),
            (tools.set_color_rgb, "set_color_rgb", "Set a light's color using RGB values (0-255 each)."),
            (tools.set_color_temp, "set_color_temp", "Set a light's white color temperature in Kelvin (~2000 warm to 9000 cool)."),
            (tools.set_scene, "set_scene", "Activate a preset light scene by name (e.g. 'Christmas', 'Party', 'Sunrise')."),
            (tools.set_toggle, "set_toggle", "Turn a named toggle feature on/off, e.g. 'oscillationToggle'."),
            (tools.set_fan_speed, "set_fan_speed", "Set a fan's speed gear: 'low', 'medium', or 'high'."),
            ]
    return [FunctionTool.from_defaults(fn=fn, name=name, description=desc) for fn, name, desc in specs]


def _info_tools(tools: InfoTools) -> list[FunctionTool]:
    specs = [
            (tools.get_weather, "get_weather", "Get current weather and a short forecast for a location. Omit location to use the configured default."),
            (tools.get_news, "get_news", "Get recent news headlines, optionally filtered by topic. Each headline includes a 'link' usable with get_article_extract."),
            (tools.get_article_extract, "get_article_extract", "Fetch and read the main body text of a specific news article, using a 'link' from a get_news result."),
            (tools.recall_memories, "recall_memories", "Recall things from earlier conversations or past weather/news lookups. Pass a query, or omit for the most recent memories."),
            ]
    return [FunctionTool.from_defaults(fn=fn, name=name, description=desc) for fn, name, desc in specs]


def build_workflow(llm: GemmaLocalLLM, govee_tools: GoveeTools, info_tools: InfoTools) -> AgentWorkflow:
    device_agent = ReActAgent(
                                name="DeviceControl",
                                description="Controls Govee smart-home devices (lights, fans, plugs, sensors).",
                                system_prompt=DEVICE_SYSTEM_PROMPT,
                                tools=_device_tools(govee_tools),
                                llm=llm,
                                can_handoff_to=["Information"],
                            )
    info_agent = ReActAgent(
                            name="Information",
                            description="Answers weather, news, and memory-recall questions.",
                            system_prompt=INFO_SYSTEM_PROMPT,
                            tools=_info_tools(info_tools),
                            llm=llm,
                            can_handoff_to=["DeviceControl"],
                            )
    return AgentWorkflow(agents=[device_agent, info_agent], root_agent="DeviceControl")


# ---------------------------------------------------------------------------
# OrchestratedAgent: same interface as GoveeAgent (drop-in for app.py / CLI)
# ---------------------------------------------------------------------------
class OrchestratedAgent:
    def __init__(self,client: GoveeClientLike,backend: Optional[ModelBackend] = None,info_tools: Optional[InfoTools] = None,llm: Optional[GemmaLocalLLM] = None):
        self.backend = backend or ModelBackend()
        self.govee_tools = GoveeTools(client)
        self.info_tools = info_tools or InfoTools()
        self.llm = llm or GemmaLocalLLM(self.backend)
        self.workflow = build_workflow(self.llm, self.govee_tools, self.info_tools)

    @staticmethod
    def _to_chat_messages(history: list[dict]) -> list[ChatMessage]:
        role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT, "system": MessageRole.SYSTEM}
        return [
                ChatMessage(role=role_map.get(m.get("role", "user"), MessageRole.USER), content=m.get("content", ""))
                for m in history
                ]

    def _run_workflow(self, user_message: str, history: list[dict]) -> str:
        chat_history = self._to_chat_messages(history) or None

        async def _go() -> Any:
            return await self.workflow.run(user_msg=user_message, chat_history=chat_history)

        result = asyncio.run(_go())
        # AgentWorkflow returns an AgentOutput; str() yields the final text, and
        # .response.content is the same when present.
        response = getattr(result, "response", None)
        if response is not None and getattr(response, "content", None):
            return response.content
        return str(result)

    def chat(self, user_message: str, history: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
        history = history or []
        try:
            reply = self._run_workflow(user_message, history)
        except Exception as e:  # noqa: BLE001
            logger.exception("AgentWorkflow run failed")
            reply = f"The multi-agent workflow hit an error: {e}. Try GOVEE_AGENT_MODE=single if this persists."

        try:
            self.info_tools.memory_store.add("chat", f"User: {user_message}\nAssistant: {reply}")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write chat turn to long-term memory")

        new_history = history + [{"role": "user", "content": user_message},{"role": "assistant", "content": reply}]
        return reply, new_history

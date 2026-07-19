"""Offline tests for the bounded writer--critic refinement loop.

No model, network, or Govee account is needed; the scripted backend verifies
that the critic cannot trigger tools and that conversation history keeps the
revised answer.
"""

from govee_assistant.agent import CritiqueAgent, WriterCriticAgent


class FakeProcessor:
    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class ScriptedBackend:
    def __init__(self, responses):
        self.processor = FakeProcessor()
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, max_new_tokens=512):
        self.prompts.append((prompt, max_new_tokens))
        return self.responses.pop(0)


class FakeWriter:
    def __init__(self, backend):
        self.backend = backend

    def chat(self, user_message, history=None):
        history = history or []
        draft = "The desk light is on."
        return draft, history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": draft},
        ]


def test_approval_skips_revision():
    backend = ScriptedBackend(["APPROVE"])
    critic = CritiqueAgent(backend, max_passes=1)
    assert critic.refine("Turn on the desk light", "The desk light is on.") == "The desk light is on."
    assert len(backend.prompts) == 1


def test_feedback_triggers_one_revision():
    backend = ScriptedBackend([
        "State which requested device changed.",
        "The desk light is now on.",
    ])
    critic = CritiqueAgent(backend, max_passes=1)
    assert critic.refine("Turn on the desk light", "Done.") == "The desk light is now on."
    assert len(backend.prompts) == 2


def test_wrapper_replaces_final_history_message():
    backend = ScriptedBackend([
        "Clarify the result.",
        "The desk light is now on.",
    ])
    agent = WriterCriticAgent(FakeWriter(backend), CritiqueAgent(backend, max_passes=1))
    reply, history = agent.chat("Turn on the desk light")
    assert reply == "The desk light is now on."
    assert history[-1] == {"role": "assistant", "content": reply}


if __name__ == "__main__":
    test_approval_skips_revision()
    test_feedback_triggers_one_revision()
    test_wrapper_replaces_final_history_message()
    print("OK: writer-critic loop approves, revises, and preserves revised history")

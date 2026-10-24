"""Offline tests for the bounded agentic long-term-memory retrieval loop.

No model, network, embedding model, or database is needed. The scripted
backend simulates only the planner/verifier decisions and the fake store makes
the selected retrieval queries observable.
"""

from govee_assistant.agent import AgenticRAG


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


class FakeMemoryStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.results_by_query.get(query, [])


def test_skip_does_not_search_memory():
    store = FakeMemoryStore({})
    rag = AgenticRAG(ScriptedBackend(["SKIP"]), store)
    assert rag.context_for("Turn the hallway light on") == ""
    assert store.queries == []


def test_verifier_selects_only_relevant_memory():
    store = FakeMemoryStore({
        "last Paris weather": [
            {"category": "weather", "when": "2026-07-19", "content": "Paris: 24C and partly cloudy."},
            {"category": "chat", "when": "2026-07-18", "content": "Turned the desk light on."},
        ],
    })
    backend = ScriptedBackend(["RETRIEVE: last Paris weather", "USE: 1"])
    rag = AgenticRAG(backend, store, top_k=3)
    context = rag.context_for("What weather did we see in Paris?")
    assert store.queries == [("last Paris weather", 3)]
    assert "Paris: 24C" in context
    assert "desk light" not in context
    assert "category=weather" in context


def test_verifier_can_reformulate_once_then_select():
    store = FakeMemoryStore({
        "weather": [{"category": "chat", "when": "", "content": "Unrelated note."}],
        "weather in Paris": [{"category": "weather", "when": "2026-07-19", "content": "Paris was rainy."}],
    })
    backend = ScriptedBackend([
        "RETRIEVE: weather",
        "RETRY: weather in Paris",
        "USE: 1",
    ])
    context = AgenticRAG(backend, store, max_searches=2).context_for("What was Paris weather?")
    assert store.queries == [("weather", 5), ("weather in Paris", 5)]
    assert "Paris was rainy." in context


if __name__ == "__main__":
    test_skip_does_not_search_memory()
    test_verifier_selects_only_relevant_memory()
    test_verifier_can_reformulate_once_then_select()
    print("OK: agentic RAG skips, verifies retrieval, and retries once")

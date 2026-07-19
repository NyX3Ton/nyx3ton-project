import json

import chromadb
from llama_index.core.embeddings import BaseEmbedding

from govee_assistant.agent import InfoTools
from govee_assistant.memory_store import COLLECTION_NAME, MemoryStore
from govee_assistant.news_client import NewsClient, NewsError
from govee_assistant.weather_client import WeatherClient, WeatherError
from govee_assistant import weather_client as weather_client_module


_FAKE_VOCAB = ["weather", "paris", "news", "election", "bedroom", "light"]


class FakeEmbedding(BaseEmbedding):
    """Tiny bag-of-words LlamaIndex embedding, deterministic and offline
    (no model download, no network) - orthogonal vectors for unrelated text.
    (BaseEmbedding is a pydantic model, so state lives at module scope rather
    than as a plain class attribute.)"""

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [1.0 if v in t else 0.0 for v in _FAKE_VOCAB]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def _get_text_embeddings(self, texts) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# weather_client.py
# ---------------------------------------------------------------------------
def test_weather_client():
    def fake_get_json(url, params):
        if "geocoding-api" in url:
            return {"results": [{"latitude": 48.85, "longitude": 2.35, "name": "Paris", "admin1": "Ile-de-France", "country": "France"}]}
        return {
            "current": {"temperature_2m": 22.5, "relative_humidity_2m": 60, "weather_code": 1, "wind_speed_10m": 10.0},
            "daily": {
                "time": ["2026-07-16", "2026-07-17"],
                "temperature_2m_max": [25.0, 26.0],
                "temperature_2m_min": [15.0, 16.0],
                "weather_code": [1, 61],
            },
        }

    client = WeatherClient(get_json_fn=fake_get_json)
    result = client.get_forecast("Paris")
    assert result["location"] == "Paris, Ile-de-France, France"
    assert result["temperature_c"] == 22.5
    assert result["condition"] == "Mainly clear"
    assert len(result["forecast"]) == 2
    assert result["forecast"][1]["condition"] == "Slight rain"
    print(f"OK: {result}")

    # geocode cache: second call for the same location shouldn't re-geocode
    calls = {"geocode": 0}
    def counting_get_json(url, params):
        if "geocoding-api" in url:
            calls["geocode"] += 1
        return fake_get_json(url, params)
    client2 = WeatherClient(get_json_fn=counting_get_json)
    client2.get_forecast("Paris")
    client2.get_forecast("paris")  # same location, different case
    assert calls["geocode"] == 1, "geocode cache should be case-insensitive and hit only once"
    print("OK: geocode cache reused across calls")

    # no location, no default -> WeatherError
    # config values are read once at import time (govee_assistant/config.py),
    # so simulating "no default set" means patching the already-loaded
    # constant directly rather than mutating os.environ at runtime.
    original_default = weather_client_module.config.GOVEE_DEFAULT_LOCATION
    weather_client_module.config.GOVEE_DEFAULT_LOCATION = ""
    try:
        WeatherClient(get_json_fn=fake_get_json).get_forecast(None)
        raise SystemExit("FAIL: expected WeatherError")
    except WeatherError as e:
        print(f"OK: {e}")
    finally:
        weather_client_module.config.GOVEE_DEFAULT_LOCATION = original_default

    # unknown location -> WeatherError
    try:
        WeatherClient(get_json_fn=lambda url, params: {"results": []}).get_forecast("Nowhereville")
        raise SystemExit("FAIL: expected WeatherError")
    except WeatherError as e:
        print(f"OK: {e}")


# ---------------------------------------------------------------------------
# news_client.py
# ---------------------------------------------------------------------------
SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
<title>Headline One &amp; More</title>
<link>https://example.com/1</link>
<pubDate>Thu, 16 Jul 2026 10:00:00 GMT</pubDate>
<description><![CDATA[<a href="x">Some</a> <b>bold</b> summary text.]]></description>
</item>
<item>
<title>Headline Two</title>
<link>https://example.com/2</link>
<pubDate>Thu, 16 Jul 2026 09:00:00 GMT</pubDate>
<description>Plain summary.</description>
</item>
</channel></rss>
"""


def test_news_client():
    client = NewsClient(get_text_fn=lambda url: SAMPLE_RSS)
    result = client.get_headlines(limit=5)
    assert len(result) == 2
    assert result[0]["title"] == "Headline One & More"
    assert result[0]["summary"] == "Some bold summary text."
    assert result[1]["title"] == "Headline Two"
    print(f"OK: {result}")

    # limit truncation
    result = client.get_headlines(limit=1)
    assert len(result) == 1
    print("OK: limit truncation")

    # topic vs default feed URL selection
    seen_urls = []
    def spy_get_text(url):
        seen_urls.append(url)
        return SAMPLE_RSS
    NewsClient(get_text_fn=spy_get_text).get_headlines(topic="climate change", limit=5)
    assert "rss/search?q=climate+change" in seen_urls[0]
    print(f"OK: topic query routed to search feed: {seen_urls[0]}")

    seen_urls.clear()
    NewsClient(get_text_fn=spy_get_text).get_headlines(limit=5)
    assert seen_urls[0].startswith("https://news.google.com/rss?")
    print(f"OK: no topic routed to default feed: {seen_urls[0]}")

    # fetch failure -> NewsError
    def boom(url):
        raise RuntimeError("network down")
    try:
        NewsClient(get_text_fn=boom).get_headlines()
        raise SystemExit("FAIL: expected NewsError")
    except NewsError as e:
        print(f"OK: {e}")


SAMPLE_ARTICLE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Big Story Breaks</title>
<meta name="author" content="Jane Reporter">
<meta property="article:published_time" content="2026-07-16T09:00:00Z">
</head>
<body>
<nav>Home | World | Sports</nav>
<header>Example News</header>
<article>
<h1>Big Story Breaks</h1>
<p class="byline">By Jane Reporter</p>
<p>This is the first paragraph of the actual article, describing what happened
in detail with enough length that it reads like real reporting prose.</p>
<p>This is the second paragraph, adding context, a quote from an official,
and background information relevant to the story.</p>
</article>
<footer>Copyright 2026 Example News</footer>
</body></html>
"""


def test_article_extract():
    client = NewsClient(get_text_fn=lambda url: SAMPLE_ARTICLE_HTML)
    extract = client.get_article_extract("https://example.com/story")
    assert "first paragraph of the actual article" in extract
    assert "second paragraph" in extract
    assert "Home | World | Sports" not in extract, "nav boilerplate should be stripped"
    print(f"OK: {extract}")

    # max_chars truncation
    short = client.get_article_extract("https://example.com/story", max_chars=20)
    assert len(short) <= 23  # 20 chars + "..."
    assert short.endswith("...")
    print(f"OK: truncated to {short!r}")

    # fetch failure -> NewsError
    def boom(url):
        raise RuntimeError("404 not found")
    try:
        NewsClient(get_text_fn=boom).get_article_extract("https://example.com/missing")
        raise SystemExit("FAIL: expected NewsError")
    except NewsError as e:
        print(f"OK: {e}")

    # page with no extractable content -> NewsError
    try:
        NewsClient(get_text_fn=lambda url: "<html><body></body></html>").get_article_extract("https://example.com/empty")
        raise SystemExit("FAIL: expected NewsError")
    except NewsError as e:
        print(f"OK: {e}")


# ---------------------------------------------------------------------------
# memory_store.py
# ---------------------------------------------------------------------------
def _fresh_memory_store(similarity_cutoff=None) -> MemoryStore:
    # chromadb's EphemeralClient() instances aren't fully isolated from each
    # other within the same process (they share collections by name), so
    # each test explicitly clears the collection first for a clean slate.
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return MemoryStore(embed_model=FakeEmbedding(), client=client, similarity_cutoff=similarity_cutoff)


def test_memory_store():
    store = _fresh_memory_store()

    store.add("weather", "Weather in Paris: 22.5C, Mainly clear.")
    store.add("news", "News: election results announced today.")
    store.add("chat", "User: turn on the bedroom light\nAssistant: Turned on the Bedroom Light.")

    recent = store.recent(limit=10)
    assert len(recent) == 3
    print(f"OK: recent() -> {[r['category'] for r in recent]}")

    weather_only = store.recent(limit=10, category="weather")
    assert len(weather_only) == 1 and weather_only[0]["category"] == "weather"
    print("OK: recent() category filter")

    hits = store.search("weather in paris", top_k=1)
    assert hits and hits[0]["category"] == "weather"
    print(f"OK: search() ranks weather memory first: {hits[0]['content']}")

    hits = store.search("bedroom light", top_k=1)
    assert hits and hits[0]["category"] == "chat"
    print(f"OK: search() ranks chat memory first: {hits[0]['content']}")

    # category filter on search()
    hits = store.search("election", top_k=5, category="news")
    assert hits and all(h["category"] == "news" for h in hits)
    print(f"OK: search() category filter -> {[h['category'] for h in hits]}")

    # opt-in similarity_cutoff drops weakly-matching (near-orthogonal) memories.
    # With the fake bag-of-words embedding, a query sharing no vocabulary with a
    # memory scores well below an exact match, so a cutoff keeps only the strong
    # hit even when a larger top_k is requested.
    cut_store = _fresh_memory_store(similarity_cutoff=0.5)
    cut_store.add("weather", "Weather in Paris: 22.5C, Mainly clear.")
    cut_store.add("news", "News: election results announced today.")
    cut_hits = cut_store.search("weather in paris", top_k=5)
    assert cut_hits and all(h["category"] == "weather" for h in cut_hits), cut_hits
    print(f"OK: similarity_cutoff drops weak matches -> {[h['category'] for h in cut_hits]}")


# ---------------------------------------------------------------------------
# InfoTools (agent.py) wiring
# ---------------------------------------------------------------------------
class FakeWeatherClient:
    def get_forecast(self, location=None):
        return {"location": "Paris", "temperature_c": 22.5, "condition": "Clear", "humidity_pct": 50, "wind_kph": 5, "forecast": []}


class FakeNewsClient:
    def get_headlines(self, topic=None, limit=5):
        return [{"title": "Big News", "link": "https://x", "published": "", "summary": "..."}]

    def get_article_extract(self, url):
        if url == "https://x":
            return "The full article body text goes here, with much more detail than the summary."
        raise NewsError(f"No article at {url}")


def test_info_tools():
    store = _fresh_memory_store()
    tools = InfoTools(weather_client=FakeWeatherClient(), news_client=FakeNewsClient(), memory_store=store)

    weather_result = tools.get_weather("Paris")
    assert weather_result["location"] == "Paris"
    assert store.recent(limit=10, category="weather"), "get_weather should auto-write a memory entry"
    print(f"OK: get_weather -> {weather_result}, memory written")

    news_result = tools.get_news()
    assert news_result["headlines"][0]["title"] == "Big News"
    assert store.recent(limit=10, category="news"), "get_news should auto-write a memory entry"
    print(f"OK: get_news -> {news_result}, memory written")

    extract_result = tools.get_article_extract(news_result["headlines"][0]["link"])
    assert "full article body text" in extract_result["extract"]
    print(f"OK: get_article_extract -> {extract_result}")

    error_result = tools.get_article_extract("https://unknown")
    assert "error" in error_result
    print(f"OK: get_article_extract error path -> {error_result}")

    recalled = tools.recall_memories(query="paris")
    assert recalled["memories"], "recall_memories should find the weather memory"
    print(f"OK: recall_memories(query) -> {recalled}")

    recalled = tools.recall_memories()
    assert len(recalled["memories"]) >= 2, "recall_memories with no query should list recent memories"
    print(f"OK: recall_memories() -> {recalled}")


def main():
    print("== weather_client.py ==")
    test_weather_client()
    print("\n== news_client.py ==")
    test_news_client()
    print("\n== news_client.py: get_article_extract ==")
    test_article_extract()
    print("\n== memory_store.py ==")
    test_memory_store()
    print("\n== InfoTools (agent.py) ==")
    test_info_tools()
    print("\nAll offline checks passed.")


if __name__ == "__main__":
    main()

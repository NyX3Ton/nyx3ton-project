#news_client.py

from __future__ import annotations

import html, logging, re
import xml.etree.ElementTree as ET
from typing import Callable, Optional
from urllib.parse import quote_plus

from . import config

logger = logging.getLogger("news_client")

DEFAULT_FEED = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
SEARCH_FEED = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
SLOVAK_SEARCH_FEED = "https://news.google.com/rss/search?q={query}&hl=sk&gl=SK&ceid=SK:sk"

_TAG_RE = re.compile(r"<[^>]+>")
_SUMMARY_MAX_CHARS = 300
_EXTRACT_MAX_CHARS = 2000


class NewsError(RuntimeError):
    pass

def _default_get_text(url: str) -> str:
    import requests

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text

def _clean_summary(raw: Optional[str]) -> str:
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = " ".join(text.split())
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS].rstrip() + "..."
    return text

def _parse_feed(xml_text: str, limit: int) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise NewsError(f"Couldn't parse RSS feed: {e}") from e

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": html.unescape(title),
            "link": (item.findtext("link") or "").strip(),
            "published": (item.findtext("pubDate") or "").strip(),
            "summary": _clean_summary(item.findtext("description")),
        })
        if len(items) >= limit:
            break
    return items

def _default_feeds() -> list[str]:
    configured = config.GOVEE_NEWS_FEEDS
    if configured:
        return [url.strip() for url in configured.split(",") if url.strip()]
    return [DEFAULT_FEED]


def _trusted_domains() -> list[str]:
    """Return a small, editable publisher allowlist without URL paths."""
    domains = []
    for raw in config.GOVEE_ACTIVE_RESEARCH_DOMAINS.split(","):
        domain = raw.strip().lower().removeprefix("https://").removeprefix("http://").split("/")[0]
        if domain and re.fullmatch(r"[a-z0-9.-]+", domain):
            domains.append(domain)
    return list(dict.fromkeys(domains))


class NewsClient:
    def __init__(self, get_text_fn: Optional[Callable[[str], str]] = None):
        self._get_text = get_text_fn or _default_get_text

    def get_headlines(self, topic: Optional[str] = None, limit: int = 5) -> list[dict]:
        if topic and topic.strip():
            feeds = [SEARCH_FEED.format(query=topic.strip().replace(" ", "+"))]
        else:
            feeds = _default_feeds()

        items: list[dict] = []
        errors: list[str] = []
        for url in feeds:
            try:
                xml_text = self._get_text(url)
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
                continue
            items.extend(_parse_feed(xml_text, limit - len(items)))
            if len(items) >= limit:
                break

        if not items and errors:
            raise NewsError(f"Couldn't fetch news: {'; '.join(errors)}")

        return items[:limit]

    def get_trusted_headlines(
        self, topic: str, limit: int = 5, domains: Optional[list[str]] = None
    ) -> dict:
        """Research a topic across an allowlisted set of publisher domains.

        Google News RSS is used only as a domain-constrained discovery index;
        returned article links and publication dates are preserved so the writer
        can name its sources. This method never attempts to work around a
        paywall or access restriction.
        """
        topic = topic.strip()
        if not topic:
            raise NewsError("A topic is required for trusted-source research")
        limit = max(1, min(20, int(limit)))
        domains = domains or _trusted_domains()
        if not domains:
            raise NewsError("No trusted research domains are configured")

        # Fetch a small slice from each source before merging, so the first
        # publisher cannot crowd out the rest of the research briefing.
        per_source = max(1, (limit + len(domains) - 1) // len(domains))
        by_source: dict[str, list[dict]] = {}
        errors: list[str] = []
        for domain in domains:
            query = quote_plus(f"site:{domain} {topic}")
            url = SLOVAK_SEARCH_FEED.format(query=query)
            try:
                source_items = _parse_feed(self._get_text(url), per_source)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{domain}: {e}")
                continue
            for item in source_items:
                item["source"] = domain
            by_source[domain] = source_items

        if not by_source and errors:
            raise NewsError(f"Couldn't research trusted sources: {'; '.join(errors)}")

        # Round-robin ordering gives a compact cross-source briefing. De-dupe
        # repeated syndicated stories while retaining the original publisher.
        articles: list[dict] = []
        seen: set[tuple[str, str]] = set()
        while len(articles) < limit and any(by_source.values()):
            for domain in domains:
                source_items = by_source.get(domain, [])
                if not source_items:
                    continue
                item = source_items.pop(0)
                fingerprint = (item.get("title", "").strip().lower(), item.get("link", ""))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                articles.append(item)
                if len(articles) >= limit:
                    break

        return {
            "topic": topic,
            "articles": articles,
            "sources_consulted": [domain for domain in domains if domain in by_source],
            "source_errors": errors,
        }

    def get_article_extract(self, url: str, max_chars: int = _EXTRACT_MAX_CHARS) -> str:
        import trafilatura

        try:
            page_html = self._get_text(url)
        except Exception as e:  # noqa: BLE001
            raise NewsError(f"Couldn't fetch article at {url}: {e}") from e

        text = trafilatura.extract(page_html, include_comments=False, include_tables=False)
        if not text or not text.strip():
            raise NewsError(f"Couldn't extract article content from {url}")

        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        return text

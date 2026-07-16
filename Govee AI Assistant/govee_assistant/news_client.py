#news_client.py

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import Callable, Optional

from . import config

logger = logging.getLogger("news_client")

DEFAULT_FEED = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
SEARCH_FEED = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

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

    def get_article_extract(self, url: str, max_chars: int = _EXTRACT_MAX_CHARS) -> str:
        """Fetch an article's page and pull out the main body text (not just
        the RSS summary), for when the user wants more than a headline."""
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

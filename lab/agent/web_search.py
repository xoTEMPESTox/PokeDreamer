"""web_search.py — Web search tool for PokeDreamer LLM System 2 planner.

Queries the Bulbapedia MediaWiki API for Pokémon game knowledge (maps, gym leaders,
types, moves, etc.) with rate limiting, timeouts, and on-disk JSON caching.
"""
from __future__ import annotations

import html
import http.client
import json
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

API_ENDPOINT: str = "https://bulbapedia.bulbagarden.net/w/api.php"
USER_AGENT: str = "PokeDreamer-research/1.0"
DEFAULT_TIMEOUT: float = 10.0
DEFAULT_CACHE_DIR: str = "work/web_cache"

_last_request_time: float = 0.0
_min_request_interval: float = 1.0


def _rate_limit(min_interval: float = _min_request_interval) -> None:
    """Enforce polite rate limiting between external API requests (<= 1 req/s)."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _clean_snippet(text: str) -> str:
    """Strip HTML tags and unescape HTML entities from a search snippet."""
    if not text:
        return ""
    stripped = re.sub(r"<[^>]+>", "", text)
    return html.unescape(stripped).strip()


def _load_cache(cache_file: Path) -> dict[str, list[dict[str, str]]]:
    """Load cached queries from on-disk JSON cache."""
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_cache(cache_file: Path, cache_data: dict[str, list[dict[str, str]]]) -> None:
    """Save queries to on-disk JSON cache."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        temp_file.replace(cache_file)
    except Exception:
        pass


def web_search(
    query: str,
    max_results: int = 3,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, str]]:
    """Search Bulbapedia MediaWiki API for query and return top results with extracts.

    Returns:
        List of dicts with {"title", "url", "snippet"}, where snippet <= 400 chars.
        Returns [] on network errors or invalid queries.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    cache_dir_path = Path(cache_dir)
    cache_file = cache_dir_path / "cache.json"
    cache_key = cleaned_query.lower()

    cached_data = _load_cache(cache_file)
    if cache_key in cached_data:
        cached_results = cached_data[cache_key]
        if isinstance(cached_results, list):
            return cached_results[:max_results]

    try:
        # Step 1: Query search list
        _rate_limit()
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": cleaned_query,
            "format": "json",
        }
        search_url = f"{API_ENDPOINT}?{urllib.parse.urlencode(search_params)}"
        search_req = urllib.request.Request(
            search_url,
            headers={"User-Agent": USER_AGENT},
        )

        with urllib.request.urlopen(search_req, timeout=timeout) as resp:
            raw_data = resp.read().decode("utf-8")
            search_data = json.loads(raw_data)

        search_items = (
            search_data.get("query", {}).get("search", [])[:max_results]
        )
        if not search_items:
            # Cache empty result for query
            cached_data[cache_key] = []
            _save_cache(cache_file, cached_data)
            return []

        page_ids = [str(item["pageid"]) for item in search_items if "pageid" in item]
        search_snippets: dict[str, str] = {
            str(item["pageid"]): _clean_snippet(item.get("snippet", ""))
            for item in search_items
            if "pageid" in item
        }

        # Step 2: Fetch plain-text extracts and canonical URLs for top page IDs
        pages: dict[str, Any] = {}
        if page_ids:
            _rate_limit()
            extract_params = {
                "action": "query",
                "prop": "extracts|info",
                "inprop": "url",
                "explaintext": "1",
                "exintro": "1",
                "pageids": "|".join(page_ids),
                "format": "json",
            }
            extract_url = f"{API_ENDPOINT}?{urllib.parse.urlencode(extract_params)}"
            extract_req = urllib.request.Request(
                extract_url,
                headers={"User-Agent": USER_AGENT},
            )

            with urllib.request.urlopen(extract_req, timeout=timeout) as resp:
                raw_extract_data = resp.read().decode("utf-8")
                extract_json = json.loads(raw_extract_data)
                pages = extract_json.get("query", {}).get("pages", {})

        results: list[dict[str, str]] = []
        for item in search_items:
            pid = str(item.get("pageid", ""))
            page_info = pages.get(pid, {})
            title = page_info.get("title") or item.get("title", "")
            raw_url = page_info.get("fullurl")
            if not raw_url:
                quoted_title = urllib.parse.quote(title.replace(" ", "_"))
                raw_url = f"https://bulbapedia.bulbagarden.net/wiki/{quoted_title}"

            extract = page_info.get("extract", "")
            if extract and extract.strip():
                snippet = extract.strip()
            else:
                snippet = search_snippets.get(pid, "")

            if len(snippet) > 400:
                snippet = snippet[:400]

            results.append({
                "title": title,
                "url": raw_url,
                "snippet": snippet,
            })

        cached_data[cache_key] = results
        _save_cache(cache_file, cached_data)
        return results

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        socket.timeout,
        TimeoutError,
        http.client.HTTPException,
        json.JSONDecodeError,
        OSError,
    ):
        return []

"""
Seed corpus utilities for benchmark-first website coverage.

Priority order:
1) Benchmark websites from WebVoyager, Mind2Web, WebDS.
2) Popular websites (LLM-generated / provided lists).
3) Long-tail websites (LLM-generated / provided lists).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from llm import chat

SOURCE_WEBVOYAGER = "webvoyager"
SOURCE_MIND2WEB = "mind2web"
SOURCE_WEBDS = "webds"
SOURCE_POPULAR = "popular"
SOURCE_LONGTAIL = "longtail"

ALL_SOURCES = {
    SOURCE_WEBVOYAGER,
    SOURCE_MIND2WEB,
    SOURCE_WEBDS,
    SOURCE_POPULAR,
    SOURCE_LONGTAIL,
}


@dataclass(frozen=True)
class SeedSite:
    url: str
    description: str
    source: str
    domain: str
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "description": self.description,
            "source": self.source,
            "domain": self.domain,
            "tags": list(self.tags),
        }


def normalize_url(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if not s.startswith(("http://", "https://")):
        s = f"https://{s}"
    p = urlparse(s)
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "/").rstrip("/")
    path = path or "/"
    # Keep homepage URLs canonical for seeds.
    if path != "/":
        path = "/"
    return f"https://{host}{path}"


def _domain_from_url(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def dedupe_seeds(items: list[SeedSite]) -> list[SeedSite]:
    seen: set[str] = set()
    out: list[SeedSite] = []
    for s in items:
        key = s.url
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _http_get_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_jsonl_lines(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def fetch_webvoyager_seeds() -> list[SeedSite]:
    """
    Parse official WebVoyager JSONL and collect unique websites.
    """
    url = "https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl"
    txt = _http_get_text(url)
    rows = _parse_jsonl_lines(txt)
    seeds: list[SeedSite] = []
    for r in rows:
        raw_url = r.get("web", "")
        norm = normalize_url(raw_url)
        if not norm:
            continue
        web_name = (r.get("web_name") or "WebVoyager website").strip()
        desc = f"Website from WebVoyager benchmark: {web_name}"
        seeds.append(
            SeedSite(
                url=norm,
                description=desc,
                source=SOURCE_WEBVOYAGER,
                domain=_domain_from_url(norm),
                tags=("benchmark", "webvoyager"),
            )
        )
    return dedupe_seeds(seeds)


_WEBDS_SLUG_TO_URL = {
    "bea": "https://www.bea.gov/",
    "arxiv": "https://arxiv.org/",
    "datausa": "https://datausa.io/",
    "noaa": "https://www.noaa.gov/",
    "understat": "https://understat.com/",
    "cfpb": "https://www.consumerfinance.gov/",
    "musicbrainz": "https://musicbrainz.org/",
    "tunebat": "https://tunebat.com/",
    "our-world-in-data": "https://ourworldindata.org/",
    "shopping": "https://www.amazon.com/",
    "apta": "https://www.apta.com/",
    "climate-gov": "https://www.climate.gov/",
    "cdc-mental-health": "https://www.cdc.gov/mentalhealth/",
    "iata": "https://www.iata.org/",
    "mit": "https://www.mit.edu/",
    "reddit": "https://www.reddit.com/",
    "st-louis-fed": "https://fred.stlouisfed.org/",
    "stocknear": "https://stocknear.com/",
    "uchicago": "https://www.uchicago.edu/",
    "unwto": "https://www.unwto.org/",
    "worldpop": "https://www.worldpop.org/",
    "cdc-covid": "https://www.cdc.gov/coronavirus/",
    "cdc-obesity": "https://www.cdc.gov/obesity/",
    "nih": "https://www.nih.gov/",
    "worldometer": "https://www.worldometers.info/",
    "fred": "https://fred.stlouisfed.org/",
    "riaa": "https://www.riaa.com/",
    "trading-economics": "https://tradingeconomics.com/",
}


def _clean_webds_slug(slug: str) -> str:
    s = slug.lower()
    s = s.replace("arkiv", "arxiv")
    s = s.replace("tunetbat", "tunebat")
    s = s.replace("st-lous-fed", "st-louis-fed")
    s = s.replace("cdc-cvoid", "cdc-covid")
    s = s.replace("cdc-obestity", "cdc-obesity")
    s = s.replace("st-louis-fed-reddit", "st-louis-fed")
    return s


def fetch_webds_seeds() -> list[SeedSite]:
    """
    Parse WebDS tasks tree and infer benchmark websites from task filenames.
    """
    api = "https://huggingface.co/api/datasets/yamhm/WebDS/tree/main/tasks"
    txt = _http_get_text(api)
    items = json.loads(txt)
    slugs: set[str] = set()
    for it in items:
        path = it.get("path", "")
        name = path.split("/")[-1]
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        # Format: 001_slug--slug2--id
        stem = re.sub(r"^\d+_", "", stem)
        parts = stem.split("--")
        for part in parts[:-1]:
            slug = _clean_webds_slug(part.strip())
            if slug:
                slugs.add(slug)

    seeds: list[SeedSite] = []
    for slug in sorted(slugs):
        url = _WEBDS_SLUG_TO_URL.get(slug)
        if not url:
            continue
        norm = normalize_url(url)
        seeds.append(
            SeedSite(
                url=norm,
                description=f"Website from WebDS benchmark ({slug})",
                source=SOURCE_WEBDS,
                domain=_domain_from_url(norm),
                tags=("benchmark", "webds"),
            )
        )
    return dedupe_seeds(seeds)


def load_mind2web_seeds_from_file(path: str | Path) -> list[SeedSite]:
    """
    Load Mind2Web websites from a local JSON/JSONL/TXT manifest.
    Expected rows:
      - {"url": "...", "description": "..."}  (json/jsonl)
      - one url per line                       (txt)
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    seeds: list[SeedSite] = []
    if p.suffix.lower() == ".json":
        raw = json.loads(txt)
        rows = raw if isinstance(raw, list) else []
    elif p.suffix.lower() == ".jsonl":
        rows = _parse_jsonl_lines(txt)
    else:
        rows = [{"url": line.strip(), "description": "Website from Mind2Web list"} for line in txt.splitlines() if line.strip()]

    for row in rows:
        raw_url = row.get("url", "")
        norm = normalize_url(raw_url)
        if not norm:
            continue
        desc = row.get("description") or "Website from Mind2Web benchmark"
        seeds.append(
            SeedSite(
                url=norm,
                description=desc,
                source=SOURCE_MIND2WEB,
                domain=_domain_from_url(norm),
                tags=("benchmark", "mind2web"),
            )
        )
    return dedupe_seeds(seeds)


_POPULAR_PROMPT = """\
Provide {n} of the most popular globally visited websites useful for web navigation agents.
Return a JSON array of objects with fields:
  - url
  - description
Rules:
  - public and broadly accessible websites
  - avoid login-only or private dashboards
  - avoid obvious bot-blocking heavy sites when possible
Return ONLY JSON."""


_LONGTAIL_PROMPT = """\
Provide {n} less-popular but useful websites for robustness testing of web agents.
Include government, university, regional, niche/public-data, and old-school UI sites.
Return a JSON array of objects with fields:
  - url
  - description
Return ONLY JSON."""


def _llm_generate_sites(prompt: str, model: str | None = None) -> list[dict]:
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        provider="openai",
        model=model,
        temperature=0.7,
        max_tokens=8192,
    )
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        arr = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict)]


def generate_popular_seeds(n: int, model: str | None = None) -> list[SeedSite]:
    rows = _llm_generate_sites(_POPULAR_PROMPT.format(n=n), model=model)
    seeds: list[SeedSite] = []
    for row in rows:
        norm = normalize_url(row.get("url", ""))
        if not norm:
            continue
        seeds.append(
            SeedSite(
                url=norm,
                description=row.get("description", "Popular website"),
                source=SOURCE_POPULAR,
                domain=_domain_from_url(norm),
                tags=("popular",),
            )
        )
    return dedupe_seeds(seeds)


def generate_longtail_seeds(n: int, model: str | None = None) -> list[SeedSite]:
    rows = _llm_generate_sites(_LONGTAIL_PROMPT.format(n=n), model=model)
    seeds: list[SeedSite] = []
    for row in rows:
        norm = normalize_url(row.get("url", ""))
        if not norm:
            continue
        seeds.append(
            SeedSite(
                url=norm,
                description=row.get("description", "Long-tail website"),
                source=SOURCE_LONGTAIL,
                domain=_domain_from_url(norm),
                tags=("longtail",),
            )
        )
    return dedupe_seeds(seeds)


def validate_seed_corpus(seeds: list[dict | SeedSite]) -> dict:
    rows: list[dict] = [s.as_dict() if isinstance(s, SeedSite) else s for s in seeds]
    by_source: dict[str, int] = {}
    domains: set[str] = set()
    invalid = 0
    for r in rows:
        src = r.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        url = normalize_url(r.get("url", ""))
        if not url:
            invalid += 1
            continue
        domains.add(_domain_from_url(url))
    return {
        "total_rows": len(rows),
        "invalid_rows": invalid,
        "unique_domains": len(domains),
        "by_source": by_source,
    }


def write_seed_corpus(path: str | Path, seeds: list[SeedSite]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s.as_dict(), ensure_ascii=False) + "\n")
    return p


def load_seed_corpus(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows

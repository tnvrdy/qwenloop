"""
Three-stage synthetic task generator for the exploration policy.

Stage 0: Ask GPT to generate diverse seed websites across many categories.
Stage 1: For each site, ask GPT what users commonly do there.
Stage 2: Create outcome-oriented browsing tasks from those activities.

Output is a JSONL file of {"url", "goal"} pairs that feeds into the
exploration loop via orchestrator.py.

Usage:
    python task_generation/task_generator.py -o tasks.jsonl --num-sites 300 -n 30
    python task_generation/task_generator.py -o tasks.jsonl --seeds custom_seeds.jsonl -n 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm import chat
from task_generation.seed_sources import (
    ALL_SOURCES,
    SOURCE_LONGTAIL,
    SOURCE_MIND2WEB,
    SOURCE_POPULAR,
    SOURCE_WEBDS,
    SOURCE_WEBVOYAGER,
    SeedSite,
    dedupe_seeds,
    fetch_webds_seeds,
    fetch_webvoyager_seeds,
    generate_longtail_seeds,
    generate_popular_seeds,
    load_mind2web_seeds_from_file,
    load_seed_corpus,
    validate_seed_corpus,
    write_seed_corpus,
)

# fallback seeds (used if benchmark fetch fails / no custom corpus provided)
DEFAULT_SEEDS: list[dict] = [
    {"url": "https://en.wikipedia.org", "description": "Free encyclopedia with articles, internal links, tables, and references", "source": SOURCE_POPULAR},
    {"url": "https://news.ycombinator.com", "description": "Tech news aggregator with ranked stories, comment threads, and job postings", "source": SOURCE_POPULAR},
    {"url": "https://github.com/explore", "description": "Code hosting with repositories, issues, pull requests, trending projects, and topics", "source": SOURCE_WEBVOYAGER},
    {"url": "https://arxiv.org", "description": "Academic preprint archive with search, abstracts, author pages, and subject categories", "source": SOURCE_WEBVOYAGER},
    {"url": "https://www.bbc.com/news", "description": "International news with articles, sections, video, and live coverage", "source": SOURCE_WEBVOYAGER},
    {"url": "https://www.allrecipes.com", "description": "Recipe site with search, categories, ingredients, instructions, and reviews", "source": SOURCE_WEBVOYAGER},
    {"url": "https://www.weather.gov", "description": "US weather service with location search, forecasts, radar maps, and alerts", "source": SOURCE_LONGTAIL},
    {"url": "https://www.openstreetmap.org", "description": "Open-source map with search, zoom, directions, and point-of-interest details", "source": SOURCE_LONGTAIL},
    {"url": "https://www.craigslist.org", "description": "Classified ads organized by city and category: housing, jobs, for sale, services", "source": SOURCE_LONGTAIL},
    {"url": "https://www.gutenberg.org", "description": "Free ebooks with search, categories, book pages, and reading formats", "source": SOURCE_LONGTAIL},
]

# stage 0: generate diverse seed websites

_SITE_GEN_PROMPT = """\
List {n} diverse, publicly accessible websites that do NOT require login to browse.

Spread them across these categories (roughly equal):
  - News & media
  - Shopping & product comparison (browsable without purchase)
  - Education & learning
  - Reference & encyclopedias
  - Government & public data
  - Food, cooking & recipes
  - Travel & maps
  - Forums & communities
  - Entertainment & culture
  - Sports
  - Finance & markets (public data)
  - Health & medical info
  - Science & research
  - Jobs & careers
  - Real estate & housing
  - Technology & tools

Requirements:
  - Include a mix of well-known and niche sites
  - Every site must be accessible without login or account creation
  - No social media that requires login (no Facebook, Instagram, Twitter/X)
  - No sites that aggressively block bots (no Amazon, Reddit, StackOverflow, IMDB)
  - Include international sites, not just US-centric ones

Return a JSON array of objects, each with "url" and "description" fields.
The description should be 1 sentence explaining what the site is and what \
interactive features it has (search, filters, categories, etc).
Return ONLY the JSON array, no other text."""


def generate_seed_sites(n: int = 300, model: str | None = None) -> list[dict]:
    """Stage 0: ask GPT to generate diverse seed websites."""
    print(f"[stage 0] generating {n} seed websites ...")
    prompt = _SITE_GEN_PROMPT.format(n=n)
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        provider="openai",
        model=model,
        temperature=0.9,
        max_tokens=16384,
    )
    sites = _parse_json_list(raw)
    valid = [s for s in sites if isinstance(s, dict) and s.get("url") and s.get("description")]
    print(f"[stage 0] got {len(valid)} valid sites")
    return valid


# stage 1: what do people commonly do on this site?

_ACTIVITIES_PROMPT = """\
What are the most common things real users do on {url}?

Site description: {description}

List 15-20 specific, concrete activities that a typical user would perform. Include a mix of:
  - Simple tasks (1-2 actions, e.g. looking something up)
  - Medium tasks (3-5 actions, e.g. comparing options, filling out a form)
  - Complex tasks (6-10 actions, e.g. researching a topic across multiple pages)

Focus on real user behavior. Include activities involving:
  - Searching and filtering
  - Reading and comparing information
  - Interacting with forms, buttons, and controls
  - Multi-page navigation

Do NOT include activities that require: logging in, creating an account, making purchases, or entering personal/payment info.

Return a JSON array of strings. Return ONLY the JSON array, no other text."""

# stage 2: create tasks from those activities

_TASK_GEN_PROMPT = """\
You are creating browsing tasks to test a web agent's ability to navigate {url}.

Here are common things real users do on this site:
{activities_text}

Using this list, create exactly {n} diverse browsing tasks.

RULES:
1. State ONLY the desired end state. Do NOT describe steps, navigation paths, or procedures.
   BAD:  "Click on the Topics link, scroll to Popular topics, and click machine-learning"
   GOOD: "Find the machine learning topic page"

   BAD:  "Search for 'Leonardo Da Vinci', click the Wikipedia article, scroll to Biography"
   GOOD: "Find the biography section of the Leonardo Da Vinci article"

   BAD:  "Navigate to the JavaScript tag page, filter for async/await questions, and find the most upvoted one"
   GOOD: "Find the most upvoted question about JavaScript async/await"

   BAD:  "Go to the recipe search, type 'pasta', filter by rating, and select the first result"
   GOOD: "Find the highest-rated pasta recipe"

   The goal is a DESTINATION, not directions. If your goal contains multiple \
   verbs chained with "and" or "then", it's probably too procedural. Rewrite it \
   as what the agent should END UP having found/seen/reached.

2. Tasks should usually require ~20 actions (longer, realistic objectives), but remain coherent and outcome-oriented.
   BAD:  "Go to the homepage"
   GOOD: "Find a highly-rated Italian recipe that takes under 30 minutes and has at least 50 reviews"

3. Mix of task types — not just "find X":
   - Searching and comparing
   - Form interaction and filtering
   - Multi-page research
   - Content navigation
   - Using site-specific features

4. No logging in, creating accounts, or entering personal/payment info.
5. Agent can: click, type into fields, scroll, navigate to URLs. No drag-and-drop or file uploads.

Return a JSON array of objects, each with a "goal" field.
Return ONLY the JSON array, no other text."""


# json response parsing

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json_list(raw: str) -> list:
    cleaned = _strip_markdown_fences(raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return result if isinstance(result, list) else []


def _parse_tasks_response(raw: str, url: str, seed_source: str | None = None) -> list[dict]:
    items = _parse_json_list(raw)
    tasks = []
    for item in items:
        goal = item.get("goal", "").strip() if isinstance(item, dict) else ""
        if goal:
            row = {"url": url, "goal": goal}
            if seed_source:
                row["seed_source"] = seed_source
            tasks.append(row)
    return tasks


# core generation functions

def _get_common_activities(url: str, description: str) -> str:
    prompt = _ACTIVITIES_PROMPT.format(url=url, description=description)
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        provider="openai",
        temperature=0.7,
        max_tokens=2048,
    )
    activities = _parse_json_list(raw)
    if not activities:
        return raw.strip()
    return "\n".join(f"  - {a}" for a in activities)


def generate_tasks_for_site(
    url: str,
    description: str,
    n: int = 30,
    model: str | None = None,
    seed_source: str | None = None,
) -> list[dict]:
    print(f"    stage 1: brainstorming activities ...")
    activities_text = _get_common_activities(url, description)

    print(f"    stage 2: generating {n} tasks ...")
    prompt = _TASK_GEN_PROMPT.format(url=url, activities_text=activities_text, n=n)
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        provider="openai",
        model=model,
        temperature=0.9,
        max_tokens=4096,
    )
    return _parse_tasks_response(raw, url, seed_source=seed_source)


def _coerce_source_set(raw: str | None) -> set[str]:
    if not raw:
        return set(ALL_SOURCES)
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return {p for p in parts if p in ALL_SOURCES}


def _limit_by_source(seeds: list[dict], max_sites_per_source: int | None) -> list[dict]:
    if not max_sites_per_source or max_sites_per_source <= 0:
        return seeds
    out: list[dict] = []
    counts: dict[str, int] = {}
    for s in seeds:
        src = s.get("source", "unknown")
        c = counts.get(src, 0)
        if c >= max_sites_per_source:
            continue
        counts[src] = c + 1
        out.append(s)
    return out


def _build_seed_pool(
    *,
    model: str | None,
    num_sites: int | None,
    seed_corpus_path: str | None,
    seed_sources_csv: str | None,
    max_sites_per_source: int | None,
    mind2web_websites_file: str | None,
    materialize_seed_corpus: str | None,
) -> list[dict]:
    if seed_corpus_path:
        seeds = load_seed_corpus(seed_corpus_path)
    else:
        source_set = _coerce_source_set(seed_sources_csv)
        bucket: list[SeedSite] = []

        if SOURCE_WEBVOYAGER in source_set:
            try:
                bucket.extend(fetch_webvoyager_seeds())
            except Exception as e:
                print(f"  WARNING: failed to fetch WebVoyager seeds: {e}")
        if SOURCE_WEBDS in source_set:
            try:
                bucket.extend(fetch_webds_seeds())
            except Exception as e:
                print(f"  WARNING: failed to fetch WebDS seeds: {e}")
        if SOURCE_MIND2WEB in source_set and mind2web_websites_file:
            try:
                bucket.extend(load_mind2web_seeds_from_file(mind2web_websites_file))
            except Exception as e:
                print(f"  WARNING: failed to load Mind2Web websites file: {e}")
        elif SOURCE_MIND2WEB in source_set:
            print("  WARNING: mind2web source requested but --mind2web-websites-file not provided")

        if SOURCE_POPULAR in source_set:
            # if num_sites is small, keep popular/longtail generation bounded.
            pop_n = 250 if not num_sites else max(50, min(500, num_sites // 2))
            bucket.extend(generate_popular_seeds(pop_n, model=model))
        if SOURCE_LONGTAIL in source_set:
            lt_n = 250 if not num_sites else max(50, min(500, num_sites // 2))
            bucket.extend(generate_longtail_seeds(lt_n, model=model))

        if not bucket:
            # strong fallback
            seeds = list(DEFAULT_SEEDS)
        else:
            seeds = [s.as_dict() for s in dedupe_seeds(bucket)]

    seeds = _limit_by_source(seeds, max_sites_per_source)
    if num_sites:
        seeds = seeds[:num_sites]

    if materialize_seed_corpus:
        serializable = [
            SeedSite(
                url=s["url"],
                description=s.get("description", ""),
                source=s.get("source", "unknown"),
                domain=s.get("domain", ""),
                tags=tuple(s.get("tags", [])),
            )
            for s in seeds
        ]
        write_seed_corpus(materialize_seed_corpus, serializable)

    stats = validate_seed_corpus(seeds)
    print(f"[seed-corpus] rows={stats['total_rows']} unique_domains={stats['unique_domains']} by_source={stats['by_source']}")
    return seeds


def generate_all_tasks(
    seeds: list[dict] | None = None,
    tasks_per_site: int = 30,
    output_path: str | Path = "tasks.jsonl",
    model: str | None = None,
    num_sites: int | None = None,
    seed_corpus_path: str | None = None,
    seed_sources_csv: str | None = None,
    max_sites_per_source: int | None = None,
    mind2web_websites_file: str | None = None,
    materialize_seed_corpus: str | None = None,
) -> int:
    if seeds is None:
        seeds = _build_seed_pool(
            model=model,
            num_sites=num_sites,
            seed_corpus_path=seed_corpus_path,
            seed_sources_csv=seed_sources_csv,
            max_sites_per_source=max_sites_per_source,
            mind2web_websites_file=mind2web_websites_file,
            materialize_seed_corpus=materialize_seed_corpus,
        )

    output_path = Path(output_path)
    total = 0

    with open(output_path, "w") as f:
        for i, seed in enumerate(seeds):
            url = seed["url"]
            desc = seed["description"]
            src = seed.get("source")
            print(f"[{i + 1}/{len(seeds)}] {url}")

            try:
                tasks = generate_tasks_for_site(url, desc, n=tasks_per_site, model=model, seed_source=src)
            except Exception as e:
                print(f"  WARNING: failed for {url}: {e}")
                continue

            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

            total += len(tasks)
            print(f"  got {len(tasks)} tasks (total: {total})\n")

    print(f"Done. {total} tasks written to {output_path}")
    return total


def _load_seeds(path: str | Path) -> list[dict]:
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic browsing tasks (three-stage)")
    parser.add_argument("-o", "--output", default="tasks.jsonl", help="Output JSONL path")
    parser.add_argument("-n", "--tasks-per-site", type=int, default=30, help="Tasks per site")
    parser.add_argument("--seeds", default=None, help="Custom seeds JSONL (overrides generation)")
    parser.add_argument("--seed-corpus", default=None, help="Canonical seed corpus JSONL to load")
    parser.add_argument(
        "--seed-sources",
        default="webvoyager,mind2web,webds,popular,longtail",
        help="Comma-separated seed sources to include",
    )
    parser.add_argument("--max-sites-per-source", type=int, default=None, help="Optional cap per source bucket")
    parser.add_argument(
        "--mind2web-websites-file",
        default=None,
        help="Path to a local list of Mind2Web websites (json/jsonl/txt)",
    )
    parser.add_argument(
        "--materialize-seed-corpus",
        default=None,
        help="Write resolved seed pool to JSONL for reproducibility",
    )
    parser.add_argument("--num-sites", type=int, default=None,
                        help="Number of seed sites to use after filtering/deduplication.")
    parser.add_argument("--model", default=None, help="LLM model override")
    parser.add_argument(
        "--validate-seeds-only",
        action="store_true",
        help="Only resolve/validate seed corpus and exit without generating tasks",
    )
    args = parser.parse_args()

    seeds = _load_seeds(args.seeds) if args.seeds else None
    if args.validate_seeds_only:
        resolved = _build_seed_pool(
            model=args.model,
            num_sites=args.num_sites,
            seed_corpus_path=args.seed_corpus,
            seed_sources_csv=args.seed_sources,
            max_sites_per_source=args.max_sites_per_source,
            mind2web_websites_file=args.mind2web_websites_file,
            materialize_seed_corpus=args.materialize_seed_corpus,
        )
        print(f"Validation complete. {len(resolved)} seeds resolved.")
    else:
        generate_all_tasks(
            seeds=seeds,
            tasks_per_site=args.tasks_per_site,
            output_path=args.output,
            model=args.model,
            num_sites=args.num_sites,
            seed_corpus_path=args.seed_corpus,
            seed_sources_csv=args.seed_sources,
            max_sites_per_source=args.max_sites_per_source,
            mind2web_websites_file=args.mind2web_websites_file,
            materialize_seed_corpus=args.materialize_seed_corpus,
        )

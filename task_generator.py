"""
Three-stage synthetic task generator for the exploration policy.

Stage 0: Ask GPT to generate diverse seed websites across many categories.
Stage 1: For each site, ask GPT what users commonly do there.
Stage 2: Create outcome-oriented browsing tasks from those activities.

Output is a JSONL file of {"url", "goal"} pairs that feeds into the
exploration loop via orchestrator.py.

Usage:
    python task_generator.py -o tasks.jsonl --num-sites 300 -n 30
    python task_generator.py -o tasks.jsonl --seeds custom_seeds.jsonl -n 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm import chat

# seed websites

DEFAULT_SEEDS: list[dict] = [
    {"url": "https://en.wikipedia.org", "description": "Free encyclopedia with articles, internal links, tables, and references"},
    {"url": "https://news.ycombinator.com", "description": "Tech news aggregator with ranked stories, comment threads, and job postings"},
    {"url": "https://github.com/explore", "description": "Code hosting with repositories, issues, pull requests, trending projects, and topics"},
    {"url": "https://www.bbc.com/news", "description": "International news with articles, sections, video, and live coverage"},
    {"url": "https://arxiv.org", "description": "Academic preprint archive with search, abstracts, author pages, and subject categories"},
    {"url": "https://www.allrecipes.com", "description": "Recipe site with search, categories, ingredients, instructions, and reviews"},
    {"url": "https://www.craigslist.org", "description": "Classified ads organized by city and category: housing, jobs, for sale, services"},
    {"url": "https://duckduckgo.com", "description": "Search engine with web search, instant answers, and image/video/news tabs"},
    {"url": "https://www.weather.gov", "description": "US weather service with location search, forecasts, radar maps, and alerts"},
    {"url": "https://www.openstreetmap.org", "description": "Open-source map with search, zoom, directions, and point-of-interest details"},
    {"url": "https://www.gutenberg.org", "description": "Free ebooks with search, categories, book pages, and reading formats"},
    {"url": "https://www.wolframalpha.com", "description": "Computational knowledge engine with queries, results, and step-by-step solutions"},
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

2. Tasks should require 3-8 actions. Not trivially easy.
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


def _parse_tasks_response(raw: str, url: str) -> list[dict]:
    items = _parse_json_list(raw)
    tasks = []
    for item in items:
        goal = item.get("goal", "").strip() if isinstance(item, dict) else ""
        if goal:
            tasks.append({"url": url, "goal": goal})
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
    return _parse_tasks_response(raw, url)


def generate_all_tasks(
    seeds: list[dict] | None = None,
    tasks_per_site: int = 30,
    output_path: str | Path = "tasks.jsonl",
    model: str | None = None,
    num_sites: int | None = None,
) -> int:
    if seeds is None:
        if num_sites and num_sites > len(DEFAULT_SEEDS):
            seeds = generate_seed_sites(n=num_sites, model=model)
        else:
            seeds = DEFAULT_SEEDS

    if num_sites:
        seeds = seeds[:num_sites]

    output_path = Path(output_path)
    total = 0

    with open(output_path, "w") as f:
        for i, seed in enumerate(seeds):
            url = seed["url"]
            desc = seed["description"]
            print(f"[{i + 1}/{len(seeds)}] {url}")

            try:
                tasks = generate_tasks_for_site(url, desc, n=tasks_per_site, model=model)
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
    parser.add_argument("--num-sites", type=int, default=None,
                        help="Number of seed sites. If > default seeds, Stage 0 generates them via LLM.")
    parser.add_argument("--model", default=None, help="LLM model override")
    args = parser.parse_args()

    seeds = _load_seeds(args.seeds) if args.seeds else None
    generate_all_tasks(
        seeds=seeds,
        tasks_per_site=args.tasks_per_site,
        output_path=args.output,
        model=args.model,
        num_sites=args.num_sites,
    )

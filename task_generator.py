"""
Two-stage synthetic task generator for the exploration policy.

Stage 1: Ask GPT what common activities real users perform on a website.
Stage 2: Ask GPT to create challenging, outcome-oriented browsing tasks
         drawn from those activities.

Output is a JSONL file of {"url", "goal"} pairs that feeds into the
exploration loop via orchestrator.py.

Usage:
    python task_generator.py -o tasks.jsonl -n 30
    python task_generator.py -o tasks.jsonl -n 50 --seeds custom_seeds.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm import chat

# seed websites

DEFAULT_SEEDS: list[dict] = [
    {
        "url": "https://en.wikipedia.org",
        "description": "Free encyclopedia with millions of articles, internal links, tables, infoboxes, categories, and references",
    },
    {
        "url": "https://www.reddit.com",
        "description": "Social platform with subreddits, posts, nested comment threads, upvoting, sorting, and sidebar info",
    },
    {
        "url": "https://news.ycombinator.com",
        "description": "Tech news aggregator with ranked story links, nested comment threads, job postings, and user pages",
    },
    {
        "url": "https://stackoverflow.com",
        "description": "Programming Q&A with questions, answers, code blocks, tags, voting, filters, and user profiles",
    },
    {
        "url": "https://github.com/explore",
        "description": "Code hosting with repositories, issues, pull requests, trending projects, topics, and user/org profiles",
    },
    {
        "url": "https://www.amazon.com",
        "description": "E-commerce with product search, categories, filters, product pages, reviews, cart, wishlists, and comparisons",
    },
    {
        "url": "https://www.bbc.com/news",
        "description": "International news with articles, sections (world, business, tech, sport), video, and live coverage",
    },
    {
        "url": "https://www.imdb.com",
        "description": "Movie/TV database with search, title pages, cast/crew lists, ratings, reviews, top-250 charts, and watchlists",
    },
    {
        "url": "https://www.espn.com",
        "description": "Sports news with live scores, standings, team pages, player stats, schedules, and fantasy tools",
    },
    {
        "url": "https://arxiv.org",
        "description": "Academic preprint archive with search, paper abstracts, author pages, subject categories, and citation links",
    },
    {
        "url": "https://www.allrecipes.com",
        "description": "Recipe site with search, categories, recipe pages with ingredients/instructions, reviews, and meal planning",
    },
    {
        "url": "https://www.craigslist.org",
        "description": "Classified ads organized by city and category: housing, jobs, for sale, services, community, gigs",
    },
    {
        "url": "https://duckduckgo.com",
        "description": "Search engine with web search, instant answers, image/video/news tabs, and settings",
    },
    {
        "url": "https://www.goodreads.com",
        "description": "Book discovery with search, book/author pages, reviews, ratings, reading lists, and genre browsing",
    },
    {
        "url": "https://www.weather.gov",
        "description": "US weather service with location search, forecasts, radar/satellite maps, alerts, and climate data",
    },
    {
        "url": "https://www.openstreetmap.org",
        "description": "Open-source map with search, zoom, pan, layer selection, directions, and point-of-interest details",
    },
]

# stage 1: what do people commonly do on this site?

_ACTIVITIES_PROMPT = """\
What are the most common things real users do on {url}?

Site description: {description}

List 15-20 specific, concrete activities that a typical user would perform on this website. Include a mix of:
  - Simple tasks (1-2 actions, e.g. looking something up)
  - Medium tasks (3-5 actions, e.g. comparing options, filling out a form)
  - Complex tasks (6-10 actions, e.g. researching a topic across multiple pages, building a cart)

Focus on real user behavior, not just browsing. Include activities that involve:
  - Searching and filtering
  - Reading and comparing information
  - Interacting with forms, buttons, and controls
  - Multi-page navigation to accomplish a goal

Do NOT include activities that require: logging in, creating an account, making real purchases, or entering personal/payment information.

Return a JSON array of strings. Return ONLY the JSON array, no other text."""

# stage 2: create tasks from those activities

_TASK_GEN_PROMPT = """\
You are creating browsing tasks to test a web agent's ability to navigate {url}.

Here are common things real users do on this site:
{activities_text}

Using this list as inspiration, create exactly {n} diverse, challenging browsing tasks.

CRITICAL RULES:
1. State the DESIRED OUTCOME, not the steps. Describe WHAT the agent should achieve, not HOW.
   BAD:  "Click on the Topics link, scroll to Popular topics, and click machine-learning"
   GOOD: "Find the machine learning topic page on GitHub"

   BAD:  "Search for 'Leonardo Da Vinci', click the Wikipedia article, scroll to the Biography section"
   GOOD: "Navigate to the biography section of the Leonardo Da Vinci article"

2. Tasks should require MULTIPLE steps (at least 3-4 actions, ideally 5-8). Simple one-click tasks are too easy.
   BAD:  "Go to the homepage"
   GOOD: "Find a highly-rated Italian restaurant recipe that takes under 30 minutes and has at least 50 reviews"

3. Include a MIX of task types — not just "find X". Include tasks that involve:
   - Searching and comparing (e.g. "find the cheapest wireless headphones with at least 4-star reviews")
   - Form interaction (e.g. "search for apartments in Boston under $2000/month")
   - Multi-page research (e.g. "find out which actor has appeared in the most top-250 IMDB films")
   - Content navigation (e.g. "navigate to the references section of the quantum computing article")
   - Using site features (e.g. "sort the recipe results by rating and find one with under 5 ingredients")

4. Do NOT require logging in, creating accounts, or entering personal/payment information.
5. The agent can: click, type into fields, scroll, and navigate to URLs. No drag-and-drop or file uploads.

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


# two-stage task generation

def _get_common_activities(url: str, description: str) -> str:
    """Stage 1: ask GPT what users commonly do on this site."""
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
    """
    Two-stage task generation for a single site.
    Stage 1 (GPT): brainstorm common user activities.
    Stage 2 (GPT): create outcome-oriented tasks from those activities.
    """
    print(f"    stage 1: brainstorming activities ...")
    activities_text = _get_common_activities(url, description)

    print(f"    stage 2: generating {n} tasks ...")
    prompt = _TASK_GEN_PROMPT.format(
        url=url,
        activities_text=activities_text,
        n=n,
    )
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
) -> int:
    seeds = seeds or DEFAULT_SEEDS
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


# load file of seed sites

def _load_seeds(path: str | Path) -> list[dict]:
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic browsing tasks (two-stage)")
    parser.add_argument("-o", "--output", default="tasks.jsonl", help="Output JSONL path")
    parser.add_argument("-n", "--tasks-per-site", type=int, default=30, help="Tasks to generate per site")
    parser.add_argument("--seeds", default=None, help="Custom seeds JSONL file (overrides defaults)")
    parser.add_argument("--model", default=None, help="LLM model override for stage 2")
    args = parser.parse_args()

    seeds = _load_seeds(args.seeds) if args.seeds else None
    generate_all_tasks(
        seeds=seeds,
        tasks_per_site=args.tasks_per_site,
        output_path=args.output,
        model=args.model,
    )

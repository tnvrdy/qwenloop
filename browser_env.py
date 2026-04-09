"""
Observable playwright browser environment for the exploration agent.

  - get_text_observation() -- numbered interactive-element list for the LLM prompt
  - get_ax_tree()          -- accessibility tree in WebArena bracket-ID format
  - get_html()             -- raw page HTML
  - screenshot()           -- viewport PNG
  - capture_full_state()   -- all of the above
  - execute_action()       -- execute a ParsedAction against the live page

Sad truth of life, playwright's accessibility snapshot is no more :( Could have instead used aria snapshot which
gives us yaml and would be lighter, but decided to use cdp's getFullAXTree instead.
can revisit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from playwright.sync_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Locator,
    Page,
    Playwright,
    sync_playwright,
)

from actions import ParsedAction


_INTERACTIVE_SELECTOR = (
    "a[href], button, input, textarea, select, "
    '[role="button"], [role="link"], [role="textbox"], [role="searchbox"], '
    '[role="checkbox"], [role="radio"], [role="tab"], [role="menuitem"]'
)

_TRUNCATION_MARK = "\n...[truncated]"

_SKIP_EMPTY_ROLES = frozenset({
    "generic", "img", "list", "strong", "paragraph",
    "banner", "navigation", "Section", "LabelText",
    "Legend", "listitem",
})

_IGNORED_PROPERTIES = frozenset({
    "focusable", "editable", "readonly", "level",
    "settable", "multiline", "invalid",
})


@dataclass
class Observation:
    """Lightweight text observation used to build the LLM prompt."""
    text: str
    n_elements: int
    truncated: bool
    element_descs: list[str]


# ax-tree via CDP

def fetch_ax_tree(cdp: CDPSession) -> list[dict]:
    """
    Fetch the full accessibility tree via CDP and deduplicate nodes.
    Returns a flat list of AX node dicts.
    """
    raw_nodes: list[dict] = cdp.send("Accessibility.getFullAXTree", {})["nodes"]

    seen: set[str] = set()
    nodes: list[dict] = []
    for node in raw_nodes:
        nid = node["nodeId"]
        if nid not in seen:
            nodes.append(node)
            seen.add(nid)
    return nodes


def format_ax_tree(nodes: list[dict]) -> str:
    """
    Format a CDP accessibility node list into a bracket-ID string,
    matching the WebArena format used in train.jsonl.

    Output example:
        [node-1] RootWebArea 'Example Domain' focused: True
            [node-5] heading 'Example Domain' level: 2
            [node-8] link 'More information...'
    """
    if not nodes:
        return "(empty accessibility tree)"

    node_id_to_idx: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        node_id_to_idx[node["nodeId"]] = idx

    def _dfs(idx: int, node_id: str, depth: int) -> str:
        node = nodes[idx]
        indent = "\t" * depth
        tree_str = ""
        valid = True

        try:
            role = node["role"]["value"]
            name = node["name"]["value"]
            node_str = f"[{node_id}] {role} {repr(name)}"

            props: list[str] = []
            for prop in node.get("properties", []):
                try:
                    if prop["name"] not in _IGNORED_PROPERTIES:
                        props.append(f'{prop["name"]}: {prop["value"]["value"]}')
                except KeyError:
                    pass
            if props:
                node_str += " " + " ".join(props)

            if not node_str.strip():
                valid = False
            if not name.strip() and not props and role in _SKIP_EMPTY_ROLES:
                valid = False

            if valid:
                tree_str += f"{indent}{node_str}"
        except Exception:
            valid = False

        for child_id in node.get("childIds", []):
            if child_id not in node_id_to_idx:
                continue
            child_depth = depth + 1 if valid else depth
            child_str = _dfs(node_id_to_idx[child_id], child_id, child_depth)
            if child_str.strip():
                if tree_str.strip():
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    return _dfs(0, nodes[0]["nodeId"], 0)


@dataclass
class BrowserEnv:
    """
    Manages a single Chromium session.  Use as a context manager:

        with BrowserEnv() as env:
            env.goto("https://example.com")
            obs = env.get_text_observation()
    """

    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720

    _playwright: Optional[Playwright] = field(default=None, repr=False)
    _browser: Optional[Browser] = field(default=None, repr=False)
    _context: Optional[BrowserContext] = field(default=None, repr=False)
    _page: Optional[Page] = field(default=None, repr=False)
    _cdp: Optional[CDPSession] = field(default=None, repr=False)
    _last_interactive_locators: List[Locator] = field(default_factory=list, repr=False)

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserEnv not started")
        return self._page

    @property
    def cdp(self) -> CDPSession:
        if self._cdp is None:
            raise RuntimeError("BrowserEnv not started")
        return self._cdp

    def start(self) -> None:
        if self._playwright is not None:
            return
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        self._page = self._context.new_page()
        self._cdp = self._page.context.new_cdp_session(self._page)
        self._cdp.send("Accessibility.enable")

    def stop(self) -> None:
        if self._cdp:
            try:
                self._cdp.detach()
            except Exception:
                pass
        self._cdp = None
        if self._context:
            self._context.close()
        self._context = None
        self._page = None
        if self._browser:
            self._browser.close()
        self._browser = None
        if self._playwright:
            self._playwright.stop()
        self._playwright = None
        self._last_interactive_locators = []

    def __enter__(self) -> BrowserEnv:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # navigation

    def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        self.page.goto(url, wait_until=wait_until)

    def _wait_settled(self, timeout: int = 8000) -> None:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except Exception:
            pass

    # text observations

    def _collect_visible_elements(self) -> tuple[list[int], list[str]]:
        """
        Single JS call: find all interactive elements, filter to only visible
        ones, return (indices_into_querySelectorAll, descriptions).
        """
        try:
            results = self.page.evaluate(
                """(selector) => {
                    const els = Array.from(document.querySelectorAll(selector));
                    const out = [];
                    for (let i = 0; i < els.length; i++) {
                        const el = els[i];
                        const r = el.getBoundingClientRect();
                        if (r.width === 0 && r.height === 0) continue;
                        if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') continue;

                        const tag = el.tagName.toLowerCase();
                        const type = el.getAttribute('type') || '';
                        const role = el.getAttribute('role') || '';
                        const placeholder = el.getAttribute('placeholder') || '';
                        const al = el.getAttribute('aria-label') || '';
                        const name = el.getAttribute('name') || '';
                        let text = (el.innerText || '').trim().replace(/\\s+/g, ' ');
                        if (text.length > 120) text = text.slice(0, 117) + '...';
                        const bits = [tag + (type ? '[' + type + ']' : '')];
                        if (role) bits.push('role=' + role);
                        const label = [al, placeholder, name, text].find(s => s && s.length);
                        out.push({idx: i, desc: bits.join(' ') + (label ? ' | ' + label : '')});
                    }
                    return out;
                }""",
                _INTERACTIVE_SELECTOR,
            )
        except Exception:
            return [], []
        indices = [r["idx"] for r in results]
        descs = [r["desc"] for r in results]
        return indices, descs

    def get_text_observation(self, max_chars: int = 16_000) -> Observation:
        """
        Build a text snapshot of the current page for the LLM prompt:
        URL, title, then numbered interactive elements (visible only).
        """
        self._wait_settled()

        try:
            title = self.page.title()
        except Exception:
            title = "(loading...)"

        visible_indices, descs = self._collect_visible_elements()

        root = self.page.locator(_INTERACTIVE_SELECTOR)
        self._last_interactive_locators = [root.nth(i) for i in visible_indices]

        lines = [
            f"URL: {self.page.url}",
            f"Title: {title}",
            f"Interactive elements: {len(descs)}",
            "",
        ]
        for i, desc in enumerate(descs):
            lines.append(f"[{i}] {desc}")

        body = "\n".join(lines)
        truncated = len(body) > max_chars
        if truncated:
            text = body[: max_chars - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
        else:
            text = body

        return Observation(
            text=text,
            n_elements=len(self._last_interactive_locators),
            truncated=truncated,
            element_descs=descs,
        )

    # richer observations

    def get_ax_tree(self) -> str:
        """
        Return the full accessibility tree as a formatted string in the
        WebArena bracket-ID notation (e.g. `[node-5] link 'Home'`).
        Uses CDP Accessibility api (like webarena/browsergym, etc.)
        """
        self._wait_settled()
        try:
            nodes = fetch_ax_tree(self.cdp)
            return format_ax_tree(nodes)
        except Exception as e:
            return f"(ax tree error: {e})"

    def get_html(self) -> str:
        """Return the full DOM HTML of the current page."""
        try:
            return self.page.content()
        except Exception:
            return ""

    def screenshot(self, path: str | Path) -> Path:
        """Save a viewport screenshot and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(p))
        return p

    def capture_full_state(self, screenshot_path: str | Path) -> dict:
        """
        Capture everything we log per step: URL, title, AX tree, HTML,
        and a viewport screenshot.
        """
        self._wait_settled()

        try:
            title = self.page.title()
        except Exception:
            title = "(loading...)"

        self.screenshot(screenshot_path)

        return {
            "url": self.page.url,
            "title": title,
            "ax_tree": self.get_ax_tree(),
            "html": self.get_html(),
            "screenshot_path": str(screenshot_path),
        }

    # action execution

    def _resolve_locator(self, index: int | None, action_name: str):
        if index is None:
            return None, {"ok": False, "error": f"{action_name}: missing index"}
        if not self._last_interactive_locators:
            return None, {"ok": False, "error": f"{action_name}: no observation yet"}
        if index < 0 or index >= len(self._last_interactive_locators):
            hi = len(self._last_interactive_locators) - 1
            return None, {"ok": False, "error": f"{action_name}: index {index} out of range (0-{hi})"}
        loc = self._last_interactive_locators[index]
        try:
            loc.scroll_into_view_if_needed(timeout=8000)
        except Exception:
            pass
        return loc, None

    def execute_action(self, action: ParsedAction) -> dict:
        """Execute a ParsedAction on the live page.  Returns {ok, error?, done?}."""
        try:
            at = action.action_type

            if at == "stop":
                return {"ok": True, "done": True}

            if at == "back":
                self.page.go_back()
                self._wait_settled(3000)
                return {"ok": True}

            if at == "scroll_up":
                self.page.mouse.wheel(0, -800)
                return {"ok": True}

            if at == "scroll_down":
                self.page.mouse.wheel(0, 800)
                return {"ok": True}

            if at == "goto":
                if not action.url:
                    return {"ok": False, "error": "goto: missing url"}
                self.goto(action.url)
                return {"ok": True}

            if at in ("click", "type"):
                loc, err = self._resolve_locator(action.index, at)
                if err:
                    return err

                if at == "click":
                    loc.click(timeout=5000)
                    self._wait_settled()
                else:
                    loc.click(timeout=5000)
                    loc.fill(action.text or "", timeout=5000)
                    if action.submit:
                        url_before = self.page.url
                        self.page.keyboard.press("Enter")
                        try:
                            self.page.wait_for_url(lambda u: u != url_before, timeout=3000)
                        except Exception:
                            pass
                        self._wait_settled()
                return {"ok": True}

            return {"ok": False, "error": f"unknown action type: {at}"}

        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

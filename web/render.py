"""Server-rendered HTML for the hosted demos.

No JavaScript framework, no build step, no CDN. Forms post back to the same URL,
which keeps every page shareable as a plain link and keeps the whole site inside
one Python function.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

REPO = "https://github.com/Arnobrizwan/applied-llm-systems"
PORTFOLIO = "https://arnobrizwan.github.io"

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0c0a;color:#f2efe6;font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
     -webkit-font-smoothing:antialiased}
a{color:#c8f73c;text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:940px;margin:0 auto;padding:48px 22px 90px}
header{border-bottom:1px solid rgba(242,239,230,.14);padding-bottom:26px;margin-bottom:34px}
.kicker{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#8f9285}
h1{font-size:clamp(28px,5vw,44px);line-height:1.1;letter-spacing:-.02em;margin:10px 0}
h2{font-size:20px;margin:34px 0 12px;letter-spacing:-.01em}
p{color:#c9c7bd;margin:10px 0}
.lede{font-size:18px;color:#e6e3d8}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:22px}
.card{display:block;padding:18px;border:1px solid rgba(242,239,230,.14);border-radius:14px;
      background:linear-gradient(160deg,rgba(200,247,60,.05),rgba(255,92,26,.03));transition:.25s}
.card:hover{border-color:#c8f73c;transform:translateY(-2px);text-decoration:none}
.card .n{font-size:12px;color:#8f9285;letter-spacing:.12em}
.card h3{font-size:17px;margin:6px 0 6px;color:#f2efe6}
.card p{font-size:14px;color:#a9a89e;margin:0}
form{margin:18px 0}
textarea{width:100%;min-height:92px;padding:14px;border-radius:12px;background:#131410;
         border:1px solid rgba(242,239,230,.18);color:#f2efe6;font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
textarea:focus{outline:none;border-color:#c8f73c}
button{margin-top:12px;padding:11px 22px;border-radius:999px;border:0;background:#c8f73c;color:#0b0c0a;
       font-weight:650;font-size:15px;cursor:pointer}
button:hover{background:#d7ff5c}
pre{white-space:pre-wrap;word-break:break-word;background:#0f100d;border:1px solid rgba(242,239,230,.14);
    border-radius:12px;padding:18px;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:#dcdad0;overflow-x:auto}
.ex{display:inline-block;margin:4px 6px 0 0;padding:6px 12px;border-radius:999px;font-size:13px;
    border:1px solid rgba(242,239,230,.2);color:#c9c7bd}
.ex:hover{border-color:#c8f73c;color:#c8f73c;text-decoration:none}
.meta{font-size:13px;color:#8f9285;margin-top:8px}
.nav{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;margin-bottom:8px}
footer{margin-top:60px;padding-top:22px;border-top:1px solid rgba(242,239,230,.14);font-size:14px;color:#8f9285}
.badge{display:inline-block;padding:5px 12px;border-radius:999px;border:1px solid rgba(200,247,60,.4);
       color:#c8f73c;font-size:12px;letter-spacing:.04em}
"""


def _page(title: str, body: str, description: str = "") -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<style>{_CSS}</style>
</head><body><div class="wrap">{body}
<footer>Built by <a href="{PORTFOLIO}">Arnob Rizwan Ahmad</a>.
Source on <a href="{REPO}">GitHub</a>. Runs on free providers, no paid API anywhere.</footer>
</div></body></html>"""


def index_page(systems: List[Dict[str, Any]]) -> str:
    cards = "\n".join(
        f"""<a class="card" href="/s/{html.escape(s['slug'])}">
  <span class="n">{s['number']:02d}</span>
  <h3>{html.escape(s['title'])}</h3>
  <p>{html.escape(s['tagline'])}</p>
</a>"""
        for s in systems
    )
    body = f"""<header>
  <div class="kicker">Applied LLM Systems</div>
  <h1>Fifteen AI systems you can run right here</h1>
  <p class="lede">Each page runs the real code on this server and shows you the actual output.
  Nothing is pre-recorded, and there is no API key behind any of it.</p>
  <p class="meta"><span class="badge">{len(systems)} live demos</span> &nbsp; <a href="{REPO}">Source on GitHub</a> &nbsp; <a href="{PORTFOLIO}">Portfolio</a></p>
</header>
<div class="grid">{cards}</div>"""
    return _page("Applied LLM Systems - 15 live demos", body,
                 "Fifteen applied LLM systems you can run in the browser: RAG, evaluation, guardrails, routing, agents.")


def system_page(s: Dict[str, Any], user_input: str = "", output: Optional[str] = None,
                elapsed_ms: Optional[float] = None) -> str:
    examples = "".join(
        f'<a class="ex" href="/s/{html.escape(s["slug"])}?q={html.escape(_q(ex))}">{html.escape(ex[:70])}</a>'
        for ex in s["examples"]
    )
    what = "".join(f"<p>{html.escape(par.strip())}</p>" for par in s["what"].split("\n\n") if par.strip())

    if s["input_label"]:
        form = f"""<h2>Try it</h2>
<form method="post">
  <label class="meta" for="q">{html.escape(s['input_label'])}</label>
  <textarea id="q" name="q" placeholder="{html.escape(s['placeholder'])}">{html.escape(user_input)}</textarea>
  <div>{examples}</div>
  <button type="submit">Run it</button>
</form>"""
    else:
        form = f"""<h2>Run it</h2>
<form method="post"><input type="hidden" name="q" value="">
  <button type="submit">Run the demo</button>
</form>"""

    result = ""
    if output is not None:
        timing = f'<p class="meta">Computed live in {elapsed_ms:.0f} ms.</p>' if elapsed_ms is not None else ""
        result = f"<h2>Output</h2><pre>{html.escape(output)}</pre>{timing}"

    body = f"""<header>
  <div class="nav"><a href="/">All fifteen</a><a href="{REPO}/tree/main/{html.escape(s['source'])}">Source code</a><a href="{PORTFOLIO}">Portfolio</a></div>
  <div class="kicker">System {s['number']:02d}</div>
  <h1>{html.escape(s['title'])}</h1>
  <p class="lede">{html.escape(s['tagline'])}</p>
</header>
<h2>What it does</h2>{what}
{form}
{result}"""
    return _page(f"{s['title']} - Applied LLM Systems", body, s["tagline"])


def not_found() -> str:
    return _page("Not found", '<header><h1>Not found</h1><p>Nothing at this address. '
                              '<a href="/">Back to the fifteen</a>.</p></header>')


def _q(text: str) -> str:
    from urllib.parse import quote
    return quote(text, safe="")

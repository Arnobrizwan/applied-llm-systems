# Web adapter contract

One adapter per system, at `web/adapters/aNN_<slug>.py`. The web app discovers
them by filename and renders every one with the same template, so an adapter is
the only file that needs to know anything about its project.

```python
NUMBER = 1
SLUG = "rag-pipeline"                 # the URL segment: /s/rag-pipeline
TITLE = "Production RAG Pipeline"
TAGLINE = "One line, plain English, what a visitor can do on this page."
WHAT_IT_DOES = """Two or three short paragraphs..."""   # plain text, blank line separated
INPUT_LABEL = "Ask a question about the Meridian docs"  # None if the page takes no input
PLACEHOLDER = "How long is a token valid?"
EXAMPLES = ["...", "..."]            # 2-4 one-click example inputs
SOURCE = "projects/p01_rag_pipeline"

def run(user_input: str) -> str:
    """Return plain text output. Must finish in under 8 seconds."""
```

Hard rules, because this runs in a serverless function:

1. **Under 8 seconds** on a cold start, every time. Trim the workload if needed.
2. **No writes outside `/tmp`.** The filesystem is read only. If the project
   writes artefacts, point it at `tempfile.mkdtemp()`.
3. **No sockets and no bound servers.** Use the project's in-process classes.
   Threads are fine.
4. **Never raise.** Catch, and return a readable error string instead.
5. Output is rendered in a `<pre>`; keep it under about 8000 characters.
6. Empty input must still work: fall back to the first example.

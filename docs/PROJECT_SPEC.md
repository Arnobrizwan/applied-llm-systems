# Build spec (applies to every project in `projects/`)

This is the contract each of the 15 systems is written against. It exists so the
repo reads as one codebase rather than fifteen unrelated demos.

## Hard constraints

1. **Zero paid APIs, zero required accounts.** Every demo and every test must run
   on a stock Python 3.9+ install with no network. Free upgrade paths (Ollama, any
   OpenAI-compatible endpoint) are supported through `llmkit`, never required.
2. **Runtime code imports only the standard library and `llmkit`.** `pytest` is a
   dev dependency and may only be imported inside `tests/`.
3. **Nothing is mocked away that the project is supposed to demonstrate.** Use
   `llmkit.EchoLLM` as the model, but the retrieval, validation, routing, caching,
   memory, guardrail and evaluation logic must be real code that would still be
   correct with a frontier model plugged in.
4. **No invented numbers.** Every figure in a README comes from running the demo.

## Layout

```
projects/pNN_slug/
├── README.md          # the document a reviewer reads first
├── __init__.py
├── <modules>.py       # the implementation
├── demo.py            # runnable end to end, prints a report
└── tests/
    ├── __init__.py
    └── test_*.py      # 6+ meaningful tests
```

Directory names are valid Python identifiers (`p01_rag_pipeline`) so tests can
import them as packages: `from projects.p01_rag_pipeline.pipeline import Pipeline`.

`demo.py` starts with the standard bootstrap so it runs directly:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```

## README sections (in this order)

1. `# NN. Title` and a one-sentence description.
2. **The problem** - the production failure this system prevents.
3. **What this builds** - the components, in order.
4. **Architecture** - an ASCII diagram.
5. **Design decisions** - 3 to 6 decisions, each with the alternative rejected and why.
6. **Running it** - exact commands and the shape of the output.
7. **Results** - real measured numbers from the demo, with the measurement method stated.
8. **Limits** - what this does not do, and what changes with a real model or budget.

## Style

Plain English. No emojis. No em dashes. Comments explain *why*, not *what*.
Say plainly when something is a simplification.

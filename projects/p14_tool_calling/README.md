# 14. Tool-Calling Framework

Typed function schemas derived from type hints and docstrings, a versioned
registry with discovery, argument coercion with structured errors, guarded
execution with an audit trail, and a bounded agent loop.

## The problem

Three failures show up in every agent that calls tools, and none of them are
model problems.

The first is schema drift. The tool schema in the prompt is a hand-written copy
of the function signature, someone adds a parameter, and only one of the two
copies changes. The model keeps sending the old argument set, the function
raises a `TypeError` deep inside the loop, and the incident gets triaged as
"model hallucinated the arguments".

The second is unguarded execution. A tool blocks on a slow dependency and the
agent hangs, holding a request thread. A tool returns a 40 MB result and the
next call blows the context window and the bill. A tool that used to work starts
failing and the agent spends its entire step budget rediscovering that on every
request.

The third is a calculator built on `eval`. It is the single most common tool in
agent demos and it hands an attacker who controls the prompt a Python
interpreter. `__import__` is the obvious route; `().__class__.__base__.__subclasses__()`
is the quiet one, and it needs no import statement at all.

## What this builds

1. **`schema.py`** - the `@tool` decorator. Generates the JSON schema from type
   hints and the Google-style docstring, and validates plus coerces model-supplied
   arguments into structured errors rather than exceptions.
2. **`registry.py`** - registration, semantic-version-aware lookup, tag
   filtering, and a stable rendering of the catalog for the prompt.
3. **`safe_math.py`** - an arithmetic evaluator built on an AST allowlist.
4. **`tools.py`** - four real tools: calculator, corpus search over
   `llmkit.corpus`, unit converter, deterministic clock.
5. **`sandbox.py`** - allowlist, wall-clock timeout, output cap, retry on
   transient failure, a circuit breaker per tool, and an audit record per call.
6. **`agent.py`** - the bounded tool-calling loop over `llmkit.EchoLLM` with
   `json_schema` output and full tracing.

## Architecture

```
   @tool decorated function
      type hints + docstring
              |
              v
        +-----------+        list(tags=...)        +-----------------+
        | ToolSpec  |------------------------------>| prompt catalog  |
        +-----+-----+                               +-----------------+
              | register                                     |
              v                                              v
        +--------------+                            +-------------------+
        | ToolRegistry |<---------------------------|    ToolAgent      |
        |  name@ver    |   find(action)             |  decide  (enum)   |
        +------+-------+                            |  arguments(schema)|
               |                                    |  observe [Sn]     |
               v                                    +---------+---------+
        +-----------------------------------+                 |
        |            ToolSandbox            |<----------------+
        |  1 allowlist                      |  call(name, args)
        |  2 circuit breaker                |
        |  3 coerce + validate arguments    |----> structured error
        |  4 retry(transient) + timeout     |----> timeout / error
        |  5 output size cap                |----> output_too_large
        |  6 audit record                   |----> ok + value
        +-----------------------------------+
```

## Design decisions

**Schema derived from the signature, never hand-written.** The rejected
alternative is a `description=` argument per parameter, which is a second copy
of information that already exists in the docstring. Both copies rot; only one
of them is read in code review. Google-style `Args:` sections were chosen
because most Python codebases already write them.

**Coerce, then validate, then execute, and return errors instead of raising.**
`{"k": "3"}` is coerced to `{"k": 3}` because a quoted integer is a formatting
slip, not a reasoning error, and a correction round trip costs a model call.
`{"k": "three"}` comes back as `{"path": "$.k", "rule": "type", ...}`, which is
what gets written into the conversation. Nothing in the call path raises, so the
agent loop has no error translation layer.

**Two model calls per step, not one.** The alternative is a single schema where
`arguments` is a union whose shape depends on `action`, which JSON Schema can
only express through `oneOf` and which constrained decoders handle poorly. The
first call picks an action from a flat enum; the second fills arguments against
that one tool's concrete schema. Every call is constrained by something simple.

**Invalid arguments do not count against tool health.** The tool never ran, so
tripping its circuit breaker on a model's bad arguments would disable a
perfectly healthy tool because the planner was confused. Only real execution
failures move the breaker.

**Oversize output is refused, not truncated.** Truncated JSON handed back to a
model is worse than an explicit error: it either fails to parse or, worse,
parses into something subtly wrong. The error names the actual size and the cap,
which is enough for the model to ask for less.

**The calculator uses an AST allowlist, not a blocklist.** A blocklist of
keywords fails open, which is the wrong direction for input an attacker
influences by definition. The allowlist covers arithmetic node types only, so
anything outside the grammar is refused without being enumerated. Two limits are
not about parsing at all: expression length and exponent size, because
`2 ** 10000000` is valid arithmetic and still a denial of service.

**The clock is injectable and pinned by default.** An agent that reads the wall
clock cannot be replayed, and replay is the only practical way to debug a
multi-step failure after the fact.

## Running it

```
python3 projects/p14_tool_calling/demo.py
python3 -m pytest projects/p14_tool_calling -q
```

The demo prints seven sections: the registry and a generated schema, two
versions of one tool side by side, real calls through the sandbox, the
adversarial calculator suite, the sandbox guarantees, six agent runs with their
step traces, and the audit log rollup.

## Results

Measured by running `demo.py`.

**Adversarial calculator: 12 of 12 expressions rejected.** The suite covers
`__import__`, the `__class__.__base__.__subclasses__()` subclass walk,
`__mro__`, attribute access with no dunder in it (`(9).real`), a call on an
attribute (`open(...).read()`), `eval`, `globals`, `2 ** 10000000`, a list
comprehension, a lambda, a conditional expression and division by zero. Ordinary
arithmetic still works: `600 * 60 / 1000` gives 36.0 and `round(pi, 4)` gives
3.1416.

**Sandbox guarantees, all observed in the demo output:**

| guarantee | configured | observed |
|---|---|---|
| allowlist | `allowlist=["flaky"]` | `sleeper` refused with `denied` before running |
| timeout | 0.2s budget, tool sleeps 1.0s | control returned after 205ms, not 1000ms |
| output cap | 1024 bytes | 20002 byte result refused as `output_too_large` |
| retry | 3 attempts, transient errors only | succeeded on attempt 3 |
| circuit breaker | threshold 3, cooldown 30s | calls 1 to 3 `error`, calls 4 and 5 `circuit_open`, half-open after the cooldown |

**Agent loop, 6 questions, step budget 5:** 47 model calls, 19 tool calls, 7
succeeded and 12 failed. Every run terminated; the deepest reached exactly 5
steps, and 3 of the 6 ended on the step budget rather than by choosing
`final_answer`.

**Audit rollup for those runs**, one record per call including refusals:

| tool | calls | ok | failed | outcomes |
|---|---|---|---|---|
| calculate | 8 | 0 | 8 | error 3, circuit_open 5 |
| convert_units | 4 | 0 | 4 | error 3, circuit_open 1 |
| current_time | 1 | 1 | 0 | ok 1 |
| search_docs | 6 | 6 | 0 | ok 6 |

The 0% success rate on `calculate` is the honest result and it is a property of
the provider, not the framework. `EchoLLM` synthesises a placeholder string such
as `expression-036` for a parameter typed `str`, which the safe evaluator
correctly rejects. After three rejections the circuit opens and the remaining
five calls are refused in microseconds instead of re-running the parser. That is
the breaker doing exactly what it exists for: it stops the loop paying for a
dependency that cannot succeed for this planner.

`search_docs` succeeds 6 times out of 6 for the opposite reason: its parameter
is named `query`, and `EchoLLM` substitutes the user's actual question for
parameters with that name, so the search receives real input. `convert_units`
receives valid units every time, because they are typed `Literal` and the
provider samples from the enum, but it picks the two units independently and so
often asks for a length-to-mass conversion, which the tool refuses.

## Limits

**The timeout does not kill anything.** It runs the tool on a daemon thread and
abandons the thread if it overruns. That bounds the agent loop, which is what
was stalling, but the work keeps going: a Python thread cannot be interrupted
from outside, so a tool wedged inside a C call or a blocking socket holds its
memory and its file handles until the process exits. Run enough of them and you
leak the process to death. A real sandbox is a subprocess you can `SIGKILL`, or
a container with a cgroup, a read-only filesystem and no network. That is the
correct fix, and it is deliberately not what this module does, because
subprocess-per-call brings pickling, cold start and platform differences that
would take over a project about tool calling.

**The allowlist is a name check, not a capability model.** It stops an agent
calling a tool it was not granted. It does nothing about a tool that was
granted and is itself dangerous. Anything with side effects needs an approval
step, and that lives above this layer.

**The agent's tool choices are not meaningful.** `EchoLLM` samples the action
enum, so the sequences in the demo are arbitrary. What the demo does prove is
the control flow: bounded steps, guaranteed termination, structured errors fed
back as observations, a circuit that opens on a tool that cannot succeed, and a
complete audit trail. Plug in a real model and the same loop runs; the tool
choices become sensible and the `calculate` failure rate collapses.

**Argument repair is out of scope here.** A malformed decision is treated as
"stop" rather than repaired, because repair and re-prompting are project 02's
job and a second implementation would drift from the first.

**`safe_math` is an arithmetic evaluator, not a computer algebra system.** No
variables, no assignment, no vectors, no statements. That is a deliberate floor:
every feature added to the grammar is a new thing to prove safe.

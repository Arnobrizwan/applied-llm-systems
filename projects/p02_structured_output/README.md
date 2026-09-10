# 02. Structured Output Engine

Schema enforcement for model output: a hand-written JSON Schema validator, a
repair pipeline for broken JSON, a retry loop that feeds validation errors back
to the model, and a typed fallback so the caller never sees an exception.

## The problem

A model that returns JSON returns JSON *most* of the time. The rest of the time
it returns JSON wrapped in a code fence, JSON with a sentence in front of it,
JSON with single quotes because a Python repr leaked into the training data, or
JSON that stops mid-token because the response hit the token limit. Every one of
those parses fine in a notebook, where you look at the output, and fails in
production, where a `json.loads` in a request handler raises and the endpoint
returns a 500.

The second half of the problem is subtler. Output that parses can still be
wrong: a score of 1.4 in a field documented as 0 to 1, a severity of "critical"
when the enum has three values and that is not one of them, an invented extra
key. That output flows downstream and breaks something further away from the
cause, which is the expensive kind of bug.

This project makes both classes of failure bounded and observable: repair what
can be repaired locally, re-prompt with the exact error when it cannot, and
return a typed fallback with a flag when the model will not comply.

## What this builds

1. **`validator.py`** - a JSON Schema subset validator written from scratch,
   returning structured errors (path, rule, got, expected, message) rather than
   a boolean, because those errors are the input to the next prompt.
2. **`schema_builder.py`** - schema generation from Python type hints and
   dataclasses, so a shape is declared once and cannot drift between the prompt
   and the code that consumes the result.
3. **`extraction.py`** - an ordered repair pipeline for realistically broken
   output: code fences, prose preambles, trailing commentary, single quotes,
   Python literals, trailing commas, truncation. Each repair that fires is
   recorded.
4. **`engine.py`** - the bounded retry loop, built on `llmkit.retry`, which
   feeds validation errors back verbatim and returns a deep-copied fallback when
   the attempt budget runs out.

## Architecture

```
  prompt + schema
        |
        v
  +-----------------+     json_schema=...     +-----------+
  | build messages  |------------------------>|  EchoLLM  |
  | (+ prior errors)|<------------------------|  raw text |
  +-----------------+                         +-----------+
        |                                            |
        |                                            v
        |                              +-----------------------------+
        |                              | extract_json                |
        |                              |  unfence                    |
        |                              |  slice_to_json              |
        |                              |  python_literals            |
        |                              |  single_quotes              |
        |                              |  trailing_commas            |
        |                              |  close_truncated            |
        |                              +-------------+---------------+
        |                                            | parsed value
        |                                            v
        |                              +-----------------------------+
        |                              | validate(value, schema)     |
        |                              +------+---------------+------+
        |                                     | errors        | none
        |          SchemaViolation            |               v
        +-------------------------------------+          valid object
        |  (llmkit.retry, bounded attempts)                   ^
        v                                                     |
  attempts exhausted ------> deep-copied fallback ------------+
```

## Design decisions

**Hand-written validator instead of `jsonschema` or `pydantic`.** Both are
third-party and pydantic v2 is a compiled extension, which this repo cannot use.
The stronger reason is that the useful output is not "valid or not" but a
machine-readable description of what is wrong, phrased so a model can act on it.
Owning the error type is most of the value, and it is about 200 lines.

**Errors go back to the model verbatim.** The rejected alternative was a generic
"that was not valid, try again". A generic retry makes the model re-roll the
whole response, which reproduces the same mistake at a similar rate. Sending
`$.findings[0].confidence: must be <= 1.0, got 1.4` turns a re-roll into a
correction. This is why the validator collects every error instead of stopping
at the first: one model call should fix all of them, not one per call.

**Repair before re-prompting, always.** A local repair costs microseconds; a
re-prompt costs a model call and its latency. The measurement below exists to
check that this ordering earns its place rather than to assume it: at every
fault rate tested, repair alone recovered at least half of the broken responses.

**The retry loop is `llmkit.retry`, not a hand-rolled loop.** A schema violation
is raised as a retryable exception so semantic failures get the same attempt
budget, backoff and exhaustion semantics as transport failures. One retry policy
in the codebase rather than two that drift apart. The default `base_delay` here
is zero: the pause between attempts is a correction round trip, not congestion
relief, so sleeping only adds latency.

**A typed fallback, never an exception.** `generate()` returns
`StructuredResult(ok=False, source="fallback", value=<your default>)` rather
than raising, and the fallback is deep-copied on the way out so one caller
cannot mutate a shared default that the next caller observes. Callers that would
rather fail loudly can check `ok` and raise themselves; callers in a request
path can degrade quietly. That choice belongs to the caller, not the library.

**`additionalProperties` defaults to False for generated schemas.** Models
invent plausible extra keys when they are unsure. Silently dropping them hides a
signal that the schema was misread; reporting them puts it in the retry prompt.

## Running it

```
python3 projects/p02_structured_output/demo.py
python3 -m pytest projects/p02_structured_output -q
```

The demo prints five sections: the schema derived from a dataclass, the repair
pipeline against seven hand-written broken payloads, the structured validation
errors for a deliberately bad object, the measured success table, and the
fallback path with its attempt trace.

## Results

All numbers below come from running `demo.py` on the offline `EchoLLM`
provider. It is deterministic, so these reproduce exactly.

**Repair pipeline, seven hand-written broken payloads: 6 of 7 recovered.** The
seventh is prose with no JSON in it, which is unrecoverable by design; inventing
a value there would be worse than failing.

**Success rate against schema, 40 prompts per fault rate, k = 3 attempts.**
`fault_rate` is `EchoLLM`'s deterministic corruption rate: the fraction of
responses returned fenced, truncated, single-quoted or with trailing prose.

| fault rate | clean at 1 | repaired at 1 | success at 1 | success at k | fixed by re-prompt | fell back | model calls |
|---|---|---|---|---|---|---|---|
| 0.00 | 100% | 0% | 100% | 100% | 0 | 0 | 40 |
| 0.25 | 75% | 20% | 95% | 100% | 2 | 0 | 42 |
| 0.50 | 52% | 25% | 78% | 100% | 9 | 0 | 50 |
| 0.75 | 22% | 40% | 62% | 98% | 14 | 1 | 58 |
| 1.00 | 0% | 50% | 50% | 88% | 15 | 5 | 68 |

**Where the recovery came from.** Of the responses that were not clean on the
first call:

| fault rate | broken responses | fixed by local repair | fixed by re-prompting | unrecovered |
|---|---|---|---|---|
| 0.25 | 10 | 80% | 20% | 0% |
| 0.50 | 19 | 53% | 47% | 0% |
| 0.75 | 31 | 52% | 45% | 3% |
| 1.00 | 40 | 50% | 38% | 12% |

Local repair carries between half and four fifths of the recovery, and it is
free. At `fault_rate` 1.00 the engine still returns a valid object 88% of the
time, at a cost of 68 model calls for 40 prompts, or 1.7 calls per prompt.

**Repair strategies that fired**, summed across all five runs: `unfence` 43,
`slice_to_json` 37, `close_truncated` 34, `single_quotes` 30,
`trailing_commas` 30. Single quotes and trailing commas fire together because
that corruption produces both at once.

## Limits

The measured schema deliberately omits the `pattern` rule. `EchoLLM` synthesises
placeholder strings such as `ticket_id-652`, so an arbitrary regex constraint
fails 100% of the time regardless of how good the repair and retry paths are;
measuring against it would measure the provider, not the engine. The validator
supports `pattern` and the demo exercises it in section 3.

`EchoLLM`'s four corruption modes are realistic but they are four modes. A real
model produces a longer tail: valid JSON with a hallucinated enum value, correct
structure with the wrong units, two JSON objects where one was asked for. The
first is caught by the validator, the second is not caught by anything in this
project, and the third would need a repair strategy this pipeline does not have.

`success@k` is not `accuracy`. This project proves the output has the right
shape, not that it says true things. Faithfulness is a different measurement and
belongs with the evaluation project.

The validator implements a subset. `$ref`, `allOf`/`anyOf`/`oneOf`, `format` and
`patternProperties` are not supported, on the view that a schema you hand to a
model should be flat enough for the model to follow. If you need `$ref`
indirection, swap in `jsonschema` for validation and keep the error projection
layer from `validator.py`, which is the part that matters for retrying.

With a real model and a real budget, two things change. Providers with native
constrained decoding (JSON mode, grammar-constrained sampling) push `clean at 1`
close to 100% and make most of the repair pipeline dead weight, though it still
earns its place as a guard against truncation, which constrained decoding does
not prevent. And re-prompting stops being free: at 1.7 calls per prompt under
heavy corruption, the cost and latency of the retry path is the thing to watch,
which is what the `model calls` column is for.

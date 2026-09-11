"""Web adapter for project 06: the model routing gateway.

The visitor types a real request. The page scores it, routes it, actually calls
the chosen tier, then prices the same request against the cheapest and the most
expensive tier, replays it under a tiny budget, and replays it again with the
chosen tier's upstream returning HTTP 503.

Read only: everything is in memory, nothing is written anywhere.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 6
SLUG = "model-router"
TITLE = "Model Routing Gateway"
TAGLINE = ("Type a request and see which size of model it gets sent to, why, and what "
           "that choice costs against the alternatives.")

WHAT_IT_DOES = """Type anything you would ask an AI assistant. The page measures the request
before any model sees it: what kind of task it is, how long the answer needs to be, whether
it involves maths or code, and how vague the wording is. Each of those adds to a single
score, and you can see exactly how much each one contributed.

That score picks one of three model sizes. The page then really calls that model and prices
the call, alongside what the cheapest and the most expensive model would have charged for
the same request, so the saving or the overspend is a number rather than a claim.

It also shows the rules that can overrule the score: a customer who pays for the big model
regardless, an overnight batch account capped at the cheap one, and endpoints pinned to a
fixed size. Then it replays your request twice more, once against a deliberately tiny
spending cap so you can watch it get downgraded and then refused, and once with the chosen
model returning an error so you can watch it fall back to another one."""

INPUT_LABEL = "Type a request to route"
PLACEHOLDER = "translate hi to French"
EXAMPLES = [
    "translate hi to French",
    "Explain why our webhook retries are double charging customers, and compare fixing it with idempotency keys against deduplicating in the consumer",
    "Classify this ticket as billing, bug or feature request: my invoice was charged twice",
    "Write a python function that verifies an HMAC-SHA256 signature header",
]
SOURCE = "projects/p06_model_router"

_TIERS = ("small", "medium", "large")
_TIER_WORDS = {"small": "cheapest", "medium": "middle", "large": "most expensive"}


def _clean(user_input: str) -> str:
    text = " ".join((user_input or "").split())
    return text[:1200]


def _run(user_input: str) -> str:
    from llmkit import FailingLLM
    from projects.p06_model_router.budget import BudgetGuard, BudgetPolicy, CostLedger
    from projects.p06_model_router.complexity import (
        has_explicit_task_signal, score_complexity,
    )
    from projects.p06_model_router.gateway import (
        ModelRouter, Request, default_backends, fallback_chain,
    )
    from projects.p06_model_router.policy import default_policy, fixed_policy

    prompt = _clean(user_input)
    note = ""
    if not prompt:
        prompt = EXAMPLES[0]
        note = "no input given, so the first example was routed"

    policy = default_policy()
    ledger = CostLedger()
    router = ModelRouter(default_backends(), policy=policy, ledger=ledger)
    result = router.handle(Request("web-1", prompt, tenant="acme", endpoint="/chat"))
    features = result.decision.features
    decision = result.decision

    out = []
    if note:
        out.append(f"note: {note}")
        out.append("")

    out.append("THE REQUEST")
    out.append(f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    out.append(f"  tenant acme, endpoint /chat, {features.input_tokens} tokens in")

    out.append("")
    out.append("WHAT WAS MEASURED, BEFORE ANY MODEL SAW IT")
    out.append("  " + f"{'feature':<22}{'value':<26}{'adds to score':>13}")
    out.append("  " + "-" * 61)
    rows = [
        ("kind of task", features.task_type,
         features.contributions.get("task_type", 0.0)),
        ("length of request", f"{features.input_tokens} tokens",
         features.contributions.get("input_length", 0.0)),
        ("answer length needed", f"{features.required_output_tokens} tokens",
         features.contributions.get("output_length", 0.0)),
        ("maths in it", "yes" if features.has_math else "no",
         features.contributions.get("math", 0.0)),
        ("code in it", "yes" if features.has_code else "no",
         features.contributions.get("code", 0.0)),
        ("needs live lookup", "yes" if features.needs_tools else "no",
         features.contributions.get("tools", 0.0)),
        ("vagueness", f"{features.ambiguity:.2f} out of 1",
         features.contributions.get("ambiguity", 0.0)),
    ]
    for label, value, contribution in rows:
        out.append("  " + f"{label:<22}{value[:25]:<26}{contribution:>+13.3f}")
    out.append("  " + "-" * 61)
    out.append("  " + f"{'complexity score':<22}{'':<26}{features.score:>13.3f}")
    if not has_explicit_task_signal(prompt):
        out.append("  nothing in the wording said what kind of task this is, so it was "
                   "treated as middling difficulty and routed on length and vagueness "
                   "alone. Knowing how often that happens matters more than the "
                   "classifier looking clever.")

    out.append("")
    out.append("WHERE THAT SCORE LANDS")
    for upper, tier in policy.bands:
        edge = f"under {min(upper, 1.0):.2f}"
        marker = "  <- this request" if tier == decision.band_tier else ""
        out.append(f"  {edge:<12}{tier} model{marker}")
    out.append(f"  chosen: {decision.tier} ({_TIER_WORDS[decision.tier]} of the three)")
    out.append(f"  why:    {decision.reason}")
    out.append(f"  rules applied: {' -> '.join(decision.applied)}")

    out.append("")
    out.append("THE CALL, PRICED AGAINST THE OTHER TWO TIERS")
    priced = {}
    for tier in _TIERS:
        tier_ledger = CostLedger()
        tier_router = ModelRouter(default_backends(), policy=fixed_policy(tier),
                                  ledger=tier_ledger)
        tier_result = tier_router.handle(Request(f"price-{tier}", prompt, tenant="pricing"))
        priced[tier] = tier_result
    out.append("  " + f"{'model':<10}{'tokens in':>11}{'tokens out':>12}{'cost usd':>12}")
    out.append("  " + "-" * 45)
    for tier in _TIERS:
        r = priced[tier]
        mark = "  <- used" if tier == result.tier_used else ""
        out.append("  " + f"{tier:<10}{r.prompt_tokens:>11}{r.completion_tokens:>12}"
                   f"{r.cost_usd:>12.6f}{mark}")
    cheapest, dearest = priced["small"].cost_usd, priced["large"].cost_usd
    used = result.cost_usd
    out.append(f"  this request cost ${used:.6f}")
    if dearest > 0 and used < dearest:
        saved = dearest - used
        out.append(f"  ${saved:.6f} less than always using the most expensive model "
                   f"({saved / dearest:.0%} cheaper)")
    elif used >= dearest > 0:
        out.append("  this request went to the most expensive model, so there is no "
                   "saving to report; the score is the argument for spending it")
    if cheapest > 0 and used > cheapest:
        out.append(f"  {used / cheapest:.1f}x the cost of always using the cheapest model, "
                   "which is the quality insurance being paid for")
    elif used <= cheapest:
        out.append("  this is already the cheapest model, so there is nothing to save")
    out.append("  the three models behind these tiers are the same offline model with "
               "different names and different price tags, so the money is real "
               "arithmetic and the quality difference is not simulated")

    out.append("")
    out.append("THE SAME REQUEST UNDER THE RULES THAT OVERRULE THE SCORE")
    out.append("  " + f"{'account and endpoint':<28}{'model':<9}why")
    out.append("  " + "-" * 74)
    for tenant, endpoint, label in (
        ("acme", "/chat", "acme, normal chat"),
        ("enterprise", "/chat", "enterprise, normal chat"),
        ("batch", "/chat", "batch, overnight backfill"),
        ("acme", "/classify", "acme, labelling endpoint"),
        ("acme", "/codegen", "acme, code endpoint"),
    ):
        d = policy.decide(features, tenant=tenant, endpoint=endpoint)
        out.append("  " + f"{label:<28}{d.tier:<9}{d.reason[:74]}")
    out.append("  a request pinned by hand on the way in beats all of these, for the "
               "engineer who needs one call on a known good model right now")

    out.append("")
    out.append("WHAT HAPPENS WHEN THE ACCOUNT RUNS OUT OF MONEY")
    if used > 0:
        cap = round(used * 3.6, 9)
        guard_ledger = CostLedger()
        guard = BudgetGuard(guard_ledger,
                            policies={"acme": BudgetPolicy(daily_usd=cap, downgrade_at=0.8)})
        guarded = ModelRouter(default_backends(), policy=policy, ledger=guard_ledger,
                              guard=guard)
        out.append(f"  daily cap set to ${cap:.6f}, under four of these requests, "
                   "so the whole arc is visible in a few lines")
        out.append("  " + f"{'request':<9}{'spent so far':>14}{'of cap':>8}   "
                          f"{'decision':<11}{'model':<9}result")
        out.append("  " + "-" * 62)
        refused_at = None
        downgraded = False
        for i in range(8):
            before = guard_ledger.spend("acme")
            r = guarded.handle(Request(f"budget-{i}", prompt, tenant="acme"))
            action = r.budget.action if r.budget else "allow"
            downgraded = downgraded or action == "downgrade"
            served = "refused" if r.refused else ("served" if r.ok else "failed")
            out.append("  " + f"{i + 1:<9}{before:>14.6f}{before / cap:>7.0%}   "
                              f"{action:<11}{str(r.tier_used or '-'):<9}{served}")
            if r.refused:
                refused_at = i + 1
                out.append(f"  stopped at ${guard_ledger.spend('acme'):.6f} of ${cap:.6f}: "
                           f"{r.reason}")
                break
        if refused_at is None:
            out.append(f"  after 8 requests it is still alive on the cheap model at "
                       f"{guard_ledger.spend('acme') / cap:.0%} of the cap, because the "
                       "downgrade made each request far cheaper")
        if not downgraded:
            out.append("  this request already routes to the cheapest model, so there was "
                       "nothing to downgrade to and the guard went straight from allowing "
                       "to refusing")
        out.append("  the guard drops to the cheap model at 80 percent of the cap and only "
                   "refuses at 100, so an account gets a worse service before it gets an "
                   "error instead of working perfectly right up to the cut off")
    else:
        out.append("  this request cost nothing, so there is no budget to run out of")

    out.append("")
    out.append("WHAT HAPPENS WHEN THAT MODEL IS DOWN")
    down_tier = result.tier_used or decision.tier
    broken = default_backends()
    broken[down_tier] = FailingLLM(fail_times=10 ** 6, status=503)
    down_router = ModelRouter(broken, policy=policy, ledger=CostLedger())
    out.append(f"  the {down_tier} model is returning HTTP 503 for every call")
    out.append(f"  order it will try: {' then '.join(fallback_chain(down_tier))} "
               "(nearest capability first, so an outage does not quietly become a "
               "quality cut)")
    for i in range(3):
        r = down_router.handle(Request(f"outage-{i}", prompt, tenant="acme"))
        chain = " then ".join(
            f"{a.tier} {'ok' if a.outcome == 'ok' else a.outcome.replace('_', ' ')}"
            for a in r.attempts)
        served = f"served by {r.tier_used}" if r.ok else "not served"
        out.append(f"  request {i + 1}: {chain}  ->  {served}")
    states = ", ".join(f"{t} {b.state}" for t, b in down_router.breakers.items())
    out.append(f"  circuit breakers now: {states}")
    out.append("  once the breaker opens, later requests skip the dead model without "
               "paying for a call that was going to fail")
    return "\n".join(out)


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo could not complete: {type(exc).__name__}: {exc}\n"
                "Try typing a plain request, for example: translate hi to French")

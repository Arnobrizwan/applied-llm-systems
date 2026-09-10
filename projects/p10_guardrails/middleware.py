"""The middleware: compose the stages, record every decision, fail closed.

This is the reusable service layer. A caller wires it once and gets input
screening, redaction, the model call, output screening and restoration, with a
decision log attached to the result. The two things it is careful about:

* Every stage is wrapped. A guardrail that raises is a guardrail an attacker can
  disable by finding an input that crashes it, so an exception inside a check
  becomes an `engine.error` decision resolved by the configured fail mode rather
  than an exception escaping to the caller.
* Redaction happens before the model call and restoration after the output
  checks. Restoring first would mean the output stage inspects text containing
  real PII and then reports "new PII in the response" for values the user
  supplied themselves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmkit import LLMProvider

from .injection import InjectionReport, scan
from .output import OutputFinding, scan_output
from .pii import PIIMatch, Vault, redact
from .policy import Action, Decision, Policy

__all__ = ["Guardrails", "InputScreen", "OutputScreen", "GuardrailResult"]


@dataclass
class InputScreen:
    action: Action
    text: str  # the (possibly redacted) prompt safe to send onward
    decisions: List[Decision] = field(default_factory=list)
    injection: Optional[InjectionReport] = None
    pii: List[PIIMatch] = field(default_factory=list)
    vault: Vault = field(default_factory=Vault)

    @property
    def allowed(self) -> bool:
        return self.action is not Action.BLOCK


@dataclass
class OutputScreen:
    action: Action
    text: str
    decisions: List[Decision] = field(default_factory=list)
    findings: List[OutputFinding] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action is not Action.BLOCK


@dataclass
class GuardrailResult:
    """One end-to-end request through the middleware."""

    allowed: bool
    action: Action
    blocked_stage: Optional[str]
    prompt_sent: str
    output: str
    decisions: List[Decision] = field(default_factory=list)
    injection_score: float = 0.0
    pii_found: List[PIIMatch] = field(default_factory=list)
    model_called: bool = False
    latency_ms: float = 0.0

    @property
    def rule_ids(self) -> List[str]:
        return [d.rule_id for d in self.decisions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action.label,
            "blocked_stage": self.blocked_stage,
            "injection_score": self.injection_score,
            "pii": [{"kind": m.kind} for m in self.pii_found],
            "model_called": self.model_called,
            "decisions": [d.to_dict() for d in self.decisions],
        }


REFUSAL = "This request was blocked by the safety policy."


class Guardrails:
    """Input filtering, PII redaction, output filtering, as one component."""

    def __init__(self, policy: Optional[Policy] = None, llm: Optional[LLMProvider] = None):
        self.policy = policy or Policy()
        self.llm = llm

    # -- input stage -----------------------------------------------------
    def screen_input(self, text: str) -> InputScreen:
        try:
            report = scan(text)
            decisions: List[Decision] = []
            if report.score >= self.policy.block_threshold:
                decisions.append(self.policy.decision(
                    "input.injection.block",
                    f"injection score {report.score:.2f} from {len(report.signals)} rule(s): "
                    + ", ".join(report.rule_ids),
                    report.signals[0].evidence if report.signals else ""))
            elif report.score >= self.policy.flag_threshold:
                decisions.append(self.policy.decision(
                    "input.injection.flag",
                    f"injection score {report.score:.2f}: " + ", ".join(report.rule_ids),
                    report.signals[0].evidence if report.signals else ""))

            redacted, matches, vault = redact(text)
            secrets = [m for m in matches if m.kind == "api_key"]
            personal = [m for m in matches if m.kind != "api_key"]
            if secrets:
                decisions.append(self.policy.decision(
                    "input.secret", f"{len(secrets)} credential(s) redacted",
                    ", ".join(sorted({m.kind for m in secrets}))))
            if personal:
                decisions.append(self.policy.decision(
                    "input.pii", f"{len(personal)} value(s) redacted",
                    ", ".join(sorted({m.kind for m in personal}))))

            action = self.policy.resolve(decisions)
            # A blocked prompt is never forwarded, redacted or not.
            forwarded = text if action is Action.BLOCK else redacted
            return InputScreen(action=action, text=forwarded, decisions=decisions,
                               injection=report, pii=matches, vault=vault)
        except Exception as exc:  # a broken check must not become an open door
            decision = self.policy.decision("engine.error", f"input stage raised: {type(exc).__name__}: {exc}")
            decision.action = self.policy.fail_action
            return InputScreen(action=decision.action, text="", decisions=[decision])

    # -- output stage ----------------------------------------------------
    def screen_output(self, text: str, *, system_prompt: str = "", input_text: str = "",
                      vault: Optional[Vault] = None) -> OutputScreen:
        try:
            findings = scan_output(
                text,
                system_prompt=system_prompt,
                input_text=input_text,
                secrets=self.policy.secrets,
                banned_topics=self.policy.banned_topics,
                echo_threshold=self.policy.system_prompt_echo_threshold,
                vault=vault,
            )
            rule_for = {
                "secret_leak": "output.secret_leak",
                "system_prompt_echo": "output.system_prompt_echo",
                "new_pii": "output.new_pii",
                "banned_topic": "output.banned_topic",
            }
            decisions = [self.policy.decision(rule_for[f.kind], f.detail, f.evidence) for f in findings]
            action = self.policy.resolve(decisions)

            safe = text
            if action is Action.BLOCK:
                safe = REFUSAL
            elif action is Action.REDACT:
                safe, _matches, _v = redact(text)
            return OutputScreen(action=action, text=safe, decisions=decisions, findings=findings)
        except Exception as exc:
            decision = self.policy.decision("engine.error", f"output stage raised: {type(exc).__name__}: {exc}")
            decision.action = self.policy.fail_action
            safe = REFUSAL if decision.action is Action.BLOCK else text
            return OutputScreen(action=decision.action, text=safe, decisions=[decision])

    # -- end to end ------------------------------------------------------
    def run(self, user_text: str, *, system_prompt: str = "", llm: Optional[LLMProvider] = None) -> GuardrailResult:
        """Screen, call the model, screen again, restore. Never raises."""
        started = time.perf_counter()
        provider = llm or self.llm
        inbound = self.screen_input(user_text)

        if not inbound.allowed:
            return GuardrailResult(
                allowed=False, action=inbound.action, blocked_stage="input",
                prompt_sent="", output=REFUSAL, decisions=inbound.decisions,
                injection_score=inbound.injection.score if inbound.injection else 0.0,
                pii_found=inbound.pii, model_called=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        raw = ""
        if provider is not None:
            messages = [{"role": "user", "content": inbound.text}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            raw = provider.complete(messages).text

        outbound = self.screen_output(raw, system_prompt=system_prompt,
                                      input_text=inbound.text, vault=inbound.vault)

        final = outbound.text
        if outbound.allowed and self.policy.restore_pii_in_output:
            # Restoration is a policy choice, not a default of nature: the user
            # already knows their own phone number, so putting it back makes the
            # answer readable. A deployment that logs responses, or shows them to
            # a different user, should set restore_pii_in_output=False.
            final = inbound.vault.restore(final)

        decisions = inbound.decisions + outbound.decisions
        action = self.policy.resolve(decisions)
        return GuardrailResult(
            allowed=outbound.allowed,
            action=action,
            blocked_stage=None if outbound.allowed else "output",
            prompt_sent=inbound.text,
            output=final,
            decisions=decisions,
            injection_score=inbound.injection.score if inbound.injection else 0.0,
            pii_found=inbound.pii,
            model_called=provider is not None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

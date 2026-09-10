"""The policy engine: rules carry severity and an action, decisions are recorded.

The point of separating this from the detectors is that "what did we find" and
"what do we do about it" change at different rates and for different reasons.
Detection changes when a new attack shows up. Policy changes when the business
decides that leaking an internal hostname is now a blocking offence rather than
a logged one. Wiring an action into a detector means every policy change is a
code change in the detection path.

Every fired rule produces a `Decision` naming the rule, the evidence and the
action, and the middleware resolves the strongest action across all of them. A
guardrail that says "blocked" without saying which rule fired is unusable: you
cannot tune a threshold you cannot attribute, and you cannot answer a customer
asking why their message was rejected.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

__all__ = ["Action", "Severity", "Rule", "Decision", "Policy", "DEFAULT_RULES"]


class Severity(enum.IntEnum):
    """Ordered so `max()` works and a numeric threshold is meaningful."""

    INFO = 10
    LOW = 20
    MEDIUM = 30
    HIGH = 40
    CRITICAL = 50


class Action(enum.IntEnum):
    """Ordered by strength. Resolution across rules is `max`, so a single
    blocking rule wins over any number of redactions, and a redaction wins over
    a flag. The alternative, first-match-wins, makes the outcome depend on rule
    registration order, which is a terrible property for a security control."""

    ALLOW = 0
    FLAG = 1
    REDACT = 2
    BLOCK = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Rule:
    id: str
    stage: str  # "input" | "output"
    category: str
    severity: Severity
    action: Action
    description: str


@dataclass
class Decision:
    """One rule firing, with enough detail to explain the outcome to a human."""

    rule_id: str
    stage: str
    action: Action
    severity: Severity
    detail: str
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"rule_id": self.rule_id, "stage": self.stage, "action": self.action.label,
                "severity": self.severity.name, "detail": self.detail, "evidence": self.evidence}


DEFAULT_RULES: Tuple[Rule, ...] = (
    Rule("input.injection.block", "input", "prompt_injection", Severity.HIGH, Action.BLOCK,
         "Injection score at or above the block threshold."),
    Rule("input.injection.flag", "input", "prompt_injection", Severity.MEDIUM, Action.FLAG,
         "Injection score above the flag threshold but below blocking."),
    Rule("input.pii", "input", "pii", Severity.MEDIUM, Action.REDACT,
         "Personal data in the user message is replaced with reversible placeholders."),
    Rule("input.secret", "input", "secret", Severity.HIGH, Action.REDACT,
         "An API key or token in the user message is redacted before it reaches the model."),
    Rule("output.secret_leak", "output", "secret", Severity.CRITICAL, Action.BLOCK,
         "The response contains a configured secret or a credential-shaped string."),
    Rule("output.system_prompt_echo", "output", "system_exfiltration", Severity.HIGH, Action.BLOCK,
         "The response repeats a substantial run of the system prompt."),
    Rule("output.new_pii", "output", "pii", Severity.HIGH, Action.REDACT,
         "The response contains personal data that was not in the user message."),
    Rule("output.banned_topic", "output", "policy", Severity.MEDIUM, Action.BLOCK,
         "The response discusses a topic this deployment does not allow."),
    Rule("engine.error", "engine", "availability", Severity.CRITICAL, Action.BLOCK,
         "A guardrail stage raised. Under fail-closed this blocks the request."),
)

DEFAULT_BANNED_TOPICS: Dict[str, Tuple[str, ...]] = {
    "competitor_pricing": ("competitor pricing", "undercut the competition", "switch to acme corp"),
    "legal_advice": ("this constitutes legal advice", "you should sue", "i am your lawyer"),
    "medical_advice": ("take this medication", "diagnose you with", "stop taking your prescription"),
}


@dataclass
class Policy:
    """Configuration for one deployment of the middleware."""

    rules: Tuple[Rule, ...] = DEFAULT_RULES
    block_threshold: float = 0.70
    flag_threshold: float = 0.35
    banned_topics: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_BANNED_TOPICS))
    secrets: Tuple[str, ...] = ()
    system_prompt_echo_threshold: float = 0.25
    restore_pii_in_output: bool = True
    # Fail closed by default. A guardrail that fails open is a guardrail that an
    # attacker can disable by finding any input that makes it raise, which turns
    # a crash bug into an authorisation bypass. Availability is the thing being
    # traded away, and that trade should be explicit: set fail_mode="open" if a
    # blocked request costs more than a leaked one, which is a real position for
    # an internal, low-sensitivity assistant and the wrong one for anything
    # handling customer data.
    fail_mode: str = "closed"

    def rule(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    def decision(self, rule_id: str, detail: str, evidence: str = "") -> Decision:
        rule = self.rule(rule_id)
        return Decision(rule_id=rule.id, stage=rule.stage, action=rule.action,
                        severity=rule.severity, detail=detail, evidence=evidence)

    @staticmethod
    def resolve(decisions: Iterable[Decision]) -> Action:
        """The strongest action across every rule that fired."""
        actions = [d.action for d in decisions]
        return max(actions) if actions else Action.ALLOW

    @property
    def fail_action(self) -> Action:
        return Action.BLOCK if self.fail_mode == "closed" else Action.ALLOW

"""The service itself: sections in, one prompt plus a receipt out.

Order of operations, and why it is this order:

  dedupe -> allocate -> compress/pack -> arrange -> verify

Deduplication runs before allocation so a section does not win budget for tokens
it was going to waste on a repeat. Allocation runs before compression so each
section knows its ceiling before deciding what to cut, rather than cutting to a
guess and finding out afterwards. Arrangement runs last because it only moves
finished blocks around and cannot change the total. Verification is a real
check, not an assertion of faith: the final string is counted, and if the
structural overhead pushed it over, the budget is reduced and the pack rerun.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import count_tokens

from .budget import BudgetAllocator, BudgetPlan
from .compression import MIN_USEFUL_TOKENS, arrange_hourglass, compress
from .dedupe import DuplicateRecord, deduplicate
from .receipt import (
    STATUS_COMPRESSED,
    STATUS_DEDUPED,
    STATUS_DROPPED,
    STATUS_INCLUDED,
    ItemRecord,
    Receipt,
    SectionRecord,
)
from .sections import Compression, ContextItem, ContextRequest, Section

SECTION_SEPARATOR = "\n\n"
ITEM_SEPARATOR = "\n\n"
# Smallest per-item slice worth handing out when a section is shared across many
# items. Below roughly this, an extractive summary is one sentence with no
# context around it.
MIN_SHARE_TOKENS = 30


@dataclass
class _PackedSection:
    section: Section
    text: str
    used: int
    record: SectionRecord
    pieces: List[Tuple[float, str]] = field(default_factory=list)


@dataclass
class AssembledContext:
    """The finished prompt and the audit trail that explains it."""

    text: str
    tokens: int
    receipt: Receipt

    def fits(self, window: int, reserve: int) -> bool:
        return self.tokens <= window - reserve


class ContextAssembler:
    """Budgets a model window across memory, documents, tools and history."""

    def __init__(
        self,
        llm=None,
        allocator: Optional[BudgetAllocator] = None,
        dedupe_threshold: float = 0.8,
        min_useful_tokens: int = MIN_USEFUL_TOKENS,
        max_verify_rounds: int = 4,
    ):
        self.llm = llm
        self.allocator = allocator or BudgetAllocator()
        self.dedupe_threshold = dedupe_threshold
        self.min_useful_tokens = min_useful_tokens
        self.max_verify_rounds = max_verify_rounds

    # -- public ----------------------------------------------------------
    def assemble(self, request: ContextRequest) -> AssembledContext:
        # Work on copies of the section containers. The caller's request stays
        # intact so the same request can be measured against the naive baseline.
        work = [Section(spec=s.spec, items=list(s.items)) for s in request.sections]
        duplicates = deduplicate(work, threshold=self.dedupe_threshold)

        structural_reserve = self._structural_reserve(work)
        text = ""
        packed: List[_PackedSection] = []
        plan: Optional[BudgetPlan] = None

        for _ in range(self.max_verify_rounds):
            budget = max(0, request.available_tokens - structural_reserve)
            plan = self.allocator.allocate(work, budget)
            packed = [self._pack(s, plan, request.query) for s in work]
            ordered = self._order(packed)
            text = SECTION_SEPARATOR.join(p.text for p in ordered if p.text)
            total = count_tokens(text)
            if total <= request.available_tokens:
                break
            # Token counting is an estimate and joining strings can round up.
            # Rather than trust the arithmetic, take the measured overrun back
            # out of the budget and pack again.
            structural_reserve += (total - request.available_tokens) + 8
        else:  # pragma: no cover - defensive, four rounds is far more than needed
            raise RuntimeError("failed to fit the context budget after repeated attempts")

        ordered = self._order(packed)
        receipt = Receipt(
            request_id=request.request_id,
            query=request.query,
            model_window=request.model_window,
            reserve_tokens=request.reserve_tokens,
            available_tokens=request.available_tokens,
            prompt_tokens=count_tokens(text),
            sections=[p.record for p in packed],
            duplicates=[vars(d) for d in duplicates],
            order=[p.section.name for p in ordered if p.text],
        )
        self._record_duplicates(receipt, duplicates, work)

        assembled = AssembledContext(text=text, tokens=receipt.prompt_tokens, receipt=receipt)
        if not assembled.fits(request.model_window, request.reserve_tokens):
            raise AssertionError("assembled context exceeded the window minus reserve")
        return assembled

    # -- internals -------------------------------------------------------
    def _structural_reserve(self, sections: Sequence[Section]) -> int:
        """Tokens set aside for separators and for token-estimate rounding.

        Small, but not zero. A budget that is exactly the window is a budget that
        overflows the first time an estimator rounds a word the other way.
        """
        return 2 * len(sections) + 8

    def _pack(self, section: Section, plan: BudgetPlan, query: str) -> _PackedSection:
        alloc = plan.allocations[section.name]
        record = SectionRecord(
            name=section.name,
            priority=section.spec.priority,
            strategy=section.spec.strategy.value,
            demand_tokens=section.demand,
            granted_tokens=alloc.granted,
            used_tokens=0,
            note=alloc.note,
        )

        if not section.items:
            return _PackedSection(section=section, text="", used=0, record=record)

        header_cost = section.spec.header_tokens
        remaining = alloc.granted - header_cost
        pieces: List[Tuple[float, str]] = []
        used = 0
        targets = self._item_targets(section, max(0, remaining))

        for item in sorted(section.items, key=lambda i: (-i.value, i.id)):
            allowance = min(remaining, targets.get(item.id, remaining))
            if allowance < self.min_useful_tokens:
                record.items.append(
                    self._item_record(
                        item, section, STATUS_DROPPED, item.tokens, 0,
                        reason=alloc.note if alloc.granted == 0 else "section budget exhausted",
                    )
                )
                continue
            result = compress(
                item.text,
                allowance,
                section.spec.strategy,
                query=query,
                llm=self.llm,
                min_useful_tokens=self.min_useful_tokens,
            )
            if result is None:
                reason = (
                    "does not fit and the section declares keep_whole"
                    if section.spec.strategy is Compression.KEEP_WHOLE
                    else f"no room to compress into {allowance} tokens"
                )
                record.items.append(
                    self._item_record(item, section, STATUS_DROPPED, item.tokens, 0, reason=reason)
                )
                continue
            status = STATUS_INCLUDED if result.method == "whole" else STATUS_COMPRESSED
            record.items.append(
                self._item_record(
                    item, section, status, result.before_tokens, result.after_tokens,
                    method=result.method,
                )
            )
            pieces.append((item.value, result.text))
            remaining -= result.after_tokens
            used += result.after_tokens

        if not pieces:
            record.used_tokens = 0
            if alloc.granted > 0:
                record.note = (
                    f"granted {alloc.granted} tokens but no item could be represented at that size"
                )
            return _PackedSection(section=section, text="", used=0, record=record)

        body_texts = self._arrange_items(section, pieces)

        body = ITEM_SEPARATOR.join(body_texts)
        text = f"{section.spec.header}\n{body}" if section.spec.header else body
        record.used_tokens = used + header_cost
        return _PackedSection(section=section, text=text, used=record.used_tokens, record=record, pieces=pieces)

    def _item_targets(self, section: Section, budget: int) -> Dict[str, int]:
        """Split a section's grant across its items before any of them is cut.

        Two policies, chosen by strategy.

        Sections that compress *within* an item (extractive, summarize) share the
        grant across as many items as can each hold a useful amount, weighted by
        value. Six documents reduced to their query-relevant sentences beat two
        documents kept whole and four thrown away: the answer-bearing sentence is
        often in the fourth document, and the first document's boilerplate was
        never going to be read.

        Sections that can only truncate (history, tool output) keep the greedy
        whole-then-stop policy, because truncating eight short turns to a third
        each produces eight fragments and no readable conversation.
        """
        if section.spec.strategy not in (Compression.EXTRACTIVE, Compression.SUMMARIZE):
            return {}
        if budget <= 0 or not section.items:
            return {}
        if section.item_tokens <= budget:
            return {i.id: i.tokens for i in section.items}

        floor = max(self.min_useful_tokens, MIN_SHARE_TOKENS)
        keep_count = max(1, min(len(section.items), budget // floor))
        selected = sorted(section.items, key=lambda i: (-i.value, i.id))[:keep_count]

        targets = {i.id: 0 for i in selected}
        remaining = budget
        for _ in range(6):
            open_items = [i for i in selected if targets[i.id] < i.tokens]
            if remaining <= 0 or not open_items:
                break
            weight_total = sum(max(0.01, i.value) for i in open_items)
            handed = 0
            for item in open_items:
                share = int(remaining * max(0.01, item.value) / weight_total)
                take = max(0, min(share, item.tokens - targets[item.id]))
                targets[item.id] += take
                handed += take
            if handed == 0:
                best = max(open_items, key=lambda i: i.value)
                take = min(remaining, best.tokens - targets[best.id])
                targets[best.id] += take
                handed += take
                if take == 0:
                    break
            remaining -= handed
        return targets

    def _arrange_items(self, section: Section, pieces: List[Tuple[float, str]]) -> List[str]:
        """Hourglass inside evidence sections, declaration order everywhere else.

        Instructions and the live user turn are read as prose and must keep their
        written order; reordering them is a correctness bug, not a tuning choice.
        """
        if section.spec.position != "flow":
            return [text for _, text in pieces]
        order = arrange_hourglass([v for v, _ in pieces])
        return [pieces[i][1] for i in order]

    def _order(self, packed: Sequence[_PackedSection]) -> List[_PackedSection]:
        firsts = [p for p in packed if p.section.spec.position == "first"]
        flow = [p for p in packed if p.section.spec.position == "flow" and p.text]
        lasts = [p for p in packed if p.section.spec.position == "last"]
        scores = [p.section.spec.priority * (1.0 + p.section.value_density) for p in flow]
        arranged = [flow[i] for i in arrange_hourglass(scores)] if flow else []
        return list(firsts) + arranged + list(lasts)

    def _item_record(
        self,
        item: ContextItem,
        section: Section,
        status: str,
        before: int,
        after: int,
        method: str = "",
        reason: str = "",
    ) -> ItemRecord:
        return ItemRecord(
            item_id=item.id,
            section=section.name,
            source=item.source,
            status=status,
            before_tokens=before,
            after_tokens=after,
            method=method,
            reason=reason,
        )

    def _record_duplicates(
        self, receipt: Receipt, duplicates: Sequence[DuplicateRecord], work: Sequence[Section]
    ) -> None:
        """Give every removed duplicate a row in its own section's item list."""
        by_name: Dict[str, SectionRecord] = {s.name: s for s in receipt.sections}
        for dup in duplicates:
            record = by_name.get(dup.dropped_section)
            if record is None:  # pragma: no cover - sections are stable within a request
                continue
            record.items.append(
                ItemRecord(
                    item_id=dup.dropped_item_id,
                    section=dup.dropped_section,
                    source="dedupe",
                    status=STATUS_DEDUPED,
                    before_tokens=dup.tokens_saved,
                    after_tokens=0,
                    method="containment",
                    reason=f"already present as {dup.kept_item_id} in {dup.kept_section}",
                )
            )

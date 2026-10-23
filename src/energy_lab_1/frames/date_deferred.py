"""date_deferred frame — effective date / compliance deadline / study window deferred.

Fires when a signal shows a schedule_change on a section referencing an
effective date, compliance deadline, or study window; new number is larger.
"""
from __future__ import annotations

import re
from typing import Any

from labs_core.reasoning.frames import Frame, FrameMatchResult

DEADLINE_KEYWORDS = re.compile(
    r"\b(effective date|compliance deadline|study window|interconnection[- ]review|interconnection[- ]study)\b",
    re.IGNORECASE,
)
NUM_UNIT_RE = re.compile(r"\b(\d{1,4})\s*(?:calendar\s+)?(days?|hours?|weeks?|months?)\b", re.IGNORECASE)


class DateDeferredFrame(Frame):
    id = "date_deferred"
    category = "date_deferred"
    urgency = "medium"

    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        best: FrameMatchResult | None = None
        for s in situation.get("signals", []):
            if s.get("shape") not in ("schedule_change", "rewrite", "insertion"):
                continue
            before = s.get("text_before", "") or ""
            after = s.get("text_after", "") or ""
            merged = before + " " + after
            if not DEADLINE_KEYWORDS.search(merged):
                continue
            nb = [(int(m.group(1)), m.group(2).lower()) for m in NUM_UNIT_RE.finditer(before)]
            na = [(int(m.group(1)), m.group(2).lower()) for m in NUM_UNIT_RE.finditer(after)]
            features = ["deadline_keyword"]
            confidence = 0.55
            if nb and na:
                # Compare max-days-equivalent
                b_max = max(_to_days(n, u) for n, u in nb)
                a_max = max(_to_days(n, u) for n, u in na)
                if a_max > b_max:
                    confidence = 0.85
                    features.append(f"days_equiv: {b_max}→{a_max}")
                elif a_max < b_max:
                    # Deadline pulled in — not "deferred", so lower confidence
                    confidence = 0.40
                    features.append(f"days_equiv: {b_max}→{a_max} (pulled in)")
            r = FrameMatchResult(
                frame_id=self.id,
                category=self.category,
                confidence=confidence,
                matched_features=features,
                predictions=[
                    "active projects with plans keyed to the old date will need re-baselining",
                    "compliance risk decreases where the old deadline was tight",
                ],
                causal_story="A compliance deadline, effective date, or study window has moved.",
                urgency=self.urgency,
                recommended_reviewer_category="date_deferred",
            )
            if best is None or r.confidence > best.confidence:
                best = r
        return best


def _to_days(n: int, unit: str) -> int:
    unit = unit.lower().rstrip("s")
    return {"day": n, "hour": n // 24, "week": n * 7, "month": n * 30}.get(unit, n)

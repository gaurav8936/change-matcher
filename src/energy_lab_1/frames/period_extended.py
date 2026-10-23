"""period_extended frame — comment / notice / review periods extended.

Fires when a signal shows a schedule_change involving a period keyword AND
the numeric portion of the "days" in v2 is larger than in v1.
"""
from __future__ import annotations

import re
from typing import Any

from labs_core.reasoning.frames import Frame, FrameMatchResult

PERIOD_KEYWORDS = re.compile(
    r"\b(comment(?:s)? period|notice period|review period|scoping meeting|consultation)\b",
    re.IGNORECASE,
)
DAYS_RE = re.compile(r"\b(\d{1,3})\s*(?:calendar\s+)?days?\b", re.IGNORECASE)


class PeriodExtendedFrame(Frame):
    id = "period_extended"
    category = "period_extended"
    urgency = "low"

    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        best: FrameMatchResult | None = None
        for s in situation.get("signals", []):
            shape = s.get("shape")
            if shape not in ("schedule_change", "rewrite", "insertion"):
                continue
            before = s.get("text_before", "") or ""
            after = s.get("text_after", "") or ""
            merged = before + " " + after
            if not PERIOD_KEYWORDS.search(merged):
                continue
            days_before = [int(m.group(1)) for m in DAYS_RE.finditer(before)]
            days_after = [int(m.group(1)) for m in DAYS_RE.finditer(after)]
            confidence = 0.55
            features = ["period_keyword"]
            if days_before and days_after and max(days_after) > max(days_before):
                confidence = 0.85
                features.append(f"days: {max(days_before)}→{max(days_after)}")
            r = FrameMatchResult(
                frame_id=self.id,
                category=self.category,
                confidence=confidence,
                matched_features=features,
                predictions=[
                    "downstream deadlines that reference this period will shift by the same delta",
                    "no immediate action; monitor for downstream obligations",
                ],
                causal_story="A comment / review / scoping period has been extended.",
                urgency=self.urgency,
                recommended_reviewer_category="period_extended",
            )
            if best is None or r.confidence > best.confidence:
                best = r
        return best

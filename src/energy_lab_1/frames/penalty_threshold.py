"""penalty_threshold frame — deviation / penalty thresholds moved.

Fires on rewrite / renumber of a section containing "penalty" / "threshold" /
"breach" AND numeric percentages differ between v1 and v2.
"""
from __future__ import annotations

import re
from typing import Any

from labs_core.reasoning.frames import Frame, FrameMatchResult

PENALTY_KEYWORDS = re.compile(r"\b(penalt(?:y|ies)|threshold|material\s+breach|deviation)\b", re.IGNORECASE)
# Trailing boundary uses lookahead — \b fails between % and ) since both are non-word.
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)(?=\W|$)", re.IGNORECASE)


class PenaltyThresholdFrame(Frame):
    id = "penalty_threshold"
    category = "penalty_threshold"
    urgency = "high"

    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        best: FrameMatchResult | None = None
        for s in situation.get("signals", []):
            if s.get("shape") not in ("rewrite", "renumber", "schedule_change"):
                continue
            before = s.get("text_before", "") or ""
            after = s.get("text_after", "") or ""
            merged = before + " " + after
            if not PENALTY_KEYWORDS.search(merged):
                continue
            pb = sorted({float(m.group(1)) for m in PERCENT_RE.finditer(before)})
            pa = sorted({float(m.group(1)) for m in PERCENT_RE.finditer(after)})
            if not (pb or pa):
                continue
            if pb == pa:
                continue
            features = [f"percents: {pb} → {pa}"]
            confidence = 0.85
            # Tightened threshold (lower percent = stricter) is more consequential
            tightened = bool(pb and pa and min(pa) < min(pb))
            if tightened:
                features.append("threshold tightened")
                confidence = 0.90
            r = FrameMatchResult(
                frame_id=self.id,
                category=self.category,
                confidence=confidence,
                matched_features=features,
                predictions=[
                    "financial exposure changes at the new threshold",
                    "historical near-compliance cases may fall out of tolerance under the new rule",
                    "counsel + CRO review warranted",
                ],
                causal_story="A penalty or compliance threshold has moved.",
                urgency=self.urgency,
                recommended_reviewer_category="penalty_threshold",
            )
            if best is None or r.confidence > best.confidence:
                best = r
        return best

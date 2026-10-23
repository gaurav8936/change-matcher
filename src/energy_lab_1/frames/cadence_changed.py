"""cadence_changed frame — reporting or settlement frequency changed.

Fires when a signal's shape is schedule_change AND text contains a cadence
keyword AND the cadence differs between before and after.
"""
from __future__ import annotations

import re
from typing import Any

from labs_core.reasoning.frames import Frame, FrameMatchResult

CADENCE_TOKENS = ["quarterly", "monthly", "annually", "annual", "semiannual", "semiannually", "weekly", "daily", "biennial", "biennially"]
CADENCE_RE = re.compile(r"\b(" + "|".join(CADENCE_TOKENS) + r")\b", re.IGNORECASE)


class CadenceChangedFrame(Frame):
    id = "cadence_changed"
    category = "cadence_changed"
    urgency = "medium"

    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        best: FrameMatchResult | None = None
        for s in situation.get("signals", []):
            if s.get("shape") not in ("schedule_change", "rewrite"):
                continue
            before = s.get("text_before", "") or ""
            after = s.get("text_after", "") or ""
            b_cad = {t.lower() for t in CADENCE_RE.findall(before)}
            a_cad = {t.lower() for t in CADENCE_RE.findall(after)}
            if not (b_cad or a_cad):
                continue
            if b_cad == a_cad:
                # Same cadences on both sides — not a cadence change
                continue
            confidence = 0.85 if b_cad and a_cad else 0.65
            features = [f"cadence: {sorted(b_cad)} → {sorted(a_cad)}"]
            r = FrameMatchResult(
                frame_id=self.id,
                category=self.category,
                confidence=confidence,
                matched_features=features,
                predictions=[
                    "downstream reporting obligations will need to shift to the new cadence",
                    "operations must retool internal reporting workflows",
                    "compliance risk elevated during transition window",
                ],
                causal_story=f"Reporting/settlement cadence changed from {sorted(b_cad) or 'unspecified'} to {sorted(a_cad) or 'unspecified'}.",
                urgency=self.urgency,
                recommended_reviewer_category="cadence_changed",
            )
            if best is None or r.confidence > best.confidence:
                best = r
        return best

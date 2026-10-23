"""definition_narrowed frame — a defined term now excludes previously covered entities.

Fires on rewrite of a section containing "means" / "shall mean" / "definitions" AND
the after-text is longer / more restrictive than the before-text.
"""
from __future__ import annotations

import re
from typing import Any

from labs_core.reasoning.frames import Frame, FrameMatchResult

DEFINITION_KEYWORDS = re.compile(
    r"\b(means|shall\s+mean|definitions?)\b",
    re.IGNORECASE,
)
RESTRICTION_KEYWORDS = re.compile(
    r"\b(provided that|except|excluding|not\s+including|shall not|shall exclude|at least|no less than)\b",
    re.IGNORECASE,
)


class DefinitionNarrowedFrame(Frame):
    id = "definition_narrowed"
    category = "definition_narrowed"
    urgency = "high"

    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        best: FrameMatchResult | None = None
        for s in situation.get("signals", []):
            if s.get("shape") not in ("rewrite", "renumber", "insertion"):
                continue
            before = s.get("text_before", "") or ""
            after = s.get("text_after", "") or ""
            if not DEFINITION_KEYWORDS.search(after):
                continue
            features = ["definition_keyword"]
            confidence = 0.55
            restrictions_after = len(RESTRICTION_KEYWORDS.findall(after))
            restrictions_before = len(RESTRICTION_KEYWORDS.findall(before))
            if restrictions_after > restrictions_before:
                confidence = 0.85
                features.append(f"restrictions: {restrictions_before}→{restrictions_after}")
            elif len(after) > len(before) * 1.15:
                # Sub-signal: the after text is materially longer — often a narrowing
                confidence = 0.72
                features.append(f"length: {len(before)}→{len(after)}")
            r = FrameMatchResult(
                frame_id=self.id,
                category=self.category,
                confidence=confidence,
                matched_features=features,
                predictions=[
                    "obligations bearing on entities that fall outside the new definition may be released",
                    "compliance status of certain projects may need to be re-evaluated",
                    "counsel review is warranted — legal exposure may change",
                ],
                causal_story="A defined term has been narrowed to exclude previously covered entities.",
                urgency=self.urgency,
                recommended_reviewer_category="definition_narrowed",
            )
            if best is None or r.confidence > best.confidence:
                best = r
        return best

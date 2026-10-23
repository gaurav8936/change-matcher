"""Frame library — recognition-mode matcher.

A frame is a whole mental model: what pattern to recognise, what predictions
to make when it fires, and a decaying accuracy history. Frames are
domain-specific (each lab supplies its registry via Lab.frames) but the mechanism is
domain-agnostic and ports upstream.

Kept simple: rule-based matchers. LLM tie-break is optional (see reasoner).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("labs_core.frames")


@dataclass
class FrameMatchResult:
    frame_id: str
    category: str
    confidence: float                         # 0.0..1.0
    matched_features: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    causal_story: str = ""
    urgency: str = "medium"                   # low · medium · high
    recommended_reviewer_category: str = ""   # matches reviewer.covers_categories entries


class Frame(ABC):
    """Base class for a domain frame."""
    id: str = "abstract"
    category: str = "abstract"
    urgency: str = "medium"

    @abstractmethod
    def match(self, situation: dict[str, Any]) -> FrameMatchResult | None:
        """Return a FrameMatchResult if this frame fires, else None."""

    def predictions_from_match(self, situation: dict[str, Any], result: FrameMatchResult) -> list[str]:
        """Testable predictions the frame makes given its match."""
        return result.predictions


class FrameRegistry:
    """Container for domain frames — the Reasoner enumerates matches."""
    def __init__(self) -> None:
        self._frames: list[Frame] = []
        # Accuracy history (per-frame). Updates on outcome feedback.
        self._accuracy: dict[str, float] = {}

    def register(self, frame: Frame) -> None:
        self._frames.append(frame)
        self._accuracy.setdefault(frame.id, 1.0)

    def frames(self) -> list[Frame]:
        return list(self._frames)

    def match_all(self, situation: dict[str, Any]) -> list[FrameMatchResult]:
        """Return every frame match, sorted by confidence descending."""
        results: list[FrameMatchResult] = []
        for f in self._frames:
            try:
                r = f.match(situation)
                if r:
                    # Decay by accuracy history
                    r.confidence *= self._accuracy.get(f.id, 1.0)
                    results.append(r)
            except Exception as e:
                logger.warning("frame %s match raised: %s", f.id, e)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def best_match(self, situation: dict[str, Any], min_confidence: float = 0.60) -> FrameMatchResult | None:
        matches = self.match_all(situation)
        if matches and matches[0].confidence >= min_confidence:
            return matches[0]
        return None

    def update_accuracy(self, frame_id: str, correct: bool, learning_rate: float = 0.1) -> None:
        """Move the frame's accuracy toward 1.0 on correct, toward 0.0 on wrong."""
        current = self._accuracy.get(frame_id, 1.0)
        target = 1.0 if correct else 0.0
        self._accuracy[frame_id] = current + learning_rate * (target - current)

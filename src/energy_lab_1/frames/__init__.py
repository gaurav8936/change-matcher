"""Energy Lab regulatory-change frames.

Five frames covering the shapes that recur in real utility regulatory work:
  - period_extended        (comment/notice periods extended)
  - date_deferred          (effective dates, compliance windows deferred)
  - definition_narrowed    (a term redefined to exclude previously covered entities)
  - cadence_changed        (reporting frequency changed)
  - penalty_threshold      (deviation / penalty thresholds moved)

Each frame reads a signal's shape and text to fire (or not). Matches ship
with predictions and a recommended reviewer category.

PUBLIC API (stable within energy-lab-1 0.x; Energy Lab 2 depends on it): `build_registry()` and the five frame
classes. Frames take a situation dict with `signals: [{shape, text_before, text_after, ...}]` (the shape
`energy_lab_1.signal.hunk_to_signal` emits) and return `labs_core.reasoning.frames.FrameMatchResult`.
"""
from __future__ import annotations

from labs_core.reasoning.frames import FrameRegistry

from energy_lab_1.frames.cadence_changed import CadenceChangedFrame
from energy_lab_1.frames.date_deferred import DateDeferredFrame
from energy_lab_1.frames.definition_narrowed import DefinitionNarrowedFrame
from energy_lab_1.frames.penalty_threshold import PenaltyThresholdFrame
from energy_lab_1.frames.period_extended import PeriodExtendedFrame

__all__ = [
    "build_registry", "CadenceChangedFrame", "DateDeferredFrame", "DefinitionNarrowedFrame",
    "PenaltyThresholdFrame", "PeriodExtendedFrame",
]


def build_registry() -> FrameRegistry:
    """A fresh Energy Lab 1 frame registry (the five frames, in match order). Public API."""
    r = FrameRegistry()
    r.register(PeriodExtendedFrame())
    r.register(DateDeferredFrame())
    r.register(DefinitionNarrowedFrame())
    r.register(CadenceChangedFrame())
    r.register(PenaltyThresholdFrame())
    return r

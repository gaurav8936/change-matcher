"""The one place Energy Lab 2 reaches into Energy Lab 1 (the `energy-lab-1` package).

Energy Lab 2 reuses two pure pieces of Energy Lab 1, by design:

  build_registry   the frame registry (period extended, date deferred, definition narrowed, cadence changed,
                   penalty threshold, ...): the hypothesis engine's priors, used by the fallback adjudicator and
                   to name the change kind in the impact stage;
  parse_sections   the regulatory `§` section parser, the first provision shape consumption tries;
  version_diff     the line diff the four-arms eval uses as arm A (eval only).

Nothing of Energy Lab 1 is copied into this repo; everything goes through this module. `reasoning_hooks()`
optionally lends Energy Lab 1's reasoning hooks to the `llm=True` impact path (prompts + evidence tools), and
falls back to labs_core's generic hooks when Energy Lab 1 does not expose them.

`ENERGY_LAB_2_EL1_STUB=1` (set by the test suite only, when `energy_lab_1` is not installed) swaps in an inert
stub: no `§` sections and an empty frame registry. Tests that need the real frames are marked `needs_el1` and
skip under the stub. The engine itself never runs on the stub.
"""
from __future__ import annotations

import os
from typing import Any

STUB = False

try:
    from energy_lab_1.frames import build_registry
    from energy_lab_1.signal import parse_sections
except ImportError:
    if os.environ.get("ENERGY_LAB_2_EL1_STUB") != "1":
        raise
    STUB = True

    def build_registry():  # type: ignore[no-redef]
        from labs_core.reasoning.frames import FrameRegistry
        return FrameRegistry()

    def parse_sections(text: str) -> dict[str, Any]:  # type: ignore[no-redef]
        return {}


def version_diff(before: str, after: str, docket_id: str = "", draft_flag: bool = False) -> list[Any]:
    """Energy Lab 1's version diff (four-arms eval, arm A)."""
    from energy_lab_1.signal import version_diff as _vd
    return _vd(before, after, docket_id, draft_flag=draft_flag)


def reasoning_hooks() -> Any:
    """Energy Lab 1's ReasoningHooks when its Lab exposes them, else None (labs_core's generic hooks)."""
    if STUB:
        return None
    try:
        from energy_lab_1.lab import LAB as _el1
    except Exception:
        return None
    return getattr(_el1, "reasoning", None)


__all__ = ["STUB", "build_registry", "parse_sections", "reasoning_hooks", "version_diff"]

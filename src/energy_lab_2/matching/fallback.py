"""Hypothesis fallback: when the cross-encoder is unsure, or the guard disagrees with it, the Energy Lab's
frames (energy_lab_1.frames) adjudicate (WS-ED-MATCH-01 · step 6).

The aligned pair becomes an Energy Lab signal (same shape as `signal.hunk_to_signal`) and runs through the frame
registry (period extended, date deferred, definition narrowed, cadence changed, penalty threshold). Decision:

  a frame fires strongly (≥ 0.8), or fires (≥ 0.5) with a guard delta → material, the frame is the evidence
                                    (frames fire on keywords alone at ~0.55: "within thirty days" in an untouched
                                    section is a keyword, not a change)
  no frame, guard sees a delta      → material, the guard is the evidence (the guard only escalates)
  neither                           → keep the model's label; needs review if still under the floor

Both opinions (model + adjudicator) are returned so the trace shows who decided.
"""
from __future__ import annotations

from functools import cache
from typing import Any

FRAME_FLOOR = 0.5
FRAME_STRONG = 0.8


@cache
def _registry():
    # Deliberate cross-lab dependency (energy-lab-1 package): the frames ARE the hypothesis engine's priors.
    from energy_lab_2._el1 import build_registry
    return build_registry()


def to_signal(old_addr: str, new_addr: str, before: str, after: str, guard_deltas: list[dict]) -> dict[str, Any]:
    if old_addr and new_addr and old_addr != new_addr:
        shape = "renumber"
    elif any(d["kind"] in ("duration", "cadence", "percent") for d in guard_deltas):
        shape = "schedule_change"
    else:
        shape = "rewrite"
    return {"shape": shape, "section_ref": old_addr or new_addr, "section_ref_after": new_addr,
            "text_before": before, "text_after": after, "draft_flag": False}


def adjudicate(*, old_addr: str, new_addr: str, before: str, after: str, model_label: str, model_confidence: float,
               guard_deltas: list[dict], floor: float) -> dict[str, Any]:
    sig = to_signal(old_addr, new_addr, before, after, guard_deltas)
    matches = _registry().match_all({"signals": [sig]})
    top = matches[0] if matches else None
    frames = [{"frame_id": m.frame_id, "confidence": round(m.confidence, 3), "features": m.matched_features,
               "causal_story": m.causal_story} for m in matches[:3]]
    if top is not None and (top.confidence >= FRAME_STRONG or (top.confidence >= FRAME_FLOOR and guard_deltas)):
        return {"label": "material", "confidence": round(max(top.confidence, model_confidence if model_label == "material" else 0.0), 4),
                "decided_by": "fallback:frame", "frames": frames, "needs_review": top.confidence < floor}
    if guard_deltas:
        return {"label": "material", "confidence": 0.8, "decided_by": "fallback:guard", "frames": frames,
                "needs_review": True}
    return {"label": model_label, "confidence": round(model_confidence, 4), "decided_by": "fallback:model_kept",
            "frames": frames, "needs_review": model_confidence < floor}

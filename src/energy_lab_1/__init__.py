"""energy_lab_1 — Energy Lab 1 · obligations, impact and action. A labs_core lab (`LAB=energy_utilities`).

A regulatory docket changes between two versions; the lab diffs it into shape-classified signals, assembles the
situation for a persona (obligations, projects, documents, reviewers), recognises one of five regulatory-change
frames, gathers evidence with hash-anchored citations, and routes a warrant to a reviewer queue.

Public API for other labs (stable within 0.x; Energy Lab 2 depends on it):

    from energy_lab_1.frames import build_registry     # the five Energy Lab frames (labs_core FrameRegistry)
    from energy_lab_1.signal import parse_sections     # docket text → {"§4.7": Section(address, heading, body, body_hash)}
    from energy_lab_1.signal import version_diff, hunk_to_signal   # diff two versions → signals frames read

See docs/public-api.md. Everything else is internal to the lab.
"""
from __future__ import annotations

__version__ = "0.1.1"

__all__ = ["__version__"]

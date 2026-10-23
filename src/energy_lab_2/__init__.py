"""Energy Lab 2 · change: alignment and classification (a labs_core lab; lab name `energy_documents`).

The Document Lab's intake + consumption joined to Energy Lab 1's reasoning, with a cross-encoder matcher between
document versions. Pipeline: arrival (a version pair) → intake gate → consumption into provisions → version
alignment (hash → BM25 candidates → cross-encoder) → change classification (+ guard + hypothesis fallback) →
impact on obligations → review → action.

Registered with labs_core through the entry point `labs_core.labs: energy_documents = energy_lab_2.lab:LAB`.
Energy Lab 1 pieces (frames, § parser) come from the `energy-lab-1` package through `energy_lab_2._el1`.
"""
__version__ = "0.1.0"

"""Version matching for the Energy and Documents Lab (WS-ED-MATCH-01): hash pass → BM25 candidates → pair scorer
(cross-encoder or stub) → one-to-one assignment → labels + confidence → guard → hypothesis fallback."""
from energy_lab_2.matching.align import LABELS, Alignment, Change, align
from energy_lab_2.matching.scorer import PairScore, PairScorer, StubScorer, load_scorer

__all__ = ["LABELS", "Alignment", "Change", "align", "PairScore", "PairScorer", "StubScorer", "load_scorer"]

"""Provenance shared by every lab (WS-LC-CORE-01 · K3).

A `SourceSpan` generalises the case engine's `Citation` (labs_core/models/schemas.py) with a page and a box, so a
citation can point into a PDF. The fidelity ladder says how strongly a cited span supports a value:

    EXACT        sha256 of the normalised quoted text equals the stored span hash (hash-anchored citations)
    NORMALISED   matches after OCR clean-up (whitespace, ligatures, confusable characters)
    VALUE_IN_SPAN the extracted value is present in, or derivable from, the quoted text
    ENTAILED     a check model confirms the quote supports the field

A field is trusted only when it reaches the rung its spec requires. No span, or a failed rung, counts as a
hallucination. The verifier is `verify.py` (`verify_field`, `trusted`); value parsing is `values.py`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import IntEnum


class Fidelity(IntEnum):
    NONE = 0
    EXACT = 1
    NORMALISED = 2
    VALUE_IN_SPAN = 3
    ENTAILED = 4


def normalise(text: str) -> str:
    """The one normaliser both labs hash with: collapse whitespace, unify quotes/dashes, drop ligatures, lowercase."""
    t = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[–—]", "-", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def span_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


@dataclass
class SourceSpan:
    """Where a value came from. `bbox` is PDF points, origin top-left: [x0, y0, x1, y1]."""
    document_id: str
    version: str
    quoted_text: str
    page: int | None = None                 # 1-based; None for plain-text sources
    bbox: list[float] | None = None
    char_start: int | None = None           # offsets into the document's extracted text, when known
    char_end: int | None = None
    paragraph_range: str | None = None      # plain-text paragraph citations
    span_hash: str = ""

    def __post_init__(self) -> None:
        if not self.span_hash:
            self.span_hash = span_hash(self.quoted_text)


@dataclass
class FieldValue:
    """One extracted value with its evidence and verdict."""
    name: str
    value: object
    span: SourceSpan | None
    confidence: float = 0.0
    fidelity: Fidelity = Fidelity.NONE
    notes: list[str] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return self.span is None or self.fidelity == Fidelity.NONE

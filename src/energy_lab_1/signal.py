"""Docket version-diff — the riskiest single item in the build.

Given two versions of a docket, emit a list of signals. Each signal is a
DocketDiffHunk with a shape classification:

  INSERTION | DELETION | RENUMBER | REWRITE | SCHEDULE_CHANGE | UNKNOWN

Strategy (per TDD §4):
  1. Structural extraction BEFORE diffing. Parse each version into
     sections keyed by their address (§1, §4.7, etc.).
  2. Diff structured sections, not raw text. This distinguishes RENUMBER
     (same body, moved address) from REWRITE (same address, changed body).
  3. Shape classifier: rules first, LLM fallback ready, UNKNOWN escape.
  4. Every hunk becomes a Signal for the pipeline.

Kept simple: pure Python, stdlib only for the base classifier.
"""
from __future__ import annotations

import difflib
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from energy_lab_1 import paths

logger = logging.getLogger("energy_lab_1.signal")

SECTION_HEADING_RE = re.compile(r"^\s*(§[0-9][0-9\.]*)\.?\s*(.*)$", re.MULTILINE)
NUMBER_TOKEN_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|days?|hours?|weeks?|months?|calendar\s+(?:day|month|quarter|year)s?)\b",
    re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(
    r"\b(?:quarterly|monthly|annually|semiannually|biennially|weekly|daily)\b",
    re.IGNORECASE,
)


@dataclass
class Section:
    """One parsed section of a docket version."""
    address: str                    # e.g. "§4.7.2"
    heading: str                    # first line after the address
    body: str                       # full text of the section (excluding heading)
    body_hash: str                  # sha256 of normalized body — identity fingerprint

    @property
    def normalized_body(self) -> str:
        return _normalize(self.body)


@dataclass
class DocketDiffHunk:
    """One material difference between v_before and v_after."""
    docket_id: str
    shape: str                      # DiffShape.value
    section_ref_before: str         # empty on INSERTION
    section_ref_after: str          # empty on DELETION
    text_before: str
    text_after: str
    draft_flag: bool
    similarity: float               # 0.0..1.0 — used for shape classification
    numbers_before: list[str] = field(default_factory=list)
    numbers_after: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────
# Parsing
# ────────────────────────────────────────────────────────────────

def parse_sections(text: str) -> dict[str, Section]:
    """Parse a docket-version text into a dict of {address: Section}."""
    sections: dict[str, Section] = {}
    matches = list(SECTION_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        address = m.group(1).rstrip(".")     # normalise "§4.7." → "§4.7"
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        # Body includes the heading text — different regulators split
        # heading/body inconsistently, so we treat address+heading+body as
        # one canonical piece for similarity + hashing purposes.
        body = (heading + "\n" + text[start:end]).strip()
        body_hash = hashlib.sha256(_normalize(body).encode("utf-8")).hexdigest()
        sections[address] = Section(address=address, heading=heading, body=body, body_hash=body_hash)
    return sections


def _normalize(text: str) -> str:
    """Whitespace + line-ending normalisation before hashing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


# ────────────────────────────────────────────────────────────────
# Diff + shape classification
# ────────────────────────────────────────────────────────────────

def version_diff(v_before: str, v_after: str, docket_id: str, draft_flag: bool = False) -> list[DocketDiffHunk]:
    """Diff two docket versions. Return a list of shape-classified hunks."""
    before = parse_sections(v_before)
    after = parse_sections(v_after)

    hunks: list[DocketDiffHunk] = []
    # Pass 0 — identical at the same address: silence, and neither side is free for the passes below.
    matched_before: set[str] = {a for a in before if a in after and after[a].body_hash == before[a].body_hash}
    matched_after: set[str] = set(matched_before)

    # Pass 1 — RENUMBER: same body (exact hash) OR high similarity, different address.
    # High-similarity renumbers with edits are common in real regulatory changes and
    # they are precisely what triggers the citation-drift failure path.
    # Renumber threshold — real-world regulatory renumbers often edit the body
    # simultaneously (see the LNG penalty structure v1→v2 example). 0.60 catches
    # these while staying above the false-positive floor for unrelated sections.
    RENUMBER_SIMILARITY_THRESHOLD = 0.60
    for addr_b, sec_b in list(before.items()):
        if addr_b in matched_before:
            continue
        best_match: tuple[str, float] | None = None
        for addr_a, sec_a in after.items():
            if addr_a in matched_after or addr_a == addr_b:
                continue
            if sec_a.body_hash == sec_b.body_hash:
                best_match = (addr_a, 1.0)
                break
            similarity = difflib.SequenceMatcher(None, sec_b.normalized_body, sec_a.normalized_body).ratio()
            if similarity >= RENUMBER_SIMILARITY_THRESHOLD and (best_match is None or similarity > best_match[1]):
                best_match = (addr_a, similarity)
        if best_match is not None:
            addr_a, similarity = best_match
            hunks.append(DocketDiffHunk(
                docket_id=docket_id, shape="renumber",
                section_ref_before=addr_b, section_ref_after=addr_a,
                text_before=sec_b.body, text_after=after[addr_a].body,
                draft_flag=draft_flag, similarity=similarity,
            ))
            matched_before.add(addr_b); matched_after.add(addr_a)

    # Pass 2 — REWRITE / SCHEDULE_CHANGE: same address, different body.
    for addr in sorted(set(before.keys()) & set(after.keys())):
        if addr in matched_before or addr in matched_after:
            continue
        sec_b = before[addr]; sec_a = after[addr]
        if sec_b.body_hash == sec_a.body_hash:
            continue                # identical — silence
        similarity = difflib.SequenceMatcher(None, _normalize(sec_b.body), _normalize(sec_a.body)).ratio()
        nb = NUMBER_TOKEN_RE.findall(sec_b.body) + DATE_TOKEN_RE.findall(sec_b.body)
        na = NUMBER_TOKEN_RE.findall(sec_a.body) + DATE_TOKEN_RE.findall(sec_a.body)
        shape = _classify_rewrite(sec_b.body, sec_a.body, similarity, nb, na)
        hunks.append(DocketDiffHunk(
            docket_id=docket_id, shape=shape,
            section_ref_before=addr, section_ref_after=addr,
            text_before=sec_b.body, text_after=sec_a.body,
            draft_flag=draft_flag, similarity=similarity,
            numbers_before=[str(x) for x in nb], numbers_after=[str(x) for x in na],
        ))
        matched_before.add(addr); matched_after.add(addr)

    # Pass 3 — INSERTION: every after-section nothing above matched. This includes an address the before version
    # also used, when its old section moved away (a section inserted mid-document renumbers the ones after it).
    for addr_a in sorted(set(after.keys()) - matched_after):
        sec_a = after[addr_a]
        hunks.append(DocketDiffHunk(
            docket_id=docket_id, shape="insertion",
            section_ref_before="", section_ref_after=addr_a,
            text_before="", text_after=sec_a.body,
            draft_flag=draft_flag, similarity=0.0,
        ))

    # Pass 4 — DELETION: every before-section nothing above matched (same reasoning as Pass 3).
    for addr_b in sorted(set(before.keys()) - matched_before):
        sec_b = before[addr_b]
        hunks.append(DocketDiffHunk(
            docket_id=docket_id, shape="deletion",
            section_ref_before=addr_b, section_ref_after="",
            text_before=sec_b.body, text_after="",
            draft_flag=draft_flag, similarity=0.0,
        ))

    return hunks


def _classify_rewrite(before: str, after: str, similarity: float, nb: list, na: list) -> str:
    """Classify a same-address rewrite as SCHEDULE_CHANGE or REWRITE.

    Heuristic: if the sets of number/date tokens differ AND similarity is high
    (mostly the same text with a numeric or cadence swap), call it a schedule
    change. Otherwise it's a general rewrite.
    """
    if similarity >= 0.60 and set(nb) != set(na):
        return "schedule_change"
    if similarity < 0.25:
        return "unknown"           # fell too far; escape hatch
    return "rewrite"


# ────────────────────────────────────────────────────────────────
# Signal object — engine-facing
# ────────────────────────────────────────────────────────────────

def hunk_to_signal(hunk: DocketDiffHunk) -> dict[str, Any]:
    """Convert a diff hunk into the engine's Signal shape (dict form)."""
    return {
        "signal_id": f"sig-{uuid.uuid4().hex[:12]}",
        "docket_id": hunk.docket_id,
        "shape": hunk.shape,
        "section_ref": hunk.section_ref_before or hunk.section_ref_after,
        "section_ref_after": hunk.section_ref_after,
        "text_before": hunk.text_before,
        "text_after": hunk.text_after,
        "draft_flag": hunk.draft_flag,
        "similarity": hunk.similarity,
        "numbers_before": hunk.numbers_before,
        "numbers_after": hunk.numbers_after,
        "created_at": time.time(),
    }


# ────────────────────────────────────────────────────────────────
# Loading dockets — corpus batches (LABS_CORPUS_ROOTS) then the canonical demos (data/dockets)
# ────────────────────────────────────────────────────────────────

def load_docket_pair(docket_ref: str, data_root: str | Path | None = None) -> tuple[str, str, dict[str, Any]]:
    """Return (v_before_text, v_after_text, manifest_dict) for a canonical demo docket
    (`{data_root}/{ref}/manifest.json` + the two version files it lists). Default root: the lab's `dockets` dir."""
    import json
    root = Path(data_root or paths.data_dir("dockets")) / docket_ref
    if not root.exists():
        raise FileNotFoundError(f"docket dir not found: {root}")
    manifest = json.loads((root / "manifest.json").read_text())
    versions = manifest.get("versions", [])
    if len(versions) < 2:
        raise ValueError(f"docket {docket_ref} needs at least two versions")
    v_before = (root / versions[0]["path"]).read_text()
    v_after = (root / versions[1]["path"]).read_text()
    return v_before, v_after, manifest


def load_corpus_docket_pair(docket_id: str, corpus_root: str | None = None) -> tuple[str, str, dict[str, Any]]:
    """Return (v_before_text, v_after_text, docket_dict) for a docket in the synthetic corpus.

    Layout: {corpus_root}/{docket_id}/{v1.txt, v2.txt, docket.json}.
    Third element is the parsed docket.json — shape-compatible with the observer's
    use of `manifest.get("docket_id")` and `versions[1].get("draft_flag")`.

    Multi-version dockets still have v1.txt + v2.txt for the diff pipeline;
    v3.txt..v5.txt exist alongside for the adjudicator and other consumers that
    walk the transition chain.
    """
    import json
    root = Path(corpus_root or paths.corpus_root()) / docket_id
    if not root.exists():
        raise FileNotFoundError(f"corpus docket dir not found: {root}")
    docket = json.loads((root / "docket.json").read_text())
    v_before = (root / "v1.txt").read_text()
    v_after = (root / "v2.txt").read_text()
    return v_before, v_after, docket


def _corpus_roots_from_env() -> list[str]:
    """Comma-separated LABS_CORPUS_ROOTS ⇒ list of paths (falls back to [LABS_CORPUS_ROOT]; HYPOTHESYS_* read too)."""
    return paths.corpus_roots()


def resolve_docket_pair(ref_or_id: str) -> tuple[str, str, dict[str, Any]]:
    """Try every corpus batch (LABS_CORPUS_ROOTS) in order, then the canonical demo dockets. First hit wins.

    Raises FileNotFoundError with every attempted path if none hit.
    """
    tried: list[str] = []
    for corpus_root in _corpus_roots_from_env():
        try:
            return load_corpus_docket_pair(ref_or_id, corpus_root=corpus_root)
        except FileNotFoundError:
            tried.append(f"{corpus_root}/{ref_or_id}")
    legacy_root = paths.data_dir("dockets")
    try:
        return load_docket_pair(ref_or_id, data_root=legacy_root)
    except FileNotFoundError:
        tried.append(f"{legacy_root}/{ref_or_id}")
    raise FileNotFoundError(
        f"docket {ref_or_id!r} not found — tried: " + " · ".join(tried)
    )


def resolve_docket_dir(docket_id: str) -> Path | None:
    """The on-disk directory for a docket across any corpus batch, or the canonical demo root. None if unknown.
    Used by the reasoning hooks + the executor (via source_text) to locate v2.txt for citation retrieval + verification.
    """
    for corpus_root in _corpus_roots_from_env():
        d = Path(corpus_root) / docket_id
        if d.exists():
            return d
    legacy = paths.data_dir("dockets") / docket_id
    if legacy.exists():
        return legacy
    return None


def source_text(docket_id: str) -> str:
    """The after-version text (v2.txt) citations are retrieved from and verified against ("" when unknown)."""
    d = resolve_docket_dir(docket_id)
    if d is not None and (d / "v2.txt").exists():
        return (d / "v2.txt").read_text()
    return ""

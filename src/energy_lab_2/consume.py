"""Consumption: a document version → provisions with provenance (WS-ED-LAB-01 · stage 3).

A provision is the unit the matcher aligns: an address (`§4.7`, `7.2`, `¶3`), a heading, a body and the shared
span hash (`labs_core.provenance.span_hash`). Three shapes are recognised, in order:

  1. regulatory `§` sections (Energy Lab 1's parser, `energy_lab_1.signal.parse_sections`, reused as is);
  2. numbered clauses: lines starting `7.`, `7.2`, `Section 7`, `Article 7`, `Clause 7` (contracts);
  3. paragraphs separated by blank lines (`¶1`, `¶2`, …) when nothing else is found.

PDFs go through the shared intake gate first (`labs_core.intake.run_intake`); text submissions get the text-level
checks (content hash, injection and PII scan) so nothing skips intake.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from labs_core.provenance import span_hash

# Deliberate cross-lab dependency: Energy Lab 1's § parser (a pure function), via the energy-lab-1 package.
from energy_lab_2._el1 import parse_sections

CLAUSE_RE = re.compile(
    r"^\s*(?:(?:section|article|clause)\s+)?(\d{1,3}(?:\.\d{1,3}){0,3})[.)]?\s+([^\n]{0,160})$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Provision:
    address: str
    heading: str
    body: str
    hash: str
    order: int
    page: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mk(address: str, heading: str, body: str, order: int, page: int | None = None) -> Provision:
    return Provision(address=address, heading=heading.strip(), body=body.strip(), hash=span_hash(body), order=order,
                     page=page)


def provisions_from_text(text: str) -> list[Provision]:
    """Split a document version into provisions (see module doc for the three shapes)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    secs = parse_sections(text)
    if secs:
        return [_mk(s.address, s.heading, s.body, i) for i, s in enumerate(secs.values())]
    matches = [m for m in CLAUSE_RE.finditer(text)]
    if len(matches) >= 2:
        out: list[Provision] = []
        seen: set[str] = set()
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            addr = m.group(1)
            if addr in seen:            # a repeated number is body text, not a new clause
                continue
            seen.add(addr)
            body = (m.group(2) + "\n" + text[m.end():end]).strip()
            out.append(_mk(addr, m.group(2), body, len(out)))
        return out
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [_mk(f"¶{i + 1}", p.split("\n", 1)[0][:80], p, i) for i, p in enumerate(paras)]


def text_intake(text: str, *, document_id: str | None = None) -> dict[str, Any]:
    """Intake record for a text submission: the checks that apply to text (the PDF checks are skipped, and say so)."""
    from labs_core.intake import scan
    from labs_core.provenance.verify import PageText

    data = (text or "").encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    pages = [PageText(page=1, text=text or "")]
    checks: list[dict[str, Any]] = [
        {"name": "file_type", "status": "pass", "detail": "plain text submission; PDF checks not applicable"},
        {"name": "size_bytes", "status": "pass" if len(data) <= 5_000_000 else "fail", "detail": f"{len(data)} bytes"},
        {"name": "empty", "status": "pass" if (text or "").strip() else "fail", "detail": "has text" if (text or "").strip() else "no text"},
    ]
    hits = scan.injection_hits(pages)
    checks.append({"name": "prompt_injection", "status": "warn" if hits else "pass",
                   "detail": f"{len(hits)} instruction-like passage(s)" if hits else "no instruction-like text"})
    pii = scan.pii_flags(pages)
    checks.append({"name": "pii", "status": "warn" if pii else "pass",
                   "detail": ", ".join(sorted(pii)) if pii else "none found"})
    decision = "reject" if any(c["status"] == "fail" for c in checks) else (
        "review" if any(c["status"] == "warn" for c in checks) else "accept")
    return {"document_id": document_id or f"txt-{sha[:12]}", "sha256": sha, "bytes": len(data), "file_type": "text",
            "checks": checks, "decision": decision}


def consume(data: bytes | str, *, document_id: str | None = None, filename: str = "") -> dict[str, Any]:
    """Intake + consumption for one version. Text (str) or PDF bytes. Returns {intake, provisions, text}."""
    if isinstance(data, str):
        rec = text_intake(data, document_id=document_id)
        provs = provisions_from_text(data) if rec["decision"] != "reject" else []
        return {"intake": rec, "provisions": provs, "text": data}
    from labs_core.intake import run_intake
    res = run_intake(data, document_id=document_id, filename=filename, register=False)
    try:
        rec = res.record.as_dict()
        if rec.get("decision") == "reject":
            return {"intake": rec, "provisions": [], "text": ""}
        provs: list[Provision] = []
        full = []
        for pg in res.pages:
            full.append(pg.text)
        text = "\n\n".join(full)
        provs = provisions_from_text(text)
        # page of each provision: first page whose text contains the provision's opening words
        for p in provs:
            head = p.body[:40]
            p.page = next((pg.page for pg in res.pages if head and head in pg.text), None)
        return {"intake": rec, "provisions": provs, "text": text}
    finally:
        res.close()

"""The guard: deterministic "did a value or a duty change?" check (WS-ED-MATCH-01 · condition 1).

Sub-word transformers read magnitudes poorly, so a small cross-encoder can call "45 days" → "75 days" equivalent.
The guard extracts the things an obligation turns on, in canonical form so restyling is not a change:

  durations     "thirty (30) days" = "30 days" = P30D        (labs_core.provenance.values.find_durations)
  percentages   "five percent (5%)" = "5%" = 5
  money         "$12,480.00" = "USD 12480" (currency marker required)
  cadence       monthly · quarterly · annually · …
  modality      shall / must / will (duty) vs may / can (permission)
  negation      not · no · never · neither … nor

and reports the deltas. The guard can only escalate a pair to the fallback; it never silences a change.
"""
from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any

from labs_core.provenance.values import _NUM_WORDS, find_durations

_PCT_RX = re.compile(r"(?:(\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s+cent\b))|(?:\b([a-z]+(?:[- ][a-z]+)?)\s+(?:percent|per\s+cent)\b)", re.I)
_PAREN_PCT = re.compile(r"\((\d+(?:\.\d+)?)\s*%\)")
_MONEY = re.compile(r"(?:[$€£]|\b(?:usd|eur|gbp)\s?)\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", re.I)
_CADENCE = re.compile(r"\b(daily|weekly|fortnightly|monthly|bi-?monthly|quarterly|semi-?annually|annually|yearly|biennially)\b", re.I)
_DUTY = re.compile(r"\b(shall|must|will be required to|is required to|are required to)\b", re.I)
_PERMIT = re.compile(r"\b(may|can|is permitted to|are permitted to|is encouraged to|are encouraged to)\b", re.I)
_NEG = re.compile(r"\b(not|no|never|neither|nor|without)\b", re.I)
_HOURS = re.compile(r"(?:\(\s*(\d+)\s*\)|\b(\d+))\s*[- ]?(?:business\s+)?hours?\b", re.I)
_CADENCE_CANON = {"yearly": "annually", "bimonthly": "bi-monthly", "semiannually": "semi-annually"}


def _pcts(text: str) -> list[str]:
    out: list[str] = []
    paren = {m.group(1) for m in _PAREN_PCT.finditer(text)}
    for m in _PCT_RX.finditer(text):
        if m.group(1):
            v = m.group(1)
        else:
            w = (m.group(2) or "").lower()
            n = _NUM_WORDS.get(w) or _NUM_WORDS.get(w.split()[-1]) or _NUM_WORDS.get(w.split("-")[-1])
            if not n or w in ("a", "an"):
                continue
            v = str(n)
            if v in paren:          # "five percent (5%)": the word form and the digit form are one value
                continue
        try:
            v = format(Decimal(v).normalize(), "f")
        except InvalidOperation:
            continue
        out.append(v)
    return out


def _money(text: str) -> list[str]:
    out = []
    for m in _MONEY.finditer(text):
        try:
            out.append(format(Decimal(m.group(1).replace(",", "")).normalize(), "f"))
        except InvalidOperation:
            pass
    return out


def _count(name: str, rx: re.Pattern, text: str) -> Counter:
    n = len(rx.findall(text))
    return Counter({name: n}) if n else Counter()


def features(text: str) -> dict[str, Counter]:
    t = text or ""
    cad = [(_CADENCE_CANON.get(c.lower().replace("-", ""), c.lower())) for c in _CADENCE.findall(t)]
    return {
        "duration": Counter(find_durations(t) + [f"PT{a or b}H" for a, b in _HOURS.findall(t)]),
        "percent": Counter(_pcts(t)),
        "money": Counter(_money(t)),
        "cadence": Counter(cad),
        "duty": _count("duty", _DUTY, t),
        "permission": _count("permission", _PERMIT, t),
        "negation": _count("negation", _NEG, t),
    }


def deltas(before: str, after: str) -> list[dict[str, Any]]:
    """Every value or duty difference between two provision texts, e.g.
    [{"kind": "duration", "before": ["P45D"], "after": ["P75D"]}]. Empty list = the guard sees no material signal."""
    fb, fa = features(before), features(after)
    out: list[dict[str, Any]] = []
    for kind in fb:
        b, a = fb[kind], fa[kind]
        if kind in ("duty", "permission", "negation"):
            # count changes in modality / negation matter; restyling keeps the count
            if sum(b.values()) != sum(a.values()):
                out.append({"kind": kind, "before": sum(b.values()), "after": sum(a.values())})
            continue
        if set(b) != set(a):
            out.append({"kind": kind, "before": sorted(b), "after": sorted(a)})
    return out


def describe(ds: list[dict[str, Any]]) -> str:
    parts = []
    for d in ds:
        b, a = d["before"], d["after"]
        b = ", ".join(map(str, b)) if isinstance(b, list) else str(b)
        a = ", ".join(map(str, a)) if isinstance(a, list) else str(a)
        parts.append(f"{d['kind']} {b or '—'} → {a or '—'}")
    return "; ".join(parts)

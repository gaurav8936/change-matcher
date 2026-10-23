"""Candidate generation: for each old provision, the new provisions worth scoring (WS-ED-MATCH-01 · step 2).

Same address first (most provisions keep their number), then BM25 top-k over the remaining new provisions
(`rank-bm25`, already an engine dependency). The cross-encoder only ever sees these pairs, so a document with
n provisions costs about n·(k+1) model calls, not n².
"""
from __future__ import annotations

import re
from collections.abc import Sequence

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the of and to a in or for by be shall any such this that with on as is are an at from".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall((text or "").lower()) if t not in _STOP]


def candidates(old: Sequence, new: Sequence, *, k: int = 4) -> dict[int, list[int]]:
    """{old index: [new indexes]} — provisions need `.address` and `.body`."""
    if not new:
        return {i: [] for i in range(len(old))}
    try:
        from rank_bm25 import BM25Okapi
        bm = BM25Okapi([tokens(n.heading + " " + n.body) or ["_"] for n in new])
    except ImportError:          # pragma: no cover - rank-bm25 ships with the engine
        bm = None
    by_addr = {n.address: j for j, n in enumerate(new)}
    out: dict[int, list[int]] = {}
    for i, o in enumerate(old):
        picks: list[int] = []
        if o.address in by_addr:
            picks.append(by_addr[o.address])
        if bm is not None:
            scores = bm.get_scores(tokens(o.heading + " " + o.body) or ["_"])
            ranked = sorted(range(len(new)), key=lambda j: -scores[j])
            for j in ranked:
                if len(picks) >= k + (1 if o.address in by_addr else 0):
                    break
                if j not in picks:
                    picks.append(j)
        else:
            picks.extend(j for j in range(len(new)) if j not in picks)
        out[i] = picks
    return out

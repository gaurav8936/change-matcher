"""Version alignment + change classification (WS-ED-MATCH-01 · steps 1–6).

    align(old_provisions, new_provisions, scorer=…) → Alignment(changes=[Change, …], stats)

1. Hash pass: equal normalised hash → unchanged (same address) or moved (other address). Never reaches the model.
2. Candidates: same address + BM25 top-k.
3. Score every candidate pair with the PairScorer (cross-encoder or stub).
4. Assign one-to-one, greedily by alignment score (ties → same address), above ALIGN_FLOOR.
5. Classify: reworded / material (+ moved when the address changed; equivalent + moved → "moved"); guard deltas.
6. Fallback: confidence < floor, or model says equivalent while the guard sees a delta → hypothesis fallback.
Leftovers: unmatched old → deleted, unmatched new → inserted.

`use_guard=False, use_fallback=False` gives eval arm C (the cross-encoder alone).
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from energy_lab_2.matching import guard as guard_mod
from energy_lab_2.matching.candidates import candidates
from energy_lab_2.matching.fallback import adjudicate
from energy_lab_2.matching.scorer import PairScore, PairScorer, StubScorer

ALIGN_FLOOR = 0.5
CONFIDENCE_FLOOR = 0.80
LABELS = ("unchanged", "moved", "reworded", "material", "inserted", "deleted")


@dataclass
class Change:
    label: str                          # one of LABELS
    old_address: str = ""
    new_address: str = ""
    old_text: str = ""
    new_text: str = ""
    old_hash: str = ""
    new_hash: str = ""
    moved: bool = False
    confidence: float = 1.0
    align_score: float = 1.0
    scores: dict[str, float] | None = None
    guard: list[dict[str, Any]] = field(default_factory=list)
    decided_by: str = "hash"            # hash · cross_encoder · stub · fallback:frame · fallback:guard · fallback:model_kept · alignment
    model_label: str = ""
    frames: list[dict[str, Any]] = field(default_factory=list)
    needs_review: bool = False

    @property
    def is_change(self) -> bool:
        return self.label in ("material", "inserted", "deleted")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["guard_summary"] = guard_mod.describe(self.guard) if self.guard else ""
        return d


@dataclass
class Alignment:
    changes: list[Change]
    scorer: str
    stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"scorer": self.scorer, "stats": self.stats, "changes": [c.as_dict() for c in self.changes]}


def align(old: Sequence, new: Sequence, *, scorer: PairScorer | None = None, k: int = 4,
          align_floor: float = ALIGN_FLOOR, floor: float = CONFIDENCE_FLOOR,
          use_guard: bool = True, use_fallback: bool = True) -> Alignment:
    t0 = time.time()
    sc = scorer or StubScorer()
    changes: list[Change] = []
    used_old: set[int] = set()
    used_new: set[int] = set()

    # 1 · hash pass
    new_by_addr = {n.address: j for j, n in enumerate(new)}
    new_by_hash: dict[str, list[int]] = {}
    for j, n in enumerate(new):
        new_by_hash.setdefault(n.hash, []).append(j)
    for i, o in enumerate(old):
        j = new_by_addr.get(o.address)
        if j is not None and j not in used_new and new[j].hash == o.hash:
            changes.append(_hash_change("unchanged", o, new[j]))
            used_old.add(i); used_new.add(j)
    for i, o in enumerate(old):
        if i in used_old:
            continue
        js = [j for j in new_by_hash.get(o.hash, []) if j not in used_new]
        if js:
            changes.append(_hash_change("moved", o, new[js[0]]))
            used_old.add(i); used_new.add(js[0])

    # 2 · candidates over what is left
    rest_old = [i for i in range(len(old)) if i not in used_old]
    rest_new = [j for j in range(len(new)) if j not in used_new]
    cand = candidates([old[i] for i in rest_old], [new[j] for j in rest_new], k=k)
    pairs: list[tuple[int, int]] = [(rest_old[a], rest_new[b]) for a, bs in cand.items() for b in bs]

    # 3 · score
    scored: list[PairScore] = sc.score([(old[i].body, new[j].body) for i, j in pairs]) if pairs else []
    by_pair = dict(zip(pairs, scored))

    # 4 · one-to-one assignment
    order = sorted(by_pair.items(), key=lambda kv: (-kv[1].align, old[kv[0][0]].address != new[kv[0][1]].address))
    best_align_old: dict[int, float] = {}
    for (i, _j), s in by_pair.items():
        best_align_old[i] = max(best_align_old.get(i, 0.0), s.align)
    n_fallback = 0
    for (i, j), s in order:
        if i in used_old or j in used_new or s.align < align_floor:
            continue
        used_old.add(i); used_new.add(j)
        o, n = old[i], new[j]
        moved = o.address != n.address
        model_label = s.change                         # equivalent | material
        g = guard_mod.deltas(o.body, n.body) if use_guard else []
        ch = Change(label="", old_address=o.address, new_address=n.address, old_text=o.body, new_text=n.body,
                    old_hash=o.hash, new_hash=n.hash, moved=moved, confidence=round(s.confidence, 4),
                    align_score=round(s.align, 4), scores=s.as_dict(), guard=g,
                    decided_by="cross_encoder" if sc.name.startswith("cross-encoder") else sc.name,
                    model_label=model_label)
        label = "material" if model_label == "material" else ("moved" if moved else "reworded")
        unsure = s.confidence < floor
        disagree = model_label == "equivalent" and bool(g)
        if use_fallback and (unsure or disagree):
            n_fallback += 1
            fb = adjudicate(old_addr=o.address, new_addr=n.address, before=o.body, after=n.body,
                            model_label=model_label, model_confidence=s.confidence, guard_deltas=g, floor=floor)
            ch.decided_by, ch.frames, ch.needs_review = fb["decided_by"], fb["frames"], fb["needs_review"]
            ch.confidence = fb["confidence"]
            label = "material" if fb["label"] == "material" else ("moved" if moved else "reworded")
        changes.append(_finish(ch, label))

    # leftovers
    for i in range(len(old)):
        if i not in used_old:
            o = old[i]
            changes.append(Change(label="deleted", old_address=o.address, old_text=o.body, old_hash=o.hash,
                                  confidence=round(1 - best_align_old.get(i, 0.0), 4), decided_by="alignment"))
    for j in range(len(new)):
        if j not in used_new:
            n = new[j]
            changes.append(Change(label="inserted", new_address=n.address, new_text=n.body, new_hash=n.hash,
                                  confidence=1.0, decided_by="alignment"))

    changes.sort(key=lambda c: (_order(old, c.old_address), _order(new, c.new_address)))
    counts: dict[str, int] = {}
    for c in changes:
        counts[c.label] = counts.get(c.label, 0) + 1
    stats = {"old_provisions": len(old), "new_provisions": len(new), "pairs_scored": len(pairs),
             "fallback": n_fallback, "counts": counts, "duration_ms": int((time.time() - t0) * 1000),
             "align_floor": align_floor, "confidence_floor": floor}
    return Alignment(changes=changes, scorer=sc.name, stats=stats)


def _hash_change(label: str, o, n) -> Change:
    return Change(label=label, old_address=o.address, new_address=n.address, old_text=o.body, new_text=n.body,
                  old_hash=o.hash, new_hash=n.hash, moved=o.address != n.address, decided_by="hash")


def _finish(ch: Change, label: str) -> Change:
    ch.label = label
    return ch


def _order(provs: Sequence, addr: str) -> int:
    for p in provs:
        if p.address == addr:
            return p.order
    return 10_000

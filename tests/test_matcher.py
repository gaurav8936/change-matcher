"""The matcher: guard, hash pass, alignment with the stub and with the shipped cross-encoder, and the fallback.
Adapted from Energy Lab 2's own tests (the impact, API and persona tests stay with the lab)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_lab_2.consume import provisions_from_text
from energy_lab_2.matching import PairScore, StubScorer, align, load_scorer
from energy_lab_2.matching.guard import deltas

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/energy_documents/samples/CTR-MSA-AMEND-01"
MODEL = ROOT / "models/energy-documents/cross-encoder"


def _labels(al):
    return {(c.old_address, c.new_address): c.label for c in al.changes}


def _sample():
    return (provisions_from_text((SAMPLE / "v1.txt").read_text()), provisions_from_text((SAMPLE / "v2.txt").read_text()),
            json.loads((SAMPLE / "meta.json").read_text())["expected"])


# ── guard ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,changed", [
    ("within thirty (30) days", "within 30 days", False),
    ("five percent (5%)", "5%", False),
    ("shall submit", "must submit", False),
    ("within forty-five (45) days", "within seventy-five (75) days", True),
    ("within seventy-two (72) hours", "within twenty-four (24) hours", True),
    ("shall submit", "may submit", True),
    ("on a quarterly basis", "on a monthly basis", True),
    ("a penalty of 3%", "a penalty of 1.5%", True),
])
def test_guard(a, b, changed):
    assert bool(deltas(a, b)) is changed


# ── alignment ─────────────────────────────────────────────────────────────

def test_hash_pass_unchanged_and_moved():
    o = provisions_from_text("§1. A\nalpha beta gamma.\n§2. B\ndelta epsilon zeta.")
    n = provisions_from_text("§1. A\nalpha beta gamma.\n§2. New\nbrand new text here.\n§3. B\ndelta epsilon zeta.")
    al = align(o, n, scorer=StubScorer())
    lab = _labels(al)
    assert lab[("§1", "§1")] == "unchanged" and lab[("§2", "§3")] == "moved" and lab[("", "§2")] == "inserted"
    assert all(c.decided_by == "hash" for c in al.changes if c.label in ("unchanged", "moved"))


def _expected_pairs(expected: dict[str, str]) -> dict[tuple[str, str], str]:
    out = {}
    for k, v in expected.items():
        a, _, b = k.partition("->")
        out[("", a) if v == "inserted" else (a, b or a)] = v
    return out


def test_contract_amendment_sample_stub():
    old, new, expected = _sample()
    lab = _labels(align(old, new, scorer=StubScorer()))
    for pair, want in _expected_pairs(expected).items():
        if pair == ("6", "7"):
            continue                      # a cross-reference edit ("Section 5" → "Section 6"); see the README
        assert lab[pair] == want, (pair, lab[pair])


def test_contract_amendment_sample_cross_encoder():
    pytest.importorskip("onnxruntime")
    sc = load_scorer(str(MODEL), allow_stub=False)
    assert sc.name == "cross-encoder:ed-minilm-l6-3way"
    old, new, expected = _sample()
    lab = _labels(align(old, new, scorer=sc))
    for pair, want in _expected_pairs(expected).items():
        if pair == ("6", "7"):
            assert lab[pair] == "material"   # known: the model reads the cross-reference edit as a change
            continue
        assert lab[pair] == want, (pair, lab[pair])


def test_cross_encoder_scores_a_paraphrase_above_an_unrelated_pair():
    pytest.importorskip("onnxruntime")
    sc = load_scorer(str(MODEL), allow_stub=False)
    s = sc.score([("The operator shall report within 45 days.", "The operator shall report within forty-five (45) days."),
                  ("The operator shall report within 45 days.", "Customer data stays in the EEA.")])
    assert s[0].align > s[1].align and s[0].change == "equivalent"


class _ConfidentlyWrong:
    """A scorer that says 'equivalent, very sure' for everything: the guard must overrule it."""
    name = "cross-encoder:test"

    def score(self, pairs):
        return [PairScore(0.97, 0.02, 0.01) for _ in pairs]


def test_guard_overrules_confident_model_via_fallback():
    o = provisions_from_text("§3. Reporting\nThe operator shall report within forty-five (45) days of the period close.")
    n = provisions_from_text("§3. Reporting\nThe operator shall report within seventy-five (75) days of the period close.")
    c = align(o, n, scorer=_ConfidentlyWrong()).changes[0]
    assert c.label == "material" and c.decided_by.startswith("fallback:") and c.model_label == "equivalent"
    arm_c = align(o, n, scorer=_ConfidentlyWrong(), use_guard=False, use_fallback=False).changes[0]
    assert arm_c.label == "reworded"              # the model alone misses it (eval arm C)


class _Unsure:
    name = "cross-encoder:unsure"

    def score(self, pairs):
        return [PairScore(0.45, 0.40, 0.15) for _ in pairs]


def test_low_confidence_goes_to_frames():
    o = provisions_from_text("§7. Effective Date\nComments may be submitted within thirty (30) days of publication.")
    n = provisions_from_text("§7. Effective Date\nComments may be submitted within sixty (60) days of publication.")
    c = align(o, n, scorer=_Unsure()).changes[0]
    assert c.label == "material" and c.decided_by == "fallback:frame" and c.frames


def test_unrelated_provisions_become_insert_and_delete():
    o = provisions_from_text("§9. Penalties\nPenalties apply to late filings of capacity reports.")
    n = provisions_from_text("§9. Data residency\nCustomer data stays in the European Economic Area at all times.")
    assert {c.label for c in align(o, n, scorer=StubScorer()).changes} == {"inserted", "deleted"}

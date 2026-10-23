# change-matcher

**Is this a real change, or the same clause reworded?** Compare two versions of a contract or a regulation with a text
diff and every edit lights up: "shall" became "must", "thirty (30) days" became "30 days", a typo was fixed, a section
was renumbered. A reviewer has to read them all to find the few that change what someone owes. This library aligns the
two versions provision by provision and labels each pair: unchanged, moved, reworded, material, inserted or deleted.

It is the matcher inside Energy Lab 2, a Hypothesys lab that takes a new version of a contract or regulatory filing,
works out what changed and which obligations that touches, and sends the result for review. This repository is a
**read-only copy** of the matcher and the model it serves, published unchanged. See `PUBLISHED_FROM` for the exact
commits.

## How it works

1. **Split into provisions.** `§` sections, numbered clauses (`7.2`, `Section 7`) or paragraphs, each with a hash of
   its normalised text.
2. **Hash pass.** Same hash at the same address → `unchanged`; same hash elsewhere → `moved`. These never reach the
   model.
3. **Candidates.** For every remaining old provision: the new provision at the same address, plus the BM25 top 4. The
   model scores about n·5 pairs instead of n².
4. **Cross-encoder.** A cross-encoder reads both passages *together* (not as two separate embeddings) and outputs
   three probabilities: equivalent, material, unrelated. This one is `cross-encoder/ms-marco-MiniLM-L-6-v2`
   fine-tuned on 10,720 labelled version pairs, exported to ONNX and quantised to int8: 23 MB, about 200 ms for a
   contract on a laptop CPU, no GPU and no PyTorch.
5. **One-to-one assignment.** Pairs are taken greedily by alignment score (ties go to the same address), above a
   floor of 0.5. Old provisions left over are `deleted`; new ones are `inserted`.
6. **The guard.** Small transformers read magnitudes badly and can call "45 days" → "75 days" equivalent. The guard
   pulls out what an obligation turns on — durations, percentages, money, cadence, duty versus permission ("shall"
   vs "may"), negation — in canonical form, so "thirty (30) days" and "30 days" are the same value. It can only
   escalate a pair, never silence one.
7. **Fallback.** When the model is unsure (confidence under 0.8), or says "equivalent" while the guard sees a changed
   value, the pair goes to rule-based regulatory-change frames (period extended, date deferred, definition narrowed,
   cadence changed, penalty threshold). The result records who decided: `hash`, `cross_encoder`, `fallback:frame`,
   `fallback:guard` or `fallback:model_kept`.

## Results

The same 120 held-out version pairs (110 energy-regulation, 10 from CUAD contracts) through five arms. Positive =
material change, insertion or deletion; cosmetic edits are rewordings that leave the obligation unchanged.

| Arm | Precision | Recall | F1 | Cosmetic edits flagged |
|---|---|---|---|---|
| Hash only | 0.31 | 0.94 | 0.47 | 100% |
| Section diff (rule-based) | 0.28 | 0.74 | 0.40 | 95% |
| Cross-encoder alone | 0.90 | 0.82 | 0.86 | 2.7% |
| **Cross-encoder + guard + fallback** (default) | 0.87 | 0.90 | 0.88 | 4.9% |
| Stub scorer + guard + fallback | 0.91 | 0.89 | 0.90 | 2.7% |

The guard and fallback buy recall (0.82 → 0.90) for a little precision. The stub scorer (token overlap and sequence
similarity, no model) does about as well on this set, because the guard and frames do most of the work on the
energy-regulation pairs, which come from the same synthetic family. On the 10 CUAD contracts the cross-encoder with
guard and fallback reaches F1 0.98 against the stub's 0.97; the full report is in `reports/`.

## Example

```python
from pathlib import Path
from energy_lab_2.consume import provisions_from_text
from energy_lab_2.matching import align, load_scorer

d = Path("data/energy_documents/samples/CTR-MSA-AMEND-01")
old, new = (provisions_from_text((d / f).read_text()) for f in ("v1.txt", "v2.txt"))
result = align(old, new, scorer=load_scorer())
for c in result.changes:
    guard = ", ".join(f"{g['kind']} {g['before']}→{g['after']}" for g in c.guard)
    print(f"{c.label:9} {c.old_address or '-':>2} → {c.new_address or '-':2} {c.decided_by:13} {guard}".rstrip())
```

prints

```text
unchanged  1 → 1  hash
reworded   2 → 2  cross_encoder
material   3 → 3  cross_encoder duration ['P30D']→['P45D']
material   4 → 4  cross_encoder cadence ['quarterly']→['monthly']
material   5 → 6  cross_encoder duration ['PT72H']→['PT24H']
material   6 → 7  cross_encoder
moved      7 → 8  hash
inserted   - → 5  alignment
```

The sample is a generated SaaS agreement and its first amendment: a numeral restyle in §2 (reworded, correctly
ignored), a payment term from 30 to 45 days, a reporting cadence from quarterly to monthly, a new clause 5 that
renumbers everything after it, and a breach-notice window cut from 72 to 24 hours inside the renumbered clause.

**One known miss:** clause 6 → 7 only changes a cross-reference ("Section 5" → "Section 6", following the renumber).
The answer key in `meta.json` calls it moved; the model calls it material. The eval scores cross-reference-only edits
separately for this reason.

## What's in it

| Path | What it is |
|---|---|
| `src/energy_lab_2/matching/` | `candidates` (BM25), `scorer` (ONNX cross-encoder and the stub), `guard`, `align` (steps 2–6), `fallback`. |
| `src/energy_lab_2/consume.py` | Provisions from text (`provisions_from_text`). Its `consume` / `text_intake` call the private engine's intake gate and do not work in this copy. |
| `src/energy_lab_2/_el1.py`, `src/energy_lab_1/` | The pieces of Energy Lab 1 the matcher uses: the `§` parser and the five frames ([docket-diff](https://github.com/gaurav8936/docket-diff)). |
| `src/labs_core/` | From the private lab engine: value parsing for the guard (`provenance.values`, see [citation-verifier](https://github.com/gaurav8936/citation-verifier)) and the frame mechanism (`reasoning.frames`). |
| `models/energy-documents/cross-encoder/` | The serving model: `model.int8.onnx`, `tokenizer.json`, `labels.json`, `MANIFEST.json` (which also lists the fp32 export, not included). |
| `data/energy_documents/samples/` | The sample contract pair and its answer key. |
| `reports/` | The five-arm eval report. |
| `tests/` | Guard cases, hash pass, the sample with the stub and with the model, the guard overruling a confidently wrong model, the frame fallback. |

Not included, and staying private: the generator that makes the labelled version pairs, the training pairs and the
eval set, the obligation-impact stage, tenant registers, the API and the pipeline around it.

## Install and test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

Python 3.11+. Without the model files (or without onnxruntime) `load_scorer()` falls back to the stub scorer.

## License

Code: MIT, see [LICENSE](LICENSE). The model is a fine-tune of `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0),
trained partly on CUAD (CC BY 4.0, The Atticus Project); see [NOTICE](NOTICE).

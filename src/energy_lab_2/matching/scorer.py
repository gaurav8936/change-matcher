"""Pair scorers: (old provision, new provision) → p(equivalent), p(material), p(unrelated)  (WS-ED-MATCH-01).

  StubScorer              lexical, deterministic, no model: tests and engines without the model installed.
  OnnxCrossEncoderScorer  the fine-tuned MiniLM cross-encoder exported to ONNX (onnxruntime + tokenizers; no torch).

`load_scorer()` picks the ONNX model when `ENERGY_DOCUMENTS_MODEL_DIR` (default /app/models/energy-documents/
cross-encoder; on a host checkout without /app, the repo's models/energy-documents/cross-encoder) holds
`model[.int8].onnx` + `tokenizer.json` + `labels.json` and onnxruntime imports; otherwise the stub.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

LABELS = ("equivalent", "material", "unrelated")
DEFAULT_MODEL_DIR = "/app/models/energy-documents/cross-encoder"
_REPO_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "energy-documents" / "cross-encoder"


def default_model_dir() -> str:
    """ENERGY_DOCUMENTS_MODEL_DIR, else the image path, else the repo checkout's copy (host runs)."""
    env = os.environ.get("ENERGY_DOCUMENTS_MODEL_DIR", "").strip()
    if env:
        return env
    if Path(DEFAULT_MODEL_DIR).exists() or not _REPO_MODEL_DIR.exists():
        return DEFAULT_MODEL_DIR
    return str(_REPO_MODEL_DIR)


@dataclass(frozen=True)
class PairScore:
    equivalent: float
    material: float
    unrelated: float

    @property
    def align(self) -> float:
        """How likely the two texts are the same provision."""
        return 1.0 - self.unrelated

    @property
    def change(self) -> str:
        return "material" if self.material > self.equivalent else "equivalent"

    @property
    def confidence(self) -> float:
        """Probability of the winning change class, renormalised over the same-provision classes."""
        s = self.equivalent + self.material
        return (max(self.equivalent, self.material) / s) if s > 0 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"equivalent": round(self.equivalent, 4), "material": round(self.material, 4),
                "unrelated": round(self.unrelated, 4)}


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[PairScore]: ...


_WORD = re.compile(r"[a-z0-9§%$]+")


def _toks(s: str) -> list[str]:
    return _WORD.findall((s or "").lower())


class StubScorer:
    """Lexical stand-in for the cross-encoder. Alignment from token overlap; change from character similarity and
    the guard. Good enough for tests and for the pipeline shape, not for the eval (it is arm B-like)."""
    name = "stub-lexical"

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[PairScore]:
        from energy_lab_2.matching.guard import deltas
        out = []
        for a, b in pairs:
            ta, tb = set(_toks(a)), set(_toks(b))
            jac = len(ta & tb) / max(1, len(ta | tb))
            sim = difflib.SequenceMatcher(None, " ".join(_toks(a)), " ".join(_toks(b))).ratio()
            same = max(0.0, min(1.0, (jac - 0.2) / 0.4))          # jaccard 0.2 → 0, 0.6 → 1
            mat = 0.85 if deltas(a, b) else (0.6 if sim < 0.85 else 0.1)
            out.append(PairScore(equivalent=same * (1 - mat), material=same * mat, unrelated=1 - same))
        return out


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


class OnnxCrossEncoderScorer:
    """Fine-tuned cross-encoder served with onnxruntime. Batches pairs; max length from labels.json (default 256)."""

    def __init__(self, model_dir: str | Path, *, batch_size: int = 32, threads: int = 2) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        meta = json.loads((d / "labels.json").read_text())
        self.labels: list[str] = list(meta["labels"])
        self.max_length = int(meta.get("max_length", 256))
        self.name = f"cross-encoder:{meta.get('name', d.name)}"
        self.meta = meta
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.max_length, strategy="longest_first")
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        onnx = d / ("model.int8.onnx" if (d / "model.int8.onnx").exists() else "model.onnx")
        self.sess = ort.InferenceSession(str(onnx), sess_options=so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.batch_size = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[PairScore]:
        import numpy as np

        out: list[PairScore] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = list(pairs[i:i + self.batch_size])
            enc = self.tok.encode_batch([(a or " ", b or " ") for a, b in chunk])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            for row in logits:
                p = dict(zip(self.labels, _softmax([float(x) for x in row])))
                out.append(PairScore(p.get("equivalent", 0.0), p.get("material", 0.0), p.get("unrelated", 0.0)))
        return out


_LOADED: dict[str, PairScorer] = {}


def load_scorer(model_dir: str | None = None, *, allow_stub: bool = True) -> PairScorer:
    d = model_dir or default_model_dir()
    if d in _LOADED:
        return _LOADED[d]
    try:
        if not (Path(d) / "labels.json").exists():
            raise FileNotFoundError(d)
        sc: PairScorer = OnnxCrossEncoderScorer(d)
    except (ImportError, FileNotFoundError, OSError, ValueError, KeyError):
        if not allow_stub:
            raise
        sc = StubScorer()
    _LOADED[d] = sc
    return sc

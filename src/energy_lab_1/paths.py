"""Where Energy Lab 1 finds its data.

In the image everything lives under `/app` (the labs_core WORKDIR): `/app/data/{dockets,context,queues,evals}`
baked in, the synthetic corpus batches at `/app/corpus-batch-0N`. In a source checkout (tests, notebooks) the
same tree sits at the repo root. `app_root()` picks whichever exists; `ENERGY_LAB_1_APP_ROOT` forces one.

Lab-specific environment (labs_core leaves these to the lab; `HYPOTHESYS_*` spellings still read as fallbacks
so today's values files and harnesses keep working):

  LABS_CORPUS_ROOTS   comma-separated corpus batch dirs, scanned in order  (was HYPOTHESYS_CORPUS_ROOTS)
  LABS_CORPUS_ROOT    single-batch fallback / primary batch                (was HYPOTHESYS_CORPUS_ROOT)
  LABS_QUEUE_ROOT     reviewer queues (labs_core reads the same variable)   (was HYPOTHESYS_QUEUE_ROOT)
  LABS_DATA_<KEY>     override one data dir (labs_core convention): DOCKETS, CONTEXT, QUEUES, EVALS
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   # src/energy_lab_1/paths.py → repo root (source checkout)

DATA_DIRS: dict[str, str] = {
    "dockets": "data/dockets",       # the three hand-built canonical demo dockets (manifest.json + v1/v2)
    "context": "data/context",       # persona (tenant) contexts: context/personas/*.json
    "queues": "data/queues",         # seed reviewer queues (runtime queues go to LABS_QUEUE_ROOT)
    "evals": "data/evals",           # labelled obligation set
    "eval_reports": "data/eval-reports",
}


def env(name: str, default: str = "") -> str:
    """`LABS_<name>`, else the legacy `HYPOTHESYS_<name>`, else `default`."""
    return os.environ.get(f"LABS_{name}") or os.environ.get(f"HYPOTHESYS_{name}") or default


def app_root() -> Path:
    forced = os.environ.get("ENERGY_LAB_1_APP_ROOT")
    if forced:
        return Path(forced)
    if Path("/app/data/context").exists():
        return Path("/app")
    if (_REPO_ROOT / "data" / "context").exists():
        return _REPO_ROOT
    return Path("/app")


def data_dir(key: str) -> Path:
    """Absolute path of a named data dir (`LABS_DATA_<KEY>` overrides it — same rule as `labs_core.Lab.data_dir`)."""
    override = os.environ.get(f"LABS_DATA_{key.upper()}")
    if override:
        return Path(override)
    return app_root() / DATA_DIRS[key]


def corpus_root() -> Path:
    return Path(env("CORPUS_ROOT", "/app/data/corpus"))


def corpus_roots() -> list[str]:
    """Corpus batch roots in scan order (`LABS_CORPUS_ROOTS`, falling back to `[LABS_CORPUS_ROOT]`)."""
    spec = env("CORPUS_ROOTS").strip()
    if spec:
        return [p.strip() for p in spec.split(",") if p.strip()]
    return [str(corpus_root())]


def queue_root() -> Path:
    return Path(env("QUEUE_ROOT", "/app/data/queues"))

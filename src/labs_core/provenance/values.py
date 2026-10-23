"""Value parsing shared by the verifier, validators and evals (WS-LC-CORE-01 · K3).

Turns the many ways a document writes a value into one canonical form, so "January 5, 2026", "05/01/2026"
(US order) and "2026-01-05" compare equal, "$12,480.00" equals "12480.00", and "twelve (12) months" equals
"P12M" (and "P1Y"). Pure Python, no dependencies.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

MONTHS = {
    m: i + 1
    for i, names in enumerate([
        ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
        ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
        ("november", "nov"), ("december", "dec"),
    ])
    for m in names
}
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"

_DATE_PATTERNS = [
    # 2026-01-05
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    # January 5, 2026 · Jan. 5 2026 · January 5th, 2026
    (re.compile(_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I), "mdy_name"),
    # 5 January 2026 · 5th day of January, 2026 · 5th of January 2026
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+)?(?:of\s+)?" + _MON + r",?\s+(\d{4})", re.I), "dmy_name"),
    # 01/05/2026 (US month first) · 1-5-2026
    (re.compile(r"\b(\d{1,2})[/](\d{1,2})[/](\d{4})\b"), "mdy_num"),
    # 05.01.2026 (European day first)
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), "dmy_num"),
]


def _mk(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str, dayfirst: bool | None = None) -> list[str]:
    """Every date in `text`, as ISO strings, in order of appearance (duplicates kept once).

    Numeric slash dates are ambiguous (03/12/2025): `dayfirst=None` returns both readings (presence checks),
    True/False picks one (parsing a document whose convention is known)."""
    found: list[tuple[int, str]] = []
    for rx, kind in _DATE_PATTERNS:
        for m in rx.finditer(text):
            g = m.groups()
            iso = None
            if kind == "ymd":
                iso = _mk(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "mdy_name":
                iso = _mk(int(g[2]), MONTHS[g[0].lower().rstrip(".")], int(g[1]))
            elif kind == "dmy_name":
                iso = _mk(int(g[2]), MONTHS[g[1].lower().rstrip(".")], int(g[0]))
            elif kind == "mdy_num":
                us, eu = _mk(int(g[2]), int(g[0]), int(g[1])), _mk(int(g[2]), int(g[1]), int(g[0]))
                if dayfirst is True:
                    iso = eu or us
                elif dayfirst is False:
                    iso = us or eu
                else:
                    iso = us or eu
                    if us and eu and us != eu:
                        found.append((m.start() + 0.5, eu))   # both readings are possible
            elif kind == "dmy_num":
                iso = _mk(int(g[2]), int(g[1]), int(g[0]))
            if iso:
                found.append((m.start(), iso))
    out: list[str] = []
    for _, iso in sorted(found):
        if iso not in out:
            out.append(iso)
    return out


def slash_dayfirst(text: str) -> bool | None:
    """The document's numeric date convention: True if any n/n/yyyy date has a first part > 12, False if any
    has a second part > 12, None if every slash date is ambiguous."""
    for a, b in re.findall(r"\b(\d{1,2})/(\d{1,2})/\d{4}\b", text):
        if int(a) > 12:
            return True
        if int(b) > 12:
            return False
    return None


def parse_date(value: object, dayfirst: bool | None = None) -> str | None:
    """Canonical ISO date for one value, or None (ambiguous slash dates read month-first unless dayfirst)."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    ds = find_dates(str(value), dayfirst=bool(dayfirst))
    return ds[0] if ds else None


_MONEY_RX = re.compile(
    r"(?<![\w.])(?:[$€£¥₹]|usd|eur|gbp|inr|jpy|cad|us\$)?\s?"
    r"(\d{1,3}(?:\.\d{3})+,\d{2}(?!\d)|\d+,\d{2}(?![\d,])|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?:usd|eur|gbp|inr|jpy|cad|[€]))?",
    re.I,
)


def _to_decimal(raw: str) -> Decimal | None:
    s = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s) or re.fullmatch(r"\d+,\d{2}", s):   # European 12.480,00 · 258,96
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def parse_money(value: object) -> Decimal | None:
    """Canonical Decimal for a money value ("$12,480.00", "12480", 12480.0); None if not money-like."""
    if value is None:
        return None
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        return Decimal(str(value))
    s = str(value).strip()
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
    m = _MONEY_RX.search(s)
    if not m:
        return None
    d = _to_decimal(m.group(1))
    if d is None:
        return None
    return -d if neg else d


def find_money(text: str) -> list[Decimal]:
    out: list[Decimal] = []
    for m in _MONEY_RX.finditer(text):
        d = _to_decimal(m.group(1))
        if d is not None:
            out.append(d)
    return out


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "a": 1, "an": 1,
    "twenty-four": 24, "thirty-six": 36, "forty-eight": 48, "one hundred twenty": 120, "one hundred eighty": 180,
}
_UNIT = {"day": "D", "days": "D", "week": "W", "weeks": "W", "month": "M", "months": "M", "year": "Y", "years": "Y"}
_DUR_RX = re.compile(
    r"(?:(?P<word>[a-z]+(?:[- ][a-z]+)?)\s*)?(?:\(\s*(?P<paren>\d+)\s*\)|(?P<num>\d+))?\s*[- ]?"
    r"(?P<unit>days?|weeks?|months?|years?)\b",
    re.I,
)
_ISO_DUR = re.compile(r"^P(\d+)([DWMY])$", re.I)


def _dur_key(n: int, unit: str) -> tuple[str, int]:
    """Comparable key: years/months → months, weeks/days → days."""
    unit = unit.upper()
    if unit == "Y":
        return ("M", n * 12)
    if unit == "W":
        return ("D", n * 7)
    return (unit, n)


def find_durations(text: str) -> list[str]:
    """Every duration in `text` as ISO 8601 (P12M, P90D, P1Y), in order."""
    out: list[str] = []
    for m in _DUR_RX.finditer(text):
        n: int | None = None
        if m.group("paren"):
            n = int(m.group("paren"))
        elif m.group("num"):
            n = int(m.group("num"))
        elif m.group("word"):
            w = m.group("word").lower()
            n = _NUM_WORDS.get(w) or _NUM_WORDS.get(w.split()[-1]) or _NUM_WORDS.get(w.split("-")[-1])
        if not n:
            continue
        iso = f"P{n}{_UNIT[m.group('unit').lower()]}"
        if iso not in out:
            out.append(iso)
    return out


def parse_duration(value: object) -> tuple[str, int] | None:
    """Comparable duration key for a value ("P12M", "P1Y", "twelve (12) months") or None."""
    if value is None:
        return None
    s = str(value).strip()
    m = _ISO_DUR.match(s)
    if m:
        return _dur_key(int(m.group(1)), m.group(2))
    ds = find_durations(s)
    if not ds:
        return None
    m = _ISO_DUR.match(ds[0])
    return _dur_key(int(m.group(1)), m.group(2)) if m else None


def parse_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if s in ("no", "none", "zero", "nil"):
        return 0
    return _NUM_WORDS.get(s)


def canonical(kind: str, value: object) -> object:
    """One comparable form per field kind; used by evals (equality) and the verifier (presence)."""
    if value is None:
        return None
    if kind == "date":
        return parse_date(value)
    if kind == "money":
        d = parse_money(value)
        return d.quantize(Decimal("0.01")) if d is not None else None
    if kind == "duration":
        return parse_duration(value)
    if kind == "int":
        return parse_int(value)
    if kind == "list":
        if isinstance(value, list | tuple):
            items = []
            for v in value:
                if isinstance(v, dict):
                    items.append(tuple(sorted((k, _text_key(str(x))) for k, x in v.items() if x is not None)))
                else:
                    items.append(_text_key(str(v)))
            return tuple(sorted(items, key=str))
        return (_text_key(str(value)),)
    if kind == "enum":
        return _text_key(str(value))
    return _text_key(str(value))


def _text_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[‘’“”\"'`]", "", s)
    s = re.sub(r"[^\w\s&./-]", " ", s)
    s = re.sub(r"\b(the|inc|llc|ltd|corp|corporation|co|company|plc|gmbh|limited)\b\.?", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def equal(kind: str, a: object, b: object) -> bool:
    """Do two values mean the same thing for a field of this kind?"""
    ca, cb = canonical(kind, a), canonical(kind, b)
    if ca is None or cb is None:
        return ca is None and cb is None
    if kind in ("text",) and isinstance(ca, str) and isinstance(cb, str):
        return ca == cb or (len(ca) > 12 and len(cb) > 12 and (ca in cb or cb in ca))
    return ca == cb

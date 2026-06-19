from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SrtEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIME_RE = re.compile(
    r"^\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
    r"(?:\s+.*)?$"
)


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    hh = int(h)
    mm = int(m)
    ss = int(s)
    mss = int(ms.ljust(3, "0"))
    return ((hh * 60 + mm) * 60 + ss) * 1000 + mss


def parse_srt_text(srt_text: str) -> List[SrtEntry]:
    """
    Parse .srt content into entries.

    Supports common SRT variants:
    - Milliseconds separated by ',' or '.'
    - Optional styling after the time range line
    """
    text = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    blocks = re.split(r"\n\s*\n", text)
    entries: List[SrtEntry] = []

    for block in blocks:
        lines = [ln.strip("\ufeff") for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue

        idx: Optional[int] = None
        time_line = lines[0]

        # Typical block: index line then time line
        if re.fullmatch(r"\d+", lines[0]) and len(lines) >= 2:
            idx = int(lines[0])
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            text_lines = lines[1:]

        m = _TIME_RE.match(time_line)
        if not m:
            continue

        start_ms = _to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end_ms = _to_ms(m.group(5), m.group(6), m.group(7), m.group(8))
        raw_text = " ".join(t.strip() for t in text_lines if t.strip())
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        if not raw_text:
            continue

        if idx is None:
            idx = len(entries) + 1

        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms

        entries.append(SrtEntry(index=idx, start_ms=start_ms, end_ms=end_ms, text=raw_text))

    entries.sort(key=lambda e: (e.start_ms, e.end_ms, e.index))
    return entries


def parse_srt_file(path: str | Path) -> List[SrtEntry]:
    p = Path(path)
    data = p.read_text(encoding="utf-8", errors="replace")
    return parse_srt_text(data)


def iter_text_chunks(entries: Iterable[SrtEntry]) -> Iterable[str]:
    for e in entries:
        if e.text:
            yield e.text

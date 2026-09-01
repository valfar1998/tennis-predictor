"""Log di avanzamento con percentuali per operazioni multi-step."""

from __future__ import annotations

import sys


def _safe_text(msg: str) -> str:
    text = str(msg or "")
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def pct(step: int, total: int) -> int:
    if total <= 0:
        return 100
    return int(round(100.0 * step / total))


def step_prefix(step: int, total: int) -> str:
    return f"[{pct(step, total):3d}%] {step}/{total}"


def format_step(step: int, total: int, msg: str) -> str:
    return f"{step_prefix(step, total)} — {msg}"


def log_step(step: int, total: int, msg: str) -> None:
    print(_safe_text(format_step(step, total, msg)), flush=True)


def log_item(current: int, total: int, msg: str, *, indent: bool = True) -> None:
    prefix = "  " if indent else ""
    print(_safe_text(f"{prefix}[{pct(current, total):3d}%] {current}/{total} — {msg}"), flush=True)


def log_done(msg: str) -> None:
    print(_safe_text(f"[100%] — {msg}"), flush=True)


class OpProgress:
    """Tracker step sequenziali (es. pipeline 1/5 … 5/5)."""

    def __init__(self, total: int, *, label: str = "") -> None:
        self.total = max(1, int(total))
        self.step = 0
        self.label = label.strip()

    def next(self, msg: str) -> int:
        self.step += 1
        line = format_step(self.step, self.total, msg)
        if self.label:
            line = f"{self.label} {line}"
        print(_safe_text(line), flush=True)
        return self.step

    def item(self, current: int, item_total: int, msg: str) -> None:
        line = f"[{pct(current, item_total):3d}%] {current}/{item_total} — {msg}"
        if self.label:
            line = f"{self.label}   {line}"
        else:
            line = f"  {line}"
        print(_safe_text(line), flush=True)

    def milestone(self, current: int, item_total: int, msg: str, *, every: int = 1) -> None:
        """Log solo a intervalli (evita spam su loop lunghi)."""
        if item_total <= 0:
            return
        every = max(1, every)
        if current == 1 or current == item_total or current % every == 0:
            self.item(current, item_total, msg)

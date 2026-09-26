"""Mise en forme Markdown commune aux rapports d'audit (CRF et NER)."""

import math
from collections.abc import Sequence


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}"


def fmt_ci(point: float, ci: Sequence[float], digits: int = 3) -> str:
    return f"{fmt(point, digits)} [{fmt(ci[0], digits)} ; {fmt(ci[1], digits)}]"


def fmt_delta(point: float, ci: Sequence[float], digits: int = 3) -> str:
    marker = " ▲" if ci[0] > 0 else " ▼" if ci[1] < 0 else ""
    return f"{point:+.{digits}f} [{ci[0]:+.{digits}f} ; {ci[1]:+.{digits}f}]{marker}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    marks = {"l": ":--", "r": "--:", "c": ":-:"}
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(marks[a] for a in align) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_code(text: str, limit: int = 90) -> str:
    text = text.replace("`", "'")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"`{text}`" if text else "∅"

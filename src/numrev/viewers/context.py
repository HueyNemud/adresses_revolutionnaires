"""Vue « Documents » du viewer d'alignement : les deux annuaires côte à côte,
chacun dans **son** ordre, avec toutes leurs lignes (titres, entrées, hors
sujet), les entrées appariées reliées par un trait dans une gouttière,
comme un outil de diff. Sert de contexte sous la carte de relecture et de
vue à part entière.

Ce module construit la fenêtre affichée (fonctions pures, testées) et monte
le composant `st.components.v2` qui la dessine (`assets/context.js|css`).
Un clic sur une entrée la sélectionne (déclencheur `pick`), pour apparier
deux entrées à la main.
"""

import html
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from numrev.alignment.records import DocLine
from numrev.ner.html import SPAN_CSS, render_tagged_html
from numrev.titles import title_text

ASSETS = Path(__file__).with_name("assets")
SIDES = ("left", "right")


@dataclass(frozen=True)
class EntryState:
    """État d'une entrée dans l'alignement courant."""

    kind: str  # "pair", "candidate" ou "alone"
    partner: str = ""  # uuid de l'autre côté (paire ou candidate)
    manual: bool = False  # décidée par le patch (versionné ou journal)
    local: bool = False  # décision du journal, non enregistrée


@dataclass
class Documents:
    """Les deux annuaires, ligne à ligne, et l'état de leurs entrées."""

    lines: dict[str, list[DocLine]]  # côté → lignes dans l'ordre du volume
    positions: dict[str, dict[str, int]]  # côté → uuid → rang dans `lines`
    states: dict[str, dict[str, EntryState]]  # côté → uuid → état (entrées alignables seulement)


def documents(lines: dict[str, list[DocLine]], states: dict[str, dict[str, EntryState]]) -> Documents:
    positions = {side: {line.uuid: index for index, line in enumerate(lines[side])} for side in SIDES}
    return Documents(lines, positions, states)


def other(side: str) -> str:
    return "right" if side == "left" else "left"


def centers(docs: Documents, left_uuid: str, right_uuid: str) -> dict[str, int]:
    """Ligne centrale de chaque côté. Un côté sans entrée est ancré sur le
    partenaire de la paire la plus proche de l'entrée présente (d'abord en
    remontant, puis en descendant), à défaut au début du document."""
    found = {side: docs.positions[side].get(uuid) for side, uuid in (("left", left_uuid), ("right", right_uuid))}
    for side in SIDES:
        if found[side] is not None or found[other(side)] is None:
            continue
        here, start = other(side), found[other(side)]
        lines = docs.lines[here]
        order = list(range(start, -1, -1)) + list(range(start + 1, len(lines)))
        for index in order:
            state = docs.states[here].get(lines[index].uuid)
            if state is not None and state.kind == "pair" and state.partner in docs.positions[side]:
                found[side] = docs.positions[side][state.partner]
                break
    return {side: found[side] if found[side] is not None else 0 for side in SIDES}


def window(docs: Documents, center: dict[str, int], before: int, after: int) -> dict[str, tuple[int, int]]:
    """Bornes [début ; fin[ de la fenêtre de chaque côté."""
    return {side: (max(0, center[side] - before), min(len(docs.lines[side]), center[side] + after + 1)) for side in SIDES}


def line_payload(line: DocLine, state: EntryState | None, direction: int) -> dict:
    """Une ligne pour le composant. `direction` : -1 / 1 si le partenaire est
    au-dessus / au-dessous de la fenêtre de l'autre côté, 0 sinon."""
    if line.entity == "TITLE":
        kind, content = "title", html.escape(title_text(line.markdown) or line.markdown)
    elif line.entity == "ENTRY":
        kind, content = "entry", render_tagged_html(line.tagged_text) or html.escape(line.markdown)
    else:
        kind, content = "out", html.escape(line.markdown)
    payload = {"u": line.uuid, "t": kind, "h": content, "p": line.page, "l": line.level or 0}
    if state is not None:
        payload |= {"s": state.kind, "x": state.partner, "m": state.manual, "d": state.local, "o": direction}
    return payload


def payload(docs: Documents, bounds: dict[str, tuple[int, int]], focus: tuple[str, str], picks: dict[str, str], height: int) -> dict:
    """Données du composant : lignes des deux fenêtres, ligne courante,
    sélection."""
    sides = {}
    for side in SIDES:
        start, end = bounds[side]
        other_start, other_end = bounds[other(side)]
        rows = []
        for line in docs.lines[side][start:end]:
            state = docs.states[side].get(line.uuid)
            direction = 0
            if state is not None and state.partner:
                position = docs.positions[other(side)].get(state.partner)
                if position is not None and not other_start <= position < other_end:
                    direction = -1 if position < other_start else 1
            rows.append(line_payload(line, state, direction))
        sides[side] = rows
    return {
        "left": sides["left"],
        "right": sides["right"],
        "focus": list(focus),
        "picks": picks,
        "height": height,
        "more": {side: [bounds[side][0] > 0, bounds[side][1] < len(docs.lines[side])] for side in SIDES},
    }


_component = None


def _diff_component():
    global _component
    if _component is None:
        css = (ASSETS / "context.css").read_text(encoding="utf-8") + SPAN_CSS
        _component = st.components.v2.component("numrev_documents_diff", js=(ASSETS / "context.js").read_text(encoding="utf-8"), css=css)
    return _component


def documents_diff(data: dict, key: str, titles: tuple[str, str]) -> dict | None:
    """Monte le composant ; renvoie l'entrée cliquée ({side, uuid}) au tour
    qui suit le clic, sinon None."""
    result = _diff_component()(key=key, data=data | {"titles": list(titles)}, on_pick_change=lambda: None)
    return result.get("pick")

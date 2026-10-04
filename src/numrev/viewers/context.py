"""Vue « Documents » du viewer d'alignement : les deux annuaires côte à côte,
chacun dans **son** ordre, avec toutes leurs lignes (titres, entrées, hors
sujet), les entrées appariées reliées par un trait dans une gouttière,
comme un outil de diff. Sert de contexte sous la carte de relecture et de
vue à part entière.

Ce module construit la fenêtre affichée (fonctions pures, testées) et monte
le composant `st.components.v2` qui la dessine (`assets/context.js|css`).
Un clic sur une entrée ou un lien en fait la ligne courante (`focus`) ; la
loupe du lien courant ouvre la relecture détaillée (`zoom`) ; en mode
« choisir le partenaire », il apparie (`pair`) ; « ⋯ » agrandit la fenêtre
(`more`).
"""

import html
from dataclasses import dataclass, field
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
    uncertain: bool = False  # paire incertaine : relue « incertaine », ou automatique d'incertitude moyenne ou forte


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
        payload |= {"s": state.kind, "x": state.partner, "m": state.manual, "d": state.local, "q": state.uncertain, "o": direction}
    return payload


@dataclass(frozen=True)
class Marks:
    """Ce que la vue signale sur les entrées, par côté."""

    tasks: dict[str, dict[str, int]] = field(default_factory=lambda: {"left": {}, "right": {}})  # uuid → niveau de tâche (1, 2)
    hits: dict[str, set[str]] = field(default_factory=lambda: {"left": set(), "right": set()})  # résultats de recherche
    eligible: dict[str, set[str]] | None = None  # mode « choisir le partenaire » : entrées cliquables


def section_before(lines: list[DocLine], start: int) -> str:
    """Rubrique (titre lisible) en vigueur juste avant la ligne `start` : le
    dernier titre de niveau 1 ou 2 au-dessus (comme
    `records.section_of`), "" s'il n'y en a pas."""
    for index in range(start - 1, -1, -1):
        line = lines[index]
        if line.entity == "TITLE" and line.level in (1, 2):
            return title_text(line.markdown) or line.markdown
    return ""


def payload(
    docs: Documents,
    bounds: dict[str, tuple[int, int]],
    focus: tuple[str, str],
    marks: Marks,
    height: str,
    info: str = "",
    center: tuple[str, str] | None = None,
) -> dict:
    """Données du composant : lignes des deux fenêtres, ligne courante,
    marqueurs (tâches, recherche) et mode d'appariement. `center` : la paire
    sur laquelle la fenêtre est alignée (la ligne courante par défaut, une
    autre après « Page »)."""
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
            item = line_payload(line, state, direction)
            if line.uuid in marks.tasks[side]:
                item["k"] = marks.tasks[side][line.uuid]
            if line.uuid in marks.hits[side]:
                item["f"] = True
            if marks.eligible is not None and line.entity == "ENTRY":
                item["e"] = line.uuid in marks.eligible[side]
            rows.append(item)
        sides[side] = rows
    return {
        "left": sides["left"],
        "right": sides["right"],
        "focus": list(focus),
        "center": list(center or focus),
        "info": info,  # la ligne courante en une phrase (infobulle de la loupe)
        "pairing": marks.eligible is not None,
        "height": height,
        "more": {side: [bounds[side][0] > 0, bounds[side][1] < len(docs.lines[side])] for side in SIDES},
        "sections": {
            side: html.escape(section_before(docs.lines[side], bounds[side][0])) for side in SIDES
        },  # rubrique au-dessus de la fenêtre
    }


_component = None


def _diff_component():
    global _component
    if _component is None:
        css = (ASSETS / "context.css").read_text(encoding="utf-8") + SPAN_CSS
        _component = st.components.v2.component("numrev_documents_diff", js=(ASSETS / "context.js").read_text(encoding="utf-8"), css=css)
    return _component


EVENTS = ("focus", "pair", "more", "page", "zoom")  # déclencheurs du composant


def documents_diff(data: dict, key: str, titles: tuple[str, str]) -> tuple[str, dict] | None:
    """Monte le composant ; renvoie (événement, valeur) au tour qui suit un
    clic, sinon None. Événements : `focus` ({side, uuid} : entrée cliquée),
    `pair` ({side, uuid} : partenaire choisi en mode d'appariement), `more`
    ({dir} : -1 lignes précédentes, 1 lignes suivantes), `page` ({dir} :
    page précédente ou suivante), `zoom` (loupe de la ligne courante)."""
    callbacks = {f"on_{event}_change": (lambda: None) for event in EVENTS}
    result = _diff_component()(key=key, data=data | {"titles": list(titles)}, **callbacks)
    for event in EVENTS:
        value = result.get(event)
        if value:
            return event, value
    return None

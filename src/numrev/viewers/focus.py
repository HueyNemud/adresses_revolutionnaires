"""Vue « Relecture » du viewer d'alignement : une ligne à la fois (paire,
candidate ou entrée seule), en grand, avec les différences de texte
surlignées, les rapprochements possibles et des boutons de décision au
clavier. Les décisions vont au journal (`numrev/alignment/decisions.py`) ;
après chacune, on passe à la ligne suivante de la file.

Les fonctions de calcul (diff de texte, rapprochements) sont pures et
testées ; `render` monte l'interface et délègue les actions à des rappels
fournis par `numrev/viewers/alignment.py`.
"""

import difflib
import html
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import streamlit as st

from numrev.alignment.decisions import ALONE, DIFFERENT, PROBABLE, SAME, UNDO
from numrev.alignment.export import CANDIDATE, PAIR
from numrev.alignment.records import Record, similarity_matrix
from numrev.alignment.review import LEVEL_LABELS, REASON_CANDIDATE, Review
from numrev.ner.html import SPAN_CSS, render_tagged_html

SIDE_NAMES = {"left": "gauche", "right": "droite"}
ALTERNATIVES = 3  # rapprochements proposés par côté

# Couleurs Alucard (clair) et Dracula (sombre), comme le thème
# (assets/theme.toml) et la vue Documents (assets/context.css).
CARD_CSS = f"""<style>
  .nr {{ --comment: #6c664b; --cyan: #036a96; --green: #14710a; --orange: #a34d14; --pink: #a3144d; --purple: #644ac9;
         --red: #cb3a2a; --yellow: #846e15; --muted: color-mix(in srgb, currentColor 58%, transparent);
         --line: color-mix(in srgb, currentColor 16%, transparent); }}
  @media (prefers-color-scheme: dark) {{
    .nr {{ --comment: #6272a4; --cyan: #8be9fd; --green: #50fa7b; --orange: #ffb86c; --pink: #ff79c6; --purple: #bd93f9;
           --red: #ff5555; --yellow: #f1fa8c; }}
  }}
  .card {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .card .side {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 0; }}
  .card .side.empty {{ border-style: dashed; color: var(--muted); font-style: italic; display: flex; align-items: center;
                      justify-content: center; }}
  .card .label {{ font-size: .75em; text-transform: uppercase; letter-spacing: .04em; color: var(--purple); }}
  .card .text {{ font-size: 1.25em; line-height: 1.45; margin: 6px 0 8px; overflow-wrap: anywhere; }}
  .card .ner {{ font-size: .85em; line-height: 1.6; }}
  .nr .meta {{ font-family: monospace; font-size: .78em; color: var(--comment); }}
  .card mark.diff {{ background: color-mix(in srgb, var(--pink) 22%, transparent); color: inherit; border-radius: 2px; padding: 0 1px;
                     text-decoration: underline; text-decoration-color: var(--pink); }}
  .verdict {{ margin: 10px 0 4px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: .92em; }}
  .verdict .status {{ font-weight: 600; padding: 2px 10px; border-radius: 12px; }}
  .verdict .pair {{ background: color-mix(in srgb, var(--green) 16%, transparent); color: var(--green); }}
  .verdict .manual {{ background: color-mix(in srgb, var(--purple) 16%, transparent); color: var(--purple); }}
  .verdict .candidate {{ background: color-mix(in srgb, var(--orange) 16%, transparent); color: var(--orange);
                         border: 1px dashed var(--orange); }}
  .verdict .alone {{ background: color-mix(in srgb, var(--comment) 18%, transparent); color: var(--muted); }}
  .verdict .level {{ padding: 1px 8px; border-radius: 10px; font-size: .85em; }}
  .verdict .level-1 {{ background: color-mix(in srgb, var(--orange) 16%, transparent); color: var(--orange); }}
  .verdict .level-2 {{ background: color-mix(in srgb, var(--red) 16%, transparent); color: var(--red); }}
  .verdict .reasons {{ color: var(--muted); }}
  .verdict .local {{ color: var(--purple); font-weight: 600; }}
  .alt {{ font-size: .92em; line-height: 1.5; }}
  .alt .meta {{ font-size: .8em; }}
  {SPAN_CSS}
  .nr mark.span {{ color: inherit !important; }}
  .nr mark.span[title="SUBJ"] {{ background: color-mix(in srgb, var(--cyan) 16%, transparent) !important;
                                 border-color: var(--cyan) !important; }}
  .nr mark.span[title="DESC"] {{ background: color-mix(in srgb, var(--yellow) 18%, transparent) !important;
                                 border-color: var(--yellow) !important; }}
  .nr mark.span[title="ADDR"] {{ background: color-mix(in srgb, var(--green) 14%, transparent) !important;
                                 border-color: var(--green) !important; }}
  .nr mark.span[title="SUBJ"] sub {{ color: var(--cyan) !important; }}
  .nr mark.span[title="DESC"] sub {{ color: var(--yellow) !important; }}
  .nr mark.span[title="ADDR"] sub {{ color: var(--green) !important; }}
</style>"""


# ----------------------------------------------------------------------
# Calculs
# ----------------------------------------------------------------------
def char_diff(left: str, right: str) -> tuple[str, str]:
    """HTML des deux textes, les caractères qui diffèrent surlignés (`mark.diff`)."""
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    pieces: dict[str, list[str]] = {"left": [], "right": []}
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        for side, text in (("left", left[i1:i2]), ("right", right[j1:j2])):
            if text:
                escaped = html.escape(text)
                pieces[side].append(escaped if operation == "equal" else f"<mark class='diff'>{escaped}</mark>")
    return "".join(pieces["left"]), "".join(pieces["right"])


@dataclass(frozen=True)
class Alternative:
    record: Record  # entrée de l'autre côté
    similarity: float
    partner: Record | None  # son partenaire actuel (paire), None si libre
    rival: bool = False  # concurrente « homonyme proche » de la paire courante


def alternatives(
    record: Record,
    side: str,
    others: list[Record],
    exclude: str,
    partners: dict[str, Record],
    subj_weight: float,
    rivals: set[str] = frozenset(),
    limit: int = ALTERNATIVES,
) -> list[Alternative]:
    """Les `limit` entrées de l'autre côté du segment les plus proches de
    `record` (côté `side`), hors `exclude` (son partenaire actuel).
    `partners` : uuid de l'autre côté → son partenaire apparié."""
    candidates = [other for other in others if other.uuid != exclude]
    if not candidates:
        return []
    if side == "left":
        scores = similarity_matrix([record], candidates, subj_weight)[0]
    else:
        scores = similarity_matrix(candidates, [record], subj_weight)[:, 0]
    best = np.argsort(-scores, kind="stable")[:limit]
    return [Alternative(candidates[i], float(scores[i]), partners.get(candidates[i].uuid), candidates[i].uuid in rivals) for i in best]


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
@dataclass
class Focus:
    """Ce que la vue affiche pour la ligne courante."""

    kind: str  # PAIR, CANDIDATE, LEFT_ONLY ou RIGHT_ONLY
    left: Record | None
    right: Record | None
    score: float | None
    source: str
    review: Review | None
    manual: bool  # décidée par le patch (versionné ou journal)
    local: bool  # touchée par le journal, non enregistrée
    alternatives: dict[str, list[Alternative]] = field(default_factory=dict)  # côté de l'entrée → rapprochements


@dataclass
class Controls:
    """Rappels (exécutés avant le tour suivant, `on_click`) et état de la file
    de tâches, communs aux vues Documents et Relecture."""

    decide: Callable[..., None]  # (action, gauche, droite, clé du champ note)
    move: Callable[..., None]  # (clé de ligne : uuid gauche, uuid droit ; vue facultative)
    undo_last: Callable[[], None]
    pairing: Callable[[bool], None]  # entre (True) / sort (False) du mode « choisir le partenaire »
    position: int | None  # rang de la ligne courante dans la file, None hors file
    remaining: int  # tâches restantes
    decided: int  # lignes décidées (patch + journal)
    pending: int  # décisions du journal, non enregistrées
    previous: tuple[str, str] | None  # clés de ligne des tâches voisines
    following: tuple[str, str] | None
    can_undo: bool
    pairing_active: bool = False


def entry_side(record: Record | None, side: str, diff: str) -> str:
    if record is None:
        return f"<div class='side empty'>aucune entrée {SIDE_NAMES[side]}</div>"
    ner = render_tagged_html(record.tagged_text)
    return (
        f"<div class='side'><div class='label'>{SIDE_NAMES[side]} · {html.escape(record.section_title or '(sans rubrique)')}</div>"
        f"<div class='text'>{diff}</div>"
        + (f"<div class='ner'>{ner}</div>" if ner else "")
        + f"<div class='meta'>p. {html.escape(record.page)} · {html.escape(record.uuid)}</div></div>"
    )


def verdict_html(focus: Focus) -> str:
    if focus.kind == CANDIDATE:
        status, css = "? candidate non appariée", "candidate"
    elif focus.kind != PAIR:
        status, css = ("✗ sans correspondance (confirmée)" if focus.manual else "✗ sans correspondance"), "alone"
    elif focus.manual:
        status, css = "✓ appariée · relue" + (", incertaine" if focus.source.endswith("incertain") else ""), "manual"
    else:
        status, css = "✓ appariée automatiquement", "pair"
    parts = [f"<span class='status {css}'>{status}</span>"]
    if focus.score is not None:
        parts.append(f"<span class='meta'>score {focus.score:.3f}</span>")
    if focus.review and focus.review.level:
        parts.append(f"<span class='level level-{focus.review.level}'>incertitude {LEVEL_LABELS[focus.review.level]}</span>")
        reasons = [reason for reason in focus.review.reasons if reason != REASON_CANDIDATE]
        if reasons:
            parts.append(f"<span class='reasons'>{html.escape(' · '.join(reasons))}</span>")
    if focus.local:
        parts.append("<span class='local'>● décision non enregistrée</span>")
    return f"<div class='verdict'>{''.join(parts)}</div>"


def task_bar(controls: Controls, back: str | None = None) -> None:
    """Bandeau commun aux deux vues : tâches voisines, progression, annulation
    et, en relecture détaillée, retour à la vue `back` (Échap)."""
    previous, progress, following, undo, zoom = st.columns([1.2, 3.4, 1.2, 1.5, 1.6], vertical_alignment="center")
    previous.button(
        "◀ Tâche",
        shortcut="P",
        width="stretch",
        disabled=controls.previous is None,
        on_click=controls.move,
        args=(controls.previous,),
        help="Tâche de relecture précédente.",
    )
    done = controls.decided
    total = done + controls.remaining
    where = "" if controls.position is None else f" · n° {controls.position + 1}"
    pending = f" (dont {controls.pending} non enregistrée(s))" if controls.pending else ""
    progress.progress(
        done / total if total else 1.0,
        text=f"{controls.remaining} tâche(s) restante(s){where} · {done} ligne(s) décidée(s){pending}",
    )
    following.button(
        "Tâche ▶",
        shortcut="N",
        width="stretch",
        disabled=controls.following is None,
        on_click=controls.move,
        args=(controls.following,),
        help="Tâche de relecture suivante, sans décider.",
    )
    undo.button(
        "↶ Annuler",
        shortcut="Ctrl+Z",
        width="stretch",
        disabled=not controls.can_undo,
        on_click=controls.undo_last,
        help="Annule la dernière décision du journal et revient sur sa ligne.",
    )
    if back:
        zoom.button(
            "Vue d'ensemble",
            icon=":material/zoom_out:",
            shortcut=None if controls.pairing_active else "Esc",
            width="stretch",
            on_click=controls.move,
            args=(None, back),
            help="Retour aux documents, sur la même ligne.",
        )


def decision_buttons(focus: Focus, controls: Controls, note_key: str | None, zoom: str | None = None) -> None:
    """Boutons de décision sur la ligne courante (mêmes raccourcis dans les deux
    vues) ; `zoom` : vue de la relecture détaillée, bouton en bout de ligne."""
    both = focus.left is not None and focus.right is not None
    widths = [1.3, 1.1, 1.1, 1.5, 1.1] if both else [2.4, 1.5, 1.1]
    columns = st.columns(widths + ([1.6] if zoom else []))
    if both:
        columns[0].button(
            "✓ Même entrée",
            type="primary",
            shortcut="V",
            width="stretch",
            on_click=controls.decide,
            args=(SAME, focus.left, focus.right, note_key),
            help="Les deux entrées se correspondent (paire du patch).",
        )
        columns[1].button(
            "≈ Incertaine",
            shortcut="I",
            width="stretch",
            on_click=controls.decide,
            args=(PROBABLE, focus.left, focus.right, note_key),
            help="Probablement la même entrée : paire retenue, marquée `certitude = incertaine` dans le patch.",
        )
        columns[2].button(
            "✗ Différentes",
            shortcut="X",
            width="stretch",
            on_click=controls.decide,
            args=(DIFFERENT, focus.left, focus.right, note_key),
            help="Pas la même entrée : les deux entrées sont déclarées sans correspondance (le patch ne sait pas dire "
            "« pas avec celle-là ») ; si l'une a un autre partenaire, l'apparier ensuite (« Autre partenaire… »).",
        )
        rest = columns[3:]
    else:
        columns[0].button(
            "✓ Confirmer sans correspondance",
            type="primary",
            shortcut="V",
            width="stretch",
            on_click=controls.decide,
            args=(ALONE, focus.left, focus.right, note_key),
            help=f"Cette entrée n'a pas de correspondance dans l'annuaire de {SIDE_NAMES['right' if focus.left else 'left']}.",
        )
        rest = columns[1:]
    if controls.pairing_active:
        rest[0].button(
            "Annuler le choix",
            shortcut="Esc",
            width="stretch",
            on_click=controls.pairing,
            args=(False,),
            help="Quitte le mode « choisir le partenaire ».",
        )
    else:
        rest[0].button(
            "⇄ Autre partenaire…",
            shortcut="A",
            width="stretch",
            on_click=controls.pairing,
            args=(True,),
            help="Choisir dans les documents l'entrée à apparier (seules les entrées des rubriques appariées sont cliquables).",
        )
    if focus.manual or focus.local:
        rest[1].button(
            "↺ Défaire",
            width="stretch",
            on_click=controls.decide,
            args=(UNDO, focus.left, focus.right, None),
            help="Annule la décision sur ces entrées (retire leurs lignes du patch) : l'alignement automatique reprend la main.",
        )
    if zoom:
        columns[-1].button(
            "Relire en détail",
            icon=":material/zoom_in:",
            shortcut=None if controls.pairing_active else "Enter",
            width="stretch",
            on_click=controls.move,
            args=(None, zoom),
            help="Ouvre la ligne courante en grand : différences surlignées, rapprochements possibles, note (aussi : 🔍 sur le lien).",
        )


def summary(focus: Focus) -> str:
    """La ligne courante en une phrase (infobulle de la loupe des documents)."""
    if focus.kind == CANDIDATE:
        parts = ["candidate non appariée"]
    elif focus.kind != PAIR:
        parts = ["sans correspondance (confirmée)" if focus.manual else "sans correspondance"]
    else:
        parts = ["appariée, relue" if focus.manual else "appariée automatiquement"]
    if focus.score is not None:
        parts.append(f"score {focus.score:.3f}")
    if focus.review and focus.review.level:
        parts.append(f"incertitude {LEVEL_LABELS[focus.review.level]}")
        parts += [reason for reason in focus.review.reasons if reason != REASON_CANDIDATE]
    return " · ".join(parts)


def render(focus: Focus, controls: Controls, key: str) -> None:
    """Vue Relecture : carte de la ligne courante, décisions et rapprochements possibles."""
    left_diff, right_diff = char_diff(focus.left.text if focus.left else "", focus.right.text if focus.right else "")
    if focus.left is None or focus.right is None:
        left_diff = html.escape(focus.left.text) if focus.left else ""
        right_diff = html.escape(focus.right.text) if focus.right else ""
    st.html(
        f"{CARD_CSS}<div class='nr'><div class='card'>{entry_side(focus.left, 'left', left_diff)}"
        f"{entry_side(focus.right, 'right', right_diff)}</div>{verdict_html(focus)}</div>"
    )
    note_key = f"note::{key}"
    decision_buttons(focus, controls, note_key)
    st.text_input(
        "Note (facultative, recopiée dans le patch)",
        key=note_key,
        placeholder="Note facultative, recopiée dans le patch : homonymes, père / fils, rue renommée…",
        label_visibility="collapsed",
    )

    shown = {side: found for side, found in focus.alternatives.items() if found}
    if shown:
        with st.expander("Rapprochements possibles", expanded=focus.kind != PAIR or bool(focus.review and focus.review.level)):
            for side, found in shown.items():
                record = focus.left if side == "left" else focus.right
                st.caption(f"Entrées de {SIDE_NAMES['right' if side == 'left' else 'left']} les plus proches de « {record.text[:60]} »")
                for number, alternative in enumerate(found):
                    alternative_row(alternative, side, record, controls, f"{key}:{side}:{number}", note_key)


def alternative_row(alternative: Alternative, side: str, record: Record, controls: Controls, key: str, note_key: str) -> None:
    text, action = st.columns([5, 1.3], vertical_alignment="center")
    rival = " · <b>homonyme proche</b>" if alternative.rival else ""
    taken = (
        f" · déjà appariée à « {html.escape(alternative.partner.text[:50])} » (ce lien sera défait)"
        if alternative.partner is not None
        else " · libre"
    )
    text.html(
        f"{CARD_CSS}<div class='nr alt'>{render_tagged_html(alternative.record.tagged_text) or html.escape(alternative.record.text)}"
        f"<br><span class='meta'>similarité {alternative.similarity:.3f} · p. {html.escape(alternative.record.page)}</span>"
        f"<span class='meta'>{rival}{taken}</span></div>"
    )
    left, right = (record, alternative.record) if side == "left" else (alternative.record, record)
    action.button("Apparier", key=key, width="stretch", on_click=controls.decide, args=(SAME, left, right, note_key))

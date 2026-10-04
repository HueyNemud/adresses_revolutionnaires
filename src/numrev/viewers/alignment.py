"""Viewer Streamlit d'un alignement entre deux annuaires (sorties de
`numrev align`), outil de relecture et de correction.

    uv run numrev view alignment

On choisit une paire d'annuaires par une sortie brute : celle de Dedupe
`annuaires/alignments/<gauche>__<droite>.dedupe.csv` ou celle de
`numrev align nw`, `<gauche>__<droite>.nw.csv`. Les deux annuaires
(`annuaires/<gauche>/`, `annuaires/<droite>/`) sont relus en entier par
`numrev/alignment/records.py`, et le patch de corrections manuelles
`data/alignment/<gauche>__<droite>.patch.csv`, augmenté des décisions du
journal, est appliqué en mémoire (`numrev/alignment/patch.py`) : ce qui est
affiché est le résultat final.

**Un curseur, trois vues.** La ligne courante (une paire, une candidate ou
une entrée seule) est partagée par les trois vues ; changer de vue la garde.

- **Relecture** (`numrev/viewers/focus.py`) : une ligne à la fois, les
  différences de texte surlignées, les rapprochements possibles, et des
  boutons de décision au clavier (V même entrée, I incertaine, X pas la
  même entrée, → passer, ← précédente, Ctrl+Z annuler la dernière). La
  file = les lignes filtrées dans la barre latérale, par défaut celles à
  vérifier et pas encore décidées. Le contexte des deux documents est
  affiché sous la carte ;
- **Table** : toutes les lignes filtrées, dans l'**ordre naturel** des
  listes (correspondances et entrées de gauche seules dans l'ordre de
  gauche, chaque entrée de droite seule après la paire qui contient
  l'entrée de droite appariée qui la précède), statut et incertitude en
  clair, bouton **Relire →** par ligne ;
- **Documents** (`numrev/viewers/context.py`) : les deux annuaires côte à
  côte, chacun dans son ordre, avec titres et lignes hors sujet, les paires
  reliées dans une gouttière (inversions en orange). Un clic sur une entrée
  de chaque côté permet de les apparier.

**Décisions** (`numrev/alignment/decisions.py`) : elles vont dans un journal
gardé dans le navigateur (`numrev/viewers/storage.py`), rejoué sur le patch
versionné. En local (`numrev view alignment` pose `NUMREV_PATCH_WRITABLE=1`),
**Enregistrer** écrit le patch et vide le journal ; sinon (copie hébergée),
**Télécharger le patch** donne le patch complet à déposer dans
`data/alignment/`. Le format du patch ne change pas.

On n'apparie qu'entre rubriques appariées (`numrev/alignment/sections.py`, avec
son patch `data/alignment/<gauche>__<droite>.sections.csv`, édité à la main,
non réécrit ici) : un lien entre rubriques qui ne se correspondent pas —
Dedupe en produit — est écarté, et compté. Un encart **Rubriques** donne,
par groupe de rubriques appariées puis par rubrique seule, la
correspondance et la part d'entrées appariées ; chaque bandeau de rubrique
a un bouton **uuid** pour alimenter le patch des rubriques.

**Relecture** (`numrev/alignment/review.py`) : chaque correspondance porte une
incertitude (faible, moyenne ou forte) et ses motifs (`déduite des voisines
(p < 0,9)`, `homonyme proche`) ; des **candidates non appariées** (deux
entrées sans correspondance, chacune la plus proche de l'autre, dans la zone
grise de similarité) sont soumises au relecteur, jamais appariées d'office.

Le bouton **Exporter en CSV** télécharge les lignes de la table (filtres et
tri appliqués) sous la forme de la jointure lisible de
`numrev/alignment/export.py`, comme `numrev join`.
"""

import csv
import html
import io
import os
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from numrev import paths as paths_module
from numrev.alignment.decisions import (
    PROBABLE,
    SAME,
    UNDO,
    Decision,
    apply_decisions,
    decide,
    from_json,
    import_patch,
    patch_digest,
    to_json,
    touched_uuids,
)
from numrev.alignment.export import CANDIDATE, LEFT_ONLY, PAIR, RIGHT_ONLY, JoinedRow, export_csv, export_encoding, natural_rows
from numrev.alignment.nw import Params
from numrev.alignment.patch import (
    PATCH_FIELDS,
    PatchEntry,
    apply_patch,
    declared_unmatched,
    patch_csv,
    read_patch,
    resolve,
    validate,
    write_patch,
)
from numrev.alignment.records import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, DocLine, Link, Record, load_lines, load_volume, read_links
from numrev.alignment.review import DEFAULT_MARGIN, LEVEL_LABELS, REASON_CANDIDATE, SEPARATOR, Review, review
from numrev.alignment.sections import (
    SECTION_PATCH_FIELDS,
    SOURCE_AUTO,
    SectionAlignment,
    load_section_alignment,
    read_section_patch,
    restrict_to_corresponding,
    segment_entries,
)
from numrev.ner.html import LABEL_COLORS, SPAN_CSS, badge, render_tagged_html
from numrev.paths import JOIN_SUFFIX, Pair
from numrev.viewers import context, focus
from numrev.viewers.common import PAGE_SIZE_OPTIONS, text_mask
from numrev.viewers.storage import browser_journal, storage_key

WRITABLE_ENV = "NUMREV_PATCH_WRITABLE"  # posée par `numrev view alignment` : enregistrement direct du patch
REVIEW, TABLE, DOCUMENTS = "Relecture", "Table", "Documents"
VIEWS = {REVIEW: ":material/rate_review: Relecture", TABLE: ":material/table_rows: Table", DOCUMENTS: ":material/difference: Documents"}
NATURAL_ORDER = "ordre naturel"
LEVEL_ORDER = "incertitude décroissante"
ORDER_OPTIONS = [NATURAL_ORDER, "score croissant", "score décroissant", LEVEL_ORDER]
LEVEL_OPTIONS = {0: "toutes les lignes", 1: "moyenne ou forte", 2: "forte seulement"}  # filtre : incertitude minimale
ALL_SECTIONS = "(toutes)"
NO_SECTION = "(sans rubrique)"
KIND_LABELS = {
    PAIR: "✓ Appariées",
    CANDIDATE: "? Candidates non appariées (à décider)",
    LEFT_ONLY: "✗ Sans correspondance à gauche",
    RIGHT_ONLY: "✗ Sans correspondance à droite",
}
LEVEL_CLASSES = {1: "level-medium", 2: "level-high"}  # badge d'incertitude moyenne / forte
MANUAL_SOURCES = {SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN}
MANUAL_COLORS = ("#ede9fe", "#6d28d9")
CONFIRMED = "confirmée sans correspondance"
ENTRY_COLUMNS = [field.name for field in fields(Record)]
DOCUMENTS_STEP = 40  # lignes de la table parcourues par ▲ / ▼ dans la vue Documents

# Clés de st.session_state
CURSOR = "cursor"  # (uuid gauche, uuid droit) de la ligne courante
PICKS = "picks"  # côté → uuid sélectionné dans la vue Documents
VIEW = "view"
PENDING_VIEW = "pending_view"  # vue à afficher au prochain tour (posée hors rappel)

CSS = f"""<style>
  .legend span {{ margin-right: 10px; }}
  .html-events {{ color: var(--st-text-color, inherit); font-family: var(--st-font, inherit); }}
  .table-scroll {{ max-height: 75vh; overflow-y: auto; border: 1px solid rgba(100,116,139,.3); border-radius: 6px; }}
  .ner-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; table-layout: fixed; }}
  .ner-table th {{ background: var(--st-secondary-background-color, #f8fafc); border-bottom: 2px solid rgba(100,116,139,.3);
                  padding: 8px; text-align: left;
                  position: sticky; top: 0; z-index: 1; }}
  .ner-table td {{ border-bottom: 1px solid rgba(100,116,139,.15); padding: 6px 8px; vertical-align: top; overflow-wrap: anywhere; }}
  .ner-table tr:hover td {{ background: rgba(100,116,139,.08); }}
  .ner-table tr.section td {{ background: var(--st-secondary-background-color, #f1f5f9); font-weight: 600; font-size: 0.85em; }}
  .ner-table tr.left td.empty, .ner-table tr.right td.empty {{ background: rgba(239,68,68,.07); }}
  .ner-table tr.candidate td {{ background: rgba(245,158,11,.10); }}
  .ner-table tr.candidate td:first-child {{ box-shadow: inset 4px 0 0 #d97706; }}
  .ner-table tr.pair td:first-child {{ box-shadow: inset 4px 0 0 #16a34a; }}
  .ner-table tr.left td:first-child, .ner-table tr.right td:first-child {{ box-shadow: inset 4px 0 0 #cbd5e1; }}
  .ner-table tr.current td {{ background: rgba(37,99,235,.10); }}
  .ner-table col.number {{ width: 3.5em; }}
  .ner-table col.score {{ width: 14em; }}
  .meta {{ font-family: monospace; font-size: 0.78em; color: #64748b; }}
  .status {{ display: inline-block; font-weight: 600; font-size: 0.85em; padding: 1px 8px; border-radius: 10px; margin-bottom: 3px; }}
  .status.pair {{ background: #dcfce7; color: #166534; }}
  .status.manual {{ background: #ede9fe; color: #5b21b6; }}
  .status.uncertain {{ background: #fef3c7; color: #92400e; }}
  .status.candidate {{ background: #fde68a; color: #78350f; border: 1px dashed #b45309; }}
  .status.alone {{ background: #f1f5f9; color: #475569; }}
  .local {{ display: inline-block; font-size: 0.75em; font-weight: 600; color: #6d28d9; }}
  .level {{ display: inline-block; font-size: 0.78em; padding: 0 6px; border-radius: 8px; margin-top: 3px; }}
  .level.level-medium {{ background: #ffedd5; color: #9a3412; }}
  .level.level-high {{ background: #fee2e2; color: #991b1b; }}
  .reasons {{ font-size: 0.78em; color: #475569; }}
  .legend-table {{ font-size: 0.85em; color: #475569; margin: 4px 0 8px; line-height: 2.1; }}
  .legend-table .status, .legend-table .level {{ margin-right: 4px; }}
  .low {{ color: #b91c1c; font-weight: 600; }}
  .empty {{ color: #b91c1c; font-size: 0.85em; font-style: italic; }}
  button.copy, button.open {{ font-size: 0.72em; padding: 0 6px; margin-left: 4px; border: 1px solid rgba(100,116,139,.45);
                border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }}
  button.open {{ font-size: 0.8em; padding: 1px 8px; margin: 4px 0 0; border-color: rgba(37,99,235,.6); color: #3b82f6; }}
  button.copy:hover, button.open:hover {{ background: rgba(100,116,139,.12); }}
  {SPAN_CSS}
</style>"""

# Copie au clic, par délégation (installée une fois par page) ; repli sur
# execCommand si l'API presse-papiers est indisponible.
COPY_SCRIPT = """<script>
if (!window.__alignmentCopy) {
  window.__alignmentCopy = true;
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("button.copy");
    if (!button) return;
    const text = button.dataset.copy;
    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const area = document.createElement("textarea");
      area.value = text; document.body.appendChild(area); area.select();
      document.execCommand("copy"); area.remove();
    }
    const label = button.textContent;
    button.textContent = "✓ copié";
    setTimeout(() => { button.textContent = label; }, 1200);
  });
}
</script>"""

# Table cliquable : HTML fourni par Python ; un bouton `[data-action]` renvoie
# {action, value} (déclencheur `action`) ; les boutons `copy` copient.
TABLE_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  if (!data) return;
  let root = parentElement.querySelector(".html-events");
  if (!root) {
    root = document.createElement("div");
    root.className = "html-events";
    parentElement.appendChild(root);
    root.addEventListener("click", async (event) => {
      const copy = event.target.closest("button.copy");
      if (copy) {
        try { await navigator.clipboard.writeText(copy.dataset.copy); } catch (error) { /* presse-papiers indisponible */ }
        const label = copy.textContent;
        copy.textContent = "✓ copié";
        setTimeout(() => { copy.textContent = label; }, 1200);
        return;
      }
      const target = event.target.closest("[data-action]");
      if (target) setTriggerValue("action", { action: target.dataset.action, value: target.dataset.value });
    });
  }
  root.innerHTML = data.html;
}
"""
_table = st.components.v2.component("numrev_alignment_table", js=TABLE_JS)


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Lecture de l'annuaire…")
def load_records(volume_dir: str) -> dict[str, Record]:
    return {record.uuid: record for record in load_volume(Path(volume_dir))}


@st.cache_resource(show_spinner="Lecture des documents…")
def load_document_lines(volume_dir: str) -> list[DocLine]:
    return load_lines(Path(volume_dir))


@dataclass
class Alignment:
    rows: pd.DataFrame  # sortie de `build_rows`
    joined: list[JoinedRow]  # mêmes lignes, même ordre (index de `rows`) : pour l'export
    reviews: dict[tuple[str, str], Review]  # motifs de relecture (`numrev/alignment/review.py`)
    entries: list[PatchEntry]  # patch effectif (versionné + journal), avant résolution
    n_base: int  # lignes du patch versionné
    n_missing: int  # liens Dedupe vers des entrées disparues, ignorés
    reanchored: int
    orphans: list[PatchEntry]
    confirmed: set[str]  # uuid déclarés sans correspondance par le patch
    local: set[str]  # uuid touchés par le journal (décisions non enregistrées)
    sections: SectionAlignment  # correspondance des rubriques (avec leur patch)
    n_section_patch: int
    dropped: list[Link]  # liens entre rubriques non appariées, écartés (automatiques et du patch)
    by_key: dict[tuple[str, str], int]  # (uuid gauche, uuid droit) → index de ligne
    by_uuid: dict[str, dict[str, int]]  # côté → uuid → index de ligne
    segments: list[tuple[list[Record], list[Record]]]  # entrées de chaque segment (rubriques appariées)
    segment_of: dict[tuple[str, str], int]  # (côté, uuid) → rang du segment
    states: dict[str, dict[str, context.EntryState]]  # côté → uuid → état (vue Documents)


@st.cache_resource(show_spinner="Application du patch…", max_entries=4)
def load_alignment(
    left_dir: str,
    right_dir: str,
    dedupe_path: str,
    dedupe_mtime: float,
    patch_path: str,
    patch_mtime: float,
    section_patch_path: str,
    section_patch_mtime: float,
    candidate_low: float,
    margin: float,
    journal: str,
) -> Alignment:
    """Résultat final (alignement + patch + journal), recalculé seulement si
    l'un des fichiers ou le journal change (`*_mtime` et `journal` font partie
    de la clé de cache). Lève ValueError si un patch est incohérent. Le
    patch des rubriques n'est pas réécrit ici."""
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    base = read_patch(Path(patch_path))
    validate(base)
    decisions = from_json(journal)[0] if journal else []
    entries = apply_decisions(base, decisions)
    validate(entries)
    resolution = resolve(entries, records["left"], records["right"])
    links, _ = apply_patch(read_links(Path(dedupe_path)), resolution.entries)
    sections = load_section_alignment(list(records["left"].values()), list(records["right"].values()), Path(section_patch_path))
    links, dropped = restrict_to_corresponding(links, records["left"], records["right"], sections)
    declared = declared_unmatched(resolution)
    params = Params()
    found = review(
        links,
        list(records["left"].values()),
        list(records["right"].values()),
        sections,
        candidate_low,
        params.residual_threshold,
        params.subj_weight,
        margin,
        declared,
    )
    joined, n_missing = natural_rows(links + found.candidates, records["left"], records["right"])
    rows = build_rows(joined, found.reviews)
    local = touched_uuids(decisions)
    segments = segment_entries(sections)
    return Alignment(
        rows=rows,
        joined=joined,
        reviews=found.reviews,
        entries=entries,
        n_base=len(base),
        n_missing=n_missing,
        reanchored=len(resolution.reanchored),
        orphans=resolution.orphans,
        confirmed=declared,
        local=local,
        sections=sections,
        n_section_patch=len(read_section_patch(Path(section_patch_path))),
        dropped=dropped,
        by_key={(row.left_uuid, row.right_uuid): index for index, row in enumerate(rows.itertuples())},
        by_uuid={side: {uuid: index for index, uuid in enumerate(rows[f"{side}_uuid"]) if uuid} for side in context.SIDES},
        segments=segments,
        segment_of={
            (side, record.uuid): number
            for number, pair in enumerate(segments)
            for side, side_records in zip(context.SIDES, pair)
            for record in side_records
        },
        states=entry_states(rows, declared, local),
    )


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def build_rows(joined: list[JoinedRow], reviews: dict[tuple[str, str], Review]) -> pd.DataFrame:
    """Une ligne par ligne de la jointure (ordre naturel), colonnes
    `left_*` / `right_*` (vides du côté absent), niveau d'incertitude et
    motifs de relecture."""

    def found(row: JoinedRow) -> Review | None:
        return reviews.get((row.link.left_uuid, row.link.right_uuid)) if row.link else None

    def side(prefix: str, record: Record | None) -> dict:
        return {f"{prefix}_{column}": getattr(record, column) if record else "" for column in ENTRY_COLUMNS}

    rows = pd.DataFrame(
        [
            {
                **side("left", row.left),
                **side("right", row.right),
                "score": row.link.score if row.link else None,
                "source": row.link.source if row.link else "",
                "kind": row.kind,
                "level": found(row).level if found(row) else 0,
                "reasons": SEPARATOR.join(found(row).reasons) if found(row) else "",
            }
            for row in joined
        ]
    )
    rows["score"] = pd.to_numeric(rows["score"])
    return rows


def entry_states(rows: pd.DataFrame, confirmed: set[str], local: set[str]) -> dict[str, dict[str, context.EntryState]]:
    """État de chaque entrée alignable, des deux côtés (vue Documents)."""
    states: dict[str, dict[str, context.EntryState]] = {"left": {}, "right": {}}
    kinds = {PAIR: "pair", CANDIDATE: "candidate", LEFT_ONLY: "alone", RIGHT_ONLY: "alone"}
    for row in rows.itertuples():
        for side, uuid, partner in (("left", row.left_uuid, row.right_uuid), ("right", row.right_uuid, row.left_uuid)):
            if uuid:
                manual = row.source in MANUAL_SOURCES or uuid in confirmed
                states[side][uuid] = context.EntryState(kinds[row.kind], partner, manual, uuid in local)
    return states


def decided_mask(rows: pd.DataFrame, confirmed: set[str]) -> pd.Series:
    """Lignes décidées par le patch (paire relue ou entrée confirmée seule)."""
    return rows["source"].isin(MANUAL_SOURCES) | rows["left_uuid"].isin(confirmed) | rows["right_uuid"].isin(confirmed)


def row_key(rows: pd.DataFrame, index: int) -> tuple[str, str]:
    return rows.at[index, "left_uuid"], rows.at[index, "right_uuid"]


def cursor_index(alignment: Alignment, cursor: tuple[str, str] | None) -> int | None:
    """Ligne de la clé `cursor`, à défaut celle qui contient son entrée de
    gauche, puis celle de droite (une décision change la forme des lignes)."""
    if not cursor:
        return None
    if tuple(cursor) in alignment.by_key:
        return alignment.by_key[tuple(cursor)]
    for side, uuid in zip(context.SIDES, cursor):
        if uuid and uuid in alignment.by_uuid[side]:
            return alignment.by_uuid[side][uuid]
    return None


# ----------------------------------------------------------------------
# Journal des décisions
# ----------------------------------------------------------------------
def journal_of(pair_name: str) -> list[Decision]:
    return st.session_state.setdefault(f"journal::{pair_name}", [])


def journal_text(pair_name: str, digest: str) -> str:
    journal = journal_of(pair_name)
    return to_json(journal, st.session_state.get(f"journal_base::{pair_name}", digest)) if journal else ""


def restore_journal(pair_name: str, digest: str) -> None:
    """Pont avec le navigateur : restaure le journal sauvegardé au premier
    tour, puis lui envoie le journal à chaque tour."""
    restored = f"restored::{pair_name}"
    saved = browser_journal(storage_key(pair_name), journal_text(pair_name, digest) if st.session_state.get(restored) else None)
    if saved is None or st.session_state.get(restored):
        return
    st.session_state[restored] = True
    if saved:
        try:
            decisions, base = from_json(saved)
        except ValueError as error:
            st.session_state[f"journal_error::{pair_name}"] = f"Journal du navigateur illisible, ignoré : {error}"
        else:
            journal_of(pair_name)[:0] = decisions
            st.session_state[f"journal_base::{pair_name}"] = base
    st.rerun()


def on_decide(pair_name: str, digest: str, following: tuple[str, str] | None, action: str, left, right, note_key: str | None) -> None:
    """Rappel d'un bouton de décision : ajoute la décision au journal et passe
    à la ligne suivante (une annulation reste sur la ligne)."""
    note = st.session_state.get(note_key, "") if note_key else ""
    journal = journal_of(pair_name)
    if not journal:
        st.session_state[f"journal_base::{pair_name}"] = digest
    journal.append(decide(action, left, right, note))
    st.session_state[PICKS] = {}
    if action == UNDO or following is None:
        st.session_state[CURSOR] = (left.uuid if left else "", right.uuid if right else "")
    else:
        st.session_state[CURSOR] = following


def on_undo_last(pair_name: str) -> None:
    journal = journal_of(pair_name)
    if journal:
        last = journal.pop()
        uuids = dict(last.touched)
        st.session_state[CURSOR] = (uuids.get("left", ""), uuids.get("right", ""))


def on_move(key: tuple[str, str] | None, view: str | None = None) -> None:
    if key is not None:
        st.session_state[CURSOR] = key
    if view is not None:
        st.session_state[VIEW] = view


def on_save(pair_name: str, patch_path: Path) -> None:
    journal = journal_of(pair_name)
    entries = apply_decisions(read_patch(patch_path), journal)
    validate(entries)
    write_patch(patch_path, entries)
    journal.clear()
    st.session_state.pop(f"journal_base::{pair_name}", None)
    st.toast(f"Patch enregistré : {patch_path} ({len(entries)} lignes)", icon=":material/save:")


def on_discard(pair_name: str) -> None:
    journal_of(pair_name).clear()
    st.session_state.pop(f"journal_base::{pair_name}", None)


def on_import(pair_name: str, digest: str, key: str) -> None:
    upload = st.session_state.get(key)
    if upload is None:
        return
    try:
        text = upload.getvalue().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        entries = [PatchEntry(**{name: (row.get(name) or "") for name in PATCH_FIELDS}) for row in rows]
        validate(entries)
    except (UnicodeDecodeError, ValueError, csv.Error) as error:
        st.session_state[f"journal_error::{pair_name}"] = f"Patch importé illisible : {error}"
        return
    journal = journal_of(pair_name)
    if not journal:
        st.session_state[f"journal_base::{pair_name}"] = digest
    journal.extend(import_patch(entries))
    st.toast(f"{len(entries)} ligne(s) importée(s) dans le journal", icon=":material/upload:")


def journal_box(pair_name: str, patch_path: Path, digest: str, writable: bool) -> None:
    journal = journal_of(pair_name)
    st.header("Décisions")
    error = st.session_state.pop(f"journal_error::{pair_name}", None)
    if error:
        st.error(error)
    if journal:
        st.markdown(f"**{len(journal)}** décision(s) non enregistrée(s)")
        base = st.session_state.get(f"journal_base::{pair_name}", digest)
        if base != digest:
            st.warning(
                "Le patch versionné a changé depuis ces décisions ; elles sont rejouées dessus (par uuid). Vérifier avant d'enregistrer."
            )
    else:
        st.caption("Aucune décision en attente.")
    if writable:
        st.button(
            f"Enregistrer dans {patch_path.name}",
            type="primary",
            icon=":material/save:",
            width="stretch",
            disabled=not journal,
            on_click=on_save,
            args=(pair_name, patch_path),
            help=f"Écrit le patch complet (patch versionné + décisions) dans `{patch_path}` et vide le journal. "
            "Git garde l'ancienne version.",
        )
    st.download_button(
        "Télécharger le patch",
        lambda: patch_csv(apply_decisions(read_patch(patch_path), journal)).encode("utf-8"),
        file_name=patch_path.name,
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
        type="secondary" if writable else "primary",
        help=f"Patch complet (patch versionné + décisions), à déposer dans `{patch_path.parent}/`.",
    )
    if not writable:
        st.caption("Copie hébergée : les décisions restent dans ce navigateur ; télécharger le patch pour les transmettre.")
    with st.expander("Journal"):
        for decision in reversed(journal[-30:]):
            st.caption(f"{decision.at.replace('T', ' ')} · {decision.summary()[:140]}")
        if len(journal) > 30:
            st.caption(f"… et {len(journal) - 30} plus ancienne(s)")
        upload_key = f"import::{pair_name}"
        st.file_uploader(
            "Importer un patch",
            type="csv",
            key=upload_key,
            on_change=on_import,
            args=(pair_name, digest, upload_key),
            help="Ajoute au journal les lignes d'un patch (celui d'un autre relecteur) : "
            "chacune remplace les lignes qui touchent ses uuid.",
        )
        with st.popover("Tout abandonner", disabled=not journal, width="stretch"):
            st.write(f"Abandonner les {len(journal)} décision(s) non enregistrée(s) ?")
            st.button("Abandonner", type="primary", on_click=on_discard, args=(pair_name,))


# ----------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------
def copy_button(label: str, text: str, title: str) -> str:
    return f'<button class="copy" data-copy="{html.escape(text, quote=True)}" title="{html.escape(title)}">{label}</button>'


def entry_html(row, side: str, confirmed: set[str], local: set[str]) -> str:
    uuid = getattr(row, f"{side}_uuid")
    if not uuid:
        return "<span class='empty'>aucune entrée appariée</span>"
    tagged, markdown = getattr(row, f"{side}_tagged_text"), getattr(row, f"{side}_markdown")
    content = render_tagged_html(tagged) or html.escape(markdown)
    page = getattr(row, f"{side}_page")
    title = getattr(row, f"{side}_section_title") or NO_SECTION
    flag = f" {badge(CONFIRMED, MANUAL_COLORS)}" if uuid in confirmed else ""
    return (
        f"{content}{flag}<br><span class='meta'>p. {html.escape(page)}</span> · "
        f"<span class='meta'>{html.escape(title)}</span>"
        f"{copy_button('uuid', uuid, f'Copier l’uuid {uuid}')}"
    )


def status_badge(kind: str, source: str = "") -> str:
    """Statut d'une ligne, en clair : appariée (automatiquement, relue, ou
    relue mais incertaine), candidate non appariée, sans correspondance."""
    if kind == CANDIDATE:
        label, css = "? non appariée · candidate à décider", "candidate"
    elif kind != PAIR:
        label, css = "✗ sans correspondance", "alone"
    elif source == SOURCE_MANUAL_UNCERTAIN:
        label, css = "✓ appariée · relue, incertaine", "uncertain"
    elif source == SOURCE_MANUAL:
        label, css = "✓ appariée · relue", "manual"
    else:
        label, css = "✓ appariée", "pair"
    return f"<span class='status {css}'>{label}</span>"


def level_badge(level: int) -> str:
    return f"<span class='level {LEVEL_CLASSES[level]}'>incertitude {LEVEL_LABELS[level]}</span>" if level else ""


def status_html(row, threshold: float, local: set[str]) -> str:
    """Statut, score (similarité ou probabilité selon la méthode), incertitude et motifs."""
    parts = [status_badge(row.kind, row.source)]
    if row.left_uuid in local or row.right_uuid in local:
        parts.append("<span class='local'>● décision non enregistrée</span>")
    if row.kind in (PAIR, CANDIDATE) and not pd.isna(row.score):
        css = " class='low'" if row.score < threshold else ""
        parts.append(f"<span class='meta'>score</span> <span{css}>{row.score:.3f}</span>")
    if row.level:
        # Le motif « candidate non appariée » répète le statut : on ne l'affiche pas.
        reasons = SEPARATOR.join(reason for reason in row.reasons.split(SEPARATOR) if reason != REASON_CANDIDATE)
        parts.append(level_badge(row.level) + (f"<br><span class='reasons'>{html.escape(reasons)}</span>" if reasons else ""))
    return "<br>".join(parts)


def legend_html() -> str:
    """Légende des statuts et de l'incertitude (bouton « Légende » au-dessus de la table)."""
    items = [
        (status_badge(PAIR), "lien retenu automatiquement (bordure verte)"),
        (status_badge(PAIR, SOURCE_MANUAL), "lien du patch"),
        (status_badge(PAIR, SOURCE_MANUAL_UNCERTAIN), "lien du patch marqué incertain"),
        (status_badge(CANDIDATE), "deux entrées <b>non appariées</b>, soumises au relecteur (fond ambre)"),
        (status_badge(LEFT_ONLY), "entrée seule (bordure grise)"),
        ("<span class='local'>● décision non enregistrée</span>", "décision du journal, pas encore dans le patch"),
        (level_badge(1) + " " + level_badge(2), "à vérifier, motif en dessous ; incertitude faible : rien n'est affiché"),
    ]
    return "<div class='legend-table'>" + "<br>".join(f"{badges} {text}" for badges, text in items) + "</div>"


def table_rows(view: pd.DataFrame, first: int, low_score: float, banners: bool, alignment: Alignment, current: int | None) -> list[str]:
    # Les bandeaux suivent la rubrique de gauche, qui fixe l'ordre de la table :
    # une entrée de droite seule, insérée dans ce fil, n'en ouvre pas de
    # nouveau (sa rubrique figure dans sa cellule), sauf en tête de table ou
    # quand seules des entrées de droite sont affichées.
    rows, section_now = [], None
    right_only = bool((view["kind"] == RIGHT_ONLY).all())
    for number, (index, row) in enumerate(zip(view.index, view.itertuples()), start=first):
        if row.kind != RIGHT_ONLY:
            section, title, uuid = row.left_section, row.left_section_title, row.left_section_uuid
        elif section_now is None or right_only:
            section, title, uuid = row.right_section, row.right_section_title, row.right_section_uuid
        else:
            section, title, uuid = section_now, None, ""
        if banners and section != section_now:
            section_now = section
            button = copy_button("uuid", uuid, f"Copier l’uuid de la rubrique {uuid}") if uuid else ""
            rows.append(f"<tr class='section'><td colspan='4'>{html.escape(title or NO_SECTION)}{button}</td></tr>")
        empty_left = " class='empty'" if row.kind == RIGHT_ONLY else ""
        empty_right = " class='empty'" if row.kind == LEFT_ONLY else ""
        value = html.escape(f"{row.left_uuid}|{row.right_uuid}", quote=True)
        current_class = " current" if index == current else ""
        rows.append(
            f"<tr class='{row.kind}{current_class}'>"
            f"<td class='meta'>{number}</td>"
            f"<td{empty_left}>{entry_html(row, 'left', alignment.confirmed, alignment.local)}</td>"
            f"<td{empty_right}>{entry_html(row, 'right', alignment.confirmed, alignment.local)}</td>"
            f"<td>{status_html(row, low_score, alignment.local)}<br>"
            f"<button class='open' data-action='focus' data-value='{value}' title='Ouvrir cette ligne dans la vue Relecture'>"
            "Relire →</button></td>"
            "</tr>"
        )
    return rows


def paginated_table(view: pd.DataFrame, headers: list[str], render, page_size: int) -> dict | None:
    """Table paginée ; renvoie l'action cliquée ({action, value}) ou None."""
    if view.empty:
        st.info("Aucune ligne ne correspond aux filtres.")
        return None
    n_pages = max(1, -(-len(view) // page_size))
    page = min(st.session_state.get("page", 0), n_pages - 1)
    previous, label, legend, following = st.columns([1, 3, 0.8, 1])
    with legend.popover("ℹ️ Légende", width="stretch"):
        st.html(f"{CSS}{legend_html()}")
    if previous.button("◀ Précédente", disabled=page == 0, width="stretch"):
        page -= 1
    if following.button("Suivante ▶", disabled=page >= n_pages - 1, width="stretch"):
        page += 1
    st.session_state.page = page
    label.markdown(
        f"<div style='text-align:center;padding-top:6px'>Page <b>{page + 1}</b> / {n_pages} · "
        f"{len(view):,} ligne(s)</div>".replace(",", " "),
        unsafe_allow_html=True,
    )
    start = page * page_size
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    table = (
        f"{CSS}<div class='table-scroll'><table class='ner-table'>"
        "<colgroup><col class='number'><col><col><col class='score'></colgroup>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(render(view.iloc[start : start + page_size], start + 1))}</tbody>"
        "</table></div>"
    )
    return _table(key="alignment_table", data={"html": table}, on_action_change=lambda: None).get("action")


def section_table(alignment: SectionAlignment, rows: pd.DataFrame) -> pd.DataFrame:
    """Rubriques : un groupe de rubriques appariées par ligne (titres et uuid
    joints par « + » s'il en compte plusieurs), puis les rubriques seules,
    avec, de chaque côté, les entrées et la part appariée."""
    pairs = rows[rows["kind"] == PAIR]
    matched = {"left": pairs["left_section_uuid"].value_counts(), "right": pairs["right_section_uuid"].value_counts()}

    def joined(sections, attribute: str) -> str:
        return " + ".join(getattr(section, attribute) or NO_SECTION for section in sections)

    def counts(side: str, sections) -> list:
        total = sum(len(section.records) for section in sections)
        paired = sum(int(matched[side].get(section.uuid, 0)) for section in sections)
        return [total, paired, round(paired / total, 3) if total else None]

    lines = [
        [
            joined(group.left, "title"),
            joined(group.right, "title"),
            "patch" if group.source != SOURCE_AUTO else "auto",
            *counts("left", group.left),
            *counts("right", group.right),
            joined(group.left, "uuid"),
            joined(group.right, "uuid"),
        ]
        for group in alignment.groups
    ]
    for side, unmatched in (("left", alignment.unmatched_left), ("right", alignment.unmatched_right)):
        for section in unmatched:
            source = "seule (patch)" if (side, section.uuid) in alignment.declared else "seule"
            empty = [None, None, None]
            if side == "left":
                lines.append([section.title or NO_SECTION, "", source, *counts("left", [section]), *empty, section.uuid, ""])
            else:
                lines.append(["", section.title or NO_SECTION, source, *empty, *counts("right", [section]), "", section.uuid])
    table = pd.DataFrame(
        lines,
        columns=[
            "gauche",
            "droite",
            "correspondance",
            "gauche : entrées",
            "gauche : appariées",
            "gauche : taux",
            "droite : entrées",
            "droite : appariées",
            "droite : taux",
            "uuid gauche",
            "uuid droite",
        ],
    )
    for column in ("gauche : entrées", "gauche : appariées", "droite : entrées", "droite : appariées"):
        table[column] = table[column].astype("Int64")
    return table


# ----------------------------------------------------------------------
# Relecture et Documents
# ----------------------------------------------------------------------
def focus_of(alignment: Alignment, records: dict[str, dict[str, Record]], index: int, subj_weight: float) -> focus.Focus:
    """La ligne `index` pour la vue Relecture, avec ses rapprochements possibles."""
    row = alignment.rows.loc[index]
    left = records["left"].get(row["left_uuid"]) if row["left_uuid"] else None
    right = records["right"].get(row["right_uuid"]) if row["right_uuid"] else None
    found = alignment.reviews.get((row["left_uuid"], row["right_uuid"]))
    rivals = {uuid for _, uuid, _ in found.rivals} if found else set()
    alternatives = {}
    for side, record, partner in (("left", left, right), ("right", right, left)):
        number = alignment.segment_of.get((side, record.uuid)) if record else None
        if number is None:
            continue
        others = alignment.segments[number][1 if side == "left" else 0]
        other_side = context.other(side)
        partners = {
            uuid: records[side][state.partner]
            for uuid, state in alignment.states[other_side].items()
            if state.kind == "pair" and state.partner in records[side]
        }
        alternatives[side] = focus.alternatives(record, side, others, partner.uuid if partner else "", partners, subj_weight, rivals)
    manual = row["source"] in MANUAL_SOURCES or row["left_uuid"] in alignment.confirmed or row["right_uuid"] in alignment.confirmed
    return focus.Focus(
        kind=row["kind"],
        left=left,
        right=right,
        score=None if pd.isna(row["score"]) else float(row["score"]),
        source=row["source"],
        review=found,
        manual=bool(manual),
        local=bool({row["left_uuid"], row["right_uuid"]} & alignment.local),
        alternatives=alternatives,
    )


def neighbours(queue: list[int], current: int) -> tuple[int | None, int | None, int | None]:
    """(rang dans la file, précédente, suivante) de la ligne `current` ; hors
    de la file, les voisines sont prises dans l'ordre naturel."""
    if current in queue:
        position = queue.index(current)
        previous = queue[position - 1] if position > 0 else None
        following = queue[position + 1] if position + 1 < len(queue) else None
        return position, previous, following
    before = [index for index in queue if index < current]
    after = [index for index in queue if index > current]
    return None, (max(before) if before else None), (min(after) if after else None)


def picks_bar(pair_name: str, digest: str, records: dict[str, dict[str, Record]], alignment: Alignment) -> None:
    """Entrées sélectionnées dans les documents, et ce qu'on peut en faire."""
    picks = st.session_state.get(PICKS) or {}
    chosen = {side: records[side].get(picks.get(side, "")) for side in context.SIDES}
    if not any(chosen.values()):
        st.caption("Cliquer une entrée de chaque côté pour les apparier à la main.")
        return
    text, actions = st.columns([3, 2], vertical_alignment="center")
    described = " ⟷ ".join(f"« {record.text[:70]} »" if record else "…" for record in chosen.values())
    text.markdown(f"**Sélection** : {described}")
    buttons = actions.columns(3)
    decide_now = partial(on_decide, pair_name, digest, None)
    if all(chosen.values()):
        buttons[0].button(
            "Apparier",
            type="primary",
            width="stretch",
            on_click=decide_now,
            args=(SAME, chosen["left"], chosen["right"], None),
            help="Paire du patch ; les lignes qui touchaient ces deux entrées sont retirées.",
        )
        buttons[1].button("≈ incertaine", width="stretch", on_click=decide_now, args=(PROBABLE, chosen["left"], chosen["right"], None))
    else:
        side = next(side for side, record in chosen.items() if record)
        index = alignment.by_uuid[side].get(chosen[side].uuid)
        if index is not None:
            buttons[0].button(
                "Relire",
                width="stretch",
                on_click=on_move,
                args=(row_key(alignment.rows, index), REVIEW),
                help="Ouvrir la ligne de cette entrée dans la vue Relecture.",
            )
    buttons[2].button("Effacer", width="stretch", on_click=lambda: st.session_state.update({PICKS: {}}))


def documents_panel(
    docs: context.Documents, cursor: tuple[str, str], before: int, after: int, height: int, key: str, titles: tuple[str, str]
) -> None:
    centre = context.centers(docs, *cursor)
    bounds = context.window(docs, centre, before, after)
    picks = st.session_state.get(PICKS) or {}
    data = context.payload(docs, bounds, cursor, picks, height)
    pick = context.documents_diff(data, key, titles)
    if pick and pick.get("side") in context.SIDES:
        picks = dict(picks)
        if picks.get(pick["side"]) == pick["uuid"]:
            picks.pop(pick["side"])
        else:
            picks[pick["side"]] = pick["uuid"]
        st.session_state[PICKS] = picks
        st.rerun()


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def choose_alignment() -> Path | None:
    paths = sorted(path for suffix in paths_module.ALIGNMENT_SUFFIXES for path in paths_module.ALIGNMENTS_DIR.glob(f"*{suffix}"))
    if not paths:
        st.sidebar.warning(
            f"Aucun `*{'` / `*'.join(paths_module.ALIGNMENT_SUFFIXES)}` sous `{paths_module.ALIGNMENTS_DIR}/` : "
            "lancer `numrev align nw` ou `numrev align dedupe`."
        )
        return None
    return st.sidebar.selectbox("Alignement", paths, format_func=lambda p: p.name.removesuffix(".csv"))


def kpis(left_name: str, right_name: str, rows: pd.DataFrame, alignment: Alignment) -> None:
    counts = rows["kind"].value_counts()
    matched, candidates_kept = counts.get(PAIR, 0), counts.get(CANDIDATE, 0)
    n_left, n_right = matched + candidates_kept + counts.get(LEFT_ONLY, 0), matched + candidates_kept + counts.get(RIGHT_ONLY, 0)
    columns = st.columns(6)
    columns[0].metric("Correspondances", f"{matched:,}".replace(",", " "))
    columns[1].metric(
        f"Appariées à gauche ({left_name})", f"{matched / n_left:.1%}", f"{n_left - matched} sans correspondance", delta_color="off"
    )
    columns[2].metric(
        f"Appariées à droite ({right_name})", f"{matched / n_right:.1%}", f"{n_right - matched} sans correspondance", delta_color="off"
    )
    columns[3].metric(
        "Liens écartés",
        str(len(alignment.dropped)),
        help="Liens entre deux rubriques non appariées, écartés : on n'apparie qu'entre rubriques appariées "
        "(correspondance des rubriques et son patch).",
    )
    columns[4].metric(
        "Lignes du patch",
        str(len(alignment.entries)),
        f"{len(alignment.entries) - alignment.n_base:+d} non enregistrées" if alignment.local else None,
        delta_color="off",
    )
    levels = rows.loc[rows["kind"].isin([PAIR, CANDIDATE]), "level"].value_counts()
    columns[5].metric(
        "À vérifier",
        f"{levels.get(1, 0) + levels.get(2, 0)}",
        f"dont {levels.get(2, 0)} d'incertitude forte",
        delta_color="off",
        help="Correspondances et candidates non appariées d'incertitude moyenne ou forte (numrev/alignment/review.py).",
    )


def warnings_and_details(alignment: Alignment, records: dict[str, dict[str, Record]]) -> None:
    section_resolution = alignment.sections.resolution
    if section_resolution.reanchored:
        st.info(
            f"↻ {len(section_resolution.reanchored)} ligne(s) du patch des rubriques réancrée(s) par le titre ; "
            "le prochain alignement réécrira le patch."
        )
    if section_resolution.orphans:
        with st.expander(f"⚠ {len(section_resolution.orphans)} ligne(s) orpheline(s) du patch des rubriques, non appliquée(s)"):
            st.dataframe(pd.DataFrame([asdict(entry) for entry in section_resolution.orphans]), width="stretch")
    if alignment.n_missing:
        st.warning(
            f"{alignment.n_missing} correspondance(s) Dedupe désignent des entrées disparues (étapes amont modifiées) : "
            "ignorées ; relancer `numrev align dedupe`."
        )
    if alignment.reanchored:
        st.info(
            f"↻ {alignment.reanchored} ligne(s) du patch réancrée(s) par le texte ; "
            "`numrev align dedupe --patch-only --apply` réécrira le patch."
        )
    if alignment.orphans:
        with st.expander(f"⚠ {len(alignment.orphans)} ligne(s) orpheline(s) du patch, non appliquée(s)"):
            st.dataframe(pd.DataFrame([asdict(entry) for entry in alignment.orphans]), width="stretch")
    manual_dropped = [link for link in alignment.dropped if link.source in MANUAL_SOURCES]
    if manual_dropped:
        with st.expander(f"⚠ {len(manual_dropped)} paire(s) du patch entre rubriques non appariées, non appliquée(s)"):
            st.caption("On n'apparie qu'entre rubriques appariées : lier d'abord les deux rubriques dans le patch des rubriques.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "gauche": records["left"][link.left_uuid].text,
                            "rubrique gauche": records["left"][link.left_uuid].section_title,
                            "droite": records["right"][link.right_uuid].text,
                            "rubrique droite": records["right"][link.right_uuid].section_title,
                        }
                        for link in manual_dropped
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def sections_expander(alignment: Alignment, rows: pd.DataFrame, section_patch_path: Path) -> None:
    with st.expander("Rubriques : correspondance et appariement"):
        groups = alignment.sections.groups
        manual = sum(group.source != SOURCE_AUTO for group in groups)
        alone = len(alignment.sections.unmatched_left) + len(alignment.sections.unmatched_right)
        st.caption(
            f"{len(groups)} groupe(s) de rubriques appariées, dont {manual} du patch des rubriques ; {alone} rubrique(s) seule(s), "
            "dont les entrées ne sont jamais appariées. Pour corriger : une ligne `left_uuid,right_uuid` par paire dans le patch "
            f"des rubriques `{section_patch_path}` ({alignment.n_section_patch} ligne(s), édité à la main ; un même uuid sur "
            "plusieurs lignes forme un groupe 1-N), ou un seul uuid pour une rubrique sans correspondance."
        )
        st.html(
            f"{CSS}{COPY_SCRIPT}"
            f"{copy_button('copier l’en-tête du patch des rubriques', ','.join(SECTION_PATCH_FIELDS), 'Copier la ligne d’en-tête')}",
            unsafe_allow_javascript=True,
        )
        st.dataframe(section_table(alignment.sections, rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Annuaires — alignement", layout="wide")
    if PENDING_VIEW in st.session_state:
        st.session_state[VIEW] = st.session_state.pop(PENDING_VIEW)
    st.session_state.setdefault(VIEW, REVIEW)

    alignment_path = choose_alignment()
    if alignment_path is None:
        st.title("Annuaires — alignement")
        st.info("Aucun alignement à afficher.")
        return
    pair = Pair.of_file(alignment_path)
    pair_name, left_name, right_name = pair.name, pair.left, pair.right
    patch_path, section_patch_path = pair.entry_patch, pair.section_patch
    left_dir, right_dir = str(pair.left_dir), str(pair.right_dir)
    writable = os.environ.get(WRITABLE_ENV) == "1"
    digest = patch_digest(patch_path)
    restore_journal(pair_name, digest)

    # Barre latérale, du plus courant au plus fin. Les réglages fins sont lus
    # d'abord (ils entrent dans le calcul mis en cache) mais affichés en bas.
    journal_container = st.sidebar.container()
    filters_box, display_box, export_box = st.sidebar.container(), st.sidebar.container(), st.sidebar.container()
    params = Params()
    with st.sidebar.expander("Réglages fins de la relecture"):
        candidate_low = st.slider(
            "Similarité minimale d'une candidate non appariée",
            0.5,
            params.residual_threshold,
            params.threshold,
            step=0.01,
            help=f"Zone grise [seuil ; {params.residual_threshold}[ : par défaut le seuil de Needleman-Wunsch d'`numrev align nw`.",
        )
        margin = st.number_input(
            "Écart « homonyme proche »",
            0.0,
            0.5,
            DEFAULT_MARGIN,
            step=0.01,
            help="Signale une paire si une autre entrée du segment est à moins de cet écart de similarité.",
        )
    try:
        alignment = load_alignment(
            left_dir,
            right_dir,
            str(alignment_path),
            mtime(alignment_path),
            str(patch_path),
            mtime(patch_path),
            str(section_patch_path),
            mtime(section_patch_path),
            candidate_low,
            margin,
            journal_text(pair_name, digest),
        )
    except (ValueError, OSError) as error:
        st.error(f"Lecture impossible : {error}")
        with journal_container:
            journal_box(pair_name, patch_path, digest, writable)
        return
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    rows = alignment.rows
    with journal_container:
        journal_box(pair_name, patch_path, digest, writable)

    title, switch = st.columns([3, 2], vertical_alignment="center")
    title.subheader(f"{left_name} ⟷ {right_name}")
    view_name = switch.segmented_control("Vue", list(VIEWS), format_func=VIEWS.get, key=VIEW, label_visibility="collapsed", width="stretch")
    view_name = view_name or REVIEW
    st.caption(
        f"Alignement `{alignment_path}` · patch `{patch_path}` ({alignment.n_base} ligne(s) enregistrée(s)"
        + (f", {len(journal_of(pair_name))} décision(s) en attente)" if journal_of(pair_name) else ")")
    )
    warnings_and_details(alignment, records)

    # Filtres (communs aux vues Relecture et Table)
    with filters_box:
        st.header("Filtres")
        st.caption("Statut des lignes")
        kinds = [kind for kind, label in KIND_LABELS.items() if st.checkbox(label, value=True, key=f"kind_{kind}")]
        min_level = st.selectbox("Incertitude", list(LEVEL_OPTIONS), format_func=LEVEL_OPTIONS.get)
        sections = sorted({record.section for side in records.values() for record in side.values()})
        section = st.selectbox("Rubrique", [ALL_SECTIONS, *sections], format_func=lambda s: s or NO_SECTION)
        query = st.text_input("Recherche (texte des entrées)")
        low, high = st.slider("Score des correspondances", 0.0, 1.0, (0.0, 1.0), step=0.01)
        only_manual = st.checkbox("Seulement les corrections manuelles")
    with display_box:
        st.header("Affichage")
        if view_name == REVIEW:
            only_doubtful = st.toggle(
                "File : seulement les lignes à vérifier",
                value=True,
                help="Candidates non appariées et correspondances d'incertitude moyenne ou forte.",
            )
            hide_decided = st.toggle("File : masquer les lignes décidées", value=True)
            radius = st.slider("Contexte : lignes de part et d'autre", 3, 30, 8)
        order = st.selectbox("Tri", ORDER_OPTIONS)
        if view_name == TABLE:
            page_size = st.selectbox("Lignes par page", PAGE_SIZE_OPTIONS, index=1)
            low_score = st.slider("Score signalé en rouge sous", 0.0, 1.0, 0.8, step=0.01)
    with export_box:
        st.header("Export")
        excel = st.checkbox("Pour un tableur en français", value=True, help="Séparateur `;` et UTF-8 avec BOM, qu'Excel ouvre directement.")

    view = rows[rows["kind"].isin(kinds)]
    view = view[(view["kind"] != PAIR) | view["score"].between(low, high) | view["score"].isna()]
    if section != ALL_SECTIONS:
        view = view[(view["left_section"] == section) | (view["right_section"] == section)]
    if query:
        view = view[text_mask(view, query, ["left_markdown", "right_markdown"])]
    if only_manual:
        view = view[decided_mask(view, alignment.confirmed)]
    if min_level:
        view = view[view["level"] >= min_level]
    if order == LEVEL_ORDER:
        view = view.sort_values("level", ascending=False, kind="stable")
    elif order != NATURAL_ORDER:
        view = view.sort_values("score", ascending=order == "score croissant", kind="stable", na_position="last")

    export_box.download_button(
        f"Exporter en CSV ({len(view):,} lignes)".replace(",", " "),
        # Généré au clic seulement : lignes affichées, dans l'ordre affiché.
        lambda: export_csv([alignment.joined[index] for index in view.index], excel=excel, reviews=alignment.reviews).encode(
            export_encoding(excel)
        ),
        file_name=f"{pair_name}{JOIN_SUFFIX}",
        mime="text/csv",
        icon=":material/download:",
        help="Jointure lisible des deux annuaires (texte sans Markdown, empans NER en colonnes), filtres et tri appliqués.",
        width="stretch",
    )
    legend = "".join(badge(label, colors) for label, colors in LABEL_COLORS.items())
    st.html(f"{CSS}<div class='legend'>{legend}</div>")

    current = cursor_index(alignment, st.session_state.get(CURSOR))
    if view_name == REVIEW:
        to_review = view
        if only_doubtful:
            to_review = to_review[(to_review["level"] >= 1) | (to_review["kind"] == CANDIDATE)]
        decided = decided_mask(to_review, alignment.confirmed)
        queue = list(to_review.index[~decided]) if hide_decided else list(to_review.index)
        if current is None and queue:
            current = queue[0]
            st.session_state[CURSOR] = row_key(rows, current)
        if current is None:
            st.success("Rien à relire avec ces filtres.")
            return
        position, previous, following = neighbours(queue, current)
        key_of = partial(row_key, rows)
        following_key = key_of(following) if following is not None else None
        controls = focus.Controls(
            decide=partial(on_decide, pair_name, digest, following_key),
            move=on_move,
            undo_last=partial(on_undo_last, pair_name),
            position=position,
            remaining=len(queue),
            total=len(to_review),
            decided=int(decided.sum()),
            previous=key_of(previous) if previous is not None else None,
            following=following_key,
            can_undo=bool(journal_of(pair_name)),
        )
        focus.queue_header(controls)
        if not queue:
            st.success("File vide : tout est décidé avec ces filtres. La ligne courante reste affichée.")
        focused = focus_of(alignment, records, current, params.subj_weight)
        focus.render(focused, controls, key=f"{pair_name}:{rows.at[current, 'left_uuid']}:{rows.at[current, 'right_uuid']}")
        st.subheader("Contexte", divider="gray")
        picks_bar(pair_name, digest, records, alignment)
        docs = context.documents({"left": load_document_lines(left_dir), "right": load_document_lines(right_dir)}, alignment.states)
        documents_panel(docs, row_key(rows, current), radius, radius, 360 + 18 * radius, "context_review", (left_name, right_name))
    elif view_name == TABLE:
        kpis(left_name, right_name, rows, alignment)
        sections_expander(alignment, rows, section_patch_path)
        # Retour à la première page quand la sélection change.
        selection = (alignment_path.name, tuple(kinds), section, query, low, high, only_manual, order, page_size, min_level)
        if st.session_state.get("_selection") != selection:
            st.session_state["_selection"] = selection
            st.session_state.page = 0
        action = paginated_table(
            view,
            ["#", f"gauche — {left_name}", f"droite — {right_name}", "statut"],
            lambda page, first: table_rows(page, first, low_score, order == NATURAL_ORDER, alignment, current),
            page_size,
        )
        if action and action.get("action") == "focus":
            left_uuid, _, right_uuid = action.get("value", "").partition("|")
            st.session_state[CURSOR] = (left_uuid, right_uuid)
            st.session_state[PENDING_VIEW] = REVIEW
            st.rerun()
    else:
        docs = context.documents({"left": load_document_lines(left_dir), "right": load_document_lines(right_dir)}, alignment.states)
        if current is None:
            current = 0
        navigation = st.columns([0.8, 0.8, 3, 1.4], vertical_alignment="bottom")
        navigation[0].button(
            "▲",
            shortcut="PageUp",
            width="stretch",
            disabled=current == 0,
            help=f"Remonter de {DOCUMENTS_STEP} lignes.",
            on_click=on_move,
            args=(row_key(rows, max(0, current - DOCUMENTS_STEP)),),
        )
        navigation[1].button(
            "▼",
            shortcut="PageDown",
            width="stretch",
            disabled=current >= len(rows) - 1,
            help=f"Descendre de {DOCUMENTS_STEP} lignes.",
            on_click=on_move,
            args=(row_key(rows, min(len(rows) - 1, current + DOCUMENTS_STEP)),),
        )
        firsts = rows.drop_duplicates("left_section").query("left_uuid != ''")
        jumps = {f"{row.left_section_title or NO_SECTION}": (row.left_uuid, row.right_uuid) for row in firsts.itertuples()}
        navigation[2].selectbox(
            "Aller à la rubrique",
            list(jumps),
            index=None,
            placeholder="Aller à la rubrique…",
            key="documents_jump",
            on_change=lambda: on_move(jumps.get(st.session_state.get("documents_jump"))),
        )
        navigation[3].button(
            "Relire cette ligne",
            icon=":material/rate_review:",
            width="stretch",
            on_click=on_move,
            args=(row_key(rows, current), REVIEW),
        )
        picks_bar(pair_name, digest, records, alignment)
        documents_panel(docs, row_key(rows, current), 25, 60, 720, "documents", (left_name, right_name))


if __name__ == "__main__":
    main()

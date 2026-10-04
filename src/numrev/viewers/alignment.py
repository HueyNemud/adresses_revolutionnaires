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

**Deux niveaux de lecture, un curseur et une file de tâches communs.**

- **Documents** (vue d'accueil, `numrev/viewers/context.py`) : les deux
  annuaires côte à côte, chacun dans son ordre, avec titres et lignes hors
  sujet ; les paires reliées dans une gouttière (inversions en orange), les
  tâches de relecture marquées « ! ». Un clic sur une entrée ou sur un lien
  en fait la ligne courante ; les boutons, au-dessus, permettent de décider
  sur place, et
  « Autre partenaire… » fait choisir le partenaire dans les documents ;
- **Relecture** (`numrev/viewers/focus.py`) : le zoom sur la ligne courante —
  différences de texte surlignées, rapprochements possibles, note — et
  l'avance automatique à la tâche suivante après chaque décision. On y entre
  par l'onglet ou par la loupe posée sur le lien courant.

Le bandeau de tâches, commun, mène d'une tâche à l'autre (P / N), annule la
dernière décision (Ctrl+Z) ; les onglets en haut changent de vue. Les décisions : V même entrée (ou confirmer sans
correspondance), I incertaine, X pas la même entrée, A apparier autrement.
La **file de tâches** se règle dans la barre latérale (candidates,
incertitude moyenne ou forte, entrées seules, paires de score faible,
rubrique, lignes déjà décidées, ordre).

**Décisions** (`numrev/alignment/decisions.py`) : elles vont dans un journal
gardé dans le navigateur (`numrev/viewers/storage.py`), rejoué sur le patch
versionné. En local (`numrev view alignment` pose `NUMREV_PATCH_WRITABLE=1`),
**Enregistrer** écrit le patch et vide le journal ; sinon (copie hébergée,
`numrev publish-viewer`), **Télécharger le patch** donne le patch complet à
déposer dans `data/alignment/`. Le format du patch ne change pas.

On n'apparie qu'entre rubriques appariées (`numrev/alignment/sections.py`, avec
son patch `data/alignment/<gauche>__<droite>.sections.csv`, édité à la main,
non réécrit ici) : un lien entre rubriques qui ne se correspondent pas —
Dedupe en produit — est écarté, et compté. La **Synthèse** (en-tête) donne
les indicateurs et, par groupe de rubriques appariées puis par rubrique
seule, la correspondance et la part d'entrées appariées ; au survol d'un
titre, le bouton **uuid** copie l'uuid de la rubrique pour son patch.

**Motifs** (`numrev/alignment/review.py`) : chaque correspondance porte une
incertitude (faible, moyenne ou forte) et ses motifs (`déduite des voisines
(p < 0,9)`, `homonyme proche`) ; des **candidates non appariées** (deux
entrées sans correspondance, chacune la plus proche de l'autre, dans la zone
grise de similarité) sont soumises au relecteur, jamais appariées d'office.

**Exporter en CSV** télécharge la jointure lisible
(`numrev/alignment/export.py`, comme `numrev join`), complète ou réduite aux
tâches de la file.
"""

import csv
import html
import io
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from numrev import paths as paths_module
from numrev.alignment.decisions import (
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
from numrev.alignment.export import (
    CANDIDATE,
    LEFT_ONLY,
    PAIR,
    RIGHT_ONLY,
    JoinedRow,
    NaturalOrder,
    export_csv,
    export_encoding,
    natural_order,
)
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
from numrev.alignment.records import (
    SOURCE_CANDIDATE,
    SOURCE_MANUAL,
    SOURCE_MANUAL_UNCERTAIN,
    DocLine,
    Link,
    Record,
    load_lines,
    load_volume,
    read_links,
)
from numrev.alignment.review import DEFAULT_MARGIN, Review, Reviewer
from numrev.alignment.sections import (
    SECTION_PATCH_FIELDS,
    SOURCE_AUTO,
    SectionAlignment,
    load_section_alignment,
    read_section_patch,
    restrict_to_corresponding,
    segment_entries,
)
from numrev.ner.html import SPAN_CSS
from numrev.paths import JOIN_SUFFIX, Pair
from numrev.viewers import context, focus
from numrev.viewers.common import text_mask
from numrev.viewers.storage import browser_journal, storage_key

WRITABLE_ENV = "NUMREV_PATCH_WRITABLE"  # posée par `numrev view alignment` : enregistrement direct du patch
DOCUMENTS, REVIEW = "Documents", "Relecture"
VIEWS = {DOCUMENTS: ":material/difference: Documents", REVIEW: ":material/zoom_in: Relecture"}
ALL_SECTIONS = "(toutes)"
NO_SECTION = "(sans rubrique)"
MANUAL_SOURCES = {SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN}
ROW_COLUMNS = ("uuid", "section", "section_title", "section_uuid", "markdown")  # champs de Record gardés par ligne, de chaque côté
WINDOW = {DOCUMENTS: (30, 70), REVIEW: (8, 8)}  # lignes affichées avant / après la ligne courante
HEIGHT = {DOCUMENTS: "68vh", REVIEW: "430px"}
MORE_STEP = 40  # lignes ajoutées par « ⋯ » (assets/context.js)
DOCUMENTS_ORDER, LEVEL_ORDER = "ordre des documents", "incertitude décroissante"

# Clés de st.session_state
CURSOR = "cursor"  # (uuid gauche, uuid droit) de la ligne courante
VIEW = "view"
PENDING_VIEW = "pending_view"  # vue à afficher au prochain tour (posée hors rappel)
PAIRING = "pairing"  # mode « choisir le partenaire »
WINDOW_EXTRA = "window_extra"  # lignes ajoutées par « ⋯ » autour de la ligne courante
SEARCH = "search"

CSS = f"""<style>
  .legend span {{ margin-right: 10px; }}
  button.copy {{ font-size: 0.72em; padding: 0 6px; margin-left: 4px; border: 1px solid rgba(100,116,139,.45);
                border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }}
  button.copy:hover {{ background: rgba(100,116,139,.12); }}
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


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Lecture de l'annuaire…")
def load_records(volume_dir: str) -> dict[str, Record]:
    return {record.uuid: record for record in load_volume(Path(volume_dir))}


@st.cache_resource(show_spinner="Lecture des documents…")
def load_document_lines(volume_dir: str) -> list[DocLine]:
    return load_lines(Path(volume_dir))


@st.cache_resource
def load_positions(volume_dir: str) -> dict[str, int]:
    """uuid → rang de la ligne dans `load_document_lines`."""
    return {line.uuid: index for index, line in enumerate(load_document_lines(volume_dir))}


# Le calcul est découpé pour qu'une décision coûte peu, même sur des annuaires
# de 100 000 entrées et plus : `load_base` (rubriques, segments, liens bruts)
# ne dépend pas des décisions ; `load_reviewer` garde les motifs de chaque
# segment et ne recalcule que les segments touchés (`Reviewer`) ;
# `load_alignment` rejoue le patch et le journal et reconstruit les lignes,
# colonne par colonne. Les fichiers entrent dans la clé de cache par leur
# date de modification (`*_mtime`).
@dataclass
class Base:
    """Ce qui ne dépend pas du patch des entrées ni des décisions."""

    links: list[Link]  # liens de l'alignement brut (Dedupe ou `numrev align nw`)
    sections: SectionAlignment  # correspondance des rubriques (avec leur patch)
    n_section_patch: int
    segments: list[tuple[list[Record], list[Record]]]  # entrées de chaque segment (rubriques appariées)
    segment_of: dict[tuple[str, str], int]  # (côté, uuid) → rang du segment
    section_keys: list[str]  # clés de rubrique des deux annuaires, triées
    records: dict[str, list[Record]]  # côté → entrées dans l'ordre de l'annuaire
    rank: dict[str, dict[str, int]]  # côté → uuid → rang dans `records`
    columns: dict[str, dict[str, np.ndarray]]  # côté → champ de ROW_COLUMNS → valeurs dans l'ordre, plus "" en dernier (rang -1)


@st.cache_resource(show_spinner="Correspondance des rubriques…", max_entries=2)
def load_base(
    left_dir: str, right_dir: str, dedupe_path: str, dedupe_mtime: float, section_patch_path: str, section_patch_mtime: float
) -> Base:
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    sections = load_section_alignment(list(records["left"].values()), list(records["right"].values()), Path(section_patch_path))
    segments = segment_entries(sections)
    ordered = {side: sorted(side_records.values(), key=lambda record: record.order) for side, side_records in records.items()}
    return Base(
        links=read_links(Path(dedupe_path)),
        sections=sections,
        n_section_patch=len(read_section_patch(Path(section_patch_path))),
        segments=segments,
        segment_of={
            (side, record.uuid): number
            for number, pair in enumerate(segments)
            for side, side_records in zip(context.SIDES, pair)
            for record in side_records
        },
        section_keys=sorted({record.section for side in records.values() for record in side.values()}),
        records=ordered,
        rank={side: {record.uuid: rank for rank, record in enumerate(side_records)} for side, side_records in ordered.items()},
        columns={
            side: {column: np.array([getattr(record, column) for record in side_records] + [""], dtype=object) for column in ROW_COLUMNS}
            for side, side_records in ordered.items()
        },
    )


@st.cache_resource(show_spinner="Similarités des entrées…", max_entries=2)
def load_reviewer(
    left_dir: str,
    right_dir: str,
    dedupe_path: str,
    dedupe_mtime: float,
    section_patch_path: str,
    section_patch_mtime: float,
    candidate_low: float,
    margin: float,
) -> Reviewer:
    base = load_base(left_dir, right_dir, dedupe_path, dedupe_mtime, section_patch_path, section_patch_mtime)
    params = Params()
    return Reviewer(base.sections, candidate_low, params.residual_threshold, params.subj_weight, margin)


class EntryStates(Mapping):
    """État de chaque entrée alignable d'un côté (uuid → `EntryState`, vue
    Documents), calculé à la demande depuis les lignes : seules les entrées
    affichées sont consultées."""

    KINDS = {PAIR: "pair", CANDIDATE: "candidate", LEFT_ONLY: "alone", RIGHT_ONLY: "alone"}

    def __init__(self, side: str, columns: dict[str, list], index: dict[str, int], confirmed: set[str], local: set[str]):
        self.columns, self.index, self.confirmed, self.local = columns, index, confirmed, local
        self.partners = columns[f"{context.other(side)}_uuid"]

    def __getitem__(self, uuid: str) -> context.EntryState:
        row = self.index[uuid]
        manual = self.columns["source"][row] in MANUAL_SOURCES or uuid in self.confirmed
        return context.EntryState(self.KINDS[self.columns["kind"][row]], self.partners[row], manual, uuid in self.local)

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)


@dataclass
class Alignment:
    rows: pd.DataFrame  # sortie de `build_rows`
    order: NaturalOrder  # mêmes lignes, même ordre (index de `rows`)
    records: dict[str, list[Record]]  # côté → entrées dans l'ordre (rangs de `order`)
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
    states: dict[str, EntryStates]  # côté → uuid → état (vue Documents)
    jumps: dict[str, tuple[str, str]]  # titre de rubrique (gauche) → clé de sa première ligne

    def joined(self, indices: list[int]) -> list[JoinedRow]:
        """Lignes `indices` de la jointure, pour l'export."""
        left, right, link = self.order.left, self.order.right, self.order.link
        return [
            JoinedRow(
                self.records["left"][left[index]] if left[index] >= 0 else None,
                self.records["right"][right[index]] if right[index] >= 0 else None,
                self.order.links[link[index]] if link[index] >= 0 else None,
            )
            for index in indices
        ]


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
    l'un des fichiers ou le journal change. Lève ValueError si un patch est
    incohérent. Le patch des rubriques n'est pas réécrit ici."""
    files = (left_dir, right_dir, dedupe_path, dedupe_mtime, section_patch_path, section_patch_mtime)
    base = load_base(*files)
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    patch = read_patch(Path(patch_path))
    validate(patch)
    decisions = from_json(journal)[0] if journal else []
    entries = apply_decisions(patch, decisions)
    validate(entries)
    resolution = resolve(entries, records["left"], records["right"])
    links, _ = apply_patch(base.links, resolution.entries)
    links, dropped = restrict_to_corresponding(links, records["left"], records["right"], base.sections)
    declared = declared_unmatched(resolution)
    found = load_reviewer(*files, candidate_low, margin)(links, declared)
    order = natural_order(links + found.candidates, base.rank["left"], base.rank["right"])
    rows = build_rows(order, found.reviews, base.columns)
    local = touched_uuids(decisions)
    columns = {column: rows[column].tolist() for column in ("left_uuid", "right_uuid", "kind", "source")}
    by_uuid = {side: {uuid: index for index, uuid in enumerate(columns[f"{side}_uuid"]) if uuid} for side in context.SIDES}
    firsts = rows.drop_duplicates("left_section").query("left_uuid != ''")
    return Alignment(
        rows=rows,
        order=order,
        records=base.records,
        reviews=found.reviews,
        entries=entries,
        n_base=len(patch),
        n_missing=order.missing,
        reanchored=len(resolution.reanchored),
        orphans=resolution.orphans,
        confirmed=declared,
        local=local,
        sections=base.sections,
        n_section_patch=base.n_section_patch,
        dropped=dropped,
        by_key=dict(zip(zip(columns["left_uuid"], columns["right_uuid"]), range(len(rows)))),
        by_uuid=by_uuid,
        segments=base.segments,
        segment_of=base.segment_of,
        states={side: EntryStates(side, columns, by_uuid[side], declared, local) for side in context.SIDES},
        jumps={
            title or NO_SECTION: (left, right)
            for title, left, right in zip(firsts["left_section_title"], firsts["left_uuid"], firsts["right_uuid"])
        },
    )


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def build_rows(order: NaturalOrder, reviews: dict[tuple[str, str], Review], columns: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Une ligne par ligne de la jointure (ordre naturel), construite par
    indexation de tableaux : `left_*` / `right_*` (`ROW_COLUMNS`, vides du
    côté absent : le rang -1 désigne le "" final de `columns`), score,
    source, statut et niveau d'incertitude."""
    data: dict[str, np.ndarray] = {}
    for side, ranks in (("left", order.left), ("right", order.right)):
        for column in ROW_COLUMNS:
            data[f"{side}_{column}"] = columns[side][column][ranks]
    # Valeurs par lien, plus une dernière pour les lignes sans lien (indice -1).
    links = order.links
    score = np.array([link.score if link.score is not None else np.nan for link in links] + [np.nan], dtype=float)
    source = np.array([link.source for link in links] + [""], dtype=object)
    level = np.zeros(len(links) + 1, dtype=np.int64)
    index = {(link.left_uuid, link.right_uuid): number for number, link in enumerate(links)}
    for key, found in reviews.items():
        if key in index:
            level[index[key]] = found.level
    linked = order.link >= 0
    data["score"] = score[order.link]
    data["source"] = source[order.link]
    data["kind"] = np.where(
        linked, np.where(data["source"] == SOURCE_CANDIDATE, CANDIDATE, PAIR), np.where(order.left >= 0, LEFT_ONLY, RIGHT_ONLY)
    ).astype(object)
    data["level"] = level[order.link]
    # Colonnes de texte en `object` : la conversion en chaînes Arrow coûterait plus que tout le reste.
    return pd.DataFrame({name: pd.Series(values, dtype=values.dtype, copy=False) for name, values in data.items()})


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
    st.session_state[PAIRING] = False
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
# Ligne courante, synthèse
# ----------------------------------------------------------------------
def focus_of(
    alignment: Alignment, records: dict[str, dict[str, Record]], index: int, subj_weight: float, with_alternatives: bool = True
) -> focus.Focus:
    """La ligne `index` pour la vue Relecture, avec ses rapprochements possibles."""
    row = alignment.rows.loc[index]
    left = records["left"].get(row["left_uuid"]) if row["left_uuid"] else None
    right = records["right"].get(row["right_uuid"]) if row["right_uuid"] else None
    found = alignment.reviews.get((row["left_uuid"], row["right_uuid"]))
    rivals = {uuid for _, uuid, _ in found.rivals} if found else set()
    alternatives = {}
    for side, record, partner in (("left", left, right), ("right", right, left)):
        number = alignment.segment_of.get((side, record.uuid)) if record else None
        if number is None or not with_alternatives:
            continue
        others = alignment.segments[number][1 if side == "left" else 0]
        other_side = context.other(side)
        partners = {}
        for other in others:
            state = alignment.states[other_side].get(other.uuid)
            if state is not None and state.kind == "pair" and state.partner in records[side]:
                partners[other.uuid] = records[side][state.partner]
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


def copy_button(label: str, text: str, title: str) -> str:
    return f'<button class="copy" data-copy="{html.escape(text, quote=True)}" title="{html.escape(title)}">{label}</button>'


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


# ----------------------------------------------------------------------
# File de tâches
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TaskOptions:
    """Ce qui fait d'une ligne une tâche de relecture (barre latérale)."""

    candidates: bool = True  # candidates non appariées
    medium: bool = True  # paires automatiques d'incertitude moyenne
    high: bool = True  # … forte
    left_only: bool = False  # entrées seules à gauche
    right_only: bool = False  # … à droite
    below: float | None = None  # paires automatiques de score inférieur
    section: str | None = None  # clé de rubrique (gauche ou droite)
    include_decided: bool = False  # revoir aussi les lignes décidées
    by_level: bool = False  # ordre : incertitude décroissante (sinon ordre des documents)


def task_levels(rows: pd.DataFrame, confirmed: set[str], options: TaskOptions) -> pd.Series:
    """Niveau de tâche de chaque ligne : 0 (pas une tâche), 1 ou 2 (marqueur
    « ! » orange ou rouge dans les documents)."""
    kind, level = rows["kind"], rows["level"]
    decided = decided_mask(rows, confirmed)
    automatic = (kind == PAIR) & ~decided
    tasks = pd.Series(0, index=rows.index)
    if options.candidates:
        tasks[kind == CANDIDATE] = 2
    if options.medium:
        tasks[automatic & (level == 1)] = 1
    if options.high:
        tasks[automatic & (level == 2)] = 2
    for wanted, alone in ((options.left_only, LEFT_ONLY), (options.right_only, RIGHT_ONLY)):
        if wanted:
            tasks[(kind == alone) & ~decided & (tasks == 0)] = 1
    if options.below is not None:
        tasks[automatic & (rows["score"] < options.below) & (tasks == 0)] = 1
    tasks[decided] = 1 if options.include_decided else 0
    if options.section is not None:
        tasks[(rows["left_section"] != options.section) & (rows["right_section"] != options.section)] = 0
    return tasks


def task_queue(levels: pd.Series, by_level: bool) -> list[int]:
    """Index des lignes à relire, dans l'ordre des documents ou par
    incertitude décroissante."""
    tasks = levels[levels > 0]
    if by_level:
        tasks = tasks.sort_values(ascending=False, kind="stable")
    return list(tasks.index)


def task_marks(rows: pd.DataFrame, levels: pd.Series) -> dict[str, dict[str, int]]:
    """Côté → uuid → niveau de tâche, pour les marqueurs des documents."""
    marks: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    tasks = rows[levels > 0]
    for side in context.SIDES:
        for uuid, level in zip(tasks[f"{side}_uuid"], levels[levels > 0]):
            if uuid:
                marks[side][uuid] = int(level)
    return marks


def task_options() -> TaskOptions:
    """Section « Tâches de relecture » de la barre latérale."""
    st.header("Tâches de relecture")
    candidates = st.checkbox("Candidates non appariées", value=True)
    medium = st.checkbox("Paires d'incertitude moyenne", value=True)
    high = st.checkbox("Paires d'incertitude forte", value=True)
    left_only = st.checkbox("Entrées seules à gauche")
    right_only = st.checkbox("Entrées seules à droite")
    low = st.checkbox("Paires automatiques de score faible")
    below = st.slider("… score inférieur à", 0.0, 1.0, 0.7, step=0.01) if low else None
    return TaskOptions(
        candidates=candidates,
        medium=medium,
        high=high,
        left_only=left_only,
        right_only=right_only,
        below=below,
        include_decided=st.toggle("Revoir aussi les lignes décidées", help="Les paires relues et les entrées confirmées seules."),
        by_level=st.radio("Ordre", [DOCUMENTS_ORDER, LEVEL_ORDER], horizontal=True) == LEVEL_ORDER,
    )


# ----------------------------------------------------------------------
# Curseur, mode d'appariement, événements des documents
# ----------------------------------------------------------------------
def set_cursor(key: tuple[str, str]) -> None:
    st.session_state[CURSOR] = key
    st.session_state[PAIRING] = False


def on_move(key: tuple[str, str] | None = None, view: str | None = None) -> None:
    if key is not None:
        set_cursor(key)
    if view is not None:
        st.session_state[VIEW] = view
        st.session_state[PAIRING] = False


def on_pairing(active: bool) -> None:
    st.session_state[PAIRING] = active


def window_extra(cursor: tuple[str, str]) -> tuple[int, int]:
    """Lignes ajoutées par « ⋯ » (avant, après), tant que le curseur ne bouge pas."""
    extra = st.session_state.get(WINDOW_EXTRA)
    if not extra or tuple(extra["cursor"]) != tuple(cursor):
        return 0, 0
    return extra["before"], extra["after"]


def eligible(alignment: Alignment, cursor: tuple[str, str]) -> dict[str, set[str]]:
    """Mode « choisir le partenaire » : entrées cliquables de chaque côté,
    celles du segment (rubriques appariées) de l'entrée d'en face."""
    found: dict[str, set[str]] = {"left": set(), "right": set()}
    for side, uuid in zip(context.SIDES, cursor):
        number = alignment.segment_of.get((side, uuid)) if uuid else None
        if number is not None:
            other = context.other(side)
            found[other] = {record.uuid for record in alignment.segments[number][1 if side == "left" else 0]} - set(cursor)
    return found


def handle_event(event: tuple[str, dict], alignment: Alignment, records: dict, pair_name: str, digest: str) -> None:
    """Clic dans les documents : ligne courante, appariement ou fenêtre agrandie."""
    name, value = event
    value = value if isinstance(value, dict) else {}
    cursor = tuple(st.session_state.get(CURSOR) or ("", ""))
    side, uuid = value.get("side"), value.get("uuid")
    if name == "focus" and side in context.SIDES:
        index = alignment.by_uuid[side].get(uuid)
        if index is not None:
            set_cursor(row_key(alignment.rows, index))
    elif name == "pair" and side in context.SIDES:
        left = records["left"].get(uuid if side == "left" else cursor[0])
        right = records["right"].get(uuid if side == "right" else cursor[1])
        if left is not None and right is not None:
            on_decide(pair_name, digest, None, SAME, left, right, None)
        st.session_state[PAIRING] = False
    elif name == "zoom":
        st.session_state[PENDING_VIEW] = REVIEW
    elif name == "more":
        before, after = window_extra(cursor)
        if value.get("dir", 1) < 0:
            before += MORE_STEP
        else:
            after += MORE_STEP
        st.session_state[WINDOW_EXTRA] = {"cursor": list(cursor), "before": before, "after": after}
    st.rerun()


def documents_panel(
    docs: context.Documents, cursor: tuple[str, str], view_name: str, marks: context.Marks, titles: tuple[str, str], info: str
) -> tuple[str, dict] | None:
    before, after = WINDOW[view_name]
    more_before, more_after = window_extra(cursor)
    bounds = context.window(docs, context.centers(docs, *cursor), before + more_before, after + more_after)
    data = context.payload(docs, bounds, cursor, marks, HEIGHT[view_name], info)
    return context.documents_diff(data, "documents_diff", titles)


# ----------------------------------------------------------------------
# Synthèse
# ----------------------------------------------------------------------
@st.dialog("Synthèse de l'alignement", width="large")
def synthesis(left_name: str, right_name: str, alignment: Alignment, section_patch_path: Path) -> None:
    kpis(left_name, right_name, alignment.rows, alignment)
    st.subheader("Rubriques : correspondance et appariement")
    groups = alignment.sections.groups
    manual = sum(group.source != SOURCE_AUTO for group in groups)
    alone = len(alignment.sections.unmatched_left) + len(alignment.sections.unmatched_right)
    st.caption(
        f"{len(groups)} groupe(s) de rubriques appariées, dont {manual} du patch des rubriques ; {alone} rubrique(s) seule(s), "
        "dont les entrées ne sont jamais appariées. Pour corriger : une ligne `left_uuid,right_uuid` par paire dans le patch "
        f"des rubriques `{section_patch_path}` ({alignment.n_section_patch} ligne(s), édité à la main ; un même uuid sur "
        "plusieurs lignes forme un groupe 1-N), ou un seul uuid pour une rubrique sans correspondance. L'uuid d'une rubrique "
        "se copie au survol de son titre dans les documents."
    )
    st.html(
        f"{CSS}{COPY_SCRIPT}"
        f"{copy_button('copier l’en-tête du patch des rubriques', ','.join(SECTION_PATCH_FIELDS), 'Copier la ligne d’en-tête')}",
        unsafe_allow_javascript=True,
    )
    st.dataframe(section_table(alignment.sections, alignment.rows), width="stretch", hide_index=True)


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Annuaires — alignement", layout="wide")
    if PENDING_VIEW in st.session_state:
        st.session_state[VIEW] = st.session_state.pop(PENDING_VIEW)
    st.session_state.setdefault(VIEW, DOCUMENTS)

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
    journal_container, tasks_box, export_box = st.sidebar.container(), st.sidebar.container(), st.sidebar.container()
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

    # File de tâches
    with tasks_box:
        options = task_options()
        section_keys = load_base(
            left_dir, right_dir, str(alignment_path), mtime(alignment_path), str(section_patch_path), mtime(section_patch_path)
        ).section_keys
        section = st.selectbox("Rubrique", [ALL_SECTIONS, *section_keys], format_func=lambda s: s or NO_SECTION)
    if section != ALL_SECTIONS:
        options = TaskOptions(**{**asdict(options), "section": section})
    levels = task_levels(rows, alignment.confirmed, options)
    queue = task_queue(levels, options.by_level)
    with export_box:
        st.header("Export")
        only_tasks = st.checkbox("Seulement les tâches de la file")
        excel = st.checkbox("Pour un tableur en français", value=True, help="Séparateur `;` et UTF-8 avec BOM, qu'Excel ouvre directement.")
        exported = queue if only_tasks else list(rows.index)
        st.download_button(
            f"Exporter en CSV ({len(exported):,} lignes)".replace(",", " "),
            # Généré au clic seulement.
            lambda: export_csv(alignment.joined(exported), excel=excel, reviews=alignment.reviews).encode(export_encoding(excel)),
            file_name=f"{pair_name}{JOIN_SUFFIX}",
            mime="text/csv",
            icon=":material/download:",
            help="Jointure lisible des deux annuaires (texte sans Markdown, empans NER en colonnes), au format de `numrev join`.",
            width="stretch",
        )

    # Ligne courante : par défaut la première tâche.
    current = cursor_index(alignment, st.session_state.get(CURSOR))
    if current is None:
        current = queue[0] if queue else 0
    cursor = row_key(rows, current)
    if tuple(st.session_state.get(CURSOR) or ()) != cursor:
        st.session_state[CURSOR] = cursor

    # En-tête
    title, switch, summary = st.columns([3.2, 2, 0.9], vertical_alignment="center")
    title.markdown(f"#### {left_name} ⟷ {right_name}")
    view_name = switch.segmented_control("Vue", list(VIEWS), format_func=VIEWS.get, key=VIEW, label_visibility="collapsed", width="stretch")
    view_name = view_name or DOCUMENTS
    if summary.button("Synthèse", icon=":material/monitoring:", width="stretch"):
        synthesis(left_name, right_name, alignment, section_patch_path)
    warnings_and_details(alignment, records)

    position, previous, following = neighbours(queue, current)
    key_of = partial(row_key, rows)
    following_key = key_of(following) if following is not None else None
    pairing_active = bool(st.session_state.get(PAIRING))
    controls = focus.Controls(
        decide=partial(on_decide, pair_name, digest, following_key if view_name == REVIEW else None),
        move=on_move,
        undo_last=partial(on_undo_last, pair_name),
        pairing=on_pairing,
        position=position,
        remaining=len(queue),
        decided=int(decided_mask(rows, alignment.confirmed).sum()),
        pending=len(journal_of(pair_name)),
        previous=key_of(previous) if previous is not None else None,
        following=following_key,
        can_undo=bool(journal_of(pair_name)),
        pairing_active=pairing_active,
    )
    focus.task_bar(controls)

    focused = focus_of(alignment, records, current, params.subj_weight, with_alternatives=view_name == REVIEW)
    if view_name == DOCUMENTS:
        focus.decision_buttons(focused, controls, None)
        jump, search, matches_box = st.columns([2.2, 2.2, 1.6], vertical_alignment="bottom")
        jumps = alignment.jumps
        jump.selectbox(
            "Aller à la rubrique",
            list(jumps),
            index=None,
            placeholder="Aller à la rubrique…",
            key="documents_jump",
            label_visibility="collapsed",
            on_change=lambda: on_move(jumps.get(st.session_state.get("documents_jump"))),
        )
        query = search.text_input("Rechercher", key=SEARCH, placeholder="Rechercher dans les deux annuaires…", label_visibility="collapsed")
        hits: dict[str, set[str]] = {"left": set(), "right": set()}
        if query:
            for side in context.SIDES:
                found = rows[text_mask(rows, query, [f"{side}_markdown"])]
                hits[side] = set(found[f"{side}_uuid"]) - {""}
            matches = list(rows.index[text_mask(rows, query, ["left_markdown", "right_markdown"])])
            rank, before, after = neighbours(matches, current)
            back, label, forth = matches_box.columns([1, 1.4, 1], vertical_alignment="center")
            back.button(
                "◀", key="hit_previous", disabled=before is None, on_click=on_move, args=(key_of(before) if before is not None else None,)
            )
            label.caption(f"{rank + 1} / {len(matches)}" if rank is not None else f"{len(matches)} résultat(s)")
            forth.button(
                "▶", key="hit_next", disabled=after is None, on_click=on_move, args=(key_of(after) if after is not None else None,)
            )
        marks = context.Marks(task_marks(rows, levels), hits, eligible(alignment, cursor) if pairing_active else None)
    else:
        if not queue:
            st.success("Aucune tâche avec ces réglages : la ligne courante reste affichée.")
        focus.render(focused, controls, key=f"{pair_name}:{cursor[0]}:{cursor[1]}")
        st.caption("Contexte dans les documents")
        marks = context.Marks(
            task_marks(rows, levels), {"left": set(), "right": set()}, eligible(alignment, cursor) if pairing_active else None
        )
    docs = context.Documents(
        {"left": load_document_lines(left_dir), "right": load_document_lines(right_dir)},
        {"left": load_positions(left_dir), "right": load_positions(right_dir)},
        alignment.states,
    )
    event = documents_panel(docs, cursor, view_name, marks, (left_name, right_name), focus.summary(focused))
    if event:
        handle_event(event, alignment, records, pair_name, digest)


if __name__ == "__main__":
    main()

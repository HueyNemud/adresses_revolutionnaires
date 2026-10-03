"""Visualiseur Streamlit d'un alignement entre deux annuaires (sorties
d'`align_directories.py`).

    uv run streamlit run tools/display_alignment.py

On choisit une paire d'annuaires par une sortie brute : celle de Dedupe
`annuaires/alignements/<gauche>__<droite>.dedupe.csv` ou celle de
`align_directories_nw.py`, `<gauche>__<droite>.nw.csv`. Les deux annuaires
(`annuaires/<gauche>/`, `annuaires/<droite>/`) sont relus en entier par
`lib/alignment.py`, et le patch de corrections manuelles
`data/alignement/<gauche>__<droite>.patch.csv` est appliqué en mémoire
(`lib/alignment_patch.py`) : ce qui est affiché est le résultat final.

On n'apparie qu'entre rubriques appariées (`lib/section_alignment.py`, avec
son patch `data/alignement/<gauche>__<droite>.sections.csv`, non réécrit
ici) : un lien entre rubriques qui ne se correspondent pas — Dedupe en
produit — est écarté à l'affichage comme à l'export, et compté. La
correspondance des rubriques est détaillée dans un encart, et chaque
bandeau de rubrique a un bouton **uuid** pour alimenter son patch.

Une seule table, dans l'**ordre naturel** des listes : les correspondances et
les entrées de gauche sans correspondance dans l'ordre de l'annuaire de
gauche, chaque entrée de droite sans correspondance insérée après la paire
qui contient l'entrée de droite appariée qui la précède. La dernière
colonne donne le **statut** de chaque ligne en clair (appariée, candidate
non appariée, sans correspondance) ; le bouton **Légende**, à côté de la
pagination, l'explique. Un encart **Rubriques** donne, par groupe de
rubriques appariées puis par rubrique seule, la correspondance (auto,
patch, seule) et la part d'entrées appariées de chaque côté.

Barre latérale, du plus courant au plus fin : choix de l'alignement ;
filtres (statut des lignes, incertitude, rubrique, recherche, score,
corrections manuelles) ; affichage ; export ; et, repliés en bas, les
réglages fins de la relecture (seuil des candidates, écart « homonyme
proche »).

Le patch s'édite à la main. Pour l'alimenter, chaque entrée a un bouton
**uuid** (copie son uuid) et chaque ligne un bouton **copier** (copie une
ligne de patch prête à coller : la paire pour une correspondance, l'entrée
seule pour une entrée sans correspondance).

**Relecture** (`lib/alignment_review.py`) : chaque correspondance porte une
incertitude (faible, moyenne ou forte) et ses motifs (`déduite des voisines
(p < 0,9)`, `homonyme proche`) ; des **candidates non appariées** (deux
entrées sans correspondance, chacune la plus proche de l'autre, dans la zone
grise de similarité) occupent une ligne à part, sur fond ambre : elles ne
sont **pas** appariées, c'est au relecteur de décider. Le filtre
« Incertitude » ne garde que les lignes d'incertitude moyenne ou forte, ou
forte seulement. Une décision incertaine se copie avec le bouton **incertaine** (ligne de patch
`certitude=incertaine`).

Le bouton **Exporter en CSV** télécharge les lignes affichées (filtres et tri
appliqués) sous la forme de la jointure lisible de `lib/alignment_export.py`,
comme `tools/export_alignment.py`.
"""

import csv
import html
import io
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from align_directories_nw import Params
from lib.alignment import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, Link, Record, load_volume, read_links
from lib.alignment_export import LEFT_ONLY, PAIR, CANDIDATE, RIGHT_ONLY, JoinedRow, export_csv, export_encoding, natural_rows
from lib.alignment_patch import PATCH_FIELDS, UNCERTAIN, PatchEntry, apply_patch, entry_from_records, read_patch, resolve, validate
from lib.alignment_review import DEFAULT_MARGIN, LEVEL_LABELS, REASON_CANDIDATE, SEPARATOR, Review, review
from lib.ner.html import LABEL_COLORS, SPAN_CSS, badge, render_tagged_html
from lib.section_alignment import (
    SECTION_PATCH_FIELDS,
    SECTION_PATCH_SUFFIX,
    SOURCE_AUTO,
    SectionAlignment,
    load_section_alignment,
    read_section_patch,
    restrict_to_corresponding,
)

ANNUAIRES_DIR = Path("annuaires")
ALIGNMENTS_DIR = ANNUAIRES_DIR / "alignements"
PATCH_DIR = Path("data/alignement")
ALIGNMENT_SUFFIXES = (".dedupe.csv", ".nw.csv")  # align_directories.py, align_directories_nw.py
PAGE_SIZE_OPTIONS = [25, 50, 100, 200]
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

CSS = f"""<style>
  .legend span {{ margin-right: 10px; }}
  .table-scroll {{ max-height: 75vh; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; }}
  .ner-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; table-layout: fixed; }}
  .ner-table th {{ background: #f8fafc; border-bottom: 2px solid #e2e8f0; padding: 8px; text-align: left;
                  position: sticky; top: 0; z-index: 1; }}
  .ner-table td {{ border-bottom: 1px solid #f1f5f9; padding: 6px 8px; vertical-align: top; overflow-wrap: anywhere; }}
  .ner-table tr:hover td {{ background: #f8fafc; }}
  .ner-table tr.section td {{ background: #f1f5f9; color: #334155; font-weight: 600; font-size: 0.85em; }}
  .ner-table tr.left td.empty, .ner-table tr.right td.empty {{ background: #fef2f2; }}
  .ner-table tr.candidate td {{ background: #fffbeb; }}
  .ner-table tr.candidate td:first-child {{ box-shadow: inset 4px 0 0 #d97706; }}
  .ner-table tr.pair td:first-child {{ box-shadow: inset 4px 0 0 #16a34a; }}
  .ner-table tr.left td:first-child, .ner-table tr.right td:first-child {{ box-shadow: inset 4px 0 0 #cbd5e1; }}
  .ner-table col.number {{ width: 3.5em; }}
  .ner-table col.score {{ width: 14em; }}
  .meta {{ font-family: monospace; font-size: 0.78em; color: #64748b; }}
  .status {{ display: inline-block; font-weight: 600; font-size: 0.85em; padding: 1px 8px; border-radius: 10px; margin-bottom: 3px; }}
  .status.pair {{ background: #dcfce7; color: #166534; }}
  .status.manual {{ background: #ede9fe; color: #5b21b6; }}
  .status.uncertain {{ background: #fef3c7; color: #92400e; }}
  .status.candidate {{ background: #fde68a; color: #78350f; border: 1px dashed #b45309; }}
  .status.alone {{ background: #f1f5f9; color: #475569; }}
  .level {{ display: inline-block; font-size: 0.78em; padding: 0 6px; border-radius: 8px; margin-top: 3px; }}
  .level.level-medium {{ background: #ffedd5; color: #9a3412; }}
  .level.level-high {{ background: #fee2e2; color: #991b1b; }}
  .reasons {{ font-size: 0.78em; color: #475569; }}
  .legend-table {{ font-size: 0.85em; color: #475569; margin: 4px 0 8px; line-height: 2.1; }}
  .legend-table .status, .legend-table .level {{ margin-right: 4px; }}
  .low {{ color: #b91c1c; font-weight: 600; }}
  .empty {{ color: #b91c1c; font-size: 0.85em; font-style: italic; }}
  button.copy {{ font-size: 0.72em; padding: 0 6px; margin-left: 4px; border: 1px solid #cbd5e1; border-radius: 4px;
                background: #fff; color: #475569; cursor: pointer; }}
  button.copy:hover {{ background: #f1f5f9; }}
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


@dataclass
class Alignment:
    rows: pd.DataFrame  # sortie de `build_rows`
    joined: list[JoinedRow]  # mêmes lignes, même ordre (index de `rows`) : pour l'export
    reviews: dict[tuple[str, str], Review]  # motifs de relecture (`lib/alignment_review.py`)
    n_patch: int
    n_missing: int  # liens Dedupe vers des entrées disparues, ignorés
    reanchored: int
    orphans: list[PatchEntry]
    confirmed: set[str]  # uuid déclarés sans correspondance par le patch
    sections: SectionAlignment  # correspondance des rubriques (avec leur patch)
    n_section_patch: int
    dropped: list[Link]  # liens entre rubriques non appariées, écartés (automatiques et du patch)


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
) -> Alignment:
    """Résultat final (alignement + patch), recalculé seulement si l'un des
    fichiers change (`*_mtime` font partie de la clé de cache). Lève
    ValueError si un patch est incohérent. Le patch des rubriques n'est pas
    réécrit ici (visualiseur en lecture seule)."""
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    entries = read_patch(Path(patch_path))
    validate(entries)
    resolution = resolve(entries, records["left"], records["right"])
    links, _ = apply_patch(read_links(Path(dedupe_path)), resolution.entries)
    sections = load_section_alignment(
        list(records["left"].values()), list(records["right"].values()), Path(section_patch_path)
    )
    links, dropped = restrict_to_corresponding(links, records["left"], records["right"], sections)
    declared = {uuid for entry in resolution.entries if not entry.is_pair for _, uuid in entry.uuids()}
    params = Params()
    found = review(
        links, list(records["left"].values()), list(records["right"].values()), sections,
        candidate_low, params.residual_threshold, params.subj_weight, margin, declared,
    )
    joined, n_missing = natural_rows(links + found.candidates, records["left"], records["right"])
    rows = build_rows(joined, found.reviews)
    confirmed = {e.left_uuid for e in resolution.entries if e.left_uuid and not e.right_uuid} | {
        e.right_uuid for e in resolution.entries if e.right_uuid and not e.left_uuid
    }
    return Alignment(
        rows, joined, found.reviews, len(entries), n_missing, len(resolution.reanchored), resolution.orphans, confirmed, sections,
        len(read_section_patch(Path(section_patch_path))), dropped,
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


def text_mask(df: pd.DataFrame, query: str, columns: list[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False, regex=False)
    return mask


# ----------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------
def copy_button(label: str, text: str, title: str) -> str:
    return f'<button class="copy" data-copy="{html.escape(text, quote=True)}" title="{html.escape(title)}">{label}</button>'


def patch_line(left: Record | None, right: Record | None, certitude: str = "") -> str:
    """Ligne CSV du patch (colonnes `PATCH_FIELDS`) pour cette paire ou cette entrée seule."""
    entry = entry_from_records(left, right, certitude=certitude)
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([getattr(entry, name) for name in PATCH_FIELDS])
    return buffer.getvalue()


def entry_html(row, side: str, confirmed: set[str]) -> str:
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


def status_html(row, threshold: float) -> str:
    """Statut, score (similarité ou probabilité selon la méthode), incertitude et motifs."""
    parts = [status_badge(row.kind, row.source)]
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
        (level_badge(1) + " " + level_badge(2), "à vérifier, motif en dessous ; incertitude faible : rien n'est affiché"),
    ]
    return "<div class='legend-table'>" + "<br>".join(f"{badges} {text}" for badges, text in items) + "</div>"


def table_rows(view: pd.DataFrame, first: int, low_score: float, banners: bool, records: dict, confirmed: set[str]) -> list[str]:
    # Les bandeaux suivent la rubrique de gauche, qui fixe l'ordre de la table :
    # une entrée de droite seule, insérée dans ce fil, n'en ouvre pas de
    # nouveau (sa rubrique figure dans sa cellule), sauf en tête de table ou
    # quand seules des entrées de droite sont affichées.
    rows, current = [], None
    right_only = bool((view["kind"] == RIGHT_ONLY).all())
    for number, row in enumerate(view.itertuples(), start=first):
        if row.kind != RIGHT_ONLY:
            section, title, uuid = row.left_section, row.left_section_title, row.left_section_uuid
        elif current is None or right_only:
            section, title, uuid = row.right_section, row.right_section_title, row.right_section_uuid
        else:
            section, title, uuid = current, None, ""
        if banners and section != current:
            current = section
            button = copy_button("uuid", uuid, f"Copier l’uuid de la rubrique {uuid}") if uuid else ""
            rows.append(f"<tr class='section'><td colspan='4'>{html.escape(title or NO_SECTION)}{button}</td></tr>")
        left_record = records["left"].get(row.left_uuid)
        right_record = records["right"].get(row.right_uuid)
        line = copy_button("copier", patch_line(left_record, right_record), "Copier une ligne de patch pour cette ligne")
        if row.kind in (PAIR, CANDIDATE):
            line += copy_button(
                "incertaine", patch_line(left_record, right_record, UNCERTAIN), "Copier une ligne de patch : paire retenue mais incertaine"
            )
        empty_left = " class='empty'" if row.kind == RIGHT_ONLY else ""
        empty_right = " class='empty'" if row.kind == LEFT_ONLY else ""
        rows.append(
            f"<tr class='{row.kind}'>"
            f"<td class='meta'>{number}</td>"
            f"<td{empty_left}>{entry_html(row, 'left', confirmed)}</td>"
            f"<td{empty_right}>{entry_html(row, 'right', confirmed)}</td>"
            f"<td>{status_html(row, low_score)}<br>{line}</td>"
            "</tr>"
        )
    return rows


def paginated_table(view: pd.DataFrame, headers: list[str], render, page_size: int) -> None:
    if view.empty:
        st.info("Aucune ligne ne correspond aux filtres.")
        return
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
    st.html(
        f"{CSS}{COPY_SCRIPT}<div class='table-scroll'><table class='ner-table'>"
        "<colgroup><col class='number'><col><col><col class='score'></colgroup>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(render(view.iloc[start : start + page_size], start + 1))}</tbody>"
        "</table></div>",
        unsafe_allow_javascript=True,
    )


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
        [joined(group.left, "title"), joined(group.right, "title"), "patch" if group.source != SOURCE_AUTO else "auto",
         *counts("left", group.left), *counts("right", group.right), joined(group.left, "uuid"), joined(group.right, "uuid")]
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
            "gauche", "droite", "correspondance",
            "gauche : entrées", "gauche : appariées", "gauche : taux",
            "droite : entrées", "droite : appariées", "droite : taux",
            "uuid gauche", "uuid droite",
        ],
    )
    for column in ("gauche : entrées", "gauche : appariées", "droite : entrées", "droite : appariées"):
        table[column] = table[column].astype("Int64")
    return table


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def pair_of(path: Path) -> str:
    """`<gauche>__<droite>` d'une sortie brute, variante éventuelle ôtée
    (`<paire>.rubriques-brutes.dedupe.csv`) : les noms de volumes n'ont pas
    de point."""
    return path.name.split(".")[0]


def choose_alignment() -> Path | None:
    paths = sorted(path for suffix in ALIGNMENT_SUFFIXES for path in ALIGNMENTS_DIR.glob(f"*{suffix}"))
    if not paths:
        st.sidebar.warning(
            f"Aucun `*{'` / `*'.join(ALIGNMENT_SUFFIXES)}` sous `{ALIGNMENTS_DIR}/` : lancer `align_directories.py` "
            "ou `align_directories_nw.py`."
        )
        return None
    return st.sidebar.selectbox("Alignement", paths, format_func=lambda p: p.name.removesuffix(".csv"))


def kpis(left_name: str, right_name: str, rows: pd.DataFrame, n_patch: int, n_dropped: int) -> None:
    counts = rows["kind"].value_counts()
    matched, candidates_kept = counts.get(PAIR, 0), counts.get(CANDIDATE, 0)
    n_left, n_right = matched + candidates_kept + counts.get(LEFT_ONLY, 0), matched + candidates_kept + counts.get(RIGHT_ONLY, 0)
    columns = st.columns(6)
    columns[0].metric("Correspondances", f"{matched:,}".replace(",", " "))
    columns[1].metric(f"Appariées à gauche ({left_name})", f"{matched / n_left:.1%}", f"{n_left - matched} sans correspondance", delta_color="off")
    columns[2].metric(f"Appariées à droite ({right_name})", f"{matched / n_right:.1%}", f"{n_right - matched} sans correspondance", delta_color="off")
    columns[3].metric(
        "Liens écartés",
        str(n_dropped),
        help="Liens entre deux rubriques non appariées, écartés : on n'apparie qu'entre rubriques appariées "
        "(correspondance des rubriques et son patch).",
    )
    columns[4].metric("Lignes du patch", str(n_patch))
    levels = rows.loc[rows["kind"].isin([PAIR, CANDIDATE]), "level"].value_counts()
    columns[5].metric(
        "À vérifier",
        f"{levels.get(1, 0) + levels.get(2, 0)}",
        f"dont {levels.get(2, 0)} d'incertitude forte",
        delta_color="off",
        help="Correspondances et candidates non appariées d'incertitude moyenne ou forte (lib/alignment_review.py).",
    )


def main() -> None:
    st.set_page_config(page_title="Annuaires — alignement", layout="wide")

    alignment_path = choose_alignment()
    if alignment_path is None:
        st.title("Annuaires — alignement")
        st.info("Aucun alignement à afficher.")
        return
    pair_name = pair_of(alignment_path)
    left_name, _, right_name = pair_name.partition("__")
    patch_path = PATCH_DIR / f"{pair_name}.patch.csv"
    section_patch_path = PATCH_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}"
    left_dir, right_dir = str(ANNUAIRES_DIR / left_name), str(ANNUAIRES_DIR / right_name)
    # Barre latérale, du plus courant au plus fin. Les réglages fins sont lus
    # d'abord (ils entrent dans le calcul mis en cache) mais affichés en bas.
    filters_box, display_box, export_box = st.sidebar.container(), st.sidebar.container(), st.sidebar.container()
    params = Params()
    with st.sidebar.expander("Réglages fins de la relecture"):
        candidate_low = st.slider(
            "Similarité minimale d'une candidate non appariée",
            0.5,
            params.residual_threshold,
            params.threshold,
            step=0.01,
            help=f"Zone grise [seuil ; {params.residual_threshold}[ : par défaut le seuil de Needleman-Wunsch d'`align_directories_nw.py`.",
        )
        margin = st.number_input(
            "Écart « homonyme proche »", 0.0, 0.5, DEFAULT_MARGIN, step=0.01,
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
        )
    except (ValueError, OSError) as error:
        st.error(f"Lecture impossible : {error}")
        return
    records = {"left": load_records(left_dir), "right": load_records(right_dir)}
    rows, confirmed = alignment.rows, alignment.confirmed

    st.title(f"{left_name} ⟷ {right_name}")
    st.html(
        f"{CSS}{COPY_SCRIPT}<span class='meta'>Alignement : {html.escape(str(alignment_path))} · patch : "
        f"{html.escape(str(patch_path))} ({alignment.n_patch} ligne(s))</span>"
        f"{copy_button('en-tête du patch', ','.join(PATCH_FIELDS), 'Copier la ligne d’en-tête du CSV de patch')}"
        f"<br><span class='meta'>patch des rubriques : {html.escape(str(section_patch_path))} "
        f"({alignment.n_section_patch} ligne(s))</span>"
        f"{copy_button('en-tête', ','.join(SECTION_PATCH_FIELDS), 'Copier la ligne d’en-tête du CSV de patch des rubriques')}",
        unsafe_allow_javascript=True,
    )
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
            "ignorées ; relancer `align_directories.py`."
        )
    if alignment.reanchored:
        st.info(f"↻ {alignment.reanchored} ligne(s) du patch réancrée(s) par le texte ; `align_directories.py --sans-dedupe --apply` réécrira le patch.")
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
    legend = "".join(badge(label, colors) for label, colors in LABEL_COLORS.items())
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    kpis(left_name, right_name, rows, alignment.n_patch, len(alignment.dropped))
    with st.expander("Rubriques : correspondance et appariement"):
        groups = alignment.sections.groups
        manual = sum(group.source != SOURCE_AUTO for group in groups)
        alone = len(alignment.sections.unmatched_left) + len(alignment.sections.unmatched_right)
        st.caption(
            f"{len(groups)} groupe(s) de rubriques appariées, dont {manual} du patch des rubriques ; {alone} rubrique(s) seule(s), "
            "dont les entrées ne sont jamais appariées. Pour corriger : une ligne `left_uuid,right_uuid` par paire dans le patch "
            "des rubriques (un même uuid sur plusieurs lignes forme un groupe 1-N), ou un seul uuid pour une rubrique sans "
            "correspondance."
        )
        st.dataframe(section_table(alignment.sections, rows), width="stretch", hide_index=True)

    # Filtres
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
        order = st.selectbox("Tri", ORDER_OPTIONS)
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
        view = view[view["source"].isin(MANUAL_SOURCES) | view["left_uuid"].isin(confirmed) | view["right_uuid"].isin(confirmed)]
    if min_level:
        view = view[view["level"] >= min_level]
    if order == LEVEL_ORDER:
        view = view.sort_values("level", ascending=False, kind="stable")
    elif order != NATURAL_ORDER:
        view = view.sort_values("score", ascending=order == "score croissant", kind="stable", na_position="last")

    # Retour à la première page quand la sélection change.
    selection = (
        alignment_path.name, tuple(kinds), section, query, low, high, only_manual, order, page_size,
        min_level, candidate_low, margin,
    )
    if st.session_state.get("_selection") != selection:
        st.session_state["_selection"] = selection
        st.session_state.page = 0

    export_box.download_button(
        f"Exporter en CSV ({len(view):,} lignes)".replace(",", " "),
        # Généré au clic seulement : lignes affichées, dans l'ordre affiché.
        lambda: export_csv([alignment.joined[index] for index in view.index], excel=excel, reviews=alignment.reviews).encode(export_encoding(excel)),
        file_name=f"{pair_name}.jointure.csv",
        mime="text/csv",
        icon=":material/download:",
        help="Jointure lisible des deux annuaires (texte sans Markdown, empans NER en colonnes), filtres et tri appliqués.",
        width="stretch",
    )

    paginated_table(
        view,
        ["#", f"gauche — {left_name}", f"droite — {right_name}", "statut"],
        lambda page, first: table_rows(page, first, low_score, order == NATURAL_ORDER, records, confirmed),
        page_size,
    )


if __name__ == "__main__":
    main()

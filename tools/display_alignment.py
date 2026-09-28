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

La correspondance des rubriques (`lib/section_alignment.py`, avec son patch
`data/alignement/<gauche>__<droite>.sections.csv`, non réécrit ici) sert à
signaler les correspondances entre rubriques **non correspondantes** —
et non entre rubriques de noms différents : « Liste » / « Listes de
non-commerçans » se correspondent. Elle est détaillée dans un encart, et
chaque bandeau de rubrique a un bouton **uuid** pour alimenter ce patch.

Une seule table, dans l'**ordre naturel** des listes : les correspondances et
les entrées de gauche sans correspondance dans l'ordre de l'annuaire de
gauche, chaque entrée de droite sans correspondance insérée après la paire
qui contient l'entrée de droite appariée qui la précède. Filtres : types de
lignes (correspondances, sans correspondance à gauche / à droite), rubrique,
recherche, plage de scores, rubriques non correspondantes, corrections
manuelles.

Le patch s'édite à la main. Pour l'alimenter, chaque entrée a un bouton
**uuid** (copie son uuid) et chaque ligne un bouton **copier** (copie une
ligne de patch prête à coller : la paire pour une correspondance, l'entrée
seule pour une entrée sans correspondance).
"""

import csv
import html
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from lib.alignment import SOURCE_MANUAL, Record, load_volume, read_links
from lib.alignment_patch import PATCH_FIELDS, PatchEntry, apply_patch, entry_from_records, read_patch, resolve, validate
from lib.ner.html import LABEL_COLORS, SPAN_CSS, badge, render_tagged_html
from lib.section_alignment import (
    SECTION_PATCH_FIELDS,
    SECTION_PATCH_SUFFIX,
    SOURCE_AUTO,
    SectionAlignment,
    corresponding,
    load_section_alignment,
    read_section_patch,
)

ANNUAIRES_DIR = Path("annuaires")
ALIGNMENTS_DIR = ANNUAIRES_DIR / "alignements"
PATCH_DIR = Path("data/alignement")
ALIGNMENT_SUFFIXES = (".dedupe.csv", ".nw.csv")  # align_directories.py, align_directories_nw.py
PAGE_SIZE_OPTIONS = [25, 50, 100, 200]
NATURAL_ORDER = "ordre naturel"
ORDER_OPTIONS = [NATURAL_ORDER, "score croissant", "score décroissant"]
ALL_SECTIONS = "(toutes)"
NO_SECTION = "(sans rubrique)"
PAIR, LEFT_ONLY, RIGHT_ONLY = "pair", "left", "right"
KIND_LABELS = {PAIR: "Correspondances", LEFT_ONLY: "Sans correspondance à gauche", RIGHT_ONLY: "Sans correspondance à droite"}
MANUAL_COLORS = ("#ede9fe", "#6d28d9")
CONFIRMED = "confirmée sans correspondance"
ENTRY_COLUMNS = ["uuid", "document", "order", "page", "section", "section_title", "section_uuid", "markdown", "tagged_text", "text"]

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
  .ner-table col.number {{ width: 3.5em; }}
  .ner-table col.score {{ width: 8em; }}
  .meta {{ font-family: monospace; font-size: 0.78em; color: #64748b; }}
  .diff {{ color: #b91c1c; }}
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


@st.cache_data(show_spinner=False)
def records_frame(volume_dir: str) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(record) for record in load_records(volume_dir).values()])
    return frame[ENTRY_COLUMNS].set_index("uuid", drop=False)


@dataclass
class Alignment:
    rows: pd.DataFrame  # sortie de `build_rows`
    n_patch: int
    n_missing: int  # liens Dedupe vers des entrées disparues, ignorés
    reanchored: int
    orphans: list[PatchEntry]
    confirmed: set[str]  # uuid déclarés sans correspondance par le patch
    sections: SectionAlignment  # correspondance des rubriques (avec leur patch)
    n_section_patch: int


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
    rows, n_missing = build_rows(links, records_frame(left_dir), records_frame(right_dir))
    sections = load_section_alignment(
        list(records["left"].values()), list(records["right"].values()), Path(section_patch_path), rewrite=False
    )
    matching = corresponding(sections)
    unmatched = pd.Series(
        [(left, right) not in matching for left, right in zip(rows["left_section_uuid"], rows["right_section_uuid"])], index=rows.index
    )
    rows["different_section"] = (rows["kind"] == PAIR) & unmatched
    confirmed = {e.left_uuid for e in resolution.entries if e.left_uuid and not e.right_uuid} | {
        e.right_uuid for e in resolution.entries if e.right_uuid and not e.left_uuid
    }
    return Alignment(
        rows, len(entries), n_missing, len(resolution.reanchored), resolution.orphans, confirmed, sections,
        len(read_section_patch(Path(section_patch_path))),
    )


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def build_rows(links, left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Une ligne par correspondance ou entrée sans correspondance, colonnes
    `left_*` / `right_*` (vides du côté absent), triées dans l'ordre naturel ;
    et le nombre de liens ignorés faute d'entrée."""
    frame = pd.DataFrame(
        [(link.left_uuid, link.right_uuid, link.score, link.source) for link in links],
        columns=["left_uuid", "right_uuid", "score", "source"],
    )
    frame = frame[frame["left_uuid"].isin(left.index) & frame["right_uuid"].isin(right.index)]
    pairs = pd.concat(
        [left.loc[frame["left_uuid"]].add_prefix("left_").reset_index(drop=True),
         right.loc[frame["right_uuid"]].add_prefix("right_").reset_index(drop=True),
         frame[["score", "source"]].reset_index(drop=True)],
        axis=1,
    ).assign(kind=PAIR)
    left_only = left[~left.index.isin(frame["left_uuid"])].add_prefix("left_").assign(kind=LEFT_ONLY)
    right_only = right[~right.index.isin(frame["right_uuid"])].add_prefix("right_").assign(kind=RIGHT_ONLY)
    rows = pd.concat([pairs, left_only.reset_index(drop=True), right_only.reset_index(drop=True)], ignore_index=True)
    rows["score"] = pd.to_numeric(rows["score"])

    # Ordre naturel : rang à gauche ; une entrée de droite seule se range après
    # la paire de l'entrée de droite appariée qui la précède (avant la première
    # paire s'il n'y en a pas).
    paired_right = dict(zip(frame["right_uuid"], left.loc[frame["left_uuid"], "order"]))
    anchors = right.sort_values("order")["uuid"].map(paired_right)
    first = anchors.dropna().iloc[0] if anchors.notna().any() else 0
    anchors = anchors.ffill().fillna(first - 0.5)
    rows["_anchor"] = rows["left_order"].where(rows["kind"] != RIGHT_ONLY, rows["right_uuid"].map(anchors))
    rows["_after"] = (rows["kind"] == RIGHT_ONLY).astype(int)
    rows["_right_order"] = rows["right_order"].where(rows["kind"] == RIGHT_ONLY, 0)
    rows = rows.sort_values(["_anchor", "_after", "_right_order"], kind="stable").drop(columns=["_anchor", "_after", "_right_order"])
    for side in ("left", "right"):
        for column in ("section", "section_title", "section_uuid", "markdown", "tagged_text", "text", "uuid", "page"):
            rows[f"{side}_{column}"] = rows[f"{side}_{column}"].fillna("").astype(str)
    return rows.reset_index(drop=True), len(links) - len(frame)


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


def patch_line(left: Record | None, right: Record | None) -> str:
    """Ligne CSV du patch (colonnes `PATCH_FIELDS`) pour cette paire ou cette entrée seule."""
    entry = entry_from_records(left, right)
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([getattr(entry, name) for name in PATCH_FIELDS])
    return buffer.getvalue()


def entry_html(row, side: str, different_section: bool, confirmed: set[str]) -> str:
    uuid = getattr(row, f"{side}_uuid")
    if not uuid:
        return "<span class='empty'>sans correspondance</span>"
    tagged, markdown = getattr(row, f"{side}_tagged_text"), getattr(row, f"{side}_markdown")
    content = render_tagged_html(tagged) or html.escape(markdown)
    section_class = "meta diff" if different_section else "meta"
    page = getattr(row, f"{side}_page")
    title = getattr(row, f"{side}_section_title") or NO_SECTION
    flag = f" {badge(CONFIRMED, MANUAL_COLORS)}" if uuid in confirmed else ""
    return (
        f"{content}{flag}<br><span class='meta'>p. {html.escape(page)}</span> · "
        f"<span class='{section_class}'>{html.escape(title)}</span>"
        f"{copy_button('uuid', uuid, f'Copier l’uuid {uuid}')}"
    )


def score_html(row, threshold: float) -> str:
    if row.kind != PAIR:
        return ""
    manual = badge(SOURCE_MANUAL, MANUAL_COLORS) if row.source == SOURCE_MANUAL else ""
    if pd.isna(row.score):
        return manual or "<span class='meta'>—</span>"
    css = " class='low'" if row.score < threshold else ""
    return f"<span{css}>{row.score:.3f}</span> {manual}"


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
        different = row.different_section
        line = copy_button("copier", patch_line(left_record, right_record), "Copier une ligne de patch pour cette ligne")
        empty_left = " class='empty'" if row.kind == RIGHT_ONLY else ""
        empty_right = " class='empty'" if row.kind == LEFT_ONLY else ""
        rows.append(
            f"<tr class='{row.kind}'>"
            f"<td class='meta'>{number}</td>"
            f"<td{empty_left}>{entry_html(row, 'left', different, confirmed)}</td>"
            f"<td{empty_right}>{entry_html(row, 'right', different, confirmed)}</td>"
            f"<td>{score_html(row, low_score)}<br>{line}</td>"
            "</tr>"
        )
    return rows


def paginated_table(view: pd.DataFrame, headers: list[str], render, page_size: int) -> None:
    if view.empty:
        st.info("Aucune ligne ne correspond aux filtres.")
        return
    n_pages = max(1, -(-len(view) // page_size))
    page = min(st.session_state.get("page", 0), n_pages - 1)
    previous, label, following = st.columns([1, 3, 1])
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


def section_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Par rubrique (nom nettoyé, tel que comparé par Dedupe) : entrées et taux d'appariement de chaque côté."""
    parts = []
    for side, name in (("left", "gauche"), ("right", "droite")):
        present = rows[rows[f"{side}_uuid"] != ""]
        grouped = present.assign(matched=present["kind"] == PAIR).groupby(f"{side}_section", sort=False)["matched"]
        part = pd.DataFrame({f"{name} : entrées": grouped.size(), f"{name} : appariées": grouped.sum()})
        part[f"{name} : taux"] = (part[f"{name} : appariées"] / part[f"{name} : entrées"]).round(3)
        parts.append(part)
    summary = parts[0].join(parts[1], how="outer")
    summary.index = summary.index.map(lambda section: section or NO_SECTION)
    summary.index.name = "rubrique"
    return summary


def section_table(alignment: SectionAlignment) -> pd.DataFrame:
    """Correspondance des rubriques : un groupe par ligne (titres et uuid
    joints par « + » s'il en compte plusieurs), puis les rubriques seules."""

    def joined(sections, attribute: str) -> str:
        return " + ".join(getattr(section, attribute) or NO_SECTION for section in sections)

    lines = [
        (joined(group.left, "title"), joined(group.right, "title"), group.source, joined(group.left, "uuid"), joined(group.right, "uuid"))
        for group in alignment.groups
    ]
    for side, unmatched in (("left", alignment.unmatched_left), ("right", alignment.unmatched_right)):
        for section in unmatched:
            source = "seule (patch)" if (side, section.uuid) in alignment.declared else "seule"
            title, uuid = section.title or NO_SECTION, section.uuid
            lines.append((title, "", source, uuid, "") if side == "left" else ("", title, source, "", uuid))
    return pd.DataFrame(lines, columns=["gauche", "droite", "source", "uuid gauche", "uuid droite"])


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


def kpis(left_name: str, right_name: str, rows: pd.DataFrame, n_patch: int) -> None:
    counts = rows["kind"].value_counts()
    matched, n_left, n_right = counts.get(PAIR, 0), counts.get(PAIR, 0) + counts.get(LEFT_ONLY, 0), counts.get(PAIR, 0) + counts.get(RIGHT_ONLY, 0)
    pairs = rows[rows["kind"] == PAIR]
    columns = st.columns(5)
    columns[0].metric("Correspondances", f"{matched:,}".replace(",", " "))
    columns[1].metric(f"Appariées à gauche ({left_name})", f"{matched / n_left:.1%}", f"{n_left - matched} sans correspondance", delta_color="off")
    columns[2].metric(f"Appariées à droite ({right_name})", f"{matched / n_right:.1%}", f"{n_right - matched} sans correspondance", delta_color="off")
    columns[3].metric(
        "Rubriques non correspondantes",
        f"{int(pairs['different_section'].sum())}",
        help="Correspondances entre deux rubriques qui ne se correspondent pas (alignement des rubriques et son patch).",
    )
    columns[4].metric("Lignes du patch", str(n_patch))


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
        st.info(f"↻ {alignment.reanchored} ligne(s) du patch réancrée(s) par le texte ; `--apply-only` réécrira le patch.")
    if alignment.orphans:
        with st.expander(f"⚠ {len(alignment.orphans)} ligne(s) orpheline(s) du patch, non appliquée(s)"):
            st.dataframe(pd.DataFrame([asdict(entry) for entry in alignment.orphans]), width="stretch")
    legend = "".join(badge(label, colors) for label, colors in LABEL_COLORS.items())
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    kpis(left_name, right_name, rows, alignment.n_patch)
    with st.expander("Bilan par rubrique"):
        st.caption("Rubriques telles que comparées (titre `##`, sinon `#`, nettoyé) : une rubrique renommée apparaît deux fois.")
        st.dataframe(section_summary(rows), width="stretch")
    with st.expander("Correspondance des rubriques"):
        groups = alignment.sections.groups
        manual = sum(group.source != SOURCE_AUTO for group in groups)
        st.caption(
            f"{len(groups)} groupe(s), dont {manual} du patch des rubriques ; une rubrique seule n'a pas de correspondance. "
            "Pour corriger : une ligne `left_uuid,right_uuid` par paire (un même uuid sur plusieurs lignes forme un "
            "groupe 1-N), ou un seul uuid pour une rubrique sans correspondance."
        )
        st.dataframe(section_table(alignment.sections), width="stretch", hide_index=True)

    # Filtres
    st.sidebar.header("Filtres")
    st.sidebar.caption("Lignes affichées")
    kinds = [kind for kind, label in KIND_LABELS.items() if st.sidebar.checkbox(label, value=True, key=f"kind_{kind}")]
    sections = sorted({record.section for side in records.values() for record in side.values()})
    section = st.sidebar.selectbox("Rubrique", [ALL_SECTIONS, *sections], format_func=lambda s: s or NO_SECTION)
    query = st.sidebar.text_input("Recherche (texte des entrées)")
    low, high = st.sidebar.slider("Score des correspondances", 0.0, 1.0, (0.0, 1.0), step=0.01)
    low_score = st.sidebar.slider("Score signalé en rouge sous", 0.0, 1.0, 0.8, step=0.01)
    only_diff = st.sidebar.checkbox("Seulement les correspondances entre rubriques non correspondantes")
    only_manual = st.sidebar.checkbox("Seulement les corrections manuelles")
    st.sidebar.header("Affichage")
    order = st.sidebar.selectbox("Tri", ORDER_OPTIONS)
    page_size = st.sidebar.selectbox("Lignes par page", PAGE_SIZE_OPTIONS, index=1)

    view = rows[rows["kind"].isin(kinds)]
    view = view[(view["kind"] != PAIR) | view["score"].between(low, high) | view["score"].isna()]
    if section != ALL_SECTIONS:
        view = view[(view["left_section"] == section) | (view["right_section"] == section)]
    if query:
        view = view[text_mask(view, query, ["left_markdown", "right_markdown"])]
    if only_diff:
        view = view[view["different_section"]]
    if only_manual:
        view = view[(view["source"] == SOURCE_MANUAL) | view["left_uuid"].isin(confirmed) | view["right_uuid"].isin(confirmed)]
    if order != NATURAL_ORDER:
        view = view.sort_values("score", ascending=order == "score croissant", kind="stable", na_position="last")

    # Retour à la première page quand la sélection change.
    selection = (alignment_path.name, tuple(kinds), section, query, low, high, only_diff, only_manual, order, page_size)
    if st.session_state.get("_selection") != selection:
        st.session_state["_selection"] = selection
        st.session_state.page = 0

    paginated_table(
        view,
        ["#", f"gauche — {left_name}", f"droite — {right_name}", "score"],
        lambda page, first: table_rows(page, first, low_score, order == NATURAL_ORDER, records, confirmed),
        page_size,
    )


if __name__ == "__main__":
    main()

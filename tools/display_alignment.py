"""Visualiseur Streamlit d'un alignement entre deux annuaires (sortie
d'`align_directories.py`).

    uv run streamlit run tools/display_alignment.py

Le CSV de correspondances (ou sa version relue `*.curated.csv`) se choisit
dans `annuaires/alignements/` (ou se téléverse). Seules ses colonnes
d'identification (`*_file`, `*_uuid`) et `score` sont lues : une ligne
ajoutée à la main peut laisser `score` vide. Les deux annuaires sont retrouvés d'après `left_file` /
`right_file` (préfixe avant le premier `.` = dossier du volume sous
`annuaires/`) et relus en entier par `lib/alignment.py`, ce qui donne aussi
les entrées **non appariées** de chaque côté.

Onglets : correspondances (gauche | droite côte à côte, score), non
appariées à gauche, non appariées à droite, bilan par rubrique. Filtres :
rubrique, recherche, plage de scores, paires dont la rubrique diffère.
"""

import html
import io
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from lib.alignment import load_volume, volume_of
from lib.ner.html import LABEL_COLORS, SPAN_CSS, badge, render_tagged_html

ANNUAIRES_DIR = Path("annuaires")
ALIGNMENTS_DIR = ANNUAIRES_DIR / "alignements"
PAGE_SIZE_OPTIONS = [25, 50, 100, 200]
ORDER_OPTIONS = ["ordre de l'annuaire de gauche", "score croissant", "score décroissant"]
LINK_COLUMNS = ["left_file", "left_uuid", "right_uuid", "right_file", "score"]
ALL_SECTIONS = "(toutes)"
NO_SECTION = "(sans rubrique)"

TABLE_CSS = """
  .legend span { margin-right: 10px; }
  .table-scroll { max-height: 72vh; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; }
  .ner-table { width: 100%; border-collapse: collapse; font-size: 0.9em; table-layout: fixed; }
  .ner-table th { background: #f8fafc; border-bottom: 2px solid #e2e8f0; padding: 8px; text-align: left;
                  position: sticky; top: 0; z-index: 1; }
  .ner-table td { border-bottom: 1px solid #f1f5f9; padding: 6px 8px; vertical-align: top; overflow-wrap: anywhere; }
  .ner-table tr:hover td { background: #f8fafc; }
  .ner-table tr.section td { background: #f1f5f9; color: #334155; font-weight: 600; font-size: 0.85em; }
  .ner-table col.score { width: 5.5em; }
  .meta { font-family: monospace; font-size: 0.78em; color: #64748b; }
  .diff { color: #b91c1c; }
  .low { color: #b91c1c; font-weight: 600; }
"""
CSS = f"<style>{TABLE_CSS}{SPAN_CSS}</style>"


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Lecture de l'annuaire…")
def load_entries(volume_dir: str) -> pd.DataFrame:
    return pd.DataFrame([asdict(record) for record in load_volume(Path(volume_dir))]).set_index("uuid", drop=False)


@st.cache_data(show_spinner="Lecture des correspondances…")
def load_links_path(path: str, modified: float) -> pd.DataFrame:  # `modified` invalide le cache si le fichier change
    return read_links(path)


@st.cache_data(show_spinner="Lecture des correspondances…")
def load_links_bytes(content: bytes) -> pd.DataFrame:
    return read_links(io.BytesIO(content))


def read_links(source) -> pd.DataFrame:
    links = pd.read_csv(source, dtype=str, keep_default_na=False)
    links["score"] = pd.to_numeric(links.get("score", ""), errors="coerce")  # vide (ajout manuel) → NaN
    return links[LINK_COLUMNS]


def build_pairs(links: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par correspondance, avec les colonnes `left_*` / `right_*` des entrées."""
    columns = ["order", "page", "section", "section_title", "markdown", "tagged_text"]
    pairs = links.join(left[columns].add_prefix("left_"), on="left_uuid").join(right[columns].add_prefix("right_"), on="right_uuid")
    missing = pairs["left_order"].isna() | pairs["right_order"].isna()
    if missing.any():
        st.warning(f"{int(missing.sum())} correspondance(s) dont une entrée est absente des annuaires (fichiers modifiés depuis ?) : ignorées.")
    return pairs[~missing]


def text_mask(df: pd.DataFrame, query: str, columns: list[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False, regex=False)
    return mask


# ----------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------
def entry_html(tagged_text: str, markdown: str, page: object, title: str, different_section: bool) -> str:
    content = render_tagged_html(tagged_text) or html.escape(markdown)
    section_class = "meta diff" if different_section else "meta"
    return (
        f"{content}<br><span class='meta'>p. {html.escape(str(page))}</span> · "
        f"<span class='{section_class}'>{html.escape(title or NO_SECTION)}</span>"
    )


def score_html(score: float, threshold: float) -> str:
    if pd.isna(score):
        return "<span class='meta' title='correspondance sans score (ajout manuel)'>—</span>"
    css = " class='low'" if score < threshold else ""
    return f"<span{css}>{score:.3f}</span>"


def pair_rows(view: pd.DataFrame, low_score: float) -> list[str]:
    return [
        "<tr>"
        f"<td>{entry_html(row.left_tagged_text, row.left_markdown, row.left_page, row.left_section_title, row.left_section != row.right_section)}</td>"
        f"<td>{entry_html(row.right_tagged_text, row.right_markdown, row.right_page, row.right_section_title, row.left_section != row.right_section)}</td>"
        f"<td>{score_html(row.score, low_score)}</td>"
        "</tr>"
        for row in view.itertuples()
    ]


def entry_rows(view: pd.DataFrame) -> list[str]:
    """Entrées non appariées, en ordre de l'annuaire, avec un bandeau à chaque changement de rubrique."""
    rows, current = [], None
    for row in view.itertuples():
        if row.section != current:
            current = row.section
            rows.append(f"<tr class='section'><td>{html.escape(row.section_title or NO_SECTION)}</td></tr>")
        content = render_tagged_html(row.tagged_text) or html.escape(row.markdown)
        rows.append(f"<tr><td>{content} <span class='meta'>p. {html.escape(str(row.page))}</span></td></tr>")
    return rows


def paginated_table(key: str, view: pd.DataFrame, headers: list[str], render, page_size: int, colgroup: str = "") -> None:
    """Table HTML d'une page de `view` ; `render(page_df)` → lignes `<tr>`."""
    if view.empty:
        st.info("Aucune ligne ne correspond aux filtres.")
        return
    n_pages = max(1, -(-len(view) // page_size))
    page = min(st.session_state.get(key, 0), n_pages - 1)
    previous, label, following = st.columns([1, 3, 1])
    if previous.button("◀ Précédente", key=f"{key}_prev", disabled=page == 0, width="stretch"):
        page -= 1
    if following.button("Suivante ▶", key=f"{key}_next", disabled=page >= n_pages - 1, width="stretch"):
        page += 1
    st.session_state[key] = page
    label.markdown(
        f"<div style='text-align:center;padding-top:6px'>Page <b>{page + 1}</b> / {n_pages} · "
        f"{len(view):,} ligne(s)</div>".replace(",", " "),
        unsafe_allow_html=True,
    )
    shown = view.iloc[page * page_size : (page + 1) * page_size]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    st.markdown(
        f"<div class='table-scroll'><table class='ner-table'>{colgroup}<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(render(shown))}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def section_summary(left: pd.DataFrame, right: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Par rubrique (nom nettoyé, tel que comparé par Dedupe) : entrées et taux d'appariement de chaque côté."""
    matched_left, matched_right = set(pairs["left_uuid"]), set(pairs["right_uuid"])

    def side(df: pd.DataFrame, matched: set, name: str) -> pd.DataFrame:
        grouped = df.assign(matched=df["uuid"].isin(matched)).groupby("section", sort=False)["matched"]
        return pd.DataFrame({f"{name} : entrées": grouped.size(), f"{name} : appariées": grouped.sum()})

    summary = side(left, matched_left, "gauche").join(side(right, matched_right, "droite"), how="outer")
    summary = summary.fillna(0).astype(int)
    for name in ("gauche", "droite"):
        total = summary[f"{name} : entrées"]
        summary[f"{name} : taux"] = (summary[f"{name} : appariées"] / total.where(total > 0)).round(3)
    summary.index = summary.index.map(lambda section: section or NO_SECTION)
    summary.index.name = "rubrique"
    return summary


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def choose_file() -> tuple[str, pd.DataFrame] | None:
    source = st.sidebar.radio("Source", [f"{ALIGNMENTS_DIR}/", "téléverser"], horizontal=True, label_visibility="collapsed")
    if source != "téléverser":
        paths = sorted(ALIGNMENTS_DIR.glob("*.csv"))
        if not paths:
            st.sidebar.warning(f"Aucun CSV sous `{ALIGNMENTS_DIR}/` : lancer `align_directories.py`.")
            return None
        path = st.sidebar.selectbox("Correspondances", paths, format_func=lambda p: p.name)
        return path.name, load_links_path(str(path), path.stat().st_mtime)
    uploaded = st.sidebar.file_uploader("CSV de correspondances", type=["csv"])
    return (uploaded.name, load_links_bytes(uploaded.getvalue())) if uploaded else None


def volume_dir(links: pd.DataFrame, column: str) -> Path:
    volumes = {volume_of(name) for name in links[column]}
    if len(volumes) != 1:
        raise ValueError(f"`{column}` doit désigner un seul annuaire, trouvé : {sorted(volumes) or 'aucun'}")
    return ANNUAIRES_DIR / volumes.pop()


def kpis(left_name: str, right_name: str, n_left: int, n_right: int, pairs: pd.DataFrame) -> None:
    matched = len(pairs)
    columns = st.columns(4)
    columns[0].metric("Correspondances", f"{matched:,}".replace(",", " "))
    columns[1].metric(f"Appariées à gauche ({left_name})", f"{matched / n_left:.1%}", f"{n_left - matched} non appariées", delta_color="off")
    columns[2].metric(f"Appariées à droite ({right_name})", f"{matched / n_right:.1%}", f"{n_right - matched} non appariées", delta_color="off")
    columns[3].metric("Rubrique différente", f"{int((pairs['left_section'] != pairs['right_section']).sum())}")


def main() -> None:
    st.set_page_config(page_title="Annuaires — alignement", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    chosen = choose_file()
    if chosen is None:
        st.title("Annuaires — alignement")
        st.info("Choisissez un CSV de correspondances produit par `align_directories.py` dans la barre latérale.")
        return
    name, links = chosen
    try:
        left_dir, right_dir = volume_dir(links, "left_file"), volume_dir(links, "right_file")
        left, right = load_entries(str(left_dir)), load_entries(str(right_dir))
    except (ValueError, OSError) as error:
        st.error(f"Impossible de relire les annuaires : {error}")
        return
    pairs = build_pairs(links, left, right)

    st.title(f"{left_dir.name} ⟷ {right_dir.name}")
    st.caption(name)
    legend = "".join(badge(label, colors) for label, colors in LABEL_COLORS.items())
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    kpis(left_dir.name, right_dir.name, len(left), len(right), pairs)

    # Filtres
    st.sidebar.header("Filtres")
    sections = sorted(set(left["section"]) | set(right["section"]))
    section = st.sidebar.selectbox("Rubrique", [ALL_SECTIONS, *sections], format_func=lambda s: s or NO_SECTION)
    query = st.sidebar.text_input("Recherche (texte des entrées)")
    low, high = st.sidebar.slider("Score", 0.0, 1.0, (0.0, 1.0), step=0.01)
    low_score = st.sidebar.slider("Score signalé en rouge sous", 0.0, 1.0, 0.8, step=0.01)
    only_diff = st.sidebar.checkbox("Seulement les paires de rubriques différentes")
    st.sidebar.header("Affichage")
    order = st.sidebar.selectbox("Tri des correspondances", ORDER_OPTIONS)
    page_size = st.sidebar.selectbox("Lignes par page", PAGE_SIZE_OPTIONS, index=1)

    pair_view = pairs[pairs["score"].between(low, high) | pairs["score"].isna()]
    left_view = left[~left["uuid"].isin(pairs["left_uuid"])]
    right_view = right[~right["uuid"].isin(pairs["right_uuid"])]
    if section != ALL_SECTIONS:
        pair_view = pair_view[(pair_view["left_section"] == section) | (pair_view["right_section"] == section)]
        left_view = left_view[left_view["section"] == section]
        right_view = right_view[right_view["section"] == section]
    if query:
        pair_view = pair_view[text_mask(pair_view, query, ["left_markdown", "right_markdown"])]
        left_view = left_view[text_mask(left_view, query, ["markdown"])]
        right_view = right_view[text_mask(right_view, query, ["markdown"])]
    if only_diff:
        pair_view = pair_view[pair_view["left_section"] != pair_view["right_section"]]
    sort_by = {ORDER_OPTIONS[0]: ("left_order", True), ORDER_OPTIONS[1]: ("score", True), ORDER_OPTIONS[2]: ("score", False)}
    column, ascending = sort_by[order]
    pair_view = pair_view.sort_values(column, ascending=ascending, kind="stable")

    # Retour à la première page quand la sélection change.
    selection = (name, section, query, low, high, only_diff, order, page_size)
    if st.session_state.get("_selection") != selection:
        st.session_state["_selection"] = selection
        for key in ("pairs_page", "left_page", "right_page"):
            st.session_state[key] = 0

    tabs = st.tabs(
        [
            f"Correspondances ({len(pair_view):,})".replace(",", " "),
            f"Non appariées à gauche ({len(left_view):,})".replace(",", " "),
            f"Non appariées à droite ({len(right_view):,})".replace(",", " "),
            "Par rubrique",
        ]
    )
    with tabs[0]:
        colgroup = "<colgroup><col><col><col class='score'></colgroup>"
        paginated_table(
            "pairs_page", pair_view, [f"gauche — {left_dir.name}", f"droite — {right_dir.name}", "score"],
            lambda shown: pair_rows(shown, low_score), page_size, colgroup,
        )
    with tabs[1]:
        paginated_table("left_page", left_view, [left_dir.name], entry_rows, page_size)
    with tabs[2]:
        paginated_table("right_page", right_view, [right_dir.name], entry_rows, page_size)
    with tabs[3]:
        st.caption("Rubriques telles que comparées (titre `##`, sinon `#`, nettoyé) : une rubrique renommée apparaît deux fois.")
        st.dataframe(section_summary(left, right, pairs), width="stretch")


if __name__ == "__main__":
    main()

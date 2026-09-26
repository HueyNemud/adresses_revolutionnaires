"""Visualiseur Streamlit pour les CSV du pipeline (merge_annotated_lines.py /
infer_gliner.py) : une vue unique, paginée, triable, avec rendu coloré des
empans détectés sur la page affichée.

Optimisations par rapport à la toute première version, qui devenait
inutilisable au-delà de quelques milliers de lignes :
  - le CSV n'est relu/reparsé qu'une seule fois par fichier (`@st.cache_data`),
    pas à chaque interaction — Streamlit réexécute tout le script du haut en
    bas à chaque clic, sans cache le fichier entier était donc reparsé à
    chaque recherche ou changement de filtre ;
  - la recherche plein texte est vectorisée par colonne (`Series.str.contains`)
    plutôt qu'appliquée ligne par ligne sur l'intégralité du DataFrame
    converti en texte (`.astype(str).apply(...)`, très coûteux) ;
  - le rendu HTML coloré (couleurs par classe), coûteux à construire et à
    afficher, n'est jamais produit pour plus d'une page à la fois (taille
    configurable) — jamais pour l'ensemble du fichier filtré d'un coup.

Une version intermédiaire ajoutait, à côté de cette table paginée, une
seconde vue basée sur `st.dataframe` (rendu virtualisé natif) pour parcourir
et trier l'ensemble du résultat filtré sans pagination — mais avoir deux
tables côte à côte doublait la place prise à l'écran et la charge cognitive
pour un gain marginal, `st.dataframe` ne pouvant de toute façon pas afficher
le surlignage coloré (pas de HTML arbitraire dans ses cellules). Cette
version fusionne les deux en une seule table, avec un tri ajouté en contrôle
explicite pour ne pas perdre cette capacité.

Schéma attendu (sortie de merge_annotated_lines.py / infer_gliner.py) :
`uuid`, `entity` (ENTRY/TITLE/OUT OF SCOPE), `markdown` (texte source),
`tagged_text` (texte balisé <SUBJ>/<DESC>/<ADDR>), `subject_count`,
`description_count`, `address_count`. Un ancien schéma `text`+`spans` (JSON
d'empans avec offsets) reste pris en charge en repli, pour ne pas casser
d'anciens exports.
"""

import html
import io
import re
import zlib
from functools import lru_cache

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visualiseur d'entités & empans CSV", layout="wide")

DEFAULT_PAGE_SIZE = 25
PAGE_SIZE_OPTIONS = [10, 25, 50, 100]
DEFAULT_VISIBLE_COLUMNS = [
    "uuid",
    "entity",
    "markdown",
    "tagged_text",
    "subject_count",
    "description_count",
    "address_count",
]
COUNT_COLUMNS = ["subject_count", "description_count", "address_count"]

TAG_PATTERN = re.compile(r"<([A-Za-z_][A-Za-z0-9_-]*)>(.*?)</\1>", re.DOTALL)

TABLE_CSS = """
<style>
    .table-scroll { max-height: 70vh; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 4px; }
    .custom-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em; }
    .custom-table th { background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; padding: 10px; text-align: left; position: sticky; top: 0; }
    .custom-table td { border-bottom: 1px solid #dee2e6; padding: 8px 10px; vertical-align: top; }
    .custom-table tr:hover { background-color: #f1f3f5; }
    .col-uuid, .col-uid { font-family: monospace; font-size: 0.85em; color: #495057; }
</style>
"""


# -----------------------------------------------------------------------------
# 1. FONCTIONS D'AFFICHAGE & COLORATION (pures, sans dépendance Streamlit —
#    testables indépendamment de l'UI)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_label_color(label: str) -> tuple[str, str]:
    """Couleur de fond et de bordure déterministe basée sur le nom du label."""
    if not label or label in ("-", "OUT OF SCOPE", "nan"):
        return "#f0f2f6", "#6c757d"

    # zlib.crc32 distribue uniformément les teintes sur [0, 359]
    hash_val = zlib.crc32(label.encode("utf-8"))
    hue = hash_val % 360

    return f"hsl({hue}, 85%, 92%)", f"hsl({hue}, 70%, 40%)"


def render_badge(label: object) -> str:
    """Badge HTML coloré pour le type d'entité."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    label = str(label)
    bg, border = get_label_color(label)
    return (
        f'<span style="background-color:{bg}; color:{border}; border:1px solid {border}; '
        f'padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.8em; '
        f'display:inline-block; margin:2px;">{html.escape(label)}</span>'
    )


def render_tagged_html(tagged_text: object) -> str:
    """Convertit un texte balisé (`<SUBJ>...</SUBJ>` etc., sortie
    d'infer_gliner.py) en HTML surligné par classe.

    Un seul passage regex (`finditer`) sur le texte brut, plutôt qu'un
    parsing JSON + reconstruction caractère par caractère : plus simple et
    nettement moins coûteux, et surtout adapté au format réellement produit
    par le pipeline (le texte est déjà balisé, pas accompagné d'une colonne
    d'empans séparée avec des offsets).

    Important : le texte brut est parcouru AVANT tout échappement HTML (pour
    reconnaître les balises `<LABEL>` littérales) ; seuls les fragments de
    texte libre et le contenu de chaque balise sont échappés individuellement
    au moment de la reconstruction.
    """
    if tagged_text is None or (isinstance(tagged_text, float) and pd.isna(tagged_text)):
        return ""
    tagged_text = str(tagged_text)
    if not tagged_text:
        return ""

    pieces: list[str] = []
    cursor = 0
    for match in TAG_PATTERN.finditer(tagged_text):
        pieces.append(html.escape(tagged_text[cursor : match.start()]))
        label, inner = match.group(1), match.group(2)
        bg, border = get_label_color(label)
        pieces.append(
            f'<mark style="background-color:{bg}; border-bottom:2px solid {border}; '
            f'padding:0 3px; border-radius:3px;" title="{html.escape(label)}">'
            f"{html.escape(inner)}"
            f'<sub style="font-size:0.65em; color:{border}; margin-left:2px;">{html.escape(label)}</sub>'
            f"</mark>"
        )
        cursor = match.end()
    pieces.append(html.escape(tagged_text[cursor:]))
    return "".join(pieces)


def highlight_spans_json(text: object, spans_raw: object) -> str:
    """Repli pour l'ancien schéma `text` + `spans` (JSON d'empans avec
    offsets `start`/`end`), au cas où un export plus ancien serait chargé."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    spans = []
    if isinstance(spans_raw, str) and spans_raw.strip():
        try:
            import json

            spans = json.loads(spans_raw)
        except (ValueError, TypeError):
            spans = []
    elif isinstance(spans_raw, list):
        spans = spans_raw

    if not spans:
        return html.escape(text)

    ordered = sorted(
        (s for s in spans if s.get("start") is not None and s.get("end") is not None and s["start"] < s["end"]),
        key=lambda s: s["start"],
    )
    pieces: list[str] = []
    cursor = 0
    for span in ordered:
        start, end, label = span["start"], span["end"], span.get("label", "SPAN")
        if start < cursor:
            continue  # empan chevauchant un empan déjà rendu : ignoré plutôt que casser le balisage
        pieces.append(html.escape(text[cursor:start]))
        bg, border = get_label_color(label)
        pieces.append(
            f'<mark style="background-color:{bg}; border-bottom:2px solid {border}; '
            f'padding:0 3px; border-radius:3px;" title="{html.escape(str(label))}">'
            f"{html.escape(text[start:end])}"
            f'<sub style="font-size:0.65em; color:{border}; margin-left:2px;">{html.escape(str(label))}</sub>'
            f"</mark>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_cell(col: str, value: object, row: pd.Series, has_tagged_text: bool, has_legacy_spans: bool) -> str:
    """Rendu HTML d'une cellule, selon le rôle de sa colonne."""
    if col == "entity":
        return render_badge(value)
    if col == "tagged_text" and has_tagged_text:
        return render_tagged_html(value)
    if col in ("text", "content", "contenu") and has_legacy_spans:
        return highlight_spans_json(value, row.get("spans"))
    if col in ("uuid", "uid"):
        return f'<span class="col-{col}">{html.escape(str(value))}</span>' if pd.notna(value) else ""
    return html.escape(str(value)) if pd.notna(value) else ""


# -----------------------------------------------------------------------------
# 2. CHARGEMENT (mis en cache) ET RECHERCHE VECTORISÉE
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner="Chargement et analyse du CSV...")
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Charge et normalise le CSV. Mis en cache par contenu du fichier : tant
    que le même fichier reste chargé, ce coût n'est payé qu'une fois, jamais
    à chaque interaction (recherche, filtre, tri...).
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "type" in df.columns and "entity" not in df.columns:
        df = df.rename(columns={"type": "entity"})
    return df


def vectorized_search(df: pd.DataFrame, query: str, columns: list[str]) -> pd.Series:
    """Masque booléen de recherche plein texte, vectorisé colonne par colonne
    (`Series.str.contains`) plutôt qu'un `.apply()` ligne par ligne sur tout
    le DataFrame converti en texte — nettement plus rapide sur un gros fichier.
    """
    if not query or not columns:
        return pd.Series(True, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.contains(query, case=False, na=False, regex=False)
    return mask


# -----------------------------------------------------------------------------
# 3. UI
# -----------------------------------------------------------------------------


def main() -> None:
    st.title("Visualiseur d'entités & empans CSV")

    uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Veuillez charger un fichier CSV depuis la barre latérale.")
        return

    df = load_csv(uploaded_file.getvalue())
    all_columns = df.columns.tolist()
    has_tagged_text = "tagged_text" in all_columns
    has_legacy_spans = "text" in all_columns and "spans" in all_columns
    available_count_columns = [c for c in COUNT_COLUMNS if c in all_columns]

    # --- KPIs ---
    st.subheader("Aperçu")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Lignes totales", f"{len(df):,}".replace(",", " "))
    if "entity" in all_columns:
        for i, (name, count) in enumerate(df["entity"].value_counts().head(4).items()):
            kpi_cols[i + 1].metric(str(name), f"{count:,}".replace(",", " "))

    # --- Filtres (sidebar) ---
    st.sidebar.header("Filtres")

    selected_entities = None
    if "entity" in all_columns:
        unique_entities = sorted(df["entity"].dropna().astype(str).unique().tolist())
        selected_entities = st.sidebar.multiselect(
            "Type d'entité", options=unique_entities, default=unique_entities
        )

    only_zero_ner = False
    if available_count_columns:
        only_zero_ner = st.sidebar.checkbox(
            "Uniquement les lignes sans aucun empan détecté",
            help="Utile pour repérer les échecs d'inférence GLiNER (infer_gliner.py).",
        )

    search_columns_default = [c for c in ["markdown", "tagged_text", "uuid", "uid"] if c in all_columns]
    search_columns = st.sidebar.multiselect(
        "Colonnes concernées par la recherche",
        options=all_columns,
        default=search_columns_default or all_columns[:3],
    )
    search_query = st.sidebar.text_input("Recherche plein texte")

    selected_columns = st.sidebar.multiselect(
        "Colonnes à afficher",
        options=all_columns,
        default=[c for c in DEFAULT_VISIBLE_COLUMNS if c in all_columns] or all_columns[:6],
    )

    # --- Application des filtres (vectorisée) ---
    mask = pd.Series(True, index=df.index)
    if selected_entities is not None:
        mask &= df["entity"].astype(str).isin(selected_entities)
    if only_zero_ner and available_count_columns:
        for col in available_count_columns:
            mask &= df[col].fillna(0).astype(float).eq(0)
    mask &= vectorized_search(df, search_query, search_columns)
    filtered_df = df[mask]

    st.write(f"**{len(filtered_df):,}** ligne(s) après filtrage sur **{len(df):,}**.".replace(",", " "))

    if not selected_columns:
        st.warning("Veuillez sélectionner au moins une colonne à afficher.")
        return

    st.subheader("Résultats")

    if filtered_df.empty:
        st.info("Aucune ligne ne correspond aux filtres actuels.")
        return

    st.download_button(
        "Télécharger le résultat filtré (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtre.csv",
        mime="text/csv",
    )

    # --- Contrôles : tri, taille de page, navigation — une seule ligne compacte ---
    ctrl_sort_col, ctrl_sort_dir, ctrl_page_size, ctrl_prev, ctrl_page_label, ctrl_next = st.columns(
        [2.2, 1.3, 1.1, 0.6, 1.6, 0.6]
    )

    with ctrl_sort_col:
        sort_column = st.selectbox("Trier par", options=["(ordre du fichier)"] + selected_columns)
    with ctrl_sort_dir:
        sort_ascending = st.selectbox("Ordre", options=["Croissant", "Décroissant"]) == "Croissant"
    with ctrl_page_size:
        page_size = st.selectbox(
            "Lignes/page", options=PAGE_SIZE_OPTIONS, index=PAGE_SIZE_OPTIONS.index(DEFAULT_PAGE_SIZE)
        )

    if sort_column != "(ordre du fichier)":
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending, na_position="last")

    n_rows = len(filtered_df)
    n_pages = max(1, (n_rows - 1) // page_size + 1)

    if "page" not in st.session_state:
        st.session_state.page = 0
    if st.session_state.get("_last_page_size") != page_size:
        st.session_state.page = 0
        st.session_state["_last_page_size"] = page_size
    st.session_state.page = min(st.session_state.page, n_pages - 1)

    with ctrl_prev:
        st.write("")
        if st.button("◀", disabled=st.session_state.page <= 0, use_container_width=True):
            st.session_state.page -= 1
    with ctrl_page_label:
        st.write("")
        st.markdown(
            f"<div style='padding-top:8px; text-align:center;'>Page {st.session_state.page + 1} / {n_pages}"
            f" &nbsp;·&nbsp; {n_rows:,} ligne(s)</div>".replace(",", " "),
            unsafe_allow_html=True,
        )
    with ctrl_next:
        st.write("")
        if st.button("▶", disabled=st.session_state.page >= n_pages - 1, use_container_width=True):
            st.session_state.page += 1

    start_row = st.session_state.page * page_size
    end_row = min(start_row + page_size, n_rows)
    page_df = filtered_df.iloc[start_row:end_row]

    # --- Table unique : structure + surlignage, une seule page à la fois ---
    html_rows = []
    for idx, row in page_df.iterrows():
        cells = "".join(
            f"<td>{render_cell(col, row.get(col), row, has_tagged_text, has_legacy_spans)}</td>"
            for col in selected_columns
        )
        html_rows.append(f"<tr><td>{idx}</td>{cells}</tr>")

    header_cells = "".join(f"<th>{html.escape(col.upper())}</th>" for col in selected_columns)
    table_html = (
        TABLE_CSS
        + "<div class='table-scroll'><table class='custom-table'><thead><tr><th>#</th>"
        + header_cells
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
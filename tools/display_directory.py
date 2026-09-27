"""Visualiseur Streamlit du résultat final d'un annuaire : le CSV NER
(`*.merged.ner.csv` d'`infer_gliner.py`, ou sa version corrigée
`*.merged.ner.curated.csv`).

    uv run streamlit run tools/display_directory.py

Le fichier se choisit dans `annuaires/` (ou se téléverse). Chaque entrée
est replacée dans sa **rubrique** (chemin des titres qui la précèdent, selon
leur niveau `#`, `##`, …) et affichée avec ses empans SUBJ / DESC / ADDR
colorés. Filtres : type de ligne, rubrique, pages, recherche, signature
(suite des classes, ex. `SUBJ,DESC,ADDR`), entrées suspectes et motifs,
confiance. En ordre du fichier, un bandeau marque chaque changement de
rubrique.

Motifs de suspicion : colonne `ner_suspect` d'`infer_gliner.py` quand elle
existe ; sinon (fichier plus ancien), ils sont recalculés depuis
`tagged_text` par `lib/ner/suspicion.py`, sans le motif `score bas` faute de
scores.

Seule une page de la table est rendue en HTML à la fois ; le CSV et les
colonnes dérivées sont calculés une fois par fichier (`st.cache_data`).
"""

import html
import io
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from lib.ner.html import DEFAULT_COLORS, LABEL_COLORS, SPAN_CSS, badge, render_tagged_html
from lib.ner.spans import parse_tagged_text, signature
from lib.ner.suspicion import DEFAULT_MIN_SCORE, SEPARATOR, suspicion_reasons

ANNUAIRES_DIR = Path("annuaires")
NER_SUFFIXES = (".merged.ner.csv", ".merged.ner.curated.csv")
PAGE_SIZE_OPTIONS = [25, 50, 100, 200]
FILE_ORDER = "ordre du fichier"
SORT_OPTIONS = [FILE_ORDER, "confiance croissante", "page", "signature"]
NO_SECTION = "(avant le premier titre)"

ENTITY_COLORS = {
    "ENTRY": ("#e0e7ff", "#3730a3"),
    "TITLE": ("#fce7f3", "#9d174d"),
    "OUT OF SCOPE": ("#f1f5f9", "#64748b"),
}
HEADING_PATTERN = re.compile(r"^\s*(#+)[\s ]*")

TABLE_CSS = """
  .legend span { margin-right: 10px; }
  .table-scroll { max-height: 72vh; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; }
  .ner-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  .ner-table th { background: #f8fafc; border-bottom: 2px solid #e2e8f0; padding: 8px; text-align: left;
                  position: sticky; top: 0; z-index: 1; }
  .ner-table td { border-bottom: 1px solid #f1f5f9; padding: 6px 8px; vertical-align: top; }
  .ner-table tr:hover td { background: #f8fafc; }
  .ner-table tr.section td { background: #f1f5f9; color: #334155; font-weight: 600; font-size: 0.85em; }
  .mono { font-family: monospace; font-size: 0.85em; color: #64748b; }
  .low { color: #b91c1c; font-weight: 600; }
"""
CSS = f"<style>{TABLE_CSS}{SPAN_CSS}</style>"


# ----------------------------------------------------------------------
# Colonnes dérivées
# ----------------------------------------------------------------------
def title_level(markdown: str) -> tuple[int, str]:
    """Niveau d'un titre (nombre de `#`, 9 sans `#`) et son texte nettoyé."""
    match = HEADING_PATTERN.match(markdown)
    level = len(match.group(1)) if match else 9
    text = markdown[match.end() :] if match else markdown
    return level, " ".join(text.replace("*", "").split())


def section_paths(entities: pd.Series, markdown: pd.Series) -> list[str]:
    """Rubrique de chaque ligne : chemin des titres en cours (« A › B »), mis
    à jour à chaque TITLE selon son niveau."""
    stack: list[tuple[int, str]] = []
    paths = []
    for entity, text in zip(entities, markdown):
        if entity == "TITLE":
            level, title = title_level(str(text))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        paths.append(" › ".join(title for _, title in stack) or NO_SECTION)
    return paths


def section_label(path: str) -> str:
    """Libellé court d'une rubrique : son dernier titre, indenté selon sa
    profondeur (le chemin complet est trop long pour une liste)."""
    parts = path.split(" › ")
    return " " * (len(parts) - 1) + parts[-1]


def derive_signature(tagged_text: str) -> str:
    if not tagged_text:
        return ""
    try:
        return signature(parse_tagged_text(tagged_text)[1]) or "(aucun empan)"
    except ValueError:
        return "(balisage invalide)"


def derive_reasons(tagged_text: str) -> str:
    """Motifs structurels recalculés (fichier sans colonne `ner_suspect`)."""
    try:
        text, spans = parse_tagged_text(tagged_text)
    except ValueError:
        return "balisage invalide"
    return SEPARATOR.join(suspicion_reasons(text, spans))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ("entity", "markdown", "tagged_text", "ner_suspect"):
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    if "page_index" in df.columns:
        first_page = df["page_index"].astype(str).str.split(",").str[0]
        df["page"] = pd.to_numeric(first_page, errors="coerce").astype("Int64")
    if "entity" in df.columns and "markdown" in df.columns:
        df["rubrique"] = section_paths(df["entity"], df["markdown"])
    is_entry = df["entity"].eq("ENTRY") if "entity" in df.columns else pd.Series(True, index=df.index)
    if "tagged_text" in df.columns:
        df["signature"] = ""
        df.loc[is_entry, "signature"] = df.loc[is_entry, "tagged_text"].map(derive_signature)
        if "ner_suspect" not in df.columns:
            df["ner_suspect"] = ""
            df.loc[is_entry, "ner_suspect"] = df.loc[is_entry, "tagged_text"].map(derive_reasons)
    if "ner_confidence" in df.columns:
        df["ner_confidence"] = pd.to_numeric(df["ner_confidence"], errors="coerce")
    return df


@st.cache_data(show_spinner="Lecture du CSV…")
def load_path(path: str, modified: float) -> pd.DataFrame:  # `modified` invalide le cache si le fichier change
    return add_derived_columns(pd.read_csv(path, dtype=str, keep_default_na=False))


@st.cache_data(show_spinner="Lecture du CSV…")
def load_bytes(content: bytes) -> pd.DataFrame:
    return add_derived_columns(pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False))


def reasons_of(series: pd.Series) -> pd.Series:
    """Série de listes de motifs."""
    return series.fillna("").map(lambda value: [r for r in value.split(SEPARATOR) if r])


def text_mask(df: pd.DataFrame, query: str, columns: list[str], regex: bool) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False, regex=regex)
    return mask


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def choose_file() -> tuple[str, pd.DataFrame] | None:
    source = st.sidebar.radio("Source", ["annuaires/", "téléverser"], horizontal=True, label_visibility="collapsed")
    if source == "annuaires/":
        paths = sorted(p for p in ANNUAIRES_DIR.glob("**/*.csv") if p.name.endswith(NER_SUFFIXES))
        if not paths:
            st.sidebar.warning(f"Aucun fichier `*.merged.ner.csv` ni `*.merged.ner.curated.csv` sous `{ANNUAIRES_DIR}/`.")
            return None
        path = st.sidebar.selectbox(
            "Fichier NER",
            paths,
            format_func=lambda p: str(p.relative_to(ANNUAIRES_DIR)),
            index=next((i for i, p in enumerate(paths) if p.name.endswith(".ner.curated.csv")), 0),
        )
        return path.name, load_path(str(path), path.stat().st_mtime)
    uploaded = st.sidebar.file_uploader("CSV NER", type=["csv"])
    return (uploaded.name, load_bytes(uploaded.getvalue())) if uploaded else None


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    columns = df.columns
    mask = pd.Series(True, index=df.index)
    state: dict = {}

    st.sidebar.header("Filtres")
    if "entity" in columns:
        types = sorted(df["entity"].unique())
        chosen = st.sidebar.multiselect("Type de ligne", types, default=[t for t in types if t == "ENTRY"] or types)
        mask &= df["entity"].isin(chosen)
        state["types"] = chosen

    if "rubrique" in columns:
        sections = list(dict.fromkeys(df["rubrique"]))
        section = st.sidebar.selectbox(
            "Rubrique (sous-rubriques comprises)", ["(toutes)", *sections], format_func=section_label
        )
        if section != "(toutes)":
            mask &= df["rubrique"].eq(section) | df["rubrique"].str.startswith(section + " › ")
        state["section"] = section

    if "page" in columns and df["page"].notna().any():
        low, high = int(df["page"].min()), int(df["page"].max())
        if low < high:
            pages = st.sidebar.slider("Pages", low, high, (low, high))
            mask &= df["page"].between(*pages)
            state["pages"] = pages

    query = st.sidebar.text_input("Recherche", placeholder="texte, uid, uuid…")
    if query:
        regex = st.sidebar.checkbox("Expression régulière")
        search_columns = [c for c in ("markdown", "tagged_text", "uid", "uuid") if c in columns]
        try:
            mask &= text_mask(df, query, search_columns, regex)
        except re.error as error:
            st.sidebar.error(f"Expression invalide : {error}")
        state["query"] = (query, regex)

    st.sidebar.header("NER")
    if "signature" in columns:
        counts = df.loc[df["signature"].ne(""), "signature"].value_counts()
        signatures = st.sidebar.multiselect(
            "Signature", counts.index.tolist(), format_func=lambda s: f"{s}  ({counts[s]})", placeholder="toutes"
        )
        if signatures:
            mask &= df["signature"].isin(signatures)
        state["signatures"] = signatures

    if "ner_suspect" in columns:
        reasons = reasons_of(df["ner_suspect"])
        if st.sidebar.checkbox("Uniquement les entrées suspectes"):
            mask &= reasons.map(bool)
            state["suspect"] = True
        all_reasons = sorted({r for rs in reasons for r in rs})
        chosen_reasons = st.sidebar.multiselect("Motifs", all_reasons, placeholder="tous")
        if chosen_reasons:
            mask &= reasons.map(lambda rs: any(r in rs for r in chosen_reasons))
        state["reasons"] = chosen_reasons

    if "ner_confidence" in columns and df["ner_confidence"].notna().any():
        maximum = st.sidebar.slider("Confiance maximale", 0.0, 1.0, 1.0, 0.01)
        if maximum < 1.0:
            mask &= df["ner_confidence"].le(maximum)
        state["confidence"] = maximum
    return mask, state


def kpis(df: pd.DataFrame) -> None:
    entries = df[df["entity"].eq("ENTRY")] if "entity" in df.columns else df
    n = len(entries)
    cells = st.columns(5)
    cells[0].metric("Entrées", f"{n:,}".replace(",", " "))
    if "entity" in df.columns:
        cells[1].metric("Titres", f"{int(df['entity'].eq('TITLE').sum()):,}".replace(",", " "))
    if "ner_suspect" in df.columns and n:
        suspects = int(entries["ner_suspect"].ne("").sum())
        cells[2].metric("Suspectes", f"{suspects:,}".replace(",", " "), f"{suspects / n:.1%}", delta_color="off")
    if "signature" in df.columns:
        cells[3].metric("Sans empan", int(entries["signature"].eq("(aucun empan)").sum()))
    if "ner_confidence" in df.columns and entries["ner_confidence"].notna().any():
        cells[4].metric("Confiance médiane", f"{entries['ner_confidence'].median():.3f}")


def statistics(df: pd.DataFrame) -> None:
    entries = df[df["entity"].eq("ENTRY")] if "entity" in df.columns else df
    if entries.empty:
        st.info("Aucune entrée.")
        return
    st.caption("Sur toutes les entrées du fichier, indépendamment des filtres.")
    left, right = st.columns(2)
    if "signature" in entries.columns:
        with left:
            st.markdown("**Signatures**")
            table = entries["signature"].value_counts().rename_axis("signature").reset_index(name="entrées")
            table["part"] = table["entrées"] / len(entries)
            st.dataframe(table, hide_index=True, column_config={"part": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    if "ner_suspect" in entries.columns:
        with right:
            st.markdown("**Motifs de suspicion**")
            reasons = reasons_of(entries["ner_suspect"]).explode().dropna()
            table = reasons.value_counts().rename_axis("motif").reset_index(name="entrées")
            table["part"] = table["entrées"] / len(entries)
            st.dataframe(table, hide_index=True, column_config={"part": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    if "rubrique" in entries.columns and "ner_suspect" in entries.columns:
        st.markdown("**Rubriques, par nombre d'entrées suspectes**")
        grouped = entries.assign(suspecte=entries["ner_suspect"].ne("")).groupby("rubrique", sort=False)
        table = grouped.agg(entrées=("suspecte", "size"), suspectes=("suspecte", "sum")).reset_index()
        table["rubrique"] = table["rubrique"].map(lambda path: " › ".join(path.split(" › ")[-2:]))
        table["taux"] = table["suspectes"] / table["entrées"]
        if "page" in entries.columns:
            table["première page"] = grouped["page"].min().values
        st.dataframe(
            table.sort_values(["suspectes", "taux"], ascending=False),
            hide_index=True,
            column_config={"taux": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
        )


def confidence_html(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    css = ' class="low"' if value < DEFAULT_MIN_SCORE else ""
    return f"<span{css}>{value:.3f}</span>"


def row_html(row: pd.Series, extra_columns: list[str], has: dict) -> str:
    entity = row.get("entity", "")
    if entity == "ENTRY" and has["tagged"]:
        content = render_tagged_html(row["tagged_text"]) or html.escape(row.get("markdown", ""))
    else:
        content = html.escape(row.get("markdown", ""))
    uuid = str(row.get("uuid", ""))
    cells = [
        f'<td class="mono">{row["page"] if has["page"] and pd.notna(row["page"]) else ""}</td>',
        f'<td class="mono" title="{html.escape(uuid)}">{html.escape(uuid[:8])}</td>',
        f"<td>{badge(entity, ENTITY_COLORS.get(entity, DEFAULT_COLORS)) if entity else ''}</td>",
        f"<td>{content}</td>",
    ]
    if has["signature"]:
        cells.append(f'<td class="mono">{html.escape(row["signature"])}</td>')
    if has["confidence"]:
        cells.append(f"<td>{confidence_html(row['ner_confidence'])}</td>")
    if has["suspect"]:
        reasons = [r for r in row["ner_suspect"].split(SEPARATOR) if r]
        cells.append("<td>" + "".join(badge(r, ("#fee2e2", "#b91c1c")) for r in reasons) + "</td>")
    cells += [f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in extra_columns]
    return "<tr>" + "".join(cells) + "</tr>"


def results_table(view: pd.DataFrame, extra_columns: list[str], page_size: int, sort: str) -> None:
    has = {
        "page": "page" in view.columns,
        "tagged": "tagged_text" in view.columns,
        "signature": "signature" in view.columns,
        "confidence": "ner_confidence" in view.columns,
        "suspect": "ner_suspect" in view.columns,
    }
    headers = ["page", "uuid", "type", "texte"]
    headers += [name for key, name in (("signature", "signature"), ("confidence", "confiance"), ("suspect", "motifs")) if has[key]]
    headers += extra_columns

    n_pages = max(1, -(-len(view) // page_size))
    st.session_state.page = min(st.session_state.get("page", 0), n_pages - 1)
    previous, label, following = st.columns([1, 3, 1])
    if previous.button("◀ Précédente", disabled=st.session_state.page == 0, use_container_width=True):
        st.session_state.page -= 1
    if following.button("Suivante ▶", disabled=st.session_state.page >= n_pages - 1, use_container_width=True):
        st.session_state.page += 1
    label.markdown(
        f"<div style='text-align:center;padding-top:6px'>Page <b>{st.session_state.page + 1}</b> / {n_pages} · "
        f"{len(view):,} ligne(s)</div>".replace(",", " "),
        unsafe_allow_html=True,
    )

    start = st.session_state.page * page_size
    shown = view.iloc[start : start + page_size]
    body, current_section = [], None
    for _, row in shown.iterrows():
        if sort == FILE_ORDER and "rubrique" in row and row["rubrique"] != current_section:
            current_section = row["rubrique"]
            body.append(f'<tr class="section"><td colspan="{len(headers)}">{html.escape(current_section)}</td></tr>')
        body.append(row_html(row, extra_columns, has))
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    st.markdown(
        f"<div class='table-scroll'><table class='ner-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Annuaire — résultat NER", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    chosen = choose_file()
    if chosen is None:
        st.title("Annuaire — résultat NER")
        st.info("Choisissez un fichier `*.merged.ner.csv` ou `*.merged.ner.curated.csv` dans la barre latérale.")
        return
    name, df = chosen
    st.title(name)
    legend = "".join(badge(label, colors) for label, colors in LABEL_COLORS.items())
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    if "ner_confidence" not in df.columns:
        st.caption("Fichier antérieur au drapeau `ner_suspect` : motifs recalculés depuis `tagged_text`, sans « score bas ».")
    kpis(df)

    mask, filter_state = sidebar_filters(df)
    st.sidebar.header("Affichage")
    sort_columns = {FILE_ORDER: None, "confiance croissante": "ner_confidence", "page": "page", "signature": "signature"}
    sort = st.sidebar.selectbox("Tri", [o for o in SORT_OPTIONS if sort_columns[o] is None or sort_columns[o] in df.columns])
    page_size = st.sidebar.selectbox("Lignes par page", PAGE_SIZE_OPTIONS, index=1)
    derived = {"page", "rubrique", "signature"}
    extra_columns = st.sidebar.multiselect(
        "Colonnes supplémentaires", [c for c in df.columns if c not in derived | {"markdown", "tagged_text", "uuid", "entity", "ner_suspect", "ner_confidence"}]
    )

    # Retour à la première page quand la sélection change.
    signature_state = (name, repr(filter_state), sort, page_size)
    if st.session_state.get("_selection") != signature_state:
        st.session_state["_selection"] = signature_state
        st.session_state.page = 0

    view = df[mask]
    if sort_columns[sort]:
        view = view.sort_values(sort_columns[sort], na_position="last", kind="stable")

    entries_tab, stats_tab = st.tabs([f"Lignes ({len(view):,})".replace(",", " "), "Statistiques"])
    with entries_tab:
        if view.empty:
            st.info("Aucune ligne ne correspond aux filtres.")
        else:
            results_table(view, extra_columns, page_size, sort)
            st.download_button(
                "Télécharger la sélection (CSV)",
                view.drop(columns=[c for c in derived if c in view.columns]).to_csv(index=False).encode("utf-8"),
                file_name=name.replace(".csv", ".selection.csv"),
                mime="text/csv",
            )
    with stats_tab:
        statistics(df)


if __name__ == "__main__":
    main()

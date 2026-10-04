"""Rendu HTML des empans NER (fonctions pures, sans Streamlit), partagé par
les visualiseurs `numrev view directory` et `numrev view alignment`."""

import html

from numrev.ner.spans import parse_tagged_text

LABEL_COLORS = {  # fond, texte/bordure
    "SUBJ": ("#dbeafe", "#1d4ed8"),
    "DESC": ("#fef3c7", "#a16207"),
    "ADDR": ("#dcfce7", "#15803d"),
}
DEFAULT_COLORS = ("#f1f5f9", "#475569")

# Styles des badges et des empans, à inclure dans la feuille de style du visualiseur.
SPAN_CSS = """
  .badge { padding: 1px 7px; border-radius: 4px; font-size: 0.78em; font-weight: 600; white-space: nowrap;
           display: inline-block; margin: 1px 2px 1px 0; border: 1px solid; }
  mark.span { padding: 0 3px; border-radius: 3px; border-bottom: 2px solid; }
  mark.span sub { font-size: 0.62em; margin-left: 2px; }
"""


def badge(label: str, colors: tuple[str, str] = DEFAULT_COLORS) -> str:
    background, foreground = colors
    return (
        f'<span class="badge" style="background:{background};color:{foreground};border-color:{foreground}">'
        f"{html.escape(label)}</span>"
    )


def render_tagged_html(tagged_text: str) -> str:
    """Texte balisé (`<SUBJ>…</SUBJ>`, avec `&lt;`/`&gt;` pour les chevrons
    du texte) → HTML surligné par classe. Un balisage invalide est affiché
    tel quel, précédé d'un avertissement."""
    if not tagged_text:
        return ""
    try:
        text, spans = parse_tagged_text(tagged_text)
    except ValueError:
        return badge("balisage invalide", ("#fee2e2", "#b91c1c")) + html.escape(tagged_text)
    pieces, cursor = [], 0
    for span in sorted(spans, key=lambda s: s.start):
        background, foreground = LABEL_COLORS.get(span.label, DEFAULT_COLORS)
        pieces.append(html.escape(text[cursor : span.start]))
        pieces.append(
            f'<mark class="span" style="background:{background};border-color:{foreground}" title="{span.label}">'
            f'{html.escape(text[span.start : span.end])}<sub style="color:{foreground}">{span.label}</sub></mark>'
        )
        cursor = span.end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)

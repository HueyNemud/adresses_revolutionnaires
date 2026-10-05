"""Outil expérimental · recompose un annuaire en LaTeX, façon facsimilé
d'époque, à partir de la sortie finale de la chaîne (les `*.ner.csv` du
volume).

    uv run numrev typeset annuaires/<volume> [--pdf] --apply

Le texte est fluide : LaTeX pagine librement, et chaque page de l'original
est repérée en marge par son folio imprimé (lu dans les en-têtes de page,
à défaut « p. <page du PDF> »). Rendu :

- titre `#` : bloc pleine largeur ; `##` : rubrique centrée en capitales,
  reprise en titre courant (« Agens de change, Architectes. — PARIS. ») ;
  `###` et plus : intertitre en italique ;
- ENTRY : paragraphe en retrait suspendu, le sujet (SUBJ) en petites
  capitales, l'emphase Markdown de l'OCR conservée ;
- OUT OF SCOPE : en-têtes et pieds de page, images et groupes vides ignorés
  (le titre courant est régénéré) ; avis, notes et tables en petit corps
  (les cellules d'une table, éclatées en lignes par `numrev extract`, sont
  rejointes par « · »).

Sortie par défaut : `annuaires/<volume>/<volume>.tex` (`--apply`) ;
`--pdf` la compile avec `pdflatex` (deux passes, pour les titres courants ;
pdfLaTeX plutôt que LuaLaTeX, qui exige `luaotfload`, absent de certaines
installations Debian).
"""

import argparse
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from numrev.command import CommandError, Writes, add_apply_argument, console, require_dir
from numrev.crf.features import EMPHASIS_PATTERN
from numrev.curation import read_csv
from numrev.ner.spans import parse_tagged_text
from numrev.paths import NER, range_dirs, range_document, typeset_output
from numrev.titles import HEADING_PATTERN, title_level, title_text

DESCRIPTION = "Recompose un annuaire en LaTeX (facsimilé d'époque) à partir de ses CSV NER."

ENTRY, TITLE, OUT_OF_SCOPE = "ENTRY", "TITLE", "OUT OF SCOPE"
SKIPPED_LABELS = {"Page-Header", "Page-Footer", "Image", "List-Group"}
NOTICE_LABELS = {"Text", "Footnote", "Table"}

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\xa0": "~",
}
MARKDOWN_ESCAPABLE = set("\\`*_{}[]()#+-.!|<>")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Folio imprimé en tête d'un en-tête de page : « 116 *Arquebusiers…* », « 513 ».
FOLIO_PATTERN = re.compile(r"^[\s#*]*(\d{1,4})\b")
SPAN_MACROS = {"SUBJ": r"\nom", "DESC": "", "ADDR": ""}


@dataclass
class Line:
    entity: str
    markdown: str
    tagged_text: str
    page: int  # page du PDF, base 0 (première page de l'entité)
    label: str  # type de bloc OCR (premier bloc de l'entité)


@dataclass
class Stats:
    pages: int = 0
    folios: int = 0
    titles: Counter = field(default_factory=Counter)
    entries: int = 0
    notices: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    fallback: int = 0  # ENTRY au balisage invalide, rendues sans empans


# ----------------------------------------------------------------------
# Lecture
# ----------------------------------------------------------------------
def read_volume(volume_dir: Path) -> list[Line]:
    """Lignes du volume dans l'ordre : plages triées par première page."""
    ranges = range_dirs(volume_dir)
    if not ranges:
        raise CommandError(f"aucun dossier de plage (`<début>-<fin>`) dans {volume_dir}.")
    lines = []
    for range_dir in ranges:
        path = range_document(range_dir, NER)
        if not path.exists():
            raise CommandError(f"CSV NER introuvable : {path}")
        for row in read_csv(path)[1]:
            lines.append(
                Line(
                    entity=row["entity"],
                    markdown=row["markdown"],
                    tagged_text=row.get("tagged_text", ""),
                    page=int(row["page_index"].split(",")[0]),
                    label=row["data_block_label"].split(",")[0],
                )
            )
    return lines


# ----------------------------------------------------------------------
# Texte en ligne
# ----------------------------------------------------------------------
def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def _styled_chars(text: str, labels: list[str | None]) -> list[tuple[str, tuple[bool, bool, str | None]]]:
    """Caractères visibles de `text` (Markdown) avec leur état (italique,
    gras, classe d'empan) : marqueurs d'emphase retirés, échappements
    Markdown résolus, blancs réduits."""
    chars = []
    italic = bold = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] in MARKDOWN_ESCAPABLE:
            chars.append((text[index + 1], (italic, bold, labels[index + 1])))
            index += 2
            continue
        marker = EMPHASIS_PATTERN.match(text, index)
        if marker:
            width = len(marker.group())
            italic ^= width in (1, 3)
            bold ^= width in (2, 3)
            index = marker.end()
            continue
        if char.isspace() and char != "\xa0":
            char = " "
            if chars and chars[-1][0] == " ":
                index += 1
                continue
        chars.append((char, (italic, bold, labels[index])))
        index += 1
    while chars and chars[0][0] in " \xa0":
        chars.pop(0)
    while chars and chars[-1][0] in " \xa0":
        chars.pop()
    return chars


def _wrap(text: str, state: tuple[bool, bool, str | None]) -> str:
    italic, bold, label = state
    result = latex_escape(text)
    if italic:
        result = rf"\textit{{{result}}}"
    if bold:
        result = rf"\textbf{{{result}}}"
    macro = SPAN_MACROS.get(label or "", "")
    if macro:
        result = rf"{macro}{{{result}}}"
    return result


def inline(markdown: str, tagged_text: str = "") -> str:
    """Texte d'une ligne en LaTeX. Chaque segment de même état (emphase,
    empan) est un groupe fermé : les groupes restent bien imbriqués même
    quand emphase et empans se chevauchent (`**<SUBJ>A**rchédéacon</SUBJ>`).
    Une emphase non fermée s'arrête en fin de ligne ; un `tagged_text` vide
    ou mal formé donne le Markdown seul."""
    try:
        text, spans = parse_tagged_text(tagged_text) if tagged_text else (markdown, [])
    except ValueError:
        text, spans = markdown, []
    if not spans:
        text = IMAGE_PATTERN.sub("", text)
    labels: list[str | None] = [None] * len(text)
    for span in spans:
        labels[span.start : span.end] = [span.label] * (span.end - span.start)
    pieces = []
    run, state = "", None
    for char, char_state in _styled_chars(text, labels):
        if char_state != state and run:
            pieces.append(_wrap(run, state))
            run = ""
        run += char
        state = char_state
    if run:
        pieces.append(_wrap(run, state))
    return "".join(pieces)


def running_title(markdown: str) -> str:
    """Nom de rubrique pour le titre courant, en casse de phrase et sans
    point final : « ## AGENS DE CHANGE. » → « Agens de change »."""
    text = title_text(markdown).rstrip(" .")
    return text[:1].upper() + text[1:].lower()


def heading_body(markdown: str) -> str:
    return HEADING_PATTERN.sub("", markdown, count=1)


def folio(line: Line) -> str | None:
    """Folio imprimé d'un en-tête de page (ou d'un titre courant que l'OCR a
    pris pour un titre de section), None sinon."""
    if not is_running_head(line) or line.label == "Page-Footer":
        return None
    match = FOLIO_PATTERN.match(line.markdown)
    return match.group(1) if match else None


def is_running_head(line: Line) -> bool:
    if line.entity != OUT_OF_SCOPE:
        return False
    if line.label in ("Page-Header", "Page-Footer"):
        return True
    return line.label == "Section-Header" and bool(FOLIO_PATTERN.match(line.markdown)) and "PARIS" in line.markdown.upper()


# ----------------------------------------------------------------------
# Blocs
# ----------------------------------------------------------------------
def body(lines: list[Line], stats: Stats) -> list[str]:
    """Corps LaTeX du volume, un bloc par ligne de sortie."""
    folios: dict[int, str] = {}
    for line in lines:
        found = folio(line)
        if found and line.page not in folios:
            folios[line.page] = found
    stats.pages = len({line.page for line in lines})
    stats.folios = len(folios)

    blocks: list[str] = []
    pending: str | None = None  # repère de page en attente d'un paragraphe
    page = None
    table: list[str] = []

    def margin() -> str:
        nonlocal pending
        mark, pending = pending, None
        return rf"\folio{{{mark}}}" if mark else ""

    def flush_table() -> None:
        if table:
            blocks.append(rf"\avis{{{margin()}{' · '.join(table)}}}")
            table.clear()

    for line in lines:
        if line.page != page:
            page = line.page
            pending = folios.get(page, f"p.~{page + 1}")
        if not (line.entity == OUT_OF_SCOPE and line.label == "Table"):
            flush_table()
        if line.entity == TITLE:
            level = title_level(line.markdown) or 4
            stats.titles[min(level, 3)] += 1
            text = inline(heading_body(line.markdown))
            if level == 1:
                blocks.append(rf"\partie{{{text}}}")
            elif level == 2:
                blocks.append(rf"\rubrique{{{text}}}{{{latex_escape(running_title(line.markdown))}}}")
            else:
                blocks.append(rf"\intertitre{{{text}}}")
        elif line.entity == ENTRY:
            stats.entries += 1
            try:
                parse_tagged_text(line.tagged_text)
            except ValueError:
                stats.fallback += 1
            blocks.append(rf"\entree{{{margin()}{inline(line.markdown, line.tagged_text)}}}")
        else:
            text = inline(line.markdown)
            if line.label in SKIPPED_LABELS or is_running_head(line) or not text:
                stats.skipped[line.label] += 1
            elif line.label == "Table":
                stats.notices[line.label] += 1
                table.append(text)
            elif line.label == "Section-Header":
                stats.notices[line.label] += 1
                blocks.append(rf"\intertitre{{{text}}}")
            else:
                stats.notices[line.label] += 1
                blocks.append(rf"\avis{{{margin()}{text}}}")
    flush_table()
    return blocks


PREAMBLE = r"""% Annuaire recomposé par `numrev typeset` à partir des CSV NER du volume.
\documentclass[9pt,twoside,twocolumn]{extarticle}
\usepackage[paperwidth=130mm,paperheight=205mm,inner=13mm,outer=13mm,top=17mm,bottom=13mm,
  headheight=15pt,headsep=4mm,columnsep=5mm,marginparwidth=9mm,marginparsep=1.5mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[osf]{ebgaramond}
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{needspace}
\usepackage{fancyhdr}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\tolerance=2000
\renewcommand{\baselinestretch}{0.98}

% Titre courant : rubriques de la page, comme l'original — celle qui court en
% haut de page (à défaut, la première ouverte), puis la dernière ouverte.
\NewMarkClass{rubrique}
\makeatletter
\newcommand{\rubriques}{%
  \protected@edef\rubriquehaut{\TopMark{rubrique}}%
  \ifx\rubriquehaut\@empty\protected@edef\rubriquehaut{\FirstMark{rubrique}}\fi
  \protected@edef\rubriquebas{\LastMark{rubrique}}%
  \rubriquehaut\ifx\rubriquehaut\rubriquebas\else, \rubriquebas\fi}
\makeatother
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[C]{\itshape\rubriques. — PARIS.}
\renewcommand{\headrulewidth}{0pt}

\newcommand{\nom}[1]{\textsc{#1}}
\newcommand{\folio}[1]{\marginpar{\raggedright\scriptsize\itshape #1}}
\newcommand{\entree}[1]{\par\RaggedRight\hangindent=1em\hangafter=1 #1\par}
\newcommand{\avis}[1]{\par\addvspace{0.4ex}{\RaggedRight\small\itshape #1\par}\addvspace{0.4ex}}
\newcommand{\intertitre}[1]{\par\Needspace{3\baselineskip}\addvspace{0.8ex}%
  {\centering\itshape #1\par}\nopagebreak\addvspace{0.4ex}}
\newcommand{\rubrique}[2]{\par\Needspace{4\baselineskip}\addvspace{1.6ex}%
  {\centering\rule{2em}{0.3pt}\par}\nopagebreak\addvspace{0.6ex}%
  \InsertMark{rubrique}{#2}%
  {\centering\MakeUppercase{#1}\par}\nopagebreak\addvspace{0.8ex}}
\newcommand{\partie}[1]{\twocolumn[{\centering\vspace*{2ex}\rule{0.6\textwidth}{0.4pt}\par\vspace{1.5ex}
  {\large\MakeUppercase{#1}\par}\vspace{1.5ex}\rule{0.6\textwidth}{0.4pt}\par\vspace{3ex}}]}

\begin{document}
"""


def title_page(volume: str, lines: list[Line]) -> str:
    year = re.match(r"\d{4}", volume)
    first = next((line for line in lines if line.entity == TITLE and title_level(line.markdown) == 1), None)
    subtitle = inline(heading_body(first.markdown)) if first else ""
    return rf"""\begin{{titlepage}}
\centering
\vspace*{{18mm}}
{{\scshape\LARGE Annuaire\par}}
\vspace{{3mm}}
{{\scshape du commerce de Paris\par}}
\vspace{{8mm}}\rule{{40mm}}{{0.4pt}}\par\vspace{{8mm}}
{{\small\itshape {subtitle}\par}}
\vfill
{{\Large {year.group() if year else ""}\par}}
\vspace{{4mm}}
{{\footnotesize\scshape {latex_escape(volume)}\par}}
\end{{titlepage}}
"""


def colophon(volume: str, stats: Stats) -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "inconnu"
    return rf"""\par\vspace{{4ex}}
{{\centering\rule{{2em}}{{0.3pt}}\par\vspace{{1ex}}
\footnotesize\itshape Recomposé par \textup{{numrev typeset}} le {date.today().strftime("%d/%m/%Y")}
(commit \texttt{{{commit}}}) d'après les {stats.entries} entrées et {sum(stats.titles.values())} titres
du volume \textup{{{latex_escape(volume)}}}, {stats.pages} pages de l'original.\par}}
"""


def compose(volume: str, lines: list[Line]) -> tuple[str, Stats]:
    stats = Stats()
    blocks = body(lines, stats)
    document = PREAMBLE + title_page(volume, lines) + "\n".join(blocks) + "\n" + colophon(volume, stats) + "\\end{document}\n"
    return document, stats


# ----------------------------------------------------------------------
# Commande
# ----------------------------------------------------------------------
def compile_pdf(tex: Path) -> Path:
    if shutil.which("pdflatex") is None:
        raise CommandError("pdflatex introuvable (TeX Live).")
    for _ in range(2):  # la seconde passe fixe les titres courants
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tex.parent,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            errors = [line for line in result.stdout.splitlines() if line.startswith("!")]
            raise CommandError(f"pdflatex a échoué ({tex.with_suffix('.log')}) : {' / '.join(errors[:3]) or 'voir le journal'}")
    log = tex.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    console.print(f"PDF : [yellow]{tex.with_suffix('.pdf')}[/yellow] ({log.count('Overfull')} lignes trop longues signalées)")
    return tex.with_suffix(".pdf")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("volume", type=Path, help="Dossier du volume (ex. annuaires/1807_AD75-PER292).")
    parser.add_argument("-o", "--output", type=Path, help="Fichier .tex (défaut : annuaires/<volume>/<volume>.tex).")
    parser.add_argument("--pdf", action="store_true", help="Compile le .tex avec pdflatex (avec --apply).")
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    volume_dir = require_dir(args.volume)
    output = args.output or typeset_output(volume_dir)
    lines = read_volume(volume_dir)
    document, stats = compose(volume_dir.name, lines)

    console.print(f"[bold]{volume_dir.name}[/bold] : {stats.pages} pages de l'original ({stats.folios} folios imprimés lus)")
    console.print(
        f"  titres : {stats.titles[1]} parties, {stats.titles[2]} rubriques, {stats.titles[3]} intertitres ; {stats.entries} entrées"
        + (f" ({stats.fallback} au balisage invalide, sans empans)" if stats.fallback else "")
    )
    console.print("  hors liste composés : " + (", ".join(f"{label} {n}" for label, n in stats.notices.most_common()) or "aucun"))
    console.print("  hors liste ignorés : " + (", ".join(f"{label} {n}" for label, n in stats.skipped.most_common()) or "aucun"))

    writes = Writes(args.apply)
    writes.add(output, lambda: output.write_text(document, encoding="utf-8"))
    writes.finish(console)
    if args.pdf:
        if args.apply:
            compile_pdf(output)
        else:
            console.print("[dim]--pdf : compilation pdflatex après écriture, avec --apply.[/dim]")

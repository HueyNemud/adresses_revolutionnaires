"""Étape 4 · entités et arbre des titres (`<document>.lines.csv` →
`<document>.entities.csv` et `<document>.entities.report.txt`).

Reconstitue les entités logiques depuis la table des lignes corrigée
(`numrev tabulate`). Deux rôles :

1. fusionner les lignes en entités (règles ci-dessous) et leur donner un
   identifiant stable (`uuid`) ;
2. rattacher chaque TITLE et chaque ENTRY à son titre parent
   (`parent_uuid`), pour reconstituer la hiérarchie des titres et compter
   les entrées par titre.

Trois types d'entités finales sont reconstitués à partir des classes ligne
par ligne :

- ENTRY  : une B-ENTRY, suivie de ses éventuelles I-ENTRY et SUB-ENTRY.
- TITLE  : une B-TITLE, suivie de ses éventuelles I-TITLE.
- OUT OF SCOPE (et toute classe non reconnue) : jamais fusionnée, recopiée
  telle quelle.

Les lignes de classe `SUPPRIMÉE` (supprimées à la main, voir
numrev/curation.py) sont ignorées ; les colonnes de curation des lignes
(`corrige`, `empreinte`) ne sont pas recopiées.

Règles de fusion :
- I-ENTRY se fusionne (avec un espace) à la ligne ENTRY/SUB-ENTRY la plus
  proche qui la précède, quitte à « remonter » au-delà de lignes
  OUT OF SCOPE ou [B|I]-TITLE intercalées.
- SUB-ENTRY se fusionne (avec un saut de ligne) à la même ligne ENTRY la
  plus proche, avec la même tolérance de remontée.
- I-TITLE se fusionne (avec un espace) à la ligne TITLE la plus proche, avec
  la même tolérance de remontée (généralisation symétrique à la règle
  ENTRY : non explicitement spécifiée à l'origine, mais plus robuste).

En pratique, ceci est implémenté par deux « pistes » indépendantes (un
groupe ENTRY actif, un groupe TITLE actif) qui restent ouvertes tant
qu'aucune nouvelle racine (B-ENTRY / B-TITLE) n'apparaît. L'ensemble des
entités créées est conservé dans une liste ordonnée pour préserver l'ordre
d'apparition original de chaque entité racine.

Un rapport d'analyse (`<document>.entities.report.txt`) recense les comptages,
l'arbre indenté des titres avec le nombre d'entrées directes et récursives
de chacun, et les familles de problèmes détectés : les lignes de continuation
sans ancre valable, les titres sans marqueur `#`, et les entrées qui rompent
l'ordre alphabétique (réinitialisé à chaque nouveau TITLE).

Identifiant d'entité (`uuid`)
-----------------------------
Déterministe (`uuid5`) : il dépend du nom du document (nom du fichier
d'entrée sans suffixe d'étape, ex. « 1808_AD75-PER292.7-186 »,
`numrev/paths.py`) et de la clé `cle`
de la **ligne racine** de l'entité (celle qui l'ouvre ; voir
numrev/curation.py : hash du texte OCR, qui ne dépend ni de la page ni du
découpage en blocs). Il est donc identique d'une exécution à l'autre,
survit aux corrections de texte, à la re-segmentation de l'OCR et au
rattachement ou détachement de lignes de continuation. La colonne `cle`
d'une entité garde la composition en clair (clés de ses lignes, séparées
par des virgules). Si deux entités ont la même racine (cas qui ne se
produit pas avec des clés uniques), la deuxième reçoit le suffixe `#2`,
etc., dans l'ordre du fichier.

Titre parent (`parent_uuid`)
----------------------------
Le niveau d'un TITLE est le nombre de `#` en tête de son texte ; un titre
sans `#` est considéré comme du niveau le plus profond (comme dans
`numrev view directory`). En parcourant les entités dans l'ordre du
document, le parent d'un TITLE est le dernier TITLE de niveau strictement
inférieur, et le parent de toute autre entité (ENTRY, OUT OF SCOPE) est le
dernier TITLE rencontré. Les titres de plus haut niveau, et les entités qui
précèdent tout titre, ont pour parent la racine artificielle `ROOT_UUID`
(UUID nul), commune à tous les documents : l'arbre a ainsi toujours une
racine unique et chaque entité y est rattachée.
"""

import argparse
import csv
import re
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from numrev.command import CommandError, Writes, add_apply_argument, console, default_output, require_file
from numrev.crf.labels import AnnotationLabel
from numrev.curation import CORRECTED_COLUMN, DELETED_CLASS, FINGERPRINT_COLUMN
from numrev.paths import ENTITIES, document_name
from numrev.pipeline.tabulate import CLASS_COLUMN, KEY_COLUMN
from numrev.titles import ROOT_UUID, title_level

UID_COLUMN = "uid"
REQUIRED_COLUMNS = (KEY_COLUMN, UID_COLUMN, "markdown", CLASS_COLUMN)

LABEL_BEGIN_ENTRY = AnnotationLabel.BENTRY.value
LABEL_INSIDE_ENTRY = AnnotationLabel.IENTRY.value
LABEL_SUB_ENTRY = AnnotationLabel.SUBENTRY.value
LABEL_BEGIN_TITLE = AnnotationLabel.BTITLE.value
LABEL_INSIDE_TITLE = AnnotationLabel.ITITLE.value
LABEL_OOS = AnnotationLabel.OOS.value

NORMALIZED_ENTRY = "ENTRY"
NORMALIZED_TITLE = "TITLE"

# Espace de noms des identifiants d'entités : le changer changerait tous les
# identifiants déjà produits.
ENTITY_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "adresses_revolutionnaires/entites")

UNMARKED_TITLE_LEVEL = 99  # titre sans `#` : niveau le plus profond


@dataclass
class MergeReport:
    """Compteurs et listes de problèmes accumulés pendant la fusion."""

    rows_read: int = 0
    rows_deleted: int = 0
    rows_written: int = 0
    final_entries: int = 0
    final_titles: int = 0
    final_oos: int = 0
    entry_lines_merged: int = 0
    subentry_lines_merged: int = 0
    title_lines_merged: int = 0
    unmarked_titles: list[str] = field(default_factory=list)
    # Arbre des titres : titres dans l'ordre du document, (uuid, parent, texte),
    # et nombre d'ENTRY directes par parent_uuid.
    titles: list[tuple[str, str, str]] = field(default_factory=list)
    direct_entries: Counter[str] = field(default_factory=Counter)
    orphan_subentries: list[str] = field(default_factory=list)
    orphan_entry_continuations: list[str] = field(default_factory=list)
    orphan_title_continuations: list[str] = field(default_factory=list)
    unknown_labels: list[tuple[str, str]] = field(default_factory=list)
    alpha_violations: list[tuple[str, str, str, str]] = field(default_factory=list)


ALPHA_KEY_WORDS = 2
ALPHA_KEY_LENGTH = 5


def alpha_sort_key(text: str) -> str:
    """Clé de tri : deux premiers mots de l'entrée avant la première virgule,
    concaténés en majuscules sans accents et tronqués à 5 caractères
    (« Le Roux, rue… » → LEROU, « Brunet (J. B.) » → BRUNE, « Adam, Pont-Neuf »
    → ADAM).
    """
    normalized = unicodedata.normalize("NFKD", text.split(",", 1)[0])
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").upper()
    words = re.findall(r"[A-Z0-9]+", ascii_text)
    return "".join(words[:ALPHA_KEY_WORDS])[:ALPHA_KEY_LENGTH]


def assign_entity_ids(entities: list[dict[str, str]], document: str) -> None:
    """Identifiant déterministe de chaque entité, d'après la clé de sa ligne
    racine (voir la docstring du module)."""
    occurrences: Counter[str] = Counter()
    for entity in entities:
        root = entity.get(KEY_COLUMN, "").split(",", 1)[0]
        key = f"{document}#{root}"
        occurrences[key] += 1
        name = key if occurrences[key] == 1 else f"{key}#{occurrences[key]}"
        entity["uuid"] = str(uuid.uuid5(ENTITY_ID_NAMESPACE, name))


def assign_parent_ids(entities: list[dict[str, str]]) -> None:
    """Titre parent de chaque entité (voir la docstring du module) ;
    à appeler après assign_entity_ids."""
    stack: list[tuple[int, str]] = []  # (niveau, uuid) des titres ouverts
    for entity in entities:
        kind = entity.get("entity")
        if kind == NORMALIZED_TITLE:
            level = title_level(entity.get("markdown", "")) or UNMARKED_TITLE_LEVEL
            while stack and stack[-1][0] >= level:
                stack.pop()
            entity["parent_uuid"] = stack[-1][1] if stack else ROOT_UUID
            stack.append((level, entity["uuid"]))
        else:
            entity["parent_uuid"] = stack[-1][1] if stack else ROOT_UUID


def _new_root(row: dict[str, str], normalized_entity: str) -> dict[str, str]:
    root = row.copy()
    root["entity"] = normalized_entity
    return root


def _merge_into(
    group: dict[str, str],
    row: dict[str, str],
    separator: str,
    fieldnames: list[str],
    strip_whitespace: bool = True,
) -> None:
    for col in fieldnames:
        sep = separator if col == "markdown" else ","
        left = group.get(col, "")
        right = row.get(col, "")

        if strip_whitespace:
            left, right = left.strip(), right.strip()

        group[col] = f"{left}{sep}{right}"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Table des lignes corrigée (<document>.lines.csv).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV de sortie (défaut : <document>.entities.csv) ; le rapport est écrit à côté (<…>.entities.report.txt).",
    )
    add_apply_argument(parser)


def process_csv(input_path: Path, output_path: Path, writes: Writes | None = None) -> MergeReport:
    """Lit le CSV, fusionne les entités ENTRY/TITLE, les rattache à leur
    titre parent, exporte le résultat et retourne le rapport d'analyse
    correspondant."""
    report = MergeReport()

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Le fichier CSV d'entrée est vide ou invalide.")

        input_fieldnames = [col for col in reader.fieldnames if col not in (CORRECTED_COLUMN, FINGERPRINT_COLUMN)]

        missing = [col for col in REQUIRED_COLUMNS if col not in input_fieldnames]
        if missing:
            raise ValueError(f"Colonne(s) absente(s) : {', '.join(missing)} " "(l'entrée est la table des lignes de `numrev tabulate`).")

        output_fieldnames = ["uuid", "parent_uuid"]
        for col in input_fieldnames:
            if col in ("uuid", "parent_uuid"):
                continue
            output_fieldnames.append(col)
            if col == "markdown":
                output_fieldnames.append("entity")

        if "entity" not in output_fieldnames:
            output_fieldnames.append("entity")

        all_entities: list[dict[str, str]] = []
        active_entry: dict[str, str] | None = None
        active_title: dict[str, str] | None = None
        last_alpha_key: str | None = None

        def identifier(row: dict[str, str]) -> str:
            return row[UID_COLUMN]

        def register_entry_root(row: dict[str, str]) -> None:
            nonlocal last_alpha_key
            key = alpha_sort_key(row.get("markdown", ""))
            if not key:
                return
            if last_alpha_key is not None and key < last_alpha_key:
                report.alpha_violations.append(
                    (
                        identifier(row),
                        key,
                        last_alpha_key,
                        row.get("markdown", "")[:60],
                    )
                )
            last_alpha_key = key

        for row in reader:
            report.rows_read += 1
            pred = row.get(CLASS_COLUMN, "").strip()
            if pred == DELETED_CLASS:
                report.rows_deleted += 1
                continue
            row = {col: row.get(col, "") for col in input_fieldnames}
            row_id = identifier(row)

            if pred == LABEL_BEGIN_ENTRY:
                register_entry_root(row)
                active_entry = _new_root(row, NORMALIZED_ENTRY)
                all_entities.append(active_entry)

            elif pred == LABEL_INSIDE_ENTRY:
                if active_entry is not None:
                    _merge_into(active_entry, row, " ", input_fieldnames)
                    report.entry_lines_merged += 1
                else:
                    report.orphan_entry_continuations.append(row_id)
                    register_entry_root(row)
                    active_entry = _new_root(row, NORMALIZED_ENTRY)
                    all_entities.append(active_entry)

            elif pred == LABEL_SUB_ENTRY:
                if active_entry is not None:
                    _merge_into(active_entry, row, "\n", input_fieldnames)
                    report.subentry_lines_merged += 1
                else:
                    report.orphan_subentries.append(row_id)
                    register_entry_root(row)
                    active_entry = _new_root(row, NORMALIZED_ENTRY)
                    all_entities.append(active_entry)

            elif pred == LABEL_BEGIN_TITLE:
                active_title = _new_root(row, NORMALIZED_TITLE)
                all_entities.append(active_title)
                last_alpha_key = None

            elif pred == LABEL_INSIDE_TITLE:
                if active_title is not None:
                    _merge_into(active_title, row, " ", input_fieldnames)
                    report.title_lines_merged += 1
                else:
                    report.orphan_title_continuations.append(row_id)
                    active_title = _new_root(row, NORMALIZED_TITLE)
                    all_entities.append(active_title)

            else:
                if pred != LABEL_OOS and pred != "":
                    report.unknown_labels.append((row_id, pred))
                out_row = row.copy()
                out_row["entity"] = LABEL_OOS
                all_entities.append(out_row)

    assign_entity_ids(all_entities, document_name(input_path))
    assign_parent_ids(all_entities)
    for entity_row in all_entities:
        report.rows_written += 1
        if entity_row.get("entity") == NORMALIZED_ENTRY:
            report.final_entries += 1
            report.direct_entries[entity_row["parent_uuid"]] += 1
        elif entity_row.get("entity") == NORMALIZED_TITLE:
            report.final_titles += 1
            markdown = entity_row.get("markdown", "")
            report.titles.append((entity_row["uuid"], entity_row["parent_uuid"], markdown))
            if title_level(markdown) is None:
                report.unmarked_titles.append(entity_row.get(UID_COLUMN, ""))
        else:
            report.final_oos += 1

    def write() -> None:
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(all_entities)

    (writes or Writes()).add(output_path, write)

    return report


TREE_TEXT_LENGTH = 70


def format_title_tree(report: MergeReport) -> list[str]:
    """Arbre indenté des titres, depuis la racine, avec pour chacun le nombre
    d'ENTRY directes et récursives (sous-arbre compris)."""
    children: dict[str, list[tuple[str, str]]] = {}
    for title_id, parent_id, markdown in report.titles:
        text = " ".join(markdown.replace("*", "").split())
        if len(text) > TREE_TEXT_LENGTH:
            text = text[: TREE_TEXT_LENGTH - 1] + "…"
        children.setdefault(parent_id, []).append((title_id, text))

    totals: dict[str, int] = {}

    def total(node_id: str) -> int:
        totals[node_id] = report.direct_entries[node_id] + sum(total(child_id) for child_id, _ in children.get(node_id, []))
        return totals[node_id]

    total(ROOT_UUID)
    lines: list[str] = []

    def walk(node_id: str, text: str, depth: int) -> None:
        lines.append(f"{'  ' * depth}{text} — {totals[node_id]} ({report.direct_entries[node_id]})")
        for child_id, child_text in children.get(node_id, []):
            walk(child_id, child_text, depth + 1)

    walk(ROOT_UUID, "[racine]", 0)
    return lines


def format_report(report: MergeReport, input_path: Path, output_path: Path) -> str:
    lines = [
        "RAPPORT — numrev assemble",
        f"Entrée  : {input_path}",
        f"Sortie  : {output_path}",
        "",
        "== Comptages ==",
        f"Lignes lues                         : {report.rows_read}",
        f"Lignes supprimées ({DELETED_CLASS})      : {report.rows_deleted}",
        f"Lignes écrites                      : {report.rows_written}",
        f"Entités finales ENTRY                : {report.final_entries}",
        f"Entités finales TITLE                : {report.final_titles}",
        f"Lignes OUT OF SCOPE (ou inconnues)   : {report.final_oos}",
        f"Lignes I-ENTRY fusionnées            : {report.entry_lines_merged}",
        f"Lignes SUB-ENTRY fusionnées          : {report.subentry_lines_merged}",
        f"Lignes I-TITLE fusionnées            : {report.title_lines_merged}",
        "",
        "== Hiérarchie des titres ==",
        "Entrées par titre : total du sous-arbre (directes).",
        *format_title_tree(report),
    ]
    if report.unmarked_titles:
        lines.append(f"Titres sans marqueur `#` ({len(report.unmarked_titles)}) " "— placés au niveau le plus profond :")
        lines.extend(f"  - {uid}" for uid in report.unmarked_titles)
    lines += [
        "",
        "== Cas problématiques ==",
    ]

    if report.orphan_subentries:
        lines.append(f"SUB-ENTRY sans [I|B]-ENTRY précédente ({len(report.orphan_subentries)}) " "— traitées comme une nouvelle ENTRY :")
        lines.extend(f"  - {uid}" for uid in report.orphan_subentries)
    else:
        lines.append("SUB-ENTRY sans ancre précédente : aucune.")

    if report.orphan_entry_continuations:
        lines.append(
            f"I-ENTRY sans [B|I|SUB]-ENTRY précédente ({len(report.orphan_entry_continuations)}) " "— traitées comme une nouvelle ENTRY :"
        )
        lines.extend(f"  - {uid}" for uid in report.orphan_entry_continuations)
    else:
        lines.append("I-ENTRY sans ancre précédente : aucune.")

    if report.orphan_title_continuations:
        lines.append(
            f"I-TITLE sans [B|I]-TITLE précédente ({len(report.orphan_title_continuations)}) " "— traitées comme un nouveau TITLE :"
        )
        lines.extend(f"  - {uid}" for uid in report.orphan_title_continuations)
    else:
        lines.append("I-TITLE sans ancre précédente : aucune.")

    if report.unknown_labels:
        lines.append(f"Classes non reconnues ({len(report.unknown_labels)}) :")
        lines.extend(f"  - {uid} : classe '{label}'" for uid, label in report.unknown_labels)

    lines.append("")
    lines.append("== Ordre alphabétique (réinitialisé à chaque nouveau TITLE) ==")
    if report.alpha_violations:
        lines.append(f"Entrées rompant l'ordre alphabétique ({len(report.alpha_violations)}) :")
        for uid, key, previous_key, excerpt in report.alpha_violations:
            lines.append(f'  - {uid} : "{key}" ("{excerpt}...") suit "{previous_key}", ' "ordre alphabétique rompu.")
    else:
        lines.append("Aucune rupture d'ordre alphabétique détectée.")

    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output(args.input, ENTITIES)
    report_path = output_path.with_name(output_path.name.removesuffix(".csv") + ".report.txt")
    writes = Writes(args.apply)
    try:
        report = process_csv(require_file(args.input), output_path, writes)
    except (csv.Error, ValueError) as error:
        raise CommandError(f"{args.input} : {error}") from error

    writes.add(report_path, lambda: report_path.write_text(format_report(report, args.input, output_path), encoding="utf-8"))

    problem_count = (
        len(report.orphan_subentries)
        + len(report.orphan_entry_continuations)
        + len(report.orphan_title_continuations)
        + len(report.alpha_violations)
        + len(report.unmarked_titles)
    )
    console.print(f"{report.rows_read} lignes lues → {report.rows_written} lignes en sortie.")
    if problem_count:
        console.print(f"[yellow]⚠ {problem_count} cas à vérifier (voir le rapport pour le détail).[/yellow]")
    writes.finish(console)

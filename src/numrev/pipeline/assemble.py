"""Étape 4 · entités et arbre des titres (`<document>.lines.csv` →
`<document>.entities.csv` et `<document>.entities.report.txt`).

Regroupe les lignes de la table des lignes en entités, chacune avec un
`uuid` stable et le `parent_uuid` de son titre :

- ENTRY : une `B-ENTRY`, puis ses `I-ENTRY` (jointes par une espace) et ses
  `SUB-ENTRY` (jointes par un saut de ligne) ;
- TITLE : une `B-TITLE`, puis ses `I-TITLE` (jointes par une espace) ;
- OUT OF SCOPE, ou classe inconnue : une ligne, jamais fusionnée.

Une entrée et un titre restent ouverts, chacun de son côté, jusqu'à la
prochaine `B-ENTRY` / `B-TITLE` : une ligne hors champ ou un titre courant
ne coupe pas une entrée. Une suite sans entité ouverte en ouvre une, et le
rapport la signale. Les lignes `SUPPRIMÉE` sont ignorées.

`uuid` = uuid5 du nom du document et de la clé `cle` de la ligne racine de
l'entité. Parent d'un titre : le dernier titre de niveau (nombre de `#`)
strictement inférieur ; de toute autre entité : le dernier titre ; à défaut,
la racine commune `ROOT_UUID`. L'arbre est propre à la plage : les entrées
avant son premier titre n'ont pas de rubrique (voir
`docs/guide_annotation_lignes.md`). Règles détaillées : `docs/pipeline.md`.
"""

import argparse
import csv
import re
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from numrev.command import (
    CommandError,
    Writes,
    add_apply_argument,
    add_folder_argument,
    console,
    require_input,
    require_ranges,
    write_or_plan,
)
from numrev.crf.labels import AnnotationLabel
from numrev.curation import CORRECTED_COLUMN, DELETED_CLASS, FINGERPRINT_COLUMN, read_csv, write_csv
from numrev.paths import ENTITIES, ENTITIES_REPORT, LINES_CSV, document_name, range_document
from numrev.pipeline.tabulate import CLASS_COLUMN, CONFIDENCE_COLUMN, DOUBTFUL_COLUMN, KEY_COLUMN
from numrev.titles import ROOT_UUID, title_level

DESCRIPTION = """Étape 4 · entrées et arbre des titres (<document>.lines.csv → <document>.entities.csv).

Regroupe les lignes en entrées et en titres, range chaque entrée sous son titre, et écrit un rapport,
<document>.entities.report.txt : l'arbre des titres et les cas à corriger dans la table des lignes."""

UID_COLUMN = "uid"
REQUIRED_COLUMNS = (KEY_COLUMN, UID_COLUMN, "markdown", CLASS_COLUMN)
ENTRY, TITLE, OUT_OF_SCOPE = "ENTRY", "TITLE", AnnotationLabel.OOS.value
ROOT_LABELS = {AnnotationLabel.BENTRY.value: ENTRY, AnnotationLabel.BTITLE.value: TITLE}
# Classe d'une ligne de suite → (entité qu'elle continue, séparateur dans le texte).
CONTINUATIONS = {
    AnnotationLabel.IENTRY.value: (ENTRY, " "),
    AnnotationLabel.SUBENTRY.value: (ENTRY, "\n"),
    AnnotationLabel.ITITLE.value: (TITLE, " "),
}
# Espace de noms des identifiants d'entités : le changer changerait tous les identifiants déjà produits.
ENTITY_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "adresses_revolutionnaires/entites")
UNMARKED_TITLE_LEVEL = 99  # titre sans `#` : niveau le plus profond
ALPHA_KEY_WORDS, ALPHA_KEY_LENGTH = 2, 5
TREE_TEXT_LENGTH = 70


@dataclass
class MergeReport:
    """Comptes et cas à vérifier d'une plage."""

    rows_read: int = 0
    rows_deleted: int = 0
    rows_written: int = 0
    merged: Counter[str] = field(default_factory=Counter)  # classe de suite → lignes fusionnées
    orphans: dict[str, list[str]] = field(default_factory=dict)  # classe de suite → uid des suites sans entité ouverte
    unknown_labels: list[tuple[str, str]] = field(default_factory=list)  # (uid, classe)
    alpha_violations: list[tuple[str, str, str, str]] = field(default_factory=list)  # (uid, clé, clé précédente, extrait)
    titles: list[tuple[str, str, str]] = field(default_factory=list)  # (uuid, parent_uuid, markdown), ordre du document
    direct_entries: Counter[str] = field(default_factory=Counter)  # parent_uuid → ENTRY directes
    unmarked_titles: list[str] = field(default_factory=list)  # uid des titres sans `#`
    untitled_entries: list[str] = field(default_factory=list)  # uid des ENTRY avant le premier titre (sans rubrique)
    entries: int = 0
    out_of_scope: int = 0

    @property
    def problem_count(self) -> int:
        """Cas à vérifier, comptés en console (les entrées sans rubrique ont leur propre message)."""
        orphans = sum(len(uids) for uids in self.orphans.values())
        return orphans + len(self.alpha_violations) + len(self.unmarked_titles) + len(self.unknown_labels)


def alpha_sort_key(text: str) -> str:
    """Clé de l'ordre alphabétique : deux premiers mots avant la première
    virgule, en majuscules sans accents, tronqués à 5 caractères
    (« Le Roux, rue… » → LEROU, « Brunet (J. B.) » → BRUNE)."""
    normalized = unicodedata.normalize("NFKD", text.split(",", 1)[0])
    words = re.findall(r"[A-Z0-9]+", normalized.encode("ascii", "ignore").decode("ascii").upper())
    return "".join(words[:ALPHA_KEY_WORDS])[:ALPHA_KEY_LENGTH]


def assign_entity_ids(entities: list[dict[str, str]], document: str) -> None:
    """`uuid` de chaque entité, d'après la clé de sa ligne racine (`#2`, `#3`… si deux entités partagent une racine)."""
    occurrences: Counter[str] = Counter()
    for entity in entities:
        key = f"{document}#{entity.get(KEY_COLUMN, '').split(',', 1)[0]}"
        occurrences[key] += 1
        name = key if occurrences[key] == 1 else f"{key}#{occurrences[key]}"
        entity["uuid"] = str(uuid.uuid5(ENTITY_ID_NAMESPACE, name))


def assign_parent_ids(entities: list[dict[str, str]]) -> None:
    """`parent_uuid` de chaque entité (voir la docstring du module) ; après `assign_entity_ids`."""
    stack: list[tuple[int, str]] = []  # (niveau, uuid) des titres ouverts
    for entity in entities:
        if entity["entity"] == TITLE:
            level = title_level(entity.get("markdown", "")) or UNMARKED_TITLE_LEVEL
            while stack and stack[-1][0] >= level:
                stack.pop()
        entity["parent_uuid"] = stack[-1][1] if stack else ROOT_UUID
        if entity["entity"] == TITLE:
            stack.append((level, entity["uuid"]))


def lowest_confidence(left: str, right: str) -> str:
    """La plus faible de deux confiances OCR (`0,44`, virgule décimale) ; vide si aucune n'a de score."""
    scores = [value for value in (left, right) if value]
    return min(scores, key=lambda value: float(value.replace(",", "."))) if scores else ""


def merge_line(entity: dict[str, str], row: dict[str, str], separator: str, columns: list[str]) -> None:
    """Ajoute une ligne à une entité : texte joint par `separator`, autres
    colonnes par une virgule, sauf `confiance` (la plus faible) et `douteuse`
    (`oui` si l'une l'est)."""
    for column in columns:
        left, right = entity.get(column, "").strip(), row.get(column, "").strip()
        if column == CONFIDENCE_COLUMN:
            entity[column] = lowest_confidence(left, right)
        elif column == DOUBTFUL_COLUMN:
            entity[column] = "oui" if "oui" in (left, right) else ""
        else:
            entity[column] = f"{left}{separator if column == 'markdown' else ','}{right}"


def merge_lines(rows: list[dict[str, str]], columns: list[str], report: MergeReport) -> list[dict[str, str]]:
    """Entités dans l'ordre du document, sans `uuid` ni `parent_uuid`."""
    entities: list[dict[str, str]] = []
    open_entities: dict[str, dict[str, str]] = {}  # ENTRY / TITLE → entité en cours
    last_alpha_key: str | None = None
    for row in rows:
        report.rows_read += 1
        label = row.get(CLASS_COLUMN, "").strip()
        if label == DELETED_CLASS:
            report.rows_deleted += 1
            continue
        row = {column: row.get(column, "") for column in columns}
        kind = ROOT_LABELS.get(label)
        if label in CONTINUATIONS:
            kind, separator = CONTINUATIONS[label]
            if kind in open_entities:
                merge_line(open_entities[kind], row, separator, columns)
                report.merged[label] += 1
                continue
            report.orphans.setdefault(label, []).append(row[UID_COLUMN])  # suite sans entité ouverte : en ouvre une
        if kind is None:
            if label not in (OUT_OF_SCOPE, ""):
                report.unknown_labels.append((row[UID_COLUMN], label))
            entities.append(row | {"entity": OUT_OF_SCOPE})
            continue
        if kind == ENTRY and (key := alpha_sort_key(row.get("markdown", ""))):
            if last_alpha_key is not None and key < last_alpha_key:
                report.alpha_violations.append((row[UID_COLUMN], key, last_alpha_key, row.get("markdown", "")[:60]))
            last_alpha_key = key
        elif label == AnnotationLabel.BTITLE.value:
            last_alpha_key = None  # l'ordre alphabétique repart à chaque titre
        open_entities[kind] = row | {"entity": kind}
        entities.append(open_entities[kind])
    return entities


def assemble_document(input_path: Path, output_path: Path, writes: Writes | None = None) -> MergeReport:
    """Fusionne les lignes en entités, les rattache à leur titre, planifie l'écriture et rend le rapport."""
    fieldnames, rows = read_csv(input_path)
    if not fieldnames:
        raise ValueError("Le fichier CSV d'entrée est vide ou invalide.")
    columns = [column for column in fieldnames if column not in (CORRECTED_COLUMN, FINGERPRINT_COLUMN)]
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Colonne(s) absente(s) : {', '.join(missing)} (l'entrée est la table des lignes de `numrev tabulate`).")
    output_fields = ["uuid", "parent_uuid"]
    for column in columns:
        if column not in ("uuid", "parent_uuid", "entity"):
            output_fields += [column, "entity"] if column == "markdown" else [column]

    report = MergeReport()
    entities = merge_lines(rows, columns, report)
    assign_entity_ids(entities, document_name(input_path))
    assign_parent_ids(entities)
    for entity in entities:
        report.rows_written += 1
        if entity["entity"] == ENTRY:
            report.entries += 1
            report.direct_entries[entity["parent_uuid"]] += 1
            if entity["parent_uuid"] == ROOT_UUID:
                report.untitled_entries.append(entity[UID_COLUMN])
        elif entity["entity"] == TITLE:
            report.titles.append((entity["uuid"], entity["parent_uuid"], entity["markdown"]))
            if title_level(entity["markdown"]) is None:
                report.unmarked_titles.append(entity[UID_COLUMN])
        else:
            report.out_of_scope += 1
    write_or_plan(writes, output_path, lambda: write_csv(output_path, output_fields, entities))
    return report


def format_title_tree(report: MergeReport) -> list[str]:
    """Arbre indenté des titres depuis la racine, chacun avec ses ENTRY : total du sous-arbre (directes)."""
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
        f"Entités finales ENTRY                : {report.entries}",
        f"Entités finales TITLE                : {len(report.titles)}",
        f"Lignes OUT OF SCOPE (ou inconnues)   : {report.out_of_scope}",
        *(f"Lignes {label + ' fusionnées':<30}: {report.merged[label]}" for label in CONTINUATIONS),
        "",
        "== Hiérarchie des titres ==",
        "Entrées par titre : total du sous-arbre (directes).",
        *format_title_tree(report),
    ]
    if report.unmarked_titles:
        lines.append(f"Titres sans marqueur `#` ({len(report.unmarked_titles)}) — placés au niveau le plus profond :")
        lines += [f"  - {uid}" for uid in report.unmarked_titles]
    if report.untitled_entries:
        first, last = report.untitled_entries[0].split(",")[0], report.untitled_entries[-1].split(",")[0]
        lines += [
            f"Entrées avant le premier titre de la plage ({len(report.untitled_entries)}, de {first} à {last}) "
            "— sans titre parent ni rubrique :",
            "  la chaîne ne les rattache à aucune rubrique (pas même celle de la fin de la plage précédente,",
            "  qui n'est souvent pas contiguë), et l'alignement de deux éditions ne les apparie jamais.",
            "  Pour les rattacher : ajouter en tête de la table des lignes la ligne de titre de leur rubrique",
            "  (voir docs/guide_annotation_lignes.md, « Une plage qui commence au milieu d'une rubrique »).",
        ]
    lines += ["", "== Cas problématiques =="]
    for label, (kind, _) in CONTINUATIONS.items():
        uids = report.orphans.get(label, [])
        opened, new = ("entrée", "une nouvelle entrée") if kind == ENTRY else ("titre", "un nouveau titre")
        if uids:
            lines.append(f"{label} sans {opened} à continuer ({len(uids)}) — chacune ouvre {new} :")
            lines += [f"  - {uid}" for uid in uids]
        else:
            lines.append(f"{label} sans {opened} à continuer : aucune.")
    if report.unknown_labels:
        lines.append(
            f"Lignes de classe incertaine ou inconnue ({len(report.unknown_labels)}) "
            "— traitées comme OUT OF SCOPE, à trancher dans la table des lignes :"
        )
        lines += [f"  - {uid} : classe '{label}'" for uid, label in report.unknown_labels]
    lines += ["", "== Ordre alphabétique (réinitialisé à chaque nouveau TITLE) =="]
    if report.alpha_violations:
        lines.append(f"Entrées rompant l'ordre alphabétique ({len(report.alpha_violations)}) :")
        lines += [
            f'  - {uid} : "{key}" ("{excerpt}...") suit "{previous}", ordre alphabétique rompu.'
            for uid, key, previous, excerpt in report.alpha_violations
        ]
    else:
        lines.append("Aucune rupture d'ordre alphabétique détectée.")
    return "\n".join(lines) + "\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_folder_argument(parser)
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    writes = Writes(args.apply)
    for range_dir in require_ranges(args.folder):
        input_path = require_input(range_dir, LINES_CSV, "tabulate")
        output_path, report_path = range_document(range_dir, ENTITIES), range_document(range_dir, ENTITIES_REPORT)
        try:
            report = assemble_document(input_path, output_path, writes)
        except (csv.Error, ValueError) as error:
            raise CommandError(f"{input_path} : {error}") from error
        text = format_report(report, input_path, output_path)
        # Valeurs liées par défaut : l'écriture a lieu après la boucle (Writes.finish).
        writes.add(report_path, lambda path=report_path, text=text: path.write_text(text, encoding="utf-8"))
        console.print(f"{input_path.name} : {report.rows_read} lignes lues → {report.rows_written} lignes en sortie.")
        if report.problem_count:
            console.print(f"[yellow]⚠ {report.problem_count} cas à vérifier (voir {report_path.name}).[/yellow]")
        if report.untitled_entries:
            console.print(
                f"[yellow]⚠ {len(report.untitled_entries)} entrée(s) avant le premier titre de la plage : sans rubrique, "
                "jamais appariées par l'alignement ; titre à restituer en tête de la table des lignes "
                f"(voir {report_path.name}).[/yellow]"
            )
    writes.finish(console)

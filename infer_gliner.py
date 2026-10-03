"""Exécute un modèle GLiNER-bi entraîné (train_gliner.py) sur les lignes
ENTRY d'un CSV fusionné (build_entity_tree.py), et ajoute au CSV le
rendu balisé des empans détectés (SUBJ/DESC/ADDR) ainsi que leurs comptes.

Colonnes ajoutées (insérées juste après la colonne `entity`) :
  - `tagged_text` : le texte intégral de la ligne, avec les empans détectés
    entourés de balises à leur position exacte, ex.
        "<SUBJ>Dupont</SUBJ>, <DESC>boulanger</DESC>, <ADDR>12 rue de la Paix</ADDR>."
    Le texte non couvert par un empan (ponctuation de liaison, espaces) est
    conservé tel quel : c'est l'ENTRY entière qui est annotée, rien n'est
    retiré. Les balises respectent l'ordre du texte source.
  - `subject_count`, `description_count`, `address_count` : nombre
    d'empans de chaque classe dans `tagged_text` ;
  - `ner_confidence` : score minimal des empans de l'entrée (0 sans empan) ;
  - `ner_suspect` : motifs de relecture prioritaire, séparés par « | »
    (vide : rien à signaler) : `aucun empan`, `score bas` (sous
    `--min-score`), `texte non couvert`, `SUBJ absent en tête`,
    `signature inhabituelle` (voir lib/ner/suspicion.py). Trier ou filtrer
    sur cette colonne donne la file de relecture.

Ces colonnes restent vides pour les lignes qui ne sont pas de type
ENTRY (`OUT OF SCOPE`, `TITLE`, ...) ou dont le texte est vide : ce sont des
lignes non traitées, pas des lignes traitées sans résultat. Une ligne ENTRY
traitée mais sans aucun empan détecté obtient son texte sans balise, des
compteurs à 0 et le motif `aucun empan` — une distinction volontaire entre
« non applicable » et « traité, rien trouvé ».

Labels et texte d'entrée
------------------------
Le modèle prédit les libellés descriptifs utilisés lors de l'entraînement,
pas les codes courts SUBJ/DESC/ADDR : la correspondance est lue dans
`<modèle>/ner_config.json`, écrit par tools/train_gliner.py (obligatoire).
Le modèle voit le texte normalisé (emphase Markdown retirée) ; les empans
sont ramenés sur le texte d'origine de la colonne pour `tagged_text`.

Corrections humaines
--------------------
Le CSV produit est aussi le fichier qu'on corrige à la main (`tagged_text`,
`parent_uuid` : rattachement d'une entrée à sa rubrique). Il suit le
protocole de `lib/curation.py` : les lignes `corrige = oui` du CSV existant
sont capturées (avant de charger le modèle) dans
`data/curation/<document>.ner.patch.csv` (versionné), puis réappliquées sur
la nouvelle inférence, par `uuid` d'entité. Une correction dont l'entité a
disparu, dont le texte a changé en amont (le `tagged_text` débalisé ne
redonne plus le texte) ou dont le titre parent n'existe plus fait paniquer
(`--force` : l'abandonner).

Inférence
---------
Par lots (`lib.ner.gliner.predict_spans`) : un lot qui échoue est retenté
texte par texte, et une ligne qui échoue encore est signalée sans perdre le
reste du lot.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from build_entity_tree import ROOT_UUID
from lib.cli import Writes, add_apply_argument
from lib.curation import (
    CORRECTED_COLUMN,
    FINGERPRINT_COLUMN,
    Check,
    Curation,
    CurationConflict,
    Step,
    is_corrected,
    patch_path,
    print_conflict,
    print_report,
    write_csv,
)
from lib.ner.gliner import NerConfig, load_model, predict_spans
from lib.ner.spans import LABELS, normalize_markdown, parse_tagged_text, render_tagged_text, unproject_spans
from lib.ner.suspicion import DEFAULT_MIN_SCORE, SEPARATOR, confidence, suspicion_reasons

console = Console()

# Colonnes du CSV fusionné (build_entity_tree.py).
ENTITY_COLUMN = "entity"
ENTRY_VALUE = "ENTRY"
TEXT_COLUMN = "markdown"
DEFAULT_BATCH_SIZE = 16
DEFAULT_THRESHOLD = 0.5

COUNT_COLUMN_FOR_LABEL = {"SUBJ": "subject_count", "DESC": "description_count", "ADDR": "address_count"}
NEW_COLUMNS = ["tagged_text", *COUNT_COLUMN_FOR_LABEL.values(), "ner_confidence", "ner_suspect", CORRECTED_COLUMN]

NER_STEP = Step("ner", "uuid", ("tagged_text", "parent_uuid"), context=("uid",))


def correction_check(rows: list[dict[str, str]]) -> Check:
    """Une correction NER ne s'applique que si son balisage redonne le texte
    actuel de l'entité et si son titre parent existe encore."""
    titles = {row["uuid"] for row in rows if row.get(ENTITY_COLUMN) == "TITLE"} | {ROOT_UUID}

    def check(correction: dict[str, str], row: dict[str, str]) -> str | None:
        tagged = correction.get("tagged_text", "")
        if tagged:
            try:
                text, _ = parse_tagged_text(tagged)
            except ValueError as error:
                return f"balisage invalide ({error})"
            if text != row.get(TEXT_COLUMN, "").strip():
                return "texte de l'entité modifié en amont depuis la correction"
        parent = correction.get("parent_uuid", "")
        if parent not in titles:
            return f"titre parent {parent!r} introuvable"
        return None

    return check


def recount(row: dict[str, str]) -> None:
    """Comptes d'empans d'une ligne corrigée, d'après son `tagged_text`."""
    if not row.get("tagged_text"):
        for column in COUNT_COLUMN_FOR_LABEL.values():
            row[column] = ""
        return
    _, spans = parse_tagged_text(row["tagged_text"])
    counts = Counter(span.label for span in spans)
    for label in LABELS:
        row[COUNT_COLUMN_FOR_LABEL[label]] = str(counts[label])


# --------------------------------------------------------------------------
# CSV : lecture, insertion de colonnes, sélection des lignes à traiter
# --------------------------------------------------------------------------


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Le fichier CSV d'entrée est vide ou invalide.")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def insert_columns_after(
    fieldnames: list[str], after: str, new_columns: list[str]
) -> list[str]:
    """Insère `new_columns` juste après la colonne `after`, sans dupliquer
    une colonne déjà présente (ré-exécution idempotente sur un fichier déjà
    traité : les colonnes existantes gardent leur position, leur contenu
    sera simplement écrasé pour les lignes retraitées).
    """
    if after not in fieldnames:
        raise ValueError(f"La colonne '{after}' est introuvable dans le CSV.")
    result = list(fieldnames)
    insert_at = result.index(after) + 1
    for offset, col in enumerate(new_columns):
        if col not in result:
            result.insert(insert_at + offset, col)
    return result


def select_entries(rows: list[dict[str, str]]) -> tuple[list[int], list[str], list[int]]:
    """Retourne (indices à traiter, textes correspondants, indices ENTRY
    écartés faute de texte)."""
    indices: list[int] = []
    texts: list[str] = []
    empty_text_indices: list[int] = []
    for index, row in enumerate(rows):
        if row.get(ENTITY_COLUMN, "").strip() != ENTRY_VALUE:
            continue
        text = row.get(TEXT_COLUMN, "").strip()
        if text:
            indices.append(index)
            texts.append(text)
        else:
            empty_text_indices.append(index)
    return indices, texts, empty_text_indices


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécute un modèle GLiNER-bi entraîné sur les lignes ENTRY d'un CSV "
            "fusionné, et ajoute le rendu balisé des empans SUBJ/DESC/ADDR détectés."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV fusionné (sortie de build_entity_tree.py).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Dossier du modèle GLiNER entraîné (sortie de train_gliner.py).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Chemin du CSV de sortie (défaut : <entrée>.ner.csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Nombre de textes par lot d'inférence (défaut : {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Seuil de confiance (défaut : {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Score minimal d'empan sous lequel une entrée est signalée « score bas » (défaut : {DEFAULT_MIN_SCORE}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprend la sortie machine pour les lignes modifiées sans « corrige = oui » et abandonne les corrections inapplicables.",
    )
    parser.add_argument(
        "--sans-capture",
        action="store_true",
        help="Ignore le CSV existant et repart du patch versionné.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche le résultat de chaque ligne traitée (déconseillé sur un gros fichier).",
    )
    add_apply_argument(parser)
    return parser.parse_args()


def apply_corrections(curation: Curation, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Réapplique le patch (lève CurationConflict) et recompte les empans
    des lignes corrigées."""
    rows = curation.apply(rows, check=correction_check(rows))
    for row in rows:
        if is_corrected(row):
            recount(row)
    return rows


def write_output(
    curation: Curation, rows: list[dict[str, str]], output_path: Path, fieldnames: list[str], writes: Writes
) -> None:
    """Corrections réappliquées puis écriture (si `--apply`) ; en cas de
    panique, rien n'est écrit et le script s'arrête."""
    try:
        rows = apply_corrections(curation, rows)
    except CurationConflict as conflict:
        print_conflict(console, conflict)
        raise SystemExit(1)
    curation.save_patch(writes)
    writes.add(output_path, lambda: write_csv(output_path, fieldnames, rows))
    print_report(console, curation)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        console.print(f"[bold red]Erreur :[/bold red] Fichier '{args.input_csv}' introuvable.")
        return
    try:
        config = NerConfig.load(args.model)
    except FileNotFoundError as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        return
    output_path = args.output or args.input_csv.with_suffix(".ner.csv")
    writes = Writes(args.apply)
    try:
        curation = Curation(
            NER_STEP, output_path, patch_path(output_path, NER_STEP.name),
            force=args.force, capture=not args.sans_capture,
        )
    except CurationConflict as conflict:
        print_conflict(console, conflict)
        raise SystemExit(1)

    console.print(f"Lecture de [yellow]{args.input_csv.name}[/yellow]...")
    try:
        fieldnames, rows = load_rows(args.input_csv)
        output_fieldnames = insert_columns_after(fieldnames, ENTITY_COLUMN, NEW_COLUMNS)
        if FINGERPRINT_COLUMN not in output_fieldnames:
            output_fieldnames.append(FINGERPRINT_COLUMN)
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        return
    for row in rows:
        for column in NEW_COLUMNS:
            row.setdefault(column, "")

    indices, texts, empty_text_indices = select_entries(rows)
    console.print(f"[green]{len(indices)}[/green] ligne(s) '{ENTRY_VALUE}' à traiter sur {len(rows)} lignes au total.")
    if empty_text_indices:
        console.print(f"[yellow]{len(empty_text_indices)} ligne(s) '{ENTRY_VALUE}' au texte vide, ignorée(s).[/yellow]")
    if not indices:
        console.print("[yellow]Aucune ligne à traiter.[/yellow]")
        write_output(curation, rows, output_path, output_fieldnames, writes)
        writes.finish(console)
        return

    console.print(f"Chargement du modèle [cyan]{args.model}[/cyan]...")
    try:
        model = load_model(args.model)
    except Exception as error:  # Surface large et imprévisible côté torch/HF Hub.
        console.print(f"[bold red]Erreur au chargement du modèle :[/bold red] {error}")
        return

    errors: list[int] = []

    def report_error(position: int, error: Exception) -> None:
        errors.append(indices[position])
        console.print(f"[red]✗[/red] ligne {indices[position]} : {error}")

    columns = (TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(),
               TextColumn("•"), TimeElapsedColumn(), TextColumn("restant :"), TimeRemainingColumn())
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("Inférence", total=len(indices))
        predictions = predict_spans(
            model, config, texts, args.threshold, args.batch_size,
            on_batch=lambda n: progress.update(task, advance=n), on_error=report_error,
        )

    span_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    suspect_rows = 0
    for row_index, text, spans in zip(indices, texts, predictions):
        if spans is None:
            continue
        row = rows[row_index]
        normalized = normalize_markdown(text)
        # Suspicion évaluée sur le texte normalisé, celui qu'a vu le modèle.
        reasons = suspicion_reasons(normalized.text, spans, args.min_score)
        score = confidence(spans)
        row["ner_confidence"] = f"{score:.4f}" if score is not None else ""
        row["ner_suspect"] = SEPARATOR.join(reasons)
        suspect_rows += bool(reasons)
        reason_counts.update(reasons)
        row["tagged_text"] = render_tagged_text(text, unproject_spans(spans, normalized))
        counts = Counter(span.label for span in spans)
        for label in LABELS:
            row[COUNT_COLUMN_FOR_LABEL[label]] = counts[label]
        span_counts.update(counts)
        if args.verbose:
            console.print(f"[green]✓[/green] ligne {row_index}  {row['tagged_text']}")

    write_output(curation, rows, output_path, output_fieldnames, writes)

    processed = len(indices) - len(errors)
    console.print(f"\nInférence : {processed}/{len(indices)} lignes traitées.")
    console.print("Empans détectés — " + "  ·  ".join(f"{label} : {span_counts[label]}" for label in LABELS))
    if processed:
        detail = ", ".join(f"{reason} {count}" for reason, count in reason_counts.most_common())
        console.print(
            f"Entrées suspectes (colonne ner_suspect) : [yellow]{suspect_rows}[/yellow] / {processed} "
            f"({suspect_rows / processed:.1%})" + (f" — {detail}" if detail else "")
        )
    if errors:
        console.print(f"[red]✗ {len(errors)} ligne(s) en échec (voir le détail ci-dessus).[/red]")
    writes.finish(console)


if __name__ == "__main__":
    main()

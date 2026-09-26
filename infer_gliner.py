"""Exécute un modèle GLiNER-bi entraîné (train_gliner.py) sur les lignes
ENTRY d'un CSV fusionné (merge_annotated_lines.py), et ajoute au CSV le
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
traitée mais sans aucun empan détecté obtient `tagged_text=""` et des
compteurs à 0 — une distinction volontaire entre « non applicable » et
« traité, rien trouvé ».

Labels et texte d'entrée
------------------------
Le modèle prédit les libellés descriptifs utilisés lors de l'entraînement,
pas les codes courts SUBJ/DESC/ADDR : la correspondance est lue dans
`<modèle>/ner_config.json`, écrit par tools/train_gliner.py (obligatoire).
Le modèle voit le texte normalisé (emphase Markdown retirée) ; les empans
sont ramenés sur le texte d'origine de la colonne pour `tagged_text`.

Performance
-----------
GLiNER tourne en local (GPU ou CPU) : contrairement à `autoclassify.py`
(appels réseau à Ollama, où un pool de threads apporte un vrai gain), le
bon levier ici est le batching natif de la bibliothèque
(`model.batch_predict_entities`), qui traite plusieurs textes en un seul
passage tenseur — un pool de threads n'apporterait rien pour de l'inférence
locale sur un seul modèle. Si un batch entier échoue (une entrée
pathologique peut faire échouer tout le lot), chaque élément du batch est
retenté individuellement pour isoler la ligne fautive sans perdre le reste.
"""

import argparse
import csv
from pathlib import Path

from lib.ner.gliner import NerConfig
from lib.ner.spans import Span, normalize_markdown, unproject_spans
from lib.ner.suspicion import DEFAULT_MIN_SCORE, SEPARATOR, confidence, suspicion_reasons

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

DEFAULT_ENTITY_COLUMN = "entity"
DEFAULT_ENTRY_VALUE = "ENTRY"
DEFAULT_TEXT_COLUMN = "markdown"
DEFAULT_BATCH_SIZE = 16
DEFAULT_THRESHOLD = 0.5


NEW_COLUMNS = ["tagged_text", "subject_count", "description_count", "address_count", "ner_confidence", "ner_suspect"]
COUNT_COLUMN_FOR_LABEL = {
    "SUBJ": "subject_count",
    "DESC": "description_count",
    "ADDR": "address_count",
}


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


def select_entries(
    rows: list[dict[str, str]], entity_column: str, entry_value: str, text_column: str
) -> tuple[list[int], list[str], list[int]]:
    """Retourne (indices à traiter, textes correspondants, indices ENTRY
    écartés faute de texte)."""
    indices: list[int] = []
    texts: list[str] = []
    empty_text_indices: list[int] = []
    for index, row in enumerate(rows):
        if row.get(entity_column, "").strip() != entry_value:
            continue
        text = row.get(text_column, "").strip()
        if text:
            indices.append(index)
            texts.append(text)
        else:
            empty_text_indices.append(index)
    return indices, texts, empty_text_indices


# --------------------------------------------------------------------------
# Rendu balisé
# --------------------------------------------------------------------------


def render_tagged_text(text: str, spans: list[dict[str, object]]) -> str:
    """Rend le texte source intégral, avec chaque empan détecté entouré de
    balises `<LABEL>...</LABEL>` à sa position exacte. Rien n'est retiré :
    le texte de liaison (ponctuation, espaces) entre/avant/après les empans
    est conservé tel quel, et les balises apparaissent dans l'ordre du
    texte source (tri par position de départ).
    """

    def escape(fragment: str) -> str:
        return fragment.replace("<", "&lt;").replace(">", "&gt;")

    ordered = sorted(spans, key=lambda span: span["start"])
    pieces: list[str] = []
    cursor = 0
    for span in ordered:
        start, end, label = span["start"], span["end"], span["label"]
        if start < cursor:
            # Empan chevauchant un empan déjà rendu (ne devrait pas arriver
            # avec une NER "plate") : ignoré plutôt que de produire un
            # balisage incohérent (balises imbriquées ou qui se recouvrent).
            continue
        pieces.append(escape(text[cursor:start]))
        pieces.append(f"<{label}>{escape(text[start:end])}</{label}>")
        cursor = end
    pieces.append(escape(text[cursor:]))
    return "".join(pieces)


def finalize_spans(raw_text: str, spans: list[dict]) -> list[dict]:
    """Empans prédits sur le texte normalisé → empans sur le texte d'origine
    (`raw_text`)."""
    normalized = normalize_markdown(raw_text)
    on_normalized = [Span(s["start"], s["end"], s["label"], s.get("score")) for s in spans]
    return [
        {"label": s.label, "text": raw_text[s.start : s.end], "start": s.start, "end": s.end, "score": s.score}
        for s in unproject_spans(on_normalized, normalized)
    ]


# --------------------------------------------------------------------------
# Inférence
# --------------------------------------------------------------------------


def run_inference(
    model,
    indices: list[int],
    texts: list[str],
    gliner_labels: list[str],
    reverse_label_text: dict[str, str],
    threshold: float,
    batch_size: int,
    verbose: bool,
    progress: Progress,
    task_id,
) -> tuple[dict[int, list[dict]], list[tuple[int, str]]]:
    """Exécute l'inférence par lots ; retourne (empans par indice de ligne,
    erreurs [(indice, message), ...])."""
    results: dict[int, list[dict]] = {}
    errors: list[tuple[int, str]] = []

    for start in range(0, len(texts), batch_size):
        chunk_indices = indices[start : start + batch_size]
        chunk_texts = texts[start : start + batch_size]

        try:
            batch_predictions = model.batch_predict_entities(
                chunk_texts, gliner_labels, threshold=threshold
            )
        except Exception:
            # Un texte pathologique peut faire échouer tout le lot : on
            # retente un par un pour isoler la ligne fautive sans perdre le
            # reste du lot.
            batch_predictions = []
            for text in chunk_texts:
                try:
                    batch_predictions.append(
                        model.predict_entities(text, gliner_labels, threshold=threshold)
                    )
                except Exception as item_error:
                    batch_predictions.append(item_error)

        for row_index, text, predictions in zip(
            chunk_indices, chunk_texts, batch_predictions
        ):
            if isinstance(predictions, Exception):
                errors.append((row_index, str(predictions)))
                console.print(f"[red]✗[/red] ligne {row_index} : {predictions}")
            else:
                spans = [
                    {
                        "label": reverse_label_text.get(pred["label"], pred["label"]),
                        "text": pred["text"],
                        "start": pred["start"],
                        "end": pred["end"],
                        "score": pred["score"],
                    }
                    for pred in predictions
                ]
                results[row_index] = spans
                if verbose:
                    console.print(
                        f"[green]✓[/green] ligne {row_index}  {render_tagged_text(text, spans)}"
                    )
                elif not spans:
                    console.print(
                        f"[yellow]⚠[/yellow] ligne {row_index} : aucun empan détecté"
                    )
            progress.update(task_id, advance=1)

    return results, errors


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
        help="CSV fusionné (sortie de merge_annotated_lines.py).",
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
        "--entity-column",
        type=str,
        default=DEFAULT_ENTITY_COLUMN,
        help=f"Colonne indiquant le type de ligne (défaut : '{DEFAULT_ENTITY_COLUMN}').",
    )
    parser.add_argument(
        "--entry-value",
        type=str,
        default=DEFAULT_ENTRY_VALUE,
        help=f"Valeur de --entity-column marquant une ligne à traiter (défaut : '{DEFAULT_ENTRY_VALUE}').",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help=f"Colonne contenant le texte à analyser (défaut : '{DEFAULT_TEXT_COLUMN}').",
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
        "--verbose",
        action="store_true",
        help="Affiche le résultat de chaque ligne traitée (déconseillé sur un gros fichier).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.input_csv}' introuvable."
        )
        return
    if not args.model.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Dossier de modèle '{args.model}' introuvable."
        )
        return

    try:
        label_text = NerConfig.load(args.model).label_text
    except FileNotFoundError as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        return
    reverse_label_text = {v: k for k, v in label_text.items()}
    gliner_labels = list(label_text.values())

    output_path = args.output or args.input_csv.with_suffix(".ner.csv")

    console.print(f"Lecture de [yellow]{args.input_csv.name}[/yellow]...")
    try:
        fieldnames, rows = load_rows(args.input_csv)
        output_fieldnames = insert_columns_after(
            fieldnames, args.entity_column, NEW_COLUMNS
        )
    except (OSError, csv.Error, ValueError) as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        return

    for row in rows:
        for col in NEW_COLUMNS:
            row.setdefault(col, "")

    indices, texts, empty_text_indices = select_entries(
        rows, args.entity_column, args.entry_value, args.text_column
    )
    console.print(
        f"[green]{len(indices)}[/green] ligne(s) '{args.entry_value}' à traiter sur "
        f"{len(rows)} lignes au total."
    )
    if empty_text_indices:
        console.print(
            f"[yellow]{len(empty_text_indices)} ligne(s) '{args.entry_value}' au texte vide, "
            "ignorée(s).[/yellow]"
        )

    if not indices:
        console.print(
            "[yellow]Aucune ligne à traiter — écriture du fichier inchangé.[/yellow]"
        )
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    try:
        from gliner import GLiNER
    except ImportError as error:
        console.print(
            f"[bold red]Erreur :[/bold red] dépendance manquante ({error}). "
            "Installez-la avec : pip install gliner torch"
        )
        return

    console.print(f"Chargement du modèle [cyan]{args.model}[/cyan]...")
    try:
        model = GLiNER.from_pretrained(str(args.model))
        model.eval()
    except Exception as error:  # Surface large et imprévisible côté torch/HF Hub.
        console.print(f"[bold red]Erreur au chargement du modèle :[/bold red] {error}")
        return

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("restant :"),
        TimeRemainingColumn(),
    )
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Inférence", total=len(indices))
        model_texts = [normalize_markdown(text).text for text in texts]
        results, errors = run_inference(
            model,
            indices,
            model_texts,
            gliner_labels,
            reverse_label_text,
            args.threshold,
            args.batch_size,
            args.verbose,
            progress,
            task_id,
        )

    total_counts = {"SUBJ": 0, "DESC": 0, "ADDR": 0}
    zero_span_rows = 0
    suspect_rows = 0
    reason_counts: dict[str, int] = {}
    for row_index, spans in results.items():
        row = rows[row_index]
        raw_text = row[args.text_column].strip()
        # Suspicion évaluée sur le texte normalisé, celui qu'a vu le modèle.
        typed = [Span(s["start"], s["end"], s["label"], s.get("score")) for s in spans]
        reasons = suspicion_reasons(normalize_markdown(raw_text).text, typed, args.min_score)
        score = confidence(typed)
        row["ner_confidence"] = f"{score:.4f}" if score is not None else ""
        row["ner_suspect"] = SEPARATOR.join(reasons)
        if reasons:
            suspect_rows += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        spans = finalize_spans(raw_text, spans)
        row["tagged_text"] = render_tagged_text(raw_text, spans)
        counts = {"SUBJ": 0, "DESC": 0, "ADDR": 0}
        for span in spans:
            counts[span["label"]] = counts.get(span["label"], 0) + 1
        for label, count in counts.items():
            row[COUNT_COLUMN_FOR_LABEL[label]] = count
            total_counts[label] += count
        if not spans:
            zero_span_rows += 1

    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(
        "\n[bold green]✅ Inférence terminée :[/bold green] "
        f"[yellow]{output_path}[/yellow] "
        f"({len(results)}/{len(indices)} lignes traitées)"
    )
    console.print(
        f"Empans détectés — SUBJ : {total_counts['SUBJ']}  ·  "
        f"DESC : {total_counts['DESC']}  ·  ADDR : {total_counts['ADDR']}"
    )
    if zero_span_rows:
        console.print(
            f"[yellow]⚠ {zero_span_rows} ligne(s) traitée(s) sans aucun empan détecté.[/yellow]"
        )
    if results:
        detail = ", ".join(f"{reason} {count}" for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1]))
        console.print(
            f"Entrées suspectes (colonne ner_suspect) : [yellow]{suspect_rows}[/yellow] / {len(results)} "
            f"({suspect_rows / len(results):.1%})" + (f" — {detail}" if detail else "")
        )
    if errors:
        console.print(
            f"[red]✗ {len(errors)} ligne(s) en échec (voir le détail ci-dessus).[/red]"
        )


if __name__ == "__main__":
    main()

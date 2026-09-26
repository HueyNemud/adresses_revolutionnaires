"""Entraîne un modèle GLiNER-bi (bi-encoder) sur les pré-annotations NER
SUBJ/DESC/ADDR produites par autoclassify_labelstudio.py (ou exportées
depuis Label Studio après relecture humaine), avec un split 80/20 et un
rapport d'évaluation .txt.

Conversion des annotations
--------------------------
Le JSON d'entrée est au format Label Studio : chaque exemple porte un texte
(`data.text`) et des empans en **caractères** (`predictions[0].result[]` ou,
si le fichier a été relu dans Label Studio, `annotations[0].result[]`).
GLiNER attend au contraire une tokenisation par espaces avec des empans en
**indices de mots inclusifs** (`{"tokenized_text": [...], "ner": [[start,
end, label], ...]}`) : ce script fait cette conversion lui-même (pas de
dépendance à une fonction interne de la bibliothèque `gliner` dont le
format d'entrée exact n'est pas garanti), et journalise tout exemple ou
empan qu'il ne peut pas convertir plutôt que de le perdre en silence.

Longueur des empans (GLiNER `max_width`)
-----------------------------------------
`max_width` (nombre max. de mots par empan candidat) vaut 12 par défaut
dans GLiNER — trop court pour ce jeu de données, où certains DESC dépassent
50 mots. Ce script calcule automatiquement la largeur maximale réellement
observée dans les données et charge le modèle avec ce `max_width` (override
documenté de `GLiNER.from_pretrained`), sauf si `--max-width` est fourni
explicitement.

Labels
------
GLiNER-bi encode sémantiquement le *texte* de chaque label (bi-encoder :
encodeur de texte + encodeur de label séparés) : utiliser tel quel les
codes courts SUBJ/DESC/ADDR prive le modèle de sens exploitable. Ce script
les fait correspondre par défaut à des libellés descriptifs en anglais
(l'encodeur de labels de `knowledgator/gliner-bi-base-v2.0` est pré-entraîné
en anglais), configurables via `--label-subj`/`--label-desc`/`--label-addr`.
Ces libellés et le texte d'entrée sont enregistrés dans
`<modèle>/ner_config.json`, relu par `infer_gliner.py` et `audit_ner.py`.

Texte d'entrée (`--input`)
--------------------------
- `normalized` (défaut) : emphase Markdown retirée (`lib.ner.spans.
  normalize_markdown`), comme le gold ; aucune frontière ne tombe dans une
  paire de marqueurs `**…**` ;
- `raw` : texte `markdown` d'origine (comme v1), quand la tâche fournit
  `data.raw_text` ; les empans y sont transportés. L'italique y reste un
  indice possible de DESC : c'est une option à mesurer, pas un réglage.

Jeu gold et validation
----------------------
Les textes du gold (`--gold`, dev et test) sont exclus des données, quelle
que soit leur source. Le split de validation se fait **par page** (les
entrées d'une même page ne se répartissent pas entre entraînement et
validation). Ce split ne sert qu'au suivi de l'entraînement ; la mesure qui
compte est `audit_ner.py --model <modèle>` sur le gold.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from rich.console import Console

from lib.ner.gliner import NerConfig
from lib.ner.spans import (
    Span,
    Token,
    char_span_to_word_span,
    ls_task_spans,
    normalize_markdown,
    project_spans,
    tokenize_with_offsets,
    unproject_spans,
)

console = Console()

DEFAULT_MODEL = "knowledgator/gliner-bi-base-v2.0"
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_THRESHOLD = 0.5
DEFAULT_INPUT = "normalized"
DEFAULT_GOLD = Path("data/ner/gold_v1.ls.json")
DEFAULT_MODELS_DIR = Path("models")

# Libellés descriptifs par défaut envoyés au modèle (voir docstring du
# module). Les clés SUBJ/DESC/ADDR restent la référence interne du script
# (rapport, CLI) ; seules les valeurs sont vues par GLiNER.
DEFAULT_LABEL_TEXT = {
    "SUBJ": "person or business name",
    "DESC": "activity description",
    "ADDR": "postal address",
}


# --------------------------------------------------------------------------
# Conversion Label Studio (empans en caractères) -> GLiNER (empans en mots)
# --------------------------------------------------------------------------
# `tokenize_with_offsets` / `char_span_to_word_span` : voir lib/ner/spans.py.


@dataclass
class ConversionReport:
    total_examples: int = 0
    kept_examples: int = 0
    skipped_examples: list[tuple[str, str]] = field(default_factory=list)  # (uid, motif)
    boundary_snapped: int = 0
    max_span_width_words: int = 0
    label_counts: Counter = field(default_factory=Counter)
    unknown_labels: Counter = field(default_factory=Counter)
    excluded_gold: int = 0


@dataclass
class ConvertedExample:
    uid: str  # "<filename>#<uid>" : uid seul n'est unique qu'au sein d'un même fichier source
    text: str
    tokens: list[Token]
    tokenized_text: list[str]
    ner: list[list]  # [[start_word, end_word, label_text], ...]
    group: str = ""  # grappe du split de validation (page)


def _dedupe_and_prune_nested(ner: list[list]) -> list[list]:
    """Supprime les doublons exacts et les empans strictement inclus dans un
    autre empan du même label. Ces cas viennent des segments de liaison
    (ponctuation/espaces) que le modèle d'annotation NER a parfois classés
    avec le même label que leur voisin (ex. une virgule isolée entre deux
    mots d'une même entité) : ce ne sont pas de nouvelles entités, juste un
    sous-empan du mot déjà couvert par l'empan réel — les laisser créerait
    un signal d'entraînement contradictoire (le même mot, tantôt entité
    complète, tantôt non).
    """
    unique: list[list] = []
    seen: set[tuple] = set()
    for start, end, label in ner:
        key = (start, end, label)
        if key not in seen:
            seen.add(key)
            unique.append([start, end, label])

    pruned = []
    for start, end, label in unique:
        contained_in_larger = any(
            (start2, end2) != (start, end)
            and label2 == label
            and start2 <= start
            and end2 >= end
            for start2, end2, label2 in unique
        )
        if not contained_in_larger:
            pruned.append([start, end, label])
    return pruned


def model_text_and_spans(data: dict, spans: list[Span], input_kind: str) -> tuple[str, list[Span]]:
    """Texte donné au modèle et empans transportés sur ce texte.

    `data.text` porte les empans. En `normalized`, il est normalisé s'il ne
    l'est pas déjà (anciens JSON dont le texte garde l'emphase) ; en `raw`,
    on revient à `data.raw_text` quand il est fourni et cohérent.
    """
    text = data.get("text", "")
    normalized = normalize_markdown(text)
    if input_kind == "normalized":
        if normalized.text == text:
            return text, spans
        return normalized.text, project_spans(spans, normalized)
    raw_text = data.get("raw_text")
    if raw_text:
        raw_normalized = normalize_markdown(raw_text)
        if raw_normalized.text == text:
            return raw_text, unproject_spans(spans, raw_normalized)
    return text, spans


def split_group(data: dict, uid: str) -> str:
    """Grappe du split de validation : la page quand elle est connue."""
    volume = data.get("volume") or data.get("filename") or ""
    page = data.get("page") or data.get("page_index") or ""
    return f"{volume}#{page}" if volume and page else uid


def convert_task(
    task: dict,
    label_text: dict[str, str],
    report: ConversionReport,
    input_kind: str,
    excluded_texts: set[str],
) -> ConvertedExample | None:
    """Convertit une tâche Label Studio en exemple GLiNER, ou None si elle
    doit être écartée (texte du gold, texte vide, aucun empan convertible)."""
    data = task.get("data", {})
    raw_uid = data.get("uid") or data.get("key") or "<inconnu>"
    filename = data.get("filename") or ""
    # `uid` (page.bloc.ligne) n'est unique qu'au sein d'un même fichier source :
    # ce jeu de données en agrège plusieurs (numérotation qui recommence à
    # chaque document), donc un identifiant sans le nom de fichier serait
    # ambigu dans le rapport.
    uid = f"{filename}#{raw_uid}" if filename else raw_uid

    if normalize_markdown(data.get("text", "")).text in excluded_texts:
        report.excluded_gold += 1
        return None

    # Relecture humaine sous "annotations", pré-annotations sous
    # "predictions" : la version relue est préférée (`ls_task_spans`).
    for block in (task.get("annotations") or []) + [p for p in task.get("predictions") or [] if isinstance(p, dict)]:
        for item in block.get("result", []):
            for raw_label in item.get("value", {}).get("labels") or []:
                if raw_label not in label_text:
                    report.unknown_labels[raw_label] += 1
    spans = ls_task_spans(task) or []
    text, spans = model_text_and_spans(data, spans, input_kind)

    tokens = tokenize_with_offsets(text)
    if not tokens:
        report.skipped_examples.append((uid, "texte vide"))
        return None

    ner: list[list] = []
    for span in spans:
        word_span = char_span_to_word_span(tokens, span.start, span.end)
        if word_span is None:
            report.skipped_examples.append((uid, f"empan {text[span.start:span.end]!r} hors limites du texte"))
            continue
        start_word, end_word = word_span
        if tokens[start_word].start_char != span.start or tokens[end_word].end_char != span.end:
            report.boundary_snapped += 1
        report.max_span_width_words = max(report.max_span_width_words, end_word - start_word + 1)
        report.label_counts[span.label] += 1
        ner.append([start_word, end_word, label_text[span.label]])

    if not ner:
        report.skipped_examples.append((uid, "aucun empan valide"))
        return None

    return ConvertedExample(
        uid=uid,
        text=text,
        tokens=tokens,
        tokenized_text=[tok.text for tok in tokens],
        ner=_dedupe_and_prune_nested(ner),
        group=split_group(data, uid),
    )


def gold_texts(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {normalize_markdown(task["data"]["text"]).text for task in json.loads(path.read_text(encoding="utf-8"))}


def load_examples(
    input_paths: list[Path], label_text: dict[str, str], input_kind: str, excluded_texts: set[str]
) -> tuple[list[ConvertedExample], ConversionReport]:
    report = ConversionReport()
    examples: list[ConvertedExample] = []
    for input_path in input_paths:
        tasks = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise ValueError(f"{input_path} : le JSON doit être une liste de tâches Label Studio.")
        report.total_examples += len(tasks)
        for task in tasks:
            example = convert_task(task, label_text, report, input_kind, excluded_texts)
            if example is not None:
                examples.append(example)
    report.kept_examples = len(examples)
    return examples, report


def split_train_test(
    examples: list[ConvertedExample], test_ratio: float, seed: int
) -> tuple[list[ConvertedExample], list[ConvertedExample]]:
    """Split reproductible **par page** : une page va entièrement en
    entraînement ou en validation (hachage de la page et de la graine), pour
    que la validation ne voie pas des voisines quasi identiques des lignes
    apprises."""

    def in_test(group: str) -> bool:
        digest = hashlib.sha1(f"{seed}:{group}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 2**32 < test_ratio

    train = [example for example in examples if not in_test(example.group)]
    test = [example for example in examples if in_test(example.group)]
    return train, test


# --------------------------------------------------------------------------
# Évaluation (P/R/F1 par classe, correspondance exacte empan+label)
# --------------------------------------------------------------------------


@dataclass
class LabelScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(
    model, examples: list[ConvertedExample], gliner_labels: list[str], threshold: float
) -> dict[str, LabelScore]:
    """Évalue le modèle sur `examples`. Correspondance exacte (empan en mots
    + label), comme l'évaluateur NER de référence de GLiNER (un empan n'est
    correct que si ses bornes ET son label correspondent exactement).
    """
    scores: dict[str, LabelScore] = {label: LabelScore() for label in gliner_labels}
    scores["__micro__"] = LabelScore()

    for example in examples:
        gold = {tuple(item) for item in example.ner}

        predicted_raw = model.predict_entities(example.text, gliner_labels, threshold=threshold)
        predicted = set()
        for pred in predicted_raw:
            span = char_span_to_word_span(example.tokens, pred["start"], pred["end"])
            if span is None:
                continue
            predicted.add((span[0], span[1], pred["label"]))

        for entry in gold & predicted:
            label = entry[2]
            scores[label].tp += 1
            scores["__micro__"].tp += 1
        for entry in gold - predicted:
            label = entry[2]
            scores[label].fn += 1
            scores["__micro__"].fn += 1
        for entry in predicted - gold:
            label = entry[2]
            scores.setdefault(label, LabelScore())
            scores[label].fp += 1
            scores["__micro__"].fp += 1

    return scores


# --------------------------------------------------------------------------
# Rapport
# --------------------------------------------------------------------------


def format_report(
    *,
    input_paths: list[Path],
    input_kind: str,
    gold_path: Path | None,
    output_dir: Path,
    model_name: str,
    label_text: dict[str, str],
    max_width: int,
    epochs: float,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    conversion_report: ConversionReport,
    n_train: int,
    n_test: int,
    scores: dict[str, LabelScore],
    reverse_label_text: dict[str, str],
) -> str:
    lines = [
        "RAPPORT D'ENTRAÎNEMENT — train_gliner.py",
        "Entrées  : " + ", ".join(str(path) for path in input_paths),
        f"Modèle   : {output_dir}",
        "",
        "== Configuration ==",
        f"Modèle de base                : {model_name}",
        f"max_width (largeur d'empan)   : {max_width} mots",
        f"Époques demandées             : {epochs} (≈ {max_steps} steps)",
        f"Taille de batch               : {batch_size}",
        f"Taux d'apprentissage          : {learning_rate}",
        f"Seuil de décision (évaluation): {threshold}",
        f"Texte d'entrée du modèle      : {input_kind}",
        "Correspondance label -> texte envoyé au modèle :",
    ]
    for short, text in label_text.items():
        lines.append(f"  - {short} -> \"{text}\"")

    lines += [
        "",
        "== Données ==",
        f"Exemples lus                  : {conversion_report.total_examples}",
        f"Exemples du gold exclus       : {conversion_report.excluded_gold} (gold : {gold_path or 'aucun'})",
        f"Exemples conservés            : {conversion_report.kept_examples}",
        f"  dont entraînement           : {n_train}",
        f"  dont validation (par page)  : {n_test}",
        f"Empans élargis à la frontière de mot la plus proche : {conversion_report.boundary_snapped}",
        f"Largeur d'empan maximale observée : {conversion_report.max_span_width_words} mots",
    ]
    if conversion_report.unknown_labels:
        lines.append("Classes non reconnues (ignorées) :")
        for label, count in conversion_report.unknown_labels.most_common():
            lines.append(f"  - {label} : {count}")
    if conversion_report.skipped_examples:
        lines.append(f"Exemples écartés ({len(conversion_report.skipped_examples)}) :")
        for uid, reason in conversion_report.skipped_examples[:50]:
            lines.append(f"  - {uid} : {reason}")
        if len(conversion_report.skipped_examples) > 50:
            lines.append(f"  ... et {len(conversion_report.skipped_examples) - 50} de plus.")

    lines += ["", "== Évaluation sur le split de validation (par page) =="]
    lines.append(f"{'Classe':<30}{'Précision':>12}{'Rappel':>12}{'F1':>10}{'TP':>8}{'FP':>8}{'FN':>8}")
    for gliner_label, score in scores.items():
        if gliner_label == "__micro__":
            continue
        display_name = reverse_label_text.get(gliner_label, gliner_label)
        lines.append(
            f"{display_name:<30}{score.precision:>12.1%}{score.recall:>12.1%}"
            f"{score.f1:>10.1%}{score.tp:>8}{score.fp:>8}{score.fn:>8}"
        )
    micro = scores["__micro__"]
    lines.append(
        f"{'MICRO (global)':<30}{micro.precision:>12.1%}{micro.recall:>12.1%}"
        f"{micro.f1:>10.1%}{micro.tp:>8}{micro.fp:>8}{micro.fn:>8}"
    )
    lines.append("")
    lines.append(
        "Note : correspondance exacte (bornes ET label) — un empan partiellement "
        "recouvrant l'empan attendu compte comme un échec, pas un succès partiel. "
        "La mesure de référence est `audit_ner.py --model` sur le gold."
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne un modèle GLiNER-bi sur des pré-annotations NER "
            "SUBJ/DESC/ADDR (format Label Studio) et évalue sur un split 80/20."
        )
    )
    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help="JSON Label Studio (ex. data/ner/train_v2.ls.json de tools/build_ner_training.py), un ou plusieurs.",
    )
    parser.add_argument(
        "--input",
        choices=("normalized", "raw"),
        default=DEFAULT_INPUT,
        help=f"Texte donné au modèle : emphase Markdown retirée ou texte d'origine (défaut : {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help=f"Gold dont les textes sont exclus de l'entraînement (défaut : {DEFAULT_GOLD}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier du modèle entraîné (défaut : models/<première entrée>-<input>.gliner-model/).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=Path,
        default=None,
        help="Chemin du rapport .txt (défaut : <output-dir>/eval_report.txt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Modèle GLiNER-bi de départ (défaut : {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Proportion réservée à l'évaluation (défaut : {DEFAULT_TEST_RATIO}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Graine du split train/test, pour reproductibilité (défaut : {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
        help=f"Nombre d'époques d'entraînement (défaut : {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Taille de batch par device (défaut : {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Taux d'apprentissage (défaut : {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help=(
            "Largeur maximale d'empan en mots (défaut : calculée automatiquement "
            "à partir de la largeur maximale observée dans les données)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Seuil de confiance pour l'évaluation finale (défaut : {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Entraînement en bfloat16 (défaut : activé automatiquement si un GPU CUDA est détecté).",
    )
    parser.add_argument(
        "--label-subj",
        type=str,
        default=DEFAULT_LABEL_TEXT["SUBJ"],
        help=f"Libellé envoyé au modèle pour SUBJ (défaut : '{DEFAULT_LABEL_TEXT['SUBJ']}').",
    )
    parser.add_argument(
        "--label-desc",
        type=str,
        default=DEFAULT_LABEL_TEXT["DESC"],
        help=f"Libellé envoyé au modèle pour DESC (défaut : '{DEFAULT_LABEL_TEXT['DESC']}').",
    )
    parser.add_argument(
        "--label-addr",
        type=str,
        default=DEFAULT_LABEL_TEXT["ADDR"],
        help=f"Libellé envoyé au modèle pour ADDR (défaut : '{DEFAULT_LABEL_TEXT['ADDR']}').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing = [path for path in args.input_paths if not path.exists()]
    if missing:
        console.print(f"[bold red]Erreur :[/bold red] Fichier(s) introuvable(s) : {', '.join(map(str, missing))}.")
        return
    if not args.gold.exists():
        console.print(f"[yellow]⚠ gold '{args.gold}' introuvable : aucune exclusion, l'audit sur ce gold serait biaisé.[/yellow]")

    label_text = {
        "SUBJ": args.label_subj,
        "DESC": args.label_desc,
        "ADDR": args.label_addr,
    }
    reverse_label_text = {v: k for k, v in label_text.items()}
    gliner_labels = list(label_text.values())

    stem = args.input_paths[0].name.split(".", 1)[0]
    output_dir = args.output_dir or DEFAULT_MODELS_DIR / f"{stem}-{args.input}.gliner-model"
    report_path = args.report or (output_dir / "eval_report.txt")

    console.print(f"Lecture et conversion de [yellow]{', '.join(p.name for p in args.input_paths)}[/yellow] (texte {args.input})...")
    try:
        examples, conversion_report = load_examples(args.input_paths, label_text, args.input, gold_texts(args.gold))
    except (json.JSONDecodeError, ValueError) as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}")
        return

    if not examples:
        console.print("[yellow]Aucun exemple exploitable après conversion.[/yellow]")
        return

    console.print(
        f"[green]{conversion_report.kept_examples}/{conversion_report.total_examples}[/green] "
        f"exemples conservés (largeur d'empan max observée : "
        f"{conversion_report.max_span_width_words} mots) ; {conversion_report.excluded_gold} exemple(s) du gold exclu(s)."
    )
    if conversion_report.skipped_examples:
        console.print(
            f"[yellow]{len(conversion_report.skipped_examples)} exemple(s) écarté(s) "
            "— détail dans le rapport final.[/yellow]"
        )

    train_examples, test_examples = split_train_test(examples, args.test_ratio, args.seed)
    console.print(f"Split : [cyan]{len(train_examples)}[/cyan] entraînement / [cyan]{len(test_examples)}[/cyan] évaluation.")

    max_width = args.max_width or conversion_report.max_span_width_words
    console.print(f"max_width utilisé : [cyan]{max_width}[/cyan] mots.")

    try:
        import torch
        from gliner import GLiNER
    except ImportError as error:
        console.print(
            f"[bold red]Erreur :[/bold red] dépendance manquante ({error}). "
            "Installez-la avec : pip install gliner torch"
        )
        return

    use_bf16 = args.bf16 if args.bf16 is not None else torch.cuda.is_available()

    console.print(f"Chargement de [cyan]{args.model}[/cyan] (max_width={max_width})...")
    try:
        model = GLiNER.from_pretrained(args.model, max_width=max_width)
    except Exception as error:  # Surface large et imprévisible côté torch/HF Hub.
        console.print(f"[bold red]Erreur au chargement du modèle :[/bold red] {error}")
        return

    train_dataset = [{"tokenized_text": ex.tokenized_text, "ner": ex.ner} for ex in train_examples]
    eval_dataset = [{"tokenized_text": ex.tokenized_text, "ner": ex.ner} for ex in test_examples]

    steps_per_epoch = max(1, math.ceil(len(train_dataset) / args.batch_size))
    max_steps = max(1, round(steps_per_epoch * args.epochs))
    console.print(
        f"Entraînement : {args.epochs} époque(s) ≈ {max_steps} steps "
        f"({steps_per_epoch} steps/époque, batch={args.batch_size})..."
    )

    try:
        trainer = model.train_model(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            bf16=use_bf16,
        )
        trainer.save_model()
        # Libellés et texte d'entrée, relus par infer_gliner.py et audit_ner.py.
        NerConfig(label_text, args.input).save(output_dir)
    except Exception as error:  # Surface large et imprévisible côté torch/HF Trainer.
        console.print(f"[bold red]Erreur pendant l'entraînement :[/bold red] {error}")
        return

    console.print("Évaluation sur le split de validation (par page)...")
    model.eval()
    scores = evaluate(model, test_examples, gliner_labels, args.threshold)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        format_report(
            input_paths=args.input_paths,
            input_kind=args.input,
            gold_path=args.gold if args.gold.exists() else None,
            output_dir=output_dir,
            model_name=args.model,
            label_text=label_text,
            max_width=max_width,
            epochs=args.epochs,
            max_steps=max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            conversion_report=conversion_report,
            n_train=len(train_examples),
            n_test=len(test_examples),
            scores=scores,
            reverse_label_text=reverse_label_text,
        ),
        encoding="utf-8",
    )

    micro = scores["__micro__"]
    console.print(
        "\n[bold green]✅ Entraînement terminé :[/bold green] "
        f"[yellow]{output_dir}[/yellow] "
        f"(F1 micro sur la validation : {micro.f1:.1%})"
    )
    console.print(f"[bold green]📄 Rapport :[/bold green] [yellow]{report_path}[/yellow]")
    console.print(f"Mesure de référence : [cyan]uv run audit_ner.py --model {output_dir} --split dev --rules[/cyan]")


if __name__ == "__main__":
    main()

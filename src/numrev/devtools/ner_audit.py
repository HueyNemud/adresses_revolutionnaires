"""Audit de la segmentation NER SUBJ/DESC/ADDR sur le jeu gold.

Compare un ou plusieurs systèmes à la relecture humaine du gold
(`data/ner/gold.ls.json`, tiré par `numrev gold ner` et corrigé
dans Label Studio), et écrit `reports/ner/rapport.md` et
`reports/ner/erreurs.csv`.

Systèmes évalués (au moins un) :
- `--model DOSSIER` : un modèle GLiNER, exécuté sur place (CPU : quelques
  secondes pour quelques centaines d'entrées) ; `--sweep` balaie le seuil ;
- `--predictions NOM=FICHIER` : des prédictions déjà calculées, JSON Label
  Studio (prédictions ou annotations) ou CSV à colonne
  `tagged_text` (sortie de `numrev tag`), appariées au gold par texte
  normalisé.

Métriques (voir `numrev.ner.metrics`) : exactitude par entrée (métrique
principale, part des entrées sans correction à faire), F1 par classe,
exactitude par token ; pondérées par le plan de sondage du gold, IC 95 %
par bootstrap des pages, Δ appariés contre la référence (▲ / ▼ =
intervalle entièrement au-dessus / au-dessous de 0). Pour un modèle avec
scores, qualité du score de confiance comme outil de tri pour la relecture,
et efficacité du drapeau `ner_suspect` d'`numrev tag`
(`numrev.ner.suspicion`, seuil `--min-score`) : part d'entrées signalées et
part des erreurs attrapées, motif par motif.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from numrev.command import CommandError, console
from numrev.curation import read_csv
from numrev.ner.gliner import DEFAULT_THRESHOLD
from numrev.ner.metrics import METRICS, Comparison, Scored, compare, page_bootstrap, paired_delta, summarize
from numrev.ner.spans import (
    Span,
    ls_task_spans,
    normalize_markdown,
    parse_tagged_text,
    project_spans,
    render_tagged_text,
    spans_from_ls_result,
)
from numrev.ner.suspicion import DEFAULT_MIN_SCORE, REASONS, SEPARATOR, suspicion_reasons
from numrev.paths import NER_GOLD, REPORTS_DIR
from numrev.reporting import fmt, fmt_ci, fmt_delta, md_code, md_table, write_table
from numrev.stats import review_capture, roc_auc

DESCRIPTION = "Audit de la NER SUBJ/DESC/ADDR sur le jeu gold relu."

DEFAULT_GOLD = NER_GOLD
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "ner"
REVIEW_BUDGETS = (0.05, 0.10, 0.20, 0.30)
SUSPICION_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
SWEEP_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class GoldEntry:
    data: dict
    gold: list[Span]

    @property
    def text(self) -> str:
        return self.data["text"]


@dataclass
class System:
    name: str
    predictions: list[list[Span] | None]  # None : pas de prédiction pour cette entrée
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def has_scores(self) -> bool:
        return any(span.score is not None for spans in self.predictions if spans for span in spans)


# ----------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------
def load_gold(path: Path, split: str) -> tuple[list[GoldEntry], int]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    entries, unreviewed = [], 0
    for task in tasks:
        data = task["data"]
        if split != "all" and data.get("split") != split:
            continue
        reviewed = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
        if not reviewed:
            unreviewed += 1
            continue
        entries.append(GoldEntry(data, spans_from_ls_result(reviewed[-1].get("result", []))))
    return entries, unreviewed


def load_prediction_file(name: str, path: Path, entries: Sequence[GoldEntry]) -> System:
    """Prédictions appariées au gold par texte normalisé."""
    by_text: dict[str, list[Span]] = {}
    if path.suffix == ".json":
        for task in json.loads(path.read_text(encoding="utf-8")):
            text = task.get("data", {}).get("text", "")
            spans = ls_task_spans(task, prefer="predictions")
            if text and spans is not None:
                normalized = normalize_markdown(text)
                by_text[normalized.text] = project_spans(spans, normalized)
    else:
        for row in read_csv(path)[1]:
            if not row.get("tagged_text"):
                continue
            try:
                text, spans = parse_tagged_text(row["tagged_text"])
            except ValueError:
                continue
            normalized = normalize_markdown(text)
            by_text[normalized.text] = project_spans(spans, normalized)
    return System(name, [by_text.get(entry.text) for entry in entries])


def model_systems(model_dir: Path, entries: Sequence[GoldEntry], threshold: float, sweep: bool) -> list[System]:
    from numrev.ner.gliner import NerConfig, load_model, predict_spans

    config = NerConfig.load(model_dir)
    console.print(f"Chargement de [cyan]{model_dir}[/cyan] (libellés : {config.label_text})...")
    model = load_model(model_dir)
    raw_texts = [entry.data["raw_text"] for entry in entries]
    # Le seuil demandé d'abord : c'est lui qui sert de référence aux Δ.
    thresholds = [threshold, *sorted(set(SWEEP_THRESHOLDS) - {threshold})] if sweep else [threshold]
    systems = []
    for value in thresholds:
        started = time.perf_counter()
        predictions = predict_spans(model, config, raw_texts, value)
        console.print(f"  seuil {value:.2f} : {len(entries)} entrées en {time.perf_counter() - started:.0f} s")
        suffix = "" if value == threshold else f"@{value:.2f}"
        systems.append(System(f"{model_dir.name}{suffix}", predictions))
    return systems


# ----------------------------------------------------------------------
# Évaluation
# ----------------------------------------------------------------------
def score_system(system: System, entries: Sequence[GoldEntry], clusters: np.ndarray) -> Scored:
    """Une entrée sans prédiction compte comme une prédiction vide (erreur)."""
    system.comparisons = [compare(entry.text, entry.gold, spans or []) for entry, spans in zip(entries, system.predictions)]
    return Scored(
        np.stack([c.vector for c in system.comparisons]),
        np.array([float(entry.data.get("weight", 1.0)) for entry in entries]),
        clusters,
    )


def uncertainty(spans: list[Span] | None) -> float:
    """1 − score minimal des empans de l'entrée (1 si aucun empan)."""
    scores = [span.score for span in spans or [] if span.score is not None]
    return 1 - min(scores) if scores else 1.0


def flag_row(label: str, flagged: Sequence[bool], errors: Sequence[bool], weights: Sequence[float]) -> list[str]:
    """Signalées (pondéré), erreurs attrapées (pondéré), précision
    (pondérée), puis les mêmes en nombres bruts."""
    caught = [f and e for f, e in zip(flagged, errors)]
    error_weight = sum(w for e, w in zip(errors, weights) if e)
    flagged_weight = sum(w for f, w in zip(flagged, weights) if f)
    caught_weight = sum(w for c, w in zip(caught, weights) if c)
    return [
        label,
        fmt(weighted_rate(flagged, weights)),
        fmt(caught_weight / error_weight if error_weight else math.nan),
        fmt(caught_weight / flagged_weight if flagged_weight else math.nan),
        f"{sum(flagged)} / {len(flagged)}",
        f"{sum(caught)} / {sum(errors)}",
    ]


def suspicion_section(entries: Sequence[GoldEntry], systems: Sequence[System], min_score: float) -> list[str]:
    headers = ["", "signalées", "erreurs attrapées", "précision", "brut : signalées", "brut : erreurs attrapées"]
    lines = [
        f"## Entrées suspectes (`ner_suspect`, `--min-score` {min_score})",
        "",
        "Drapeau d'`numrev tag` (`numrev/ner/suspicion.py`). *Signalées* : part des entrées signalées ; *erreurs "
        "attrapées* : part des entrées erronées qui sont signalées ; *précision* : part des signalées qui sont erronées. "
        "Valeurs pondérées (corpus entier), puis nombres bruts sur l'échantillon gold. Une ligne par motif, puis le seuil "
        "de score balayé pour le drapeau complet.",
        "",
    ]
    weights = [float(entry.data.get("weight", 1.0)) for entry in entries]
    for system in systems:
        errors = [not c.exact for c in system.comparisons]
        reasons = [suspicion_reasons(e.text, spans or [], min_score) for e, spans in zip(entries, system.predictions)]
        rows = [flag_row("**drapeau (≥ 1 motif)**", [bool(r) for r in reasons], errors, weights)]
        rows += [flag_row(reason, [reason in r for r in reasons], errors, weights) for reason in REASONS]
        if system.has_scores:
            for threshold in SUSPICION_THRESHOLDS:
                flagged = [bool(suspicion_reasons(e.text, spans or [], threshold)) for e, spans in zip(entries, system.predictions)]
                rows.append(flag_row(f"drapeau, `--min-score` {threshold}", flagged, errors, weights))
        lines += [f"### `{system.name}`", "", md_table(headers, rows), ""]
    return lines


def weighted_rate(flags: Sequence[bool], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(w for flag, w in zip(flags, weights) if flag) / total if total else math.nan


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------
def breakdown(entries: Sequence[GoldEntry], systems: Sequence[System], key: str, limit: int | None = None) -> str:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[str(entry.data.get(key, "?"))].append(index)
    ordered = sorted(groups.items(), key=lambda item: -len(item[1]))[:limit]
    rows = []
    for name, indices in ordered:
        weights = [float(entries[i].data.get("weight", 1.0)) for i in indices]
        rows.append(
            [md_code(name, 40), len(indices)]
            + [fmt(weighted_rate([s.comparisons[i].exact for i in indices], weights)) for s in systems]
        )
    return md_table([key, "n", *(s.name for s in systems)], rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    entries: Sequence[GoldEntry],
    unreviewed: int,
    systems: Sequence[System],
    scored: Sequence[Scored],
    bootstrap: np.ndarray,
    reference: int,
) -> None:
    n_pages = int(scored[0].clusters.max()) + 1
    lines = [
        "# Audit NER SUBJ / DESC / ADDR",
        "",
        f"Généré par `numrev audit ner` le {time.strftime('%Y-%m-%d %H:%M')} : gold `{args.gold}`, split **{args.split}**, "
        f"{len(entries)} entrées relues ({unreviewed} non relues ignorées) sur {n_pages} pages, "
        f"{args.bootstrap} rééchantillonnages bootstrap par page. Métriques pondérées par le plan de sondage "
        "(représentatives du corpus entier). Conventions : `docs/guide_annotation_ner.md`.",
        "",
    ]

    lines += ["## Résumé", "", "Exactitude = part des entrées sans aucune correction à faire (métrique principale).", ""]
    rows = []
    for system, score in zip(systems, scored):
        summary = summarize(score, bootstrap)
        missing = sum(p is None for p in system.predictions)
        rows.append([system.name, *(fmt_ci(v, ci) for v, ci in summary), missing])
    lines += [md_table(["système", *METRICS, "sans prédiction"], rows), ""]

    lines += [f"## Comparaison appariée avec `{systems[reference].name}`", ""]
    rows = []
    for index, (system, score) in enumerate(zip(systems, scored)):
        if index != reference:
            rows.append([system.name, *(fmt_delta(v, ci) for v, ci in paired_delta(scored[reference], score, bootstrap))])
    lines += [md_table(["système", *(f"Δ {m}" for m in METRICS)], rows) if rows else "_Un seul système._", ""]

    lines += ["## Types d'erreurs", "", "Part pondérée des entrées. *Signature* : segment manquant, en trop ou mal classé ; *frontière* : mêmes classes, bornes différentes.", ""]
    weights = [float(entry.data.get("weight", 1.0)) for entry in entries]
    rows = []
    for system in systems:
        kinds = [c.error_kind for c in system.comparisons]
        rows.append([system.name, *(fmt(weighted_rate([k == kind for k in kinds], weights)) for kind in ("ok", "signature", "frontière"))])
    lines += [md_table(["système", "correct", "signature", "frontière"], rows), ""]

    lines += ["## Exactitude par volume", "", breakdown(entries, systems, "volume"), ""]
    lines += ["## Exactitude par strate du tirage", "", "Au sein d'une strate, toutes les entrées ont le même poids.", "", breakdown(entries, systems, "stratum"), ""]
    lines += ["## Exactitude par profil typographique (15 plus fréquents)", "", breakdown(entries, systems, "profile", 15), ""]

    scored_systems = [s for s in systems if s.has_scores]
    if scored_systems:
        lines += [
            "## Score de confiance comme outil de relecture",
            "",
            "Incertitude d'une entrée = 1 − score minimal de ses empans. AUC : probabilité qu'une entrée erronée soit plus incertaine "
            "qu'une entrée correcte. Colonnes suivantes : part des erreurs trouvées en relisant les k % d'entrées les plus incertaines "
            "(non pondéré, sur l'échantillon gold).",
            "",
        ]
        rows = []
        for system in scored_systems:
            u = np.array([uncertainty(p) for p in system.predictions])
            errors = np.array([not c.exact for c in system.comparisons])
            capture = review_capture(u, errors, REVIEW_BUDGETS)
            rows.append([system.name, fmt(roc_auc(u, errors)), *(fmt(share) for _, _, share in capture)])
        lines += [md_table(["système", "AUC", *(f"relire {b:.0%}" for b in REVIEW_BUDGETS)], rows), ""]

    lines += suspicion_section(entries, systems, args.min_score)

    lines += [
        "## Exemples d'erreurs",
        "",
        f"Liste complète : `{args.output / 'erreurs.csv'}`. Ci-dessous, jusqu'à 15 erreurs de `{systems[reference].name}`.",
        "",
    ]
    rows = []
    for entry, spans, comparison in zip(entries, systems[reference].predictions, systems[reference].comparisons):
        if not comparison.exact and len(rows) < 15:
            rows.append([comparison.error_kind, md_code(render_tagged_text(entry.text, entry.gold), 160), md_code(render_tagged_text(entry.text, spans or []), 160)])
    lines += [md_table(["type", "gold", "prédit"], rows, align="lll") if rows else "_Aucune erreur._", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_errors(path: Path, entries: Sequence[GoldEntry], systems: Sequence[System], min_score: float) -> None:
    rows = [
        [
            system.name,
            comparison.error_kind,
            entry.data.get("key", ""),
            entry.data.get("stratum", ""),
            entry.data.get("volume", ""),
            render_tagged_text(entry.text, entry.gold),
            render_tagged_text(entry.text, spans or []),
            f"{uncertainty(spans):.4f}" if system.has_scores else "",
            SEPARATOR.join(suspicion_reasons(entry.text, spans or [], min_score)),
        ]
        for system in systems
        for entry, spans, comparison in zip(entries, system.predictions, system.comparisons)
        if not comparison.exact
    ]
    write_table(path, ["système", "type", "clé", "strate", "volume", "gold", "prédit", "incertitude", "suspect"], rows)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help=f"Gold Label Studio (défaut : {DEFAULT_GOLD}).")
    parser.add_argument("--split", choices=("test", "dev", "all"), default="test", help="Partie du gold évaluée (défaut : test).")
    parser.add_argument("--model", type=Path, action="append", default=[], help="Dossier d'un modèle GLiNER à évaluer (répétable).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Score minimal d'un empan GLiNER (défaut : {DEFAULT_THRESHOLD}).")
    parser.add_argument("--sweep", action="store_true", help=f"Évalue aussi les seuils {SWEEP_THRESHOLDS}.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help=f"Seuil « score bas » du drapeau ner_suspect (défaut : {DEFAULT_MIN_SCORE}).")
    parser.add_argument("--predictions", action="append", default=[], metavar="NOM=FICHIER", help="Prédictions précalculées (JSON Label Studio ou CSV tagged_text), répétable.")
    parser.add_argument("--reference", default=None, help="Système de référence des Δ (défaut : le premier système).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Rééchantillonnages bootstrap (défaut : 1000).")
    parser.add_argument("--seed", type=int, default=0, help="Graine du bootstrap (défaut : 0).")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Dossier du rapport (défaut : {DEFAULT_OUTPUT_DIR}).")


def run(args: argparse.Namespace) -> None:
    if not args.gold.exists():
        raise CommandError(f"gold '{args.gold}' introuvable (voir numrev gold ner).")
    entries, unreviewed = load_gold(args.gold, args.split)
    if not entries:
        console.print(f"[yellow]Aucune entrée relue dans le split '{args.split}' ({unreviewed} non relues).[/yellow]")
        return
    console.print(f"{len(entries)} entrées gold relues (split {args.split}), {unreviewed} non relues ignorées.")

    systems: list[System] = []
    for model_dir in args.model:
        systems += model_systems(model_dir, entries, args.threshold, args.sweep)
    for item in args.predictions:
        name, _, path = item.partition("=")
        if not path:
            raise CommandError(f"--predictions attend NOM=FICHIER, reçu '{item}'.")
        systems.append(load_prediction_file(name, Path(path), entries))
    if not systems:
        raise CommandError("aucun système à évaluer (--model ou --predictions).")

    names = [system.name for system in systems]
    if args.reference and args.reference not in names:
        raise CommandError(f"référence '{args.reference}' inconnue ({', '.join(names)}).")
    reference = names.index(args.reference or names[0])

    pages = {key: index for index, key in enumerate(dict.fromkeys(f"{e.data['document']}#{e.data['page']}" for e in entries))}
    clusters = np.array([pages[f"{e.data['document']}#{e.data['page']}"] for e in entries])
    scored = [score_system(system, entries, clusters) for system in systems]
    bootstrap = page_bootstrap(len(pages), args.bootstrap, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "rapport.md"
    write_report(report_path, args, entries, unreviewed, systems, scored, bootstrap, reference)
    write_errors(args.output / "erreurs.csv", entries, systems, args.min_score)

    for system, score in zip(systems, scored):
        (value, ci), *_ = summarize(score, bootstrap)
        console.print(f"  {system.name:40s} exactitude {fmt_ci(value, ci)}")
    console.print(f"[bold green]📄 Rapport :[/bold green] [yellow]{report_path}[/yellow]")

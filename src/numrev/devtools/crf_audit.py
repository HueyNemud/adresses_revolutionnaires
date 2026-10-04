"""Audit des performances du CRF de classification de lignes et de la
pertinence de ses features, sur les silver datasets
(`*.lines.csv`, colonne `classe`).

Produit, dans un dossier de sortie :
  - `rapport.md` : rapport d'analyse commenté, avec recommandations ;
  - des tables CSV détaillées (expériences, classes, features, erreurs...) ;
  - `resume.json` : les chiffres clés, lisibles par un programme.

Démarche (détaillée dans la section « Méthodologie » du rapport) :
  1. performances du CRF de production et de baselines selon deux protocoles
     de validation croisée : intra-document et inter-volumes ;
  2. qualité des probabilités (calibration, détection d'erreurs), qui
     pilotent l'apprentissage actif et la relecture ;
  3. audit des features : statistiques sans modèle (information mutuelle,
     constance, redondance), ablations groupe par groupe, groupe seul,
     poids appris — les verdicts sont calibrés par des features placebo ;
  4. évaluation de groupes de features candidats, puis d'une sélection ;
  5. courbe d'apprentissage (régime de peu d'annotations, celui de
     l'apprentissage actif) et sensibilité aux hyperparamètres ;
  6. analyse d'erreurs pour guider la conception de nouvelles features.
"""

import argparse
import json
import math
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from numrev.command import CommandError, console
from numrev.crf.evaluation import (
    LABEL_INDEX,
    N_CLASSES,
    Experiment,
    Predictions,
    ScoredLines,
    Split,
    accuracy,
    cross_volume_splits,
    entity_f1,
    entropy_bits,
    factorize,
    heuristic_rule,
    macro_f1,
    majority_rule,
    miller_madow_bias_bits,
    mutual_information_bits,
    page_boundaries,
    page_confusions,
    page_entity_counts,
    prf,
    rule_predictions,
    run_experiments,
    score_lines,
    subsample_training_pages,
    within_document_splits,
)
from numrev.crf.features import (
    CANDIDATE_GROUPS,
    FEATURE_GROUPS,
    LEGACY_GROUPS,
    PLACEBO_GROUPS,
    PRODUCTION_GROUPS,
    extract_features_from_context,
    extract_grouped_features,
)
from numrev.crf.labels import CLASSES
from numrev.crf.model import DEFAULT_CONFIG, TrainingConfig, iter_labeled_segments, open_tagger, train_model
from numrev.crf.silver import SilverDocument, discover_silver_csvs, load_silver_corpus
from numrev.paths import ANNUAIRES_DIR, REPORTS_DIR
from numrev.reporting import fmt, fmt_ci, fmt_delta, md_code, md_table, write_table
from numrev.stats import bootstrap_weights, calibration_table, expected_calibration_error, interval, resample, review_capture, roc_auc

DESCRIPTION = "Audit du CRF de lignes et de ses features sur les silver datasets."

INTRA = "intra-document"
INTER = "inter-volumes"
PROTOCOL_LABELS = {
    INTRA: "Intra-document (plis de pages contiguës, un modèle par document)",
    INTER: "Inter-volumes (modèle entraîné sur les autres volumes)",
}
SHORT = {INTRA: "intra", INTER: "inter"}
ABSENT = "<absent>"
FREQUENT_CLASS_MIN_SUPPORT = 100
VERDICT_METRIC = "macro_f1_freq"
METRIC_LABELS = {
    "macro_f1": "macro-F1 (toutes classes)",
    "macro_f1_freq": "macro-F1 (classes fréquentes)",
    "accuracy": "exactitude",
    "entry_f1": "F1 ENTRY (entités)",
    "title_f1": "F1 TITLE (entités)",
}
TITLE_CANDIDATES = ("small_word_start", "block_continuation", "uppercase")
TITLE_METRICS = ("f1_B-TITLE", "f1_I-TITLE", "title_f1")


def names(items: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in items) if items else "aucun"


# ----------------------------------------------------------------------
# Évaluation d'une expérience (points + bootstrap)
# ----------------------------------------------------------------------
@dataclass
class Evaluated:
    name: str
    protocol: str
    scored: ScoredLines
    cm_pages: np.ndarray
    entity_pages: np.ndarray

    @property
    def confusion(self) -> np.ndarray:
        return self.cm_pages.sum(axis=0)

    @property
    def entity_counts(self) -> np.ndarray:
        return self.entity_pages.sum(axis=0)


def evaluate(predictions: Predictions, documents: Sequence[SilverDocument]) -> Evaluated:
    scored = score_lines(predictions, documents)
    return Evaluated(
        predictions.experiment.name,
        predictions.protocol,
        scored,
        page_confusions(scored),
        page_entity_counts(scored, documents),
    )


Delta = tuple[float, tuple[float, float]]


class Statistics:
    """Métriques ponctuelles et intervalles bootstrap (par page) d'un protocole."""

    def __init__(self, reference: Evaluated, n_bootstrap: int, seed: int) -> None:
        self.page_names = reference.scored.page_names
        support = reference.confusion.sum(axis=1)
        self.classes_mask = support > 0
        self.frequent_mask = support >= FREQUENT_CLASS_MIN_SUPPORT
        self.weights = bootstrap_weights(len(self.page_names), n_bootstrap, seed)
        self._samples: dict[int, dict[str, np.ndarray]] = {}

    @property
    def frequent_classes(self) -> list[str]:
        return [CLASSES[k] for k in range(N_CLASSES) if self.frequent_mask[k]]

    def metrics(self, cm: np.ndarray, ent: np.ndarray) -> dict[str, np.ndarray]:
        entity = entity_f1(ent)[2]
        class_f1 = prf(cm)[2]
        return {
            "macro_f1": macro_f1(cm, self.classes_mask),
            "macro_f1_freq": macro_f1(cm, self.frequent_mask),
            "accuracy": accuracy(cm),
            "entry_f1": entity[..., 0],
            "title_f1": entity[..., 1],
            **{f"f1_{label}": class_f1[..., k] for k, label in enumerate(CLASSES) if self.classes_mask[k]},
        }

    def point(self, evaluated: Evaluated) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics(evaluated.confusion, evaluated.entity_counts).items()}

    def samples(self, evaluated: Evaluated) -> dict[str, np.ndarray]:
        if evaluated.scored.page_names != self.page_names:
            raise ValueError(f"{evaluated.name} : lignes évaluées différentes de la référence.")
        key = id(evaluated)
        if key not in self._samples:
            self._samples[key] = self.metrics(resample(evaluated.cm_pages, self.weights), resample(evaluated.entity_pages, self.weights))
        return self._samples[key]

    def with_ci(self, evaluated: Evaluated) -> dict[str, Delta]:
        point, samples = self.point(evaluated), self.samples(evaluated)
        return {key: (point[key], interval(samples[key])) for key in point}

    def delta(self, a: Evaluated, b: Evaluated) -> dict[str, Delta]:
        """Différence appariée a − b (mêmes rééchantillonnages de pages)."""
        pa, pb = self.point(a), self.point(b)
        sa, sb = self.samples(a), self.samples(b)
        return {key: (pa[key] - pb[key], interval(sa[key] - sb[key])) for key in pa}

    def class_f1_ci(self, evaluated: Evaluated) -> list[tuple[float, float]]:
        f1 = prf(resample(evaluated.cm_pages, self.weights))[2]
        return [interval(f1[:, k]) for k in range(N_CLASSES)]


def verdict(delta: Delta, floor: float) -> str:
    """« améliore » / « dégrade » si l'IC exclut 0 ET si l'effet dépasse le
    plancher de bruit mesuré par les placebos ; « neutre » sinon."""
    point, (low, high) = delta
    if high < 0 and -point > floor:
        return "dégrade"
    if low > 0 and point > floor:
        return "améliore"
    return "neutre"


# ----------------------------------------------------------------------
# Expériences
# ----------------------------------------------------------------------
def without(groups: Sequence[str], *removed: str) -> tuple[str, ...]:
    return tuple(g for g in groups if g not in removed)


def config_name(config: TrainingConfig) -> str:
    return f"c1={config.c1:g}, c2={config.c2:g}, iter={config.max_iterations}"


def build_experiments(args: argparse.Namespace) -> dict[str, list[Experiment]]:
    ablatable = [g for g in PRODUCTION_GROUPS if g != "bias"]
    families: dict[str, list[Experiment]] = {
        "reference": [
            Experiment("production", PRODUCTION_GROUPS, description="Features de production"),
            Experiment("production_v1", LEGACY_GROUPS, description="Features de production historiques (v1)"),
        ],
        "placebo": [Experiment(f"+{g}", PRODUCTION_GROUPS + (g,), description="Témoin") for g in PLACEBO_GROUPS],
        "ablation": [Experiment(f"−{g}", without(PRODUCTION_GROUPS, g), description=f"Production sans {g}") for g in ablatable],
    }
    if args.single_groups:
        families["single"] = [Experiment(f"seul:{g}", ("bias", g), description=f"bias + {g}") for g in ablatable]
    if args.candidates:
        families["candidates"] = [
            Experiment(f"+{g}", PRODUCTION_GROUPS + (g,), description=f"Production + {g}") for g in CANDIDATE_GROUPS
        ] + [Experiment("+tous_candidats", PRODUCTION_GROUPS + CANDIDATE_GROUPS, description="Production + tous les candidats")]
    configs = [TrainingConfig(max_iterations=200), TrainingConfig(max_iterations=500)]
    if args.hyperparams:
        configs += [
            TrainingConfig(c1=c1, c2=c2, max_iterations=200)
            for c1 in (0.0, 0.05, 0.1, 0.5, 1.0)
            for c2 in (0.001, 0.01, 0.1, 1.0)
            if (c1, c2) != (DEFAULT_CONFIG.c1, DEFAULT_CONFIG.c2)
        ]
    families["hyperparams"] = [Experiment(config_name(c), PRODUCTION_GROUPS, c, description="Autres hyperparamètres") for c in configs]
    return families


def run_with_progress(
    documents: Sequence[SilverDocument],
    experiments: Sequence[Experiment],
    splits: Sequence[Split],
    workers: int,
    title: str,
    cache_dir: Path | None = None,
) -> dict[tuple[str, str], Predictions]:
    with Progress(
        TextColumn(f"[bold]{title}[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(title, total=len(experiments) * len(splits))
        return run_experiments(
            documents,
            experiments,
            splits,
            workers=workers,
            progress=lambda done, total: progress.update(task, completed=done),
            cache_dir=cache_dir,
        )


# ----------------------------------------------------------------------
# Analyses sans modèle
# ----------------------------------------------------------------------
@dataclass
class FeatureTable:
    """Valeurs de chaque attribut (clé) sur les lignes labellisées du corpus."""

    keys: list[str]
    key_group: dict[str, str]
    values: dict[str, list[str]]
    gold: np.ndarray


def build_feature_table(documents: Sequence[SilverDocument]) -> FeatureTable:
    groups = PRODUCTION_GROUPS + CANDIDATE_GROUPS
    key_group: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    gold: list[int] = []
    for document in documents:
        grouped = extract_grouped_features(document.context, groups)
        for line_groups, label in zip(grouped, document.gold):
            if label is None:
                continue
            row: dict[str, str] = {}
            for group, feat in line_groups.items():
                for key, value in feat.items():
                    key_group.setdefault(key, group)
                    row[key] = value
            rows.append(row)
            gold.append(LABEL_INDEX[label])
    keys = sorted(key_group, key=lambda k: (groups.index(key_group[k]), k))
    values = {key: [row.get(key, ABSENT) for row in rows] for key in keys}
    return FeatureTable(keys, key_group, values, np.array(gold))


def feature_statistics(table: FeatureTable) -> list[dict[str, object]]:
    y = table.gold
    h_y = entropy_bits(np.bincount(y))
    stats = []
    for key in table.keys:
        values = table.values[key]
        codes = factorize(values)
        counts = Counter(values)
        top_value, top_count = counts.most_common(1)[0]
        mi = mutual_information_bits(codes, y)
        corrected = max(0.0, mi - miller_madow_bias_bits(codes, y))
        stats.append(
            {
                "key": key,
                "group": table.key_group[key],
                "production": table.key_group[key] in PRODUCTION_GROUPS,
                "coverage": 1 - counts.get(ABSENT, 0) / len(values),
                "n_values": len([v for v in counts if v != ABSENT]),
                "top_value": top_value,
                "top_share": top_count / len(values),
                "mi_bits": mi,
                "mi_corrected": corrected,
                "nmi": corrected / h_y if h_y else 0.0,
            }
        )
    return stats


def constant_groups(stats: list[dict[str, object]]) -> list[str]:
    """Groupes de production dont tous les attributs sont constants (hors bias)."""
    by_group: dict[str, list[bool]] = defaultdict(list)
    for s in stats:
        if s["production"] and s["group"] != "bias":
            by_group[str(s["group"])].append(int(s["n_values"]) <= 1)
    return [g for g, flags in by_group.items() if all(flags)]


def redundancy_pairs(table: FeatureTable, threshold: float) -> list[tuple[str, str, float]]:
    codes = {key: factorize(table.values[key]) for key in table.keys}
    entropies = {key: entropy_bits(np.bincount(c)) for key, c in codes.items()}
    keys = [k for k in table.keys if entropies[k] > 1e-3]
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            su = 2 * mutual_information_bits(codes[a], codes[b]) / (entropies[a] + entropies[b])
            if su >= threshold:
                pairs.append((a, b, su))
    return sorted(pairs, key=lambda p: -p[2])


def discriminative_values(table: FeatureTable, min_support: int, top: int) -> dict[str, list[tuple]]:
    """Pour chaque classe, les règles « attribut=valeur ⇒ classe » de meilleure
    F1 (moyenne harmonique de P(classe | valeur) et P(valeur | classe))."""
    class_totals = np.bincount(table.gold, minlength=N_CLASSES)
    candidates = []
    for key in table.keys:
        by_value: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_CLASSES))
        for value, label in zip(table.values[key], table.gold):
            if value != ABSENT:
                by_value[value][label] += 1
        candidates.extend((key, value, counts, counts.sum()) for value, counts in by_value.items() if counts.sum() >= min_support)
    result: dict[str, list[tuple]] = {}
    for k, label in enumerate(CLASSES):
        if class_totals[k] == 0:
            continue
        rows = []
        for key, value, counts, total in candidates:
            precision, recall = counts[k] / total, counts[k] / class_totals[k]
            if precision > 0:
                rows.append((key, value, precision, recall, 2 * precision * recall / (precision + recall), int(total)))
        rows.sort(key=lambda r: -r[4])
        result[label] = rows[:top]
    return result


def model_weights(documents: Sequence[SilverDocument], key_group: dict[str, str]):
    """Entraîne le modèle de production sur tout le corpus et renvoie ses poids."""
    sequences = []
    for document in documents:
        features = extract_features_from_context(document.context, PRODUCTION_GROUPS)
        sequences.extend(iter_labeled_segments(features, document.gold))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.crfsuite"
        train_model(sequences, path)
        tagger = open_tagger(path)
        info = tagger.info()
        state = dict(info.state_features)
        transitions = dict(info.transitions)
        tagger.close()
    group_mass: Counter[str] = Counter()
    for (attribute, _), weight in state.items():
        group_mass[key_group.get(attribute.split(":", 1)[0], "?")] += abs(weight)
    return state, transitions, group_mass


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, *texts: str) -> None:
        self.parts.extend(text for text in texts if text)

    def replace(self, placeholder: str, text: str) -> None:
        self.parts[self.parts.index(placeholder)] = text

    def render(self) -> str:
        return "\n\n".join(self.parts).rstrip() + "\n"


@dataclass
class Context:
    """État partagé entre les sections du rapport."""

    documents: Sequence[SilverDocument]
    protocols: list[str]
    evaluated: dict[tuple[str, str], Evaluated]
    stats: dict[str, Statistics]
    floors: dict[str, dict[str, float]]
    out: Path

    def ev(self, name: str, protocol: str) -> Evaluated:
        return self.evaluated[(name, protocol)]

    def delta(self, name: str, protocol: str, reference: str = "production") -> dict[str, Delta]:
        return self.stats[protocol].delta(self.ev(name, protocol), self.ev(reference, protocol))

    def verdict(self, name: str, protocol: str, reference: str = "production") -> str:
        return verdict(self.delta(name, protocol, reference)[VERDICT_METRIC], self.floors[protocol][VERDICT_METRIC])


def corpus_section(report: Report, documents: Sequence[SilverDocument], out: Path) -> dict:
    total: Counter[str] = Counter()
    rows, csv_rows = [], []
    for d in documents:
        counts = Counter(label for label in d.gold if label is not None)
        total.update(counts)
        s = d.stats
        n_pages = len(page_boundaries(d)) - 1
        rows.append(
            [
                d.name,
                n_pages,
                d.labeled_count,
                s.original_human_lines,
                s.label_changed_vs_original,
                s.text_edited,
                s.heading_marker_changed,
                s.unmatched_source_lines,
            ]
        )
        csv_rows.append(
            [
                d.name,
                d.volume,
                n_pages,
                len(d),
                d.labeled_count,
                *[counts.get(c, 0) for c in CLASSES],
                s.original_human_lines,
                s.original_model_lines,
                s.label_changed_vs_original,
                s.text_edited,
                s.heading_marker_changed,
                s.unmatched_source_lines,
                s.added_lines,
            ]
        )
    write_table(
        out / "corpus.csv",
        [
            "document",
            "volume",
            "pages",
            "lignes",
            "lignes_labellisees",
            *CLASSES,
            "labels_humains_origine",
            "labels_modele_origine",
            "classes_corrigees",
            "textes_corriges",
            "marqueurs_titre_modifies",
            "lignes_non_alignees",
            "lignes_ajoutees",
        ],
        csv_rows,
    )
    labeled = sum(total.values())
    changed = sum(d.stats.label_changed_vs_original for d in documents)
    model_lines = sum(d.stats.original_model_lines for d in documents)
    human = sum(d.stats.original_human_lines for d in documents)
    report.add(
        "## 1. Données",
        f"{len(documents)} documents ({len({d.volume for d in documents})} volumes), {labeled} lignes non vides labellisées. "
        "Les observations (texte, bloc OCR, page) sont relues dans le JSON `*.lines.json` vu par "
        "l'annotateur ; la vérité vient de la colonne `classe` du CSV corrigé, rattachée par la clé de ligne `cle`.",
        md_table(
            [
                "Document",
                "Pages",
                "Lignes lab.",
                "Labels humains (orig.)",
                "Classes corrigées",
                "Textes corrigés",
                "dont marqueur « # »",
                "Non alignées",
            ],
            rows,
        ),
        md_table(["Classe", "Lignes", "Part"], [[c, total[c], f"{total[c] / labeled:.2%}"] for c in CLASSES if total[c]]),
        "**Nature du silver dataset — à garder en tête pour lire tout le rapport.** "
        f"Sur {model_lines} lignes dont la classe d'origine venait du modèle (et {human} d'un humain), "
        f"seules **{changed} ({changed / max(labeled, 1):.2%})** ont vu leur classe corrigée à la curation. "
        "La référence est donc fortement ancrée sur les prédictions du CRF d'origine : une erreur que la "
        "curation n'a pas repérée est comptée comme juste. Les scores mesurent l'accord avec une référence "
        "« CRF + corrections » ; ils sont optimistes en absolu, mais les **comparaisons entre configurations** "
        "restent informatives, car toutes sont mesurées contre la même référence.",
        "La curation a aussi corrigé le **texte** de certaines lignes, surtout les marqueurs de titre « # » "
        "(ajoutés, retirés ou changés de niveau) : signal direct que la feature `heading`, fondée sur les « # » "
        "produits par l'OCR, est imparfaite. Le modèle évalué ne voit pas ces corrections : il voit le texte "
        "OCR brut, comme en production.",
    )
    return {
        "lignes_labellisees": labeled,
        "classes_corrigees": changed,
        "repartition": {c: total[c] / labeled for c in CLASSES if total[c]},
    }


def performance_section(report: Report, ctx: Context) -> dict:
    summary: dict[str, dict] = {}
    report.add(
        "## 2. Performances du CRF de production",
        "Intervalles de confiance à 95 % par bootstrap sur les pages. `macro-F1` : moyenne non pondérée des "
        "F1 des classes présentes, très sensible aux classes rares (I-TITLE, SUB-ENTRY : quelques dizaines de "
        f"lignes) ; `macro-F1 fréq.` : même moyenne restreinte aux classes d'au moins {FREQUENT_CLASS_MIN_SUPPORT} lignes "
        f"({', '.join(ctx.stats[ctx.protocols[0]].frequent_classes)}), plus stable. `F1 ENTRY` / `F1 TITLE` : F1 au "
        "niveau des entités reconstituées selon les règles de numrev assemble (une entité est juste si "
        "elle regroupe exactement les mêmes lignes) — la métrique la plus proche du livrable.",
    )
    csv_rows = []
    for protocol in ctx.protocols:
        st = ctx.stats[protocol]
        rows = []
        for name in ("majoritaire", "heuristique", "production_v1", "production"):
            m = st.with_ci(ctx.ev(name, protocol))
            rows.append(
                [
                    name,
                    fmt(m["accuracy"][0]),
                    fmt_ci(*m["macro_f1"]),
                    fmt_ci(*m["macro_f1_freq"]),
                    fmt_ci(*m["entry_f1"]),
                    fmt_ci(*m["title_f1"]),
                ]
            )
        report.add(
            f"### 2.{ctx.protocols.index(protocol) + 1} {PROTOCOL_LABELS[protocol]}",
            md_table(["Système", "Exactitude", "macro-F1", "macro-F1 fréq.", "F1 ENTRY (entités)", "F1 TITLE (entités)"], rows),
        )

        production = ctx.ev("production", protocol)
        precision, recall, f1, support = prf(production.confusion)
        f1_ci = st.class_f1_ci(production)
        class_rows = []
        for k, label in enumerate(CLASSES):
            if support[k]:
                class_rows.append([label, int(support[k]), fmt(precision[k]), fmt(recall[k]), fmt_ci(f1[k], f1_ci[k])])
                csv_rows.append([protocol, label, int(support[k]), precision[k], recall[k], f1[k], *f1_ci[k]])
        present = [k for k in range(N_CLASSES) if production.confusion[k].sum() or production.confusion[:, k].sum()]
        report.add(
            "Par classe (production) :",
            md_table(["Classe", "Support", "Précision", "Rappel", "F1 [IC 95 %]"], class_rows),
            "Matrice de confusion (lignes = vérité, colonnes = prédiction) :",
            md_table(
                ["vérité \\ prédit", *[CLASSES[p] for p in present]],
                [[CLASSES[g], *[int(production.confusion[g, p]) for p in present]] for g in present],
            ),
        )
        detail_rows = []
        scored = production.scored
        for d, document in enumerate(ctx.documents):
            mask = scored.doc_index == d
            if not mask.any():
                continue
            cm = np.zeros((N_CLASSES, N_CLASSES))
            np.add.at(cm, (scored.gold[mask], scored.pred[mask]), 1)
            pages = np.unique(scored.page_code[mask])
            ent = production.entity_pages[pages].sum(axis=0)
            detail_rows.append(
                [
                    document.name,
                    int(mask.sum()),
                    int((scored.gold[mask] != scored.pred[mask]).sum()),
                    fmt(float(accuracy(cm))),
                    fmt(float(macro_f1(cm, cm.sum(1) > 0))),
                    fmt(float(entity_f1(ent)[2][0])),
                ]
            )
        report.add(
            "Par document (macro-F1 sur les classes présentes dans le document) :",
            md_table(["Document", "Lignes", "Erreurs", "Exactitude", "macro-F1", "F1 ENTRY"], detail_rows),
        )
        summary[protocol] = {key: {"valeur": value, "ic95": list(ci)} for key, (value, ci) in st.with_ci(production).items()}
        summary[protocol]["f1_par_classe"] = {CLASSES[k]: float(f1[k]) for k in range(N_CLASSES) if support[k]}
    write_table(
        ctx.out / "classes.csv", ["protocole", "classe", "support", "precision", "rappel", "f1", "f1_ic_bas", "f1_ic_haut"], csv_rows
    )
    return summary


def calibration_section(report: Report, ctx: Context) -> dict:
    report.add(
        "## 3. Qualité des probabilités",
        "Les marginales du CRF servent à choisir les lignes à annoter (marge top-1 − top-2) et sont exportées "
        "comme `probability`. On mesure l'erreur de calibration attendue (ECE, 10 tranches de confiance), "
        "l'AUROC de la détection d'erreurs (probabilité qu'une ligne fausse soit jugée plus incertaine qu'une "
        "ligne juste ; 0,5 = hasard) et la part des erreurs retrouvées en relisant les x % de lignes les plus "
        "incertaines — c'est l'efficacité d'une relecture guidée par le modèle.",
    )
    summary, csv_rows = {}, []
    for protocol in ctx.protocols:
        scored = ctx.ev("production", protocol).scored
        correct = scored.correct
        errors = ~correct
        ece = expected_calibration_error(scored.confidence, correct)
        auc_conf = roc_auc(1 - scored.confidence, errors)
        auc_margin = roc_auc(-scored.margin, errors)
        capture = review_capture(-scored.margin, errors, (0.005, 0.01, 0.02, 0.05, 0.10))
        summary[protocol] = {
            "ece": ece,
            "auroc_confiance": auc_conf,
            "auroc_marge": auc_margin,
            "capture": {f"{b:.1%}": c for b, _, c in capture},
        }
        table = calibration_table(scored.confidence, correct)
        csv_rows += [[protocol, *row] for row in table]
        report.add(
            f"### 3.{ctx.protocols.index(protocol) + 1} {PROTOCOL_LABELS[protocol]}",
            f"ECE = **{fmt(ece, 4)}** · AUROC détection d'erreurs : confiance **{fmt(auc_conf)}**, marge **{fmt(auc_margin)}** · "
            f"{int(errors.sum())} erreurs sur {len(scored)} lignes.",
            md_table(
                ["Confiance", "Lignes", "Confiance moy.", "Exactitude", "Écart"],
                [[f"{low:.1f}–{high:.1f}", n, fmt(conf), fmt(acc), f"{acc - conf:+.3f}"] for low, high, n, conf, acc in table],
            ),
            md_table(
                ["Lignes relues (les plus incertaines)", "Nombre", "Part des erreurs trouvées"],
                [[f"{b:.1%}", n, f"{c:.1%}" if not math.isnan(c) else "–"] for b, n, c in capture],
            ),
        )
    write_table(
        ctx.out / "calibration.csv", ["protocole", "borne_basse", "borne_haute", "lignes", "confiance_moyenne", "exactitude"], csv_rows
    )
    return summary


def inventory_section(report: Report, table: FeatureTable, stats: list[dict], redundancy: list, out: Path) -> dict:
    write_table(
        out / "features_statistiques.csv",
        [
            "cle",
            "groupe",
            "production",
            "couverture",
            "n_valeurs",
            "valeur_dominante",
            "part_dominante",
            "mi_bits",
            "mi_corrigee_bits",
            "nmi",
        ],
        [
            [
                s["key"],
                s["group"],
                s["production"],
                s["coverage"],
                s["n_values"],
                s["top_value"],
                s["top_share"],
                s["mi_bits"],
                s["mi_corrected"],
                s["nmi"],
            ]
            for s in stats
        ],
    )
    dead = [s["key"] for s in stats if s["production"] and s["group"] != "bias" and s["n_values"] <= 1]
    rows = []
    for s in stats:
        flag = ""
        if s["group"] != "bias":
            flag = "**constante**" if s["n_values"] <= 1 else "quasi constante" if s["top_share"] >= 0.999 else ""
        rows.append(
            [
                ("" if s["production"] else "★ ") + f"`{s['key']}`",
                s["group"],
                f"{s['coverage']:.1%}",
                s["n_values"],
                f"{md_code(str(s['top_value']), 20)} ({s['top_share']:.1%})",
                fmt(s["mi_corrected"], 4),
                fmt(s["nmi"], 3),
                flag,
            ]
        )
    report.add(
        "### 4.1 Inventaire des attributs (sans modèle)",
        "Pour chaque attribut : couverture (part des lignes où il est défini), nombre de valeurs, valeur "
        "dominante, information mutuelle (IM) avec la classe corrigée du biais de l'estimateur "
        "(Miller–Madow), en bits, et normalisée par l'entropie des classes (NMI). L'IM mesure l'information "
        "*individuelle* d'un attribut ; elle ignore interactions et redondance, que mesurent les ablations. "
        "Pour un attribut à très nombreuses valeurs (`first_word`...) la correction peut ramener l'IM à 0 : "
        "l'estimateur ne peut alors pas conclure. ★ = attribut candidat.",
        md_table(["Attribut", "Groupe", "Couverture", "Valeurs", "Valeur dominante", "IM (bits)", "NMI", "Alerte"], rows),
    )
    notes = []
    if "previous_source_gap" in dead:
        notes.append(
            "- `previous_source_gap` est **constant par construction** : `source_row` n'est incrémenté que pour les "
            "lignes non vides (`load_json_lines`), donc l'écart entre deux lignes consécutives vaut toujours 1. "
            "L'intention (repérer une ligne vide intercalée) est reprise par le candidat `blank_before`."
        )
    if "is_page_marker" in dead:
        notes.append(
            "- `is_page_marker` ne se déclenche jamais : le motif « {n}--- » vient de la pagination de l'ancien "
            "service OCR ; la sortie Chandra locale ne le produit pas. Même remarque pour la règle correspondante "
            "de `get_heuristic_label`."
        )
    if dead:
        report.add(f"**Attributs de production constants** (aucune information) : {names(dead)}.", "\n".join(notes))
    report.add(
        "`BOS` / `EOS` ne valent `True` que sur la première / dernière ligne de la séquence ; en production la "
        "séquence est le document entier, ils ne portent donc sur qu'une ligne par document.",
        "### 4.2 Redondance entre attributs",
        "Incertitude symétrique U(X,Y) = 2·I(X;Y)/(H(X)+H(Y)) ∈ [0, 1] (1 = l'un détermine l'autre). Des "
        "attributs très redondants se partagent le poids dans le CRF et rendent l'ablation de chacun peu parlante.",
        (
            md_table(
                ["Attribut A", "Attribut B", "U"],
                [[f"`{a}` ({table.key_group[a]})", f"`{b}` ({table.key_group[b]})", fmt(u)] for a, b, u in redundancy[:30]],
                "llr",
            )
            if redundancy
            else "Aucune paire au-dessus du seuil."
        ),
    )
    return {"attributs_constants": dead}


def evolution_section(report: Report, ctx: Context) -> dict:
    """Production actuelle (v2) contre features historiques (v1)."""
    report.add(
        "## 2bis. Évolution des features : v2 contre v1",
        "v1 = features de production historiques (texte brut, `previous_source_gap`, `is_page_marker`, "
        "`prev_ends_dash`) ; v2 = production actuelle : features calculées sur le texte sans typographie Markdown, "
        "proportion d'italique (`italic`), fin de la ligne précédente et transition fin → début (`prev_line`), "
        "features constantes supprimées. Δ = v2 − v1, bootstrap apparié par page.",
    )
    summary = {}
    for protocol in ctx.protocols:
        st = ctx.stats[protocol]
        v1, v2 = ctx.ev("production_v1", protocol), ctx.ev("production", protocol)
        delta = st.delta(v2, v1)
        p1, p2 = st.point(v1), st.point(v2)
        rows = [
            [METRIC_LABELS[m], fmt(p1[m]), fmt(p2[m]), fmt_delta(*delta[m]), verdict(delta[m], ctx.floors[protocol][m])]
            for m in ("macro_f1_freq", "macro_f1", "entry_f1", "title_f1")
        ]
        rows += [
            [
                f"F1 {CLASSES[k]}",
                fmt(p1[f"f1_{CLASSES[k]}"]),
                fmt(p2[f"f1_{CLASSES[k]}"]),
                fmt_delta(*delta[f"f1_{CLASSES[k]}"]),
                verdict(delta[f"f1_{CLASSES[k]}"], ctx.floors[protocol][f"f1_{CLASSES[k]}"]),
            ]
            for k in range(N_CLASSES)
            if st.classes_mask[k]
        ]
        errors_v1 = int((~v1.scored.correct).sum())
        errors_v2 = int((~v2.scored.correct).sum())
        report.add(
            f"**{PROTOCOL_LABELS[protocol]}** — erreurs : {errors_v1} (v1) → {errors_v2} (v2).",
            md_table(["Métrique", "v1", "v2", "Δ v2 − v1 [IC 95 %]", "Verdict (vs plancher de bruit)"], rows),
        )
        summary[protocol] = {
            "erreurs_v1": errors_v1,
            "erreurs_v2": errors_v2,
            **{m: {"v1": p1[m], "v2": p2[m], "delta": delta[m][0], "ic95": list(delta[m][1])} for m in p1},
        }
    return summary


def title_candidates_section(report: Report, ctx: Context, experiments: Sequence[Experiment]) -> dict:
    """Candidats ciblant les titres, jugés sur les F1 des classes de titre."""
    report.add(
        "### 5.2 Candidats ciblant les titres (I-TITLE)",
        "La macro-F1 des classes fréquentes, qui sert aux verdicts, exclut I-TITLE (quelques dizaines de lignes) : "
        "un candidat ciblant les titres y paraît toujours neutre. Ils sont donc jugés ici sur le F1 de B-TITLE, "
        "d'I-TITLE et des entités TITLE, chacun contre son propre plancher de bruit (placebos, section 4.3). "
        "Avec si peu de lignes I-TITLE, les intervalles sont larges : un verdict « neutre » signifie surtout "
        "« pas démontrable sur ce corpus ».",
    )
    verdicts: dict = {}
    for protocol in ctx.protocols:
        rows = []
        for experiment in experiments:
            delta = ctx.delta(experiment.name, protocol)
            row = [f"`{experiment.name}`"]
            for metric in TITLE_METRICS:
                v = verdict(delta[metric], ctx.floors[protocol][metric])
                verdicts.setdefault(experiment.name, {}).setdefault(protocol, {})[metric] = v
                row.append(fmt_delta(*delta[metric]) + (f" **{v}**" if v != "neutre" else ""))
            row.append(fmt_delta(*delta[VERDICT_METRIC]))
            rows.append(row)
        floors = ", ".join(f"{m} {ctx.floors[protocol][m]:.3f}" for m in TITLE_METRICS)
        report.add(
            f"**{PROTOCOL_LABELS[protocol]}** (planchers de bruit : {floors})",
            md_table(["Candidat", "Δ F1 B-TITLE", "Δ F1 I-TITLE", "Δ F1 TITLE (entités)", "Δ macro-F1 fréq."], rows),
        )
    return verdicts


def placebo_section(report: Report, ctx: Context, placebos: Sequence[Experiment]) -> None:
    rows = []
    for protocol in ctx.protocols:
        for experiment in placebos:
            delta = ctx.delta(experiment.name, protocol)
            f1_exp = prf(ctx.ev(experiment.name, protocol).confusion)[2]
            f1_ref = prf(ctx.ev("production", protocol).confusion)[2]
            diffs = [(f1_exp[k] - f1_ref[k], CLASSES[k]) for k in range(N_CLASSES) if ctx.stats[protocol].classes_mask[k]]
            biggest = max(diffs, key=lambda d: abs(d[0]))
            rows.append(
                [
                    SHORT[protocol],
                    f"`{experiment.name}`",
                    fmt_delta(*delta["macro_f1_freq"]),
                    fmt_delta(*delta["macro_f1"]),
                    fmt_delta(*delta["entry_f1"]),
                    f"{biggest[1]} {biggest[0]:+.3f}",
                ]
            )
    floors = [[SHORT[p], *[fmt(ctx.floors[p][m]) for m in ("macro_f1_freq", "macro_f1", "entry_f1")]] for p in ctx.protocols]
    report.add(
        "### 4.3 Témoins (placebos) et plancher de bruit",
        "Ajouter un attribut sans information (constant, ou bit pseudo-aléatoire) ne devrait rien changer. En "
        "pratique, toute modification de l'espace de features déplace l'optimum régularisé du CRF et peut faire "
        "basculer quelques lignes — ce qui pèse lourd sur les classes rares. Les intervalles bootstrap, qui ne "
        "rendent compte que de la variabilité des pages évaluées, n'en tiennent pas compte. Le **plancher de "
        "bruit** est donc mesuré ici : la plus grande |Δ| observée sur ces témoins. Un effet n'est jugé réel "
        f"que si son IC exclut 0 **et** s'il dépasse ce plancher (verdicts calculés sur la {METRIC_LABELS[VERDICT_METRIC]}).",
        md_table(["Protocole", "Témoin", "Δ macro-F1 fréq.", "Δ macro-F1", "Δ F1 ENTRY", "Classe la plus affectée (Δ F1)"], rows),
        md_table(["Protocole", "Plancher macro-F1 fréq.", "Plancher macro-F1", "Plancher F1 ENTRY"], floors),
    )


def comparison_section(
    report: Report, ctx: Context, title: str, intro: str, experiments: Sequence[Experiment], reference: str = "production"
) -> tuple[list, dict]:
    report.add(title, intro)
    csv_rows, verdicts = [], {}
    for protocol in ctx.protocols:
        st = ctx.stats[protocol]
        rows = []
        for experiment in experiments:
            ev = ctx.ev(experiment.name, protocol)
            ref = ctx.ev(reference, protocol)
            delta = st.delta(ev, ref)
            point = st.point(ev)
            f1_exp, f1_ref = prf(ev.confusion)[2], prf(ref.confusion)[2]
            diffs = [(f1_exp[k] - f1_ref[k], CLASSES[k]) for k in range(N_CLASSES) if st.classes_mask[k]]
            biggest = max(diffs, key=lambda d: abs(d[0]))
            v = verdict(delta[VERDICT_METRIC], ctx.floors[protocol][VERDICT_METRIC])
            verdicts.setdefault(experiment.name, {})[protocol] = v
            rows.append(
                [
                    f"`{experiment.name}`",
                    fmt(point["macro_f1_freq"]),
                    fmt_delta(*delta["macro_f1_freq"]),
                    f"{delta['macro_f1'][0]:+.3f}",
                    fmt_delta(*delta["entry_f1"]),
                    f"{delta['title_f1'][0]:+.3f}",
                    f"{biggest[1]} {biggest[0]:+.3f}" if abs(biggest[0]) >= 0.001 else "–",
                    f"**{v}**" if v != "neutre" else v,
                ]
            )
            csv_rows.append(
                [
                    experiment.name,
                    protocol,
                    *[point[m] for m in METRIC_LABELS],
                    *[x for m in METRIC_LABELS for x in (delta[m][0], *delta[m][1])],
                    v,
                    *[f1_exp[k] - f1_ref[k] for k in range(N_CLASSES)],
                ]
            )
        report.add(
            f"**{PROTOCOL_LABELS[protocol]}**",
            md_table(
                [
                    "Configuration",
                    "macro-F1 fréq.",
                    "Δ macro-F1 fréq. [IC 95 %]",
                    "Δ macro-F1",
                    "Δ F1 ENTRY [IC 95 %]",
                    "Δ F1 TITLE",
                    "Classe la plus affectée (Δ F1)",
                    "Verdict",
                ],
                rows,
            ),
        )
    return csv_rows, verdicts


COMPARISON_CSV_HEADERS = [
    "experience",
    "protocole",
    *METRIC_LABELS,
    *[f"{prefix}{m}" for m in METRIC_LABELS for prefix in ("delta_", "delta_ic_bas_", "delta_ic_haut_")],
    "verdict",
    *[f"delta_f1_{c}" for c in CLASSES],
]


def single_group_section(report: Report, ctx: Context, experiments: Sequence[Experiment]) -> None:
    headers, rows = ["Configuration"], []
    for protocol in ctx.protocols:
        headers += [f"macro-F1 fréq. ({SHORT[protocol]})", f"F1 ENTRY ({SHORT[protocol]})"]
    for experiment in experiments:
        row = [f"`{experiment.name}`"]
        for protocol in ctx.protocols:
            m = ctx.stats[protocol].with_ci(ctx.ev(experiment.name, protocol))
            row += [fmt_ci(*m["macro_f1_freq"]), fmt(m["entry_f1"][0])]
        rows.append(row)
    reference = ["production"]
    for protocol in ctx.protocols:
        m = ctx.stats[protocol].point(ctx.ev("production", protocol))
        reference += [fmt(m["macro_f1_freq"]), fmt(m["entry_f1"])]
    rows.append(reference)
    report.add(
        "### 4.5 Chaque groupe seul (bias + groupe)",
        "Information apportée par un groupe à lui seul (transitions CRF comprises), à comparer à la dernière "
        "ligne (production complète). Un groupe fort seul mais neutre en ablation est redondant avec d'autres.",
        md_table(headers, rows),
    )


def discriminative_section(report: Report, table: FeatureTable, discriminative: dict) -> dict:
    rows, best = [], {}
    for label, rules in discriminative.items():
        best[label] = rules[0][4] if rules else 0.0
        for key, value, precision, recall, f1, support in rules:
            star = "" if table.key_group[key] in PRODUCTION_GROUPS else "★ "
            rows.append([label, f"{star}`{key}={value}`", support, f"{precision:.1%}", f"{recall:.1%}", fmt(f1)])
    report.add(
        "### 4.6 Règles « attribut = valeur ⇒ classe » les plus informatives",
        "Pour chaque classe, les valeurs d'attributs (support ≥ 30 lignes) dont la règle « valeur ⇒ classe » a "
        "la meilleure F1 : P(classe | valeur) = précision, part de la classe couverte = rappel. Une classe dont "
        "la meilleure règle reste faible n'a pas de signal simple dans les features : c'est là qu'une feature "
        "dédiée a le plus de valeur (★ = attribut candidat).",
        md_table(["Classe", "Attribut = valeur", "Lignes", "P(classe | valeur)", "Part de la classe", "F1 règle"], rows, "llrrrr"),
    )
    return best


def weights_section(report: Report, state: dict, transitions: dict, group_mass: Counter, out: Path) -> None:
    total = sum(group_mass.values()) or 1
    rows = []
    for label in CLASSES:
        items = sorted(((w, a) for (a, name), w in state.items() if name == label), reverse=True)
        if items:
            rows.append(
                [label, ", ".join(f"`{a}` {w:+.2f}" for w, a in items[:6]), ", ".join(f"`{a}` {w:+.2f}" for w, a in items[-3:][::-1])]
            )
    labels = [c for c in CLASSES if any(c in pair for pair in transitions)]
    report.add(
        "### 4.7 Poids appris par le modèle de production (entraîné sur tout le corpus)",
        "Masse des |poids| d'état par groupe : où le modèle *place* sa confiance (pas l'utilité causale, que "
        "mesurent les ablations ; des attributs redondants se partagent le poids).",
        md_table(["Groupe", "Σ |poids|", "Part"], [[g, fmt(m, 2), f"{m / total:.1%}"] for g, m in group_mass.most_common()]),
        "Attributs les plus fortement associés à chaque classe :",
        md_table(["Classe", "Poids positifs les plus forts", "Poids négatifs les plus forts"], rows, "lll"),
        "Transitions apprises (ligne = classe précédente, colonne = suivante ; positif = favorisée) :",
        md_table(["de \\ vers", *labels], [[a, *[fmt(transitions.get((a, b), 0.0), 2) for b in labels]] for a in labels]),
    )
    write_table(
        out / "poids_modele.csv",
        ["attribut", "classe", "poids"],
        [[a, name, w] for (a, name), w in sorted(state.items(), key=lambda x: -abs(x[1]))],
    )


def learning_curve(
    documents: Sequence[SilverDocument],
    base_splits: Sequence[Split],
    budgets: Sequence[int],
    seeds: int,
    experiments: Sequence[Experiment],
    workers: int,
    cache_dir: Path | None,
) -> list[tuple[str, int, int, float, float]]:
    """(expérience, budget en pages, tirage, macro-F1 fréq., F1 ENTRY) par tirage."""
    splits = []
    for budget in budgets:
        for seed in range(seeds):
            for split in base_splits:
                sub = subsample_training_pages(split, documents, budget, seed)
                splits.append(Split(f"lc-{budget}-{seed}", sub.name, sub.train, sub.test))
    predictions = run_with_progress(documents, experiments, splits, workers, "Courbe d'apprentissage", cache_dir)
    rows = []
    for (name, protocol), pred in predictions.items():
        _, budget, seed = protocol.split("-")
        ev = evaluate(pred, documents)
        cm = ev.confusion
        rows.append(
            (
                name,
                int(budget),
                int(seed),
                float(macro_f1(cm, cm.sum(1) >= FREQUENT_CLASS_MIN_SUPPORT)),
                float(entity_f1(ev.entity_counts)[2][0]),
            )
        )
    return rows


def error_analysis_section(report: Report, ctx: Context, protocol: str, examples: int, title: str, file_name: str) -> dict:
    documents = ctx.documents
    scored = ctx.ev("production", protocol).scored
    errors = np.flatnonzero(~scored.correct)

    def context_of(i: int) -> tuple[SilverDocument, int]:
        return documents[scored.doc_index[i]], int(scored.line_index[i])

    csv_rows = []
    for i in errors:
        document, line = context_of(i)
        record = document.records[line]
        prev_text = document.records[line - 1].text if line > 0 else ""
        next_text = document.records[line + 1].text if line + 1 < len(document) else ""
        csv_rows.append(
            [
                document.name,
                record.uid,
                record.page_index,
                record.data_block_label,
                CLASSES[scored.gold[i]],
                CLASSES[scored.pred[i]],
                round(float(scored.confidence[i]), 4),
                prev_text,
                record.text,
                next_text,
            ]
        )
    write_table(
        ctx.out / file_name,
        ["document", "uid", "page", "bloc_ocr", "verite", "prediction", "confiance", "ligne_precedente", "ligne", "ligne_suivante"],
        csv_rows,
    )

    pairs = Counter((CLASSES[scored.gold[i]], CLASSES[scored.pred[i]]) for i in errors)
    by_doc = Counter(documents[scored.doc_index[i]].name for i in errors)
    report.add(
        f"### {title}",
        f"{len(errors)} erreurs (détail complet : `{file_name}`), réparties par document : "
        + ", ".join(f"{name} {count}" for name, count in by_doc.most_common())
        + ". Paires (vérité → prédiction) les plus fréquentes, avec des exemples tirés au hasard (contexte : "
        "ligne précédente / **ligne** / suivante) :",
    )
    rng = np.random.default_rng(0)
    for (gold, pred), count in pairs.most_common(8):
        members = [i for i in errors if CLASSES[scored.gold[i]] == gold and CLASSES[scored.pred[i]] == pred]
        rows = []
        for i in rng.choice(members, size=min(examples, len(members)), replace=False):
            document, line = context_of(i)
            record = document.records[line]
            prev_text = document.records[line - 1].text if line > 0 else ""
            next_text = document.records[line + 1].text if line + 1 < len(document) else ""
            rows.append(
                [
                    f"{document.volume} p.{record.page_index} {record.data_block_label}",
                    md_code(prev_text, 50),
                    f"**{md_code(record.text, 70)}**",
                    md_code(next_text, 50),
                    fmt(float(scored.confidence[i]), 2),
                ]
            )
        report.add(
            f"**{gold} → {pred}** : {count} erreurs ({count / max(len(errors), 1):.1%})",
            md_table(["Volume, page, bloc", "Précédente", "Ligne", "Suivante", "Conf."], rows, "lllll"),
        )
    return {f"{g} → {p}": c for (g, p), c in pairs.most_common(10)}


def strata_section(report: Report, ctx: Context, protocol: str) -> None:
    """Taux d'erreur par strate d'observation : où le modèle manque d'information."""
    scored = ctx.ev("production", protocol).scored
    keys = (
        "ocr_data_block_label",
        "is_page_start",
        "is_heading",
        "first_in_block",
        "follows_blank",
        "prev_end",
        "end_start",
        "italic",
        "uppercase",
        "first_is_small_word",
    )
    groups = PRODUCTION_GROUPS + CANDIDATE_GROUPS
    per_doc = {d: extract_features_from_context(doc.context, groups) for d, doc in enumerate(ctx.documents)}
    total_errors = int((~scored.correct).sum())
    rows = []
    for key in keys:
        by_value: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for d, line, ok in zip(scored.doc_index, scored.line_index, scored.correct):
            cell = by_value[per_doc[d][line].get(key, ABSENT)]
            cell[0] += 1
            cell[1] += int(not ok)
        for value, (n, e) in sorted(by_value.items(), key=lambda kv: -kv[1][1]):
            if n >= 20 and e:
                rows.append([f"`{key}`", md_code(value, 25), n, e, f"{e / n:.2%}", f"{e / max(total_errors, 1):.1%}"])
    report.add(
        f"### 7.{len(ctx.protocols) + 1} Taux d'erreur par strate ({SHORT[protocol]})",
        "Une strate à la fois fréquente parmi les erreurs et à fort taux d'erreur désigne une situation que les "
        "features actuelles ne permettent pas de trancher (attributs candidats inclus, pour décrire les strates).",
        md_table(["Attribut", "Valeur", "Lignes", "Erreurs", "Taux d'erreur", "Part des erreurs"], rows),
    )


def select_groups(
    ctx: Context, ablation_verdicts: dict, candidate_verdicts: dict, dead_groups: Sequence[str]
) -> tuple[tuple[str, ...], list[str]]:
    """Construit une sélection de features à partir des verdicts :
    retire les groupes constants ou nuisibles, ajoute les candidats utiles."""

    def improves(v: dict) -> bool:
        return "améliore" in v.values() and "dégrade" not in v.values()

    reasons = []
    groups = list(PRODUCTION_GROUPS)
    for group in dead_groups:
        groups.remove(group)
        reasons.append(f"retrait de `{group}` (constant)")
    for name, v in ablation_verdicts.items():
        group = name.lstrip("−")
        if improves(v) and group in groups:
            groups.remove(group)
            reasons.append(f"retrait de `{group}` (son retrait améliore)")
    for name, v in candidate_verdicts.items():
        group = name.lstrip("+")
        if group in CANDIDATE_GROUPS and improves(v) and group not in groups:
            groups.append(group)
            reasons.append(f"ajout de `{group}`")
    return tuple(groups), reasons


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv_files", nargs="*", type=Path, help="CSV curés (défaut : tous les *.lines.csv sous --root).")
    parser.add_argument(
        "--root", type=Path, default=ANNUAIRES_DIR, help="Dossier où chercher les tables de lignes curées (défaut : annuaires)."
    )
    parser.add_argument("-o", "--output", type=Path, default=REPORTS_DIR / "crf", help="Dossier du rapport (défaut : reports/crf).")
    parser.add_argument("--folds", type=int, default=5, help="Plis de pages contiguës par document (défaut : 5).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Rééchantillonnages bootstrap (défaut : 1000).")
    parser.add_argument("--workers", type=int, default=12, help="Processus parallèles (défaut : 12).")
    parser.add_argument("--seed", type=int, default=0, help="Graine des plis et du bootstrap (défaut : 0).")
    parser.add_argument("--no-single-groups", dest="single_groups", action="store_false", help="Ne pas évaluer chaque groupe seul.")
    parser.add_argument(
        "--no-candidates", dest="candidates", action="store_false", help="Ne pas évaluer les groupes candidats ni la sélection."
    )
    parser.add_argument(
        "--no-learning-curve", dest="learning_curve", action="store_false", help="Ne pas calculer la courbe d'apprentissage."
    )
    parser.add_argument("--learning-curve-seeds", type=int, default=3, help="Tirages par budget de la courbe d'apprentissage (défaut : 3).")
    parser.add_argument("--hyperparams", action="store_true", help="Ajouter une grille c1 × c2 (plus long).")
    parser.add_argument("--no-cache", action="store_true", help="Ne pas réutiliser ni écrire le cache des prédictions (<sortie>/cache).")
    parser.add_argument("--examples", type=int, default=6, help="Exemples par type d'erreur dans le rapport (défaut : 6).")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    paths = args.csv_files or discover_silver_csvs(args.root)
    missing = [p for p in paths if not p.exists()]
    if missing or not paths:
        raise CommandError(f"aucun CSV curé trouvé, ou fichier(s) introuvable(s) : {missing}")
    try:
        documents = load_silver_corpus(paths)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise CommandError(f"chargement : {error}") from error
    for document in documents:
        console.print(f"• {document.name} : {document.labeled_count}/{len(document)} lignes labellisées")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    splits = within_document_splits(documents, args.folds) + cross_volume_splits(documents)
    protocols = [p for p in (INTRA, INTER) if any(s.protocol == p for s in splits)]
    families = build_experiments(args)
    experiments = [e for family in families.values() for e in family]
    cache_dir = None if args.no_cache else out / "cache"
    predictions = run_with_progress(documents, experiments, splits, args.workers, "Validation croisée", cache_dir)
    for protocol in protocols:
        predictions[("majoritaire", protocol)] = rule_predictions("majoritaire", documents, majority_rule, protocol)
        predictions[("heuristique", protocol)] = rule_predictions("heuristique", documents, heuristic_rule, protocol)
    console.print("Calcul des métriques…")
    evaluated = {key: evaluate(pred, documents) for key, pred in predictions.items()}
    stats = {p: Statistics(evaluated[("production", p)], args.bootstrap, args.seed) for p in protocols}
    floors = {
        p: {
            m: max(abs(stats[p].delta(evaluated[(e.name, p)], evaluated[("production", p)])[m][0]) for e in families["placebo"])
            for m in stats[p].point(evaluated[("production", p)])
        }
        for p in protocols
    }
    ctx = Context(documents, protocols, evaluated, stats, floors, out)

    report = Report()
    summary: dict[str, object] = {"plancher_bruit": floors}
    report.add(
        "# Audit du CRF de classification de lignes",
        f"Généré par `numrev audit crf` le {time.strftime('%Y-%m-%d %H:%M')} : {len(documents)} documents, "
        f"{args.folds} plis intra-document, {args.bootstrap} rééchantillonnages bootstrap. Légende des Δ : "
        "différence avec la production [IC 95 % bootstrap apparié par page] ; ▲ / ▼ = intervalle entièrement "
        "au-dessus / au-dessous de 0. **Verdict** : effet significatif *et* supérieur au plancher de bruit (4.3).",
        "<!-- RÉSUMÉ -->",
        "<!-- RECOMMANDATIONS -->",
    )
    summary["corpus"] = corpus_section(report, documents, out)
    summary["performances"] = performance_section(report, ctx)
    summary["evolution_v1_v2"] = evolution_section(report, ctx)
    summary["calibration"] = calibration_section(report, ctx)

    # 4. Audit des features
    console.print("Statistiques de features…")
    table = build_feature_table(documents)
    feature_stats = feature_statistics(table)
    dead_groups = constant_groups(feature_stats)
    report.add(
        "## 4. Audit des features",
        "Sauf mention contraire, les attributs sont calculés sur le texte normalisé : sans marqueurs d'emphase "
        "Markdown, sans « # » de tête ni espaces de fin. Les « # » restent lus par `heading`, l'italique est "
        "résumé par `italic`. Les groupes `*_v1` reproduisent les features historiques et ne servent qu'à la "
        "comparaison (section 2bis).",
        md_table(["Groupe", "Statut", "Description"], [[g.name, g.kind, g.description] for g in FEATURE_GROUPS.values()], "lll"),
    )
    summary["features"] = inventory_section(report, table, feature_stats, redundancy_pairs(table, 0.8), out)
    placebo_section(report, ctx, families["placebo"])
    all_comparisons: list[list[object]] = []
    rows, ablation_verdicts = comparison_section(
        report,
        ctx,
        "### 4.4 Ablations : production moins un groupe",
        "Chaque groupe est retiré à tour de rôle ; Δ = (sans le groupe) − production. Verdict « dégrade » = le "
        "groupe est **utile** ; « neutre » = inutile *ou* redondant avec d'autres (voir 4.2 et 4.5) ; « améliore » "
        "= le groupe **nuit**.",
        families["ablation"],
    )
    all_comparisons += rows
    summary["verdicts_ablation"] = ablation_verdicts
    if "single" in families:
        single_group_section(report, ctx, families["single"])
    best_rules = discriminative_section(report, table, discriminative_values(table, min_support=30, top=5))
    console.print("Poids du modèle complet…")
    weights_section(report, *model_weights(documents, table.key_group), out)

    # 5. Candidats et sélection
    candidate_verdicts: dict = {}
    selection_reasons: list[str] = []
    selection: Experiment | None = None
    if "candidates" in families:
        rows, candidate_verdicts = comparison_section(
            report,
            ctx,
            "## 5. Features candidates\n\n### 5.1 Candidats ajoutés un à un",
            "Chaque candidat est ajouté *seul* à la production ; Δ = (production + candidat) − production. Un gain "
            "en inter-volumes est le meilleur indice qu'une feature capture une régularité générale plutôt qu'une "
            "particularité d'un document.",
            families["candidates"],
        )
        all_comparisons += rows
        summary["candidats_titres"] = title_candidates_section(
            report, ctx, [e for e in families["candidates"] if e.name.lstrip("+") in TITLE_CANDIDATES]
        )
        selected_groups, selection_reasons = select_groups(ctx, ablation_verdicts, candidate_verdicts, dead_groups)
        if selected_groups != PRODUCTION_GROUPS:
            selection = Experiment("sélection", selected_groups, description="Sélection issue des verdicts")
            console.print(f"Évaluation de la sélection : {', '.join(selection_reasons)}")
            extra = run_with_progress(documents, [selection], splits, args.workers, "Sélection", cache_dir)
            for key, pred in extra.items():
                evaluated[key] = evaluate(pred, documents)
            rows, selection_verdict = comparison_section(
                report,
                ctx,
                "### 5.3 Sélection proposée",
                "Combinaison construite automatiquement à partir des verdicts ci-dessus : "
                + "; ".join(selection_reasons)
                + f". Groupes retenus : {names(list(selected_groups))}. ⚠ La sélection est choisie et évaluée sur les "
                "mêmes plis : son gain est une estimation optimiste, à confirmer sur un volume curé non encore utilisé.",
                [selection],
            )
            all_comparisons += rows
            summary["selection"] = {"groupes": list(selected_groups), "raisons": selection_reasons, "verdicts": selection_verdict}
    summary["verdicts_candidats"] = candidate_verdicts

    # 6. Courbe d'apprentissage et hyperparamètres
    report.add("## 6. Régime de peu d'annotations et hyperparamètres")
    if args.learning_curve:
        lines_per_page = sum(d.labeled_count for d in documents) / sum(len(page_boundaries(d)) - 1 for d in documents)
        budgets = [1, 2, 4, 8, 16, 32]
        lc_experiments = [families["reference"][1], families["reference"][0]]
        if selection is not None:
            lc_experiments.append(selection)
        if "candidates" in families:
            lc_experiments.append(families["candidates"][-1])
        lc_rows = learning_curve(
            documents,
            [s for s in splits if s.protocol == INTRA],
            budgets,
            args.learning_curve_seeds,
            lc_experiments,
            args.workers,
            cache_dir,
        )
        write_table(out / "courbe_apprentissage.csv", ["experience", "pages", "tirage", "macro_f1_freq", "f1_entry"], lc_rows)
        aggregated = defaultdict(list)
        for name, budget, _, mf1, ef1 in lc_rows:
            aggregated[(name, budget)].append((mf1, ef1))
        headers = ["Pages d'entraînement", "≈ lignes"]
        for e in lc_experiments:
            headers += [f"macro-F1 fréq. `{e.name}`", f"F1 ENTRY `{e.name}`"]
        rows = []
        for budget in budgets:
            row: list[object] = [budget, round(budget * lines_per_page)]
            for e in lc_experiments:
                values = np.array(aggregated[(e.name, budget)])
                row += [f"{values[:, 0].mean():.3f} ± {values[:, 0].std():.3f}", f"{values[:, 1].mean():.3f} ± {values[:, 1].std():.3f}"]
            rows.append(row)
        full_row: list[object] = ["toutes (≈ 80 % du document)", "–"]
        for e in lc_experiments:
            point = stats[INTRA].point(evaluated[(e.name, INTRA)])
            full_row += [fmt(point["macro_f1_freq"]), fmt(point["entry_f1"])]
        rows.append(full_row)
        report.add(
            "### 6.1 Courbe d'apprentissage (intra-document)",
            f"Chaque pli intra-document est réentraîné sur n pages tirées au hasard parmi ses pages "
            f"d'entraînement ({args.learning_curve_seeds} tirages, moyenne ± écart-type ; ≈ {lines_per_page:.0f} lignes "
            "par page). En production, l'apprentissage actif s'arrête après quelques centaines de lignes humaines "
            "(section 1) : c'est ce régime qui compte, et c'est là que la qualité des features pèse le plus. Les "
            "pages tirées sont entièrement annotées, alors que l'apprentissage actif annote des paires de lignes "
            "choisies : la courbe est indicative, pas une simulation de la boucle d'annotation.",
            md_table(headers, rows),
        )
    rows, hyper_verdicts = comparison_section(
        report,
        ctx,
        "### 6.2 Hyperparamètres",
        f"Production : {config_name(DEFAULT_CONFIG)}. Un gain avec plus d'itérations indiquerait que l'optimisation "
        "L-BFGS est arrêtée avant convergence.",
        families["hyperparams"],
    )
    all_comparisons += rows
    summary["verdicts_hyperparametres"] = hyper_verdicts
    write_table(out / "experiences.csv", COMPARISON_CSV_HEADERS, all_comparisons)

    # 7. Analyse d'erreurs
    report.add("## 7. Analyse d'erreurs")
    for number, protocol in enumerate(protocols, start=1):
        summary[f"erreurs_{SHORT[protocol]}"] = error_analysis_section(
            report, ctx, protocol, args.examples, f"7.{number} {PROTOCOL_LABELS[protocol]}", f"erreurs_{SHORT[protocol]}.csv"
        )
    strata_section(report, ctx, INTRA)

    # 8. Méthodologie
    report.add(
        "## 8. Méthodologie et limites",
        "\n".join(
            [
                "- **Observations** : lignes non vides du JSON `*.lines.json` (celui qu'a vu l'annotateur), features "
                "recalculées avec `numrev.crf.features` — strictement celles de production pour la configuration "
                "`production`. La vérité (`classe`) est rattachée par la clé de ligne `cle` ; les "
                "lignes supprimées à la curation restent dans la séquence observée mais ne sont ni apprises ni évaluées.",
                "- **Protocoles** : *intra-document* = K plis de pages contiguës par document, chacun prédit par un "
                "modèle entraîné sur les autres plis du même document ; *inter-volumes* = chaque volume prédit par un "
                "modèle entraîné sur les autres volumes (3 volumes : 3 modèles seulement, d'où une forte variance). "
                "Chaque ligne est prédite exactement une fois par protocole. Les segments d'entraînement sont coupés aux "
                "lignes sans label, comme en production ; les séquences de test sont les plis entiers.",
                "- **Entraînement** : identique à la production (`numrev.crf.model`, L-BFGS, c1 = 0,1, c2 = 0,01, 50 "
                "itérations), décodage postérieur (argmax des marginales). Contrairement à la production, toutes les "
                "lignes des plis d'entraînement sont utilisées : les sections 2 à 5 décrivent un régime riche en "
                "données ; le régime de l'apprentissage actif est approché par la courbe d'apprentissage (6.1).",
                "- **Incertitude** : bootstrap par page (les lignes d'une page ne sont pas indépendantes), "
                "différences appariées. Ces intervalles ignorent l'instabilité de l'entraînement : d'où le plancher de "
                "bruit mesuré par des placebos (4.3), exigé en plus de la significativité pour tout verdict. Les "
                "comparaisons sont nombreuses et non corrigées pour la multiplicité : un verdict isolé proche du "
                "plancher est à confirmer.",
                "- **Métrique de verdict** : macro-F1 sur les classes fréquentes (≥ "
                f"{FREQUENT_CLASS_MIN_SUPPORT} lignes). Les classes rares (SUB-ENTRY, I-TITLE) ont des F1 à très larges "
                "intervalles ; elles sont suivies à part (colonne « classe la plus affectée », sections 2 et 4.6).",
                "- **Référence silver** : ancrée sur le CRF d'origine (section 1) ; scores absolus optimistes, "
                "comparaisons plus fiables.",
                "- **Information mutuelle** : estimateur plug-in corrigé (Miller–Madow), ligne à ligne ; information "
                "individuelle d'un attribut, pas sa contribution dans le modèle.",
            ]
        ),
    )

    report.replace("<!-- RÉSUMÉ -->", "## Résumé\n\n" + "\n".join(executive_summary(summary, ctx, ablation_verdicts, candidate_verdicts)))
    report.replace(
        "<!-- RECOMMANDATIONS -->",
        "## Recommandations\n\n" + "\n".join(recommendations(summary, ctx, ablation_verdicts, candidate_verdicts, dead_groups, best_rules)),
    )
    (out / "rapport.md").write_text(report.render(), encoding="utf-8")
    (out / "resume.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    console.print(
        f"\n[bold green]✅ Rapport :[/bold green] [yellow]{out / 'rapport.md'}[/yellow] "
        f"({time.time() - started:.0f} s ; tables CSV dans {out})"
    )


def executive_summary(summary: dict, ctx: Context, ablation_verdicts: dict, candidate_verdicts: dict) -> list[str]:
    lines = []
    for protocol in ctx.protocols:
        e = summary["evolution_v1_v2"][protocol]
        m, entry = e["macro_f1_freq"], e["entry_f1"]
        lines.append(
            f"- **v2 contre v1 ({SHORT[protocol]})** : macro-F1 fréq. {m['v1']:.3f} → {m['v2']:.3f} "
            f"({fmt_delta(m['delta'], m['ic95'])}), F1 ENTRY {entry['v1']:.3f} → {entry['v2']:.3f}, "
            f"erreurs {e['erreurs_v1']} → {e['erreurs_v2']}."
        )
    perf = summary["performances"]
    for protocol in ctx.protocols:
        p = perf[protocol]
        weak = sorted((f1, c) for c, f1 in p["f1_par_classe"].items() if f1 < 0.8)
        lines.append(
            f"- **{PROTOCOL_LABELS[protocol]}** : macro-F1 fréq. {fmt_ci(p['macro_f1_freq']['valeur'], p['macro_f1_freq']['ic95'])}, "
            f"macro-F1 {fmt(p['macro_f1']['valeur'])}, F1 ENTRY (entités) {fmt_ci(p['entry_f1']['valeur'], p['entry_f1']['ic95'])}, "
            f"F1 TITLE {fmt(p['title_f1']['valeur'])}. Classes faibles (F1 < 0,8) : "
            + (", ".join(f"{c} ({f1:.2f})" for f1, c in weak) or "aucune")
            + "."
        )
    for protocol in ctx.protocols:
        by_verdict = defaultdict(list)
        for name, v in ablation_verdicts.items():
            by_verdict[v[protocol]].append(name.lstrip("−"))
        lines.append(
            f"- Ablations ({SHORT[protocol]}, plancher de bruit {ctx.floors[protocol][VERDICT_METRIC]:.3f}) — groupes **utiles** : "
            f"{names(by_verdict['dégrade'])} ; **sans effet mesurable** : {names(by_verdict['neutre'])} ; "
            f"**nuisibles** : {names(by_verdict['améliore'])}."
        )
    if summary["features"]["attributs_constants"]:
        lines.append(f"- Attributs de production **constants** (aucune information) : {names(summary['features']['attributs_constants'])}.")
    if candidate_verdicts:
        for protocol in ctx.protocols:
            gains = [n for n, v in candidate_verdicts.items() if v[protocol] == "améliore"]
            losses = [n for n, v in candidate_verdicts.items() if v[protocol] == "dégrade"]
            lines.append(f"- Candidats ({SHORT[protocol]}) — gain : {names(gains)} ; perte : {names(losses)}.")
    if "selection" in summary:
        verdicts = summary["selection"]["verdicts"]["sélection"]
        deltas = []
        for protocol in ctx.protocols:
            d = ctx.delta("sélection", protocol)
            deltas.append(
                f"{SHORT[protocol]} Δ macro-F1 fréq. {fmt_delta(*d['macro_f1_freq'])}, "
                f"Δ F1 ENTRY {d['entry_f1'][0]:+.3f} ({verdicts[protocol]})"
            )
        lines.append("- **Sélection proposée** : " + "; ".join(deltas) + " (estimation optimiste, voir 5.3).")
    for protocol in ctx.protocols:
        c = summary["calibration"][protocol]
        lines.append(
            f"- Probabilités ({SHORT[protocol]}) : ECE {fmt(c['ece'], 4)}, "
            f"AUROC de détection d'erreurs par la marge {fmt(c['auroc_marge'])} ; "
            f"relire les 5 % de lignes les plus incertaines retrouve {c['capture']['5.0%']:.0%} des erreurs."
        )
    lines.append(
        f"- Rappel : la référence est ancrée sur le CRF d'origine ({summary['corpus']['classes_corrigees']} classes corrigées sur "
        f"{summary['corpus']['lignes_labellisees']} lignes) — scores absolus optimistes, comparaisons plus fiables."
    )
    return lines


def recommendations(
    summary: dict, ctx: Context, ablation_verdicts: dict, candidate_verdicts: dict, dead_groups: Sequence[str], best_rules: dict
) -> list[str]:
    """Recommandations dérivées mécaniquement des résultats (à discuter)."""
    items = []
    if dead_groups:
        items.append(
            f"1. **Corriger ou retirer les features constantes** ({names(list(dead_groups))}) : elles n'apportent aucune "
            "information et leur présence modifie néanmoins l'optimum régularisé (voir 4.3)."
        )
    useful_everywhere = [n.lstrip("−") for n, v in ablation_verdicts.items() if all(x == "dégrade" for x in v.values())]
    if useful_everywhere:
        items.append(f"1. **Conserver** les groupes utiles dans tous les protocoles : {names(useful_everywhere)}.")
    harmful = [n.lstrip("−") for n, v in ablation_verdicts.items() if "améliore" in v.values()]
    for group in harmful:
        v = ablation_verdicts[f"−{group}"]
        where = ", ".join(SHORT[p] for p, x in v.items() if x == "améliore")
        also = " mais utile en " + ", ".join(SHORT[p] for p, x in v.items() if x == "dégrade") if "dégrade" in v.values() else ""
        items.append(
            f"1. **Repenser `{group}`** : le retirer améliore les résultats ({where}){also} — il encode probablement "
            "une particularité de volume plutôt qu'une régularité générale."
        )
    gains = [
        n.lstrip("+")
        for n, v in candidate_verdicts.items()
        if n != "+tous_candidats" and "améliore" in v.values() and "dégrade" not in v.values()
    ]
    if gains:
        items.append(
            f"1. **Intégrer les candidats validés** : {names(gains)} (gain significatif et supérieur au plancher de bruit "
            "dans au moins un protocole, sans perte au-delà du bruit dans l'autre — voir le détail en 5.1, une perte "
            "inférieure au plancher reste possible)."
        )
    neutral_everywhere = [
        n.lstrip("−") for n, v in ablation_verdicts.items() if all(x == "neutre" for x in v.values()) and n.lstrip("−") not in dead_groups
    ]
    if neutral_everywhere:
        items.append(
            f"1. Groupes **sans effet mesurable** en régime riche : {names(neutral_everywhere)}. Ne pas les retirer sur cette "
            "seule base : vérifier d'abord leur effet en régime de peu d'annotations (6.1), où la redondance aide."
        )
    weak = sorted({c for p in ctx.protocols for c, f1 in summary["performances"][p]["f1_par_classe"].items() if f1 < 0.8})
    if weak:
        detail = ", ".join(f"{c} (meilleure règle simple : F1 {best_rules.get(c, 0):.2f})" for c in weak)
        items.append(
            f"1. **Classes faibles** : {detail}. Sans règle simple informative, ces classes demandent une feature "
            "dédiée (voir les exemples de 7.x) et davantage d'exemples annotés — l'apprentissage actif en propose peu, "
            "car elles sont rares."
        )
    if INTER in ctx.protocols:
        c = summary["calibration"][INTER]
        if c["auroc_marge"] < 0.8:
            items.append(
                f"1. **Transfert entre volumes** : un modèle appris sur d'autres volumes est mal calibré sur un volume "
                f"nouveau (ECE {c['ece']:.3f}, AUROC {c['auroc_marge']:.2f}) : ses probabilités ne peuvent pas guider la "
                "sélection des lignes à annoter. Pré-entraîner sur les volumes passés n'est envisageable qu'avec des "
                "features robustes à la mise en forme (5.2) et une phase d'annotation propre au nouveau volume."
            )
    items.append(
        "1. **Fiabiliser la référence** : relire en priorité les lignes où la prédiction inter-volumes contredit la "
        "référence avec une forte confiance (`erreurs_inter.csv`, colonne `confiance`) — elles concentrent les "
        "erreurs de curation plausibles, que l'ancrage du silver rend invisibles."
    )
    return items

#!/usr/bin/env python3
"""CLI d'Apprentissage Actif minimaliste (Word Shapes) avec Métriques & Debug complets."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

console = Console()

CLASSES = {
    "d": "DEBUT_ENTREE",
    "s": "SUITE_ENTREE",
    "t": "TITRE",
    "b": "BRUIT",
}


# ==============================================================================
# 1. EXTRACTION MINIMALISTE : WORD SHAPES
# ==============================================================================


def get_token_shape(token: str) -> str:
    """Détermine la catégorie typographique d'un jeton."""
    if token in (",", "-", "—", "."):
        return "PUNCT"
    if token.isdigit():
        return "NUM"
    if token.isupper():
        return "UPPER"
    if token.istitle():
        return "TITLECASE"
    if token.islower():
        return "LOWER"
    return "SYM"


import numpy as np
from treeinterpreter import treeinterpreter as ti


def debug_line_explanation(clf, feature_names, X_sample, top_k=3):
    """Affiche les features qui tirent la probabilité vers le haut (+) ou vers le bas (-)

    pour chaque classe sur une ligne donnée.
    """
    # X_sample doit être en 2D shape (1, n_features)
    prediction, bias, contributions = ti.predict(clf, X_sample)

    classes = clf.classes_
    contribs = contributions[0]  # Shape: (n_features, n_classes)
    x_vec = X_sample[0] if hasattr(X_sample, "shape") else X_sample

    print("\n🔍 EXPLICATION DÉTAILLÉE PAR CLASSE (Local Feature Attribution) :")
    print("=" * 70)

    for c_idx, class_name in enumerate(classes):
        base_p = bias[0][c_idx]
        final_p = prediction[0][c_idx]

        print(
            f"\n--- Classe: {class_name} (Probabilité: {final_p * 100:.1f}% | Base: {base_p * 100:.1f}%) ---"
        )

        # Récupérer les contributions pour cette classe
        class_contribs = contribs[:, c_idx]

        # Ne garder que les features qui sont réellement présentes (valeur != 0)
        active_indices = np.where(x_vec != 0)[0]

        # Trier par importance
        sorted_indices = active_indices[
            np.argsort(np.abs(class_contribs[active_indices]))[::-1]
        ]

        for idx in sorted_indices[:top_k]:
            feat_name = feature_names[idx]
            val = class_contribs[idx]
            direction = "🟢 +" if val > 0 else "🔴 "
            print(f"  {direction}{val * 100:+6.1f}%  ->  {feat_name}")


def extract_word_shapes(line: str, k: int = 10) -> dict[str, str]:
    """Extrait uniquement les formes des k premiers et k derniers mots."""
    tokens = re.findall(r"\w+|[^\w\s]", line)
    shapes = [get_token_shape(t) for t in tokens]
    n = len(shapes)

    f = {}
    for i in range(k):
        f[f"shape_start_{i}"] = shapes[i] if i < n else "<PAD>"
        idx = n - k + i
        f[f"shape_end_{i}"] = shapes[idx] if (0 <= idx < n) else "<PAD>"
    return f


def extract_features(lines: list[str]) -> tuple[csr_matrix, list[str]]:
    """Construit la matrice de features (Word Shapes) avec fenêtre [i-1, i, i+1]."""
    raw_dicts = [extract_word_shapes(line) for line in lines]

    vec = DictVectorizer(sparse=True)
    X_base = vec.fit_transform(raw_dicts).tocsr()
    base_names = list(vec.get_feature_names_out())

    # Fenêtre contextuelle [i-1, i, i+1]
    zero = csr_matrix((1, X_base.shape[1]))
    X_prev = vstack([zero, X_base[:-1]])
    X_next = vstack([X_base[1:], zero])

    X = hstack([X_prev, X_base, X_next]).tocsr()
    feature_names = (
        [f"prev_{n}" for n in base_names]
        + [f"curr_{n}" for n in base_names]
        + [f"next_{n}" for n in base_names]
    )

    return X, feature_names


# ==============================================================================
# 2. AFFICHAGE DE DEBUG ET MÉTRIQUES DE CONVERGENCE
# ==============================================================================


def print_step_debug(
    step_info: dict,
    clf: RandomForestClassifier | None,
    feature_names: list[str],
    X: csr_matrix,
    idx: int,
) -> None:
    """Affiche le tableau de bord de convergence et l'inspection interne du modèle."""
    console.print("\n" + "─" * 80)

    # 1. Panel de Métriques Globale & Convergence
    metrics_tb = Table(show_header=False, box=None)
    metrics_tb.add_column("Key", style="bold cyan")
    metrics_tb.add_column("Value", style="yellow")

    metrics_tb.add_row("Avancement :", f"{step_info['progress']}")
    metrics_tb.add_row("Distribution Train :", f"{step_info['class_dist']}")

    if step_info["oob_acc"] is not None:
        metrics_tb.add_row(
            "OOB Accuracy (Train) :",
            f"[bold blue]{step_info['oob_acc']:.1f}%[/bold blue]",
        )
    if step_info["pool_conf"] is not None:
        metrics_tb.add_row(
            "Confiance Moyenne Pool :",
            f"[bold green]{step_info['pool_conf']:.1f}%[/bold green]",
        )

    console.print(
        Panel(
            metrics_tb,
            title="📈 [bold yellow]CONVERGENCE DU MODÈLE[/bold yellow]",
        )
    )

    # 2. Décomposition Prédictive sur le Candidat
    if clf is not None and "probs_candidate" in step_info:
        probs_tb = Table(
            title=f"🎯 Probas estimées pour la ligne [{idx + 1}]",
            header_style="bold magenta",
        )
        probs_tb.add_column("Classe")
        probs_tb.add_column("Probabilité", justify="right")

        for cls_name, prob in sorted(
            step_info["probs_candidate"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            probs_tb.add_row(cls_name, f"{prob * 100:.1f}%")

        console.print(probs_tb)

        # Top 5 Features RF globales
        if hasattr(clf, "feature_importances_"):
            top_idx = np.argsort(clf.feature_importances_)[::-1][:5]
            top_feat_str = ", ".join(
                [
                    f"[yellow]{feature_names[i]}[/yellow] ({clf.feature_importances_[i]:.3f})"
                    for i in top_idx
                ]
            )
            console.print(
                f"🔥 [dim]Top 5 Shapes globales les plus discriminantes :[/dim] {top_feat_str}"
            )

    # 3. EXPLICATION LOCALE DES PROBABILITÉS (Impact des features sur la ligne)
    if clf is not None and hasattr(clf, "estimators_"):
        try:
            from treeinterpreter import treeinterpreter as ti

            # ti.predict gère directement les csr_matrix
            _, _, contributions = ti.predict(clf, X[idx])
            contribs = contributions[0]  # Matrix (n_features, n_classes)

            explain_tb = Table(
                title=f"💡 Impact local des Shapes sur les probabilités [Ligne {idx + 1}]",
                header_style="bold green",
            )
            explain_tb.add_column("Classe Target", style="cyan")
            explain_tb.add_column("Top 3 Shapes qui poussent (+) ou freinent (-)")

            row_indices = X[idx].indices  # Seules les features valant 1

            for c_idx, class_name in enumerate(clf.classes_):
                class_contribs = contribs[:, c_idx]

                # Trier les features actives par magnitude d'impact
                sorted_feat_idx = row_indices[
                    np.argsort(np.abs(class_contribs[row_indices]))[::-1]
                ][:3]

                impact_strings = []
                for f_idx in sorted_feat_idx:
                    val = class_contribs[f_idx]
                    if abs(val) < 0.005:  # Ignorer les impacts négligeables
                        continue
                    color = "bold green" if val > 0 else "bold red"
                    sign = "+" if val > 0 else ""
                    feat_name = feature_names[f_idx]
                    impact_strings.append(
                        f"[{color}]{sign}{val * 100:.1f}%[/] [yellow]({feat_name})[/yellow]"
                    )

                explain_tb.add_row(
                    class_name,
                    (
                        "  |  ".join(impact_strings)
                        if impact_strings
                        else "[dim]Impact neutre[/dim]"
                    ),
                )

            console.print(explain_tb)

        except ImportError:
            console.print(
                "[dim italic]💡 Installez 'treeinterpreter' (pip install treeinterpreter) pour voir l'explication locale.[/dim italic]"
            )

    # 4. Shapes Actives de la Ligne Cible
    row = X[idx]
    shapes_tb = Table(
        title=f"🔍 Shapes actives [Ligne {idx + 1}]",
        header_style="bold cyan",
        padding=(0, 1),
    )
    shapes_tb.add_column("Feature (Shape)", style="yellow")
    shapes_tb.add_column("Valeur", style="green")

    for col_idx, val in zip(row.indices, row.data):
        shapes_tb.add_row(feature_names[col_idx], str(int(val)))

    # console.print(shapes_tb)


def ask_label(idx: int, lines: list[str], reason: str, context_size: int = 2) -> str:
    """Affiche le contexte textuel et demande le label."""
    start = max(0, idx - context_size)
    end = min(len(lines), idx + context_size + 1)

    txt = Text()
    for i in range(start, end):
        prefix = " ▶" if i == idx else "  "
        style = "bold cyan" if i == idx else "dim white"
        txt.append(f"{prefix} [{i + 1}] {lines[i]}\n", style=style)

    console.print(
        Panel(txt, title=f"[bold yellow]{reason}[/bold yellow]", border_style="blue")
    )

    choices = list(CLASSES.keys()) + ["q"]
    ans = Prompt.ask(
        "[d] Début  [s] Suite  [t] Titre  [b] Bruit  [q] Quitter", choices=choices
    )
    return CLASSES[ans] if ans != "q" else "q"


# ==============================================================================
# 3. BOUCLE D'APPRENTISSAGE ACTIF ET DEBUGS
# ==============================================================================


def active_learning(
    lines: list[str], X: csr_matrix, feature_names: list[str], max_queries: int = 0
) -> tuple[RandomForestClassifier | None, dict[int, str]]:
    """Active Learning basé sur l'incertitude avec traçabilité complète."""
    annotations: dict[int, str] = {}
    unannotated = list(range(len(lines)))
    total_lines = len(lines)

    clf = RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    # Amorçage tirage aléatoire de 5 exemples
    init_seeds = np.random.choice(
        total_lines, size=min(5, total_lines), replace=False
    ).tolist()
    console.print(
        "\n[bold magenta]=== PHASE D'AMORÇAGE (5 tirages aléatoires) ===[/bold magenta]"
    )

    for idx in init_seeds:
        step_info = {
            "progress": f"{len(annotations)}/{total_lines} ({(len(annotations)/total_lines)*100:.1f}%)",
            "class_dist": dict(Counter(annotations.values())),
            "oob_acc": None,
            "pool_conf": None,
        }
        print_step_debug(step_info, None, feature_names, X, idx)

        label = ask_label(idx, lines, "[Amorçage]")
        if label == "q":
            return None, annotations
        annotations[idx] = label
        unannotated.remove(idx)

    # Boucle d'Incertitude
    queries = 0
    while unannotated and (max_queries == 0 or queries < max_queries):
        known_classes = set(annotations.values())
        annotated_idx = list(annotations.keys())
        y_train = [annotations[i] for i in annotated_idx]

        step_info = {
            "progress": f"{len(annotations)}/{total_lines} ({(len(annotations)/total_lines)*100:.1f}%)",
            "class_dist": dict(Counter(y_train)),
            "oob_acc": None,
            "pool_conf": None,
        }

        # S'il y a moins de 2 classes distinctes, on ne peut pas entraîner
        if len(known_classes) < 2:
            target_idx = int(np.random.choice(unannotated))
            reason = "[Exploration : 1 seule classe connue]"
            current_clf = None
        else:
            clf.fit(X[annotated_idx], y_train)
            current_clf = clf

            if hasattr(clf, "oob_score_") and clf.oob_score_:
                step_info["oob_acc"] = float(clf.oob_score_) * 100.0

            # Prédictions sur le pool non annoté
            probs = clf.predict_proba(X[unannotated])
            step_info["pool_conf"] = float(np.mean(np.max(probs, axis=1))) * 100.0

            # Calcul de la marge entre les 2 meilleures classes
            sorted_p = np.sort(probs, axis=1)
            margins = (
                sorted_p[:, -1] - sorted_p[:, -2]
                if sorted_p.shape[1] > 1
                else sorted_p[:, -1]
            )

            best_sub = int(np.argmin(margins))
            target_idx = unannotated[best_sub]

            # Enregistrement des détails prédictifs du candidat
            cand_probs = probs[best_sub]
            step_info["probs_candidate"] = {
                clf.classes_[i]: float(cand_probs[i]) for i in range(len(clf.classes_))
            }

            top1_cls = clf.classes_[np.argmax(cand_probs)]
            margin_val = margins[best_sub] * 100
            reason = f"[Incertitude Max] Top: {top1_cls} | Marge Δ: {margin_val:.1f}%"

        queries += 1
        print_step_debug(step_info, current_clf, feature_names, X, target_idx)
        label = ask_label(target_idx, lines, f"[Q{queries}] {reason}")

        if label == "q":
            break

        annotations[target_idx] = label
        unannotated.remove(target_idx)

    # Entraînement final
    if len(set(annotations.values())) < 2:
        return None, annotations

    clf_final = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf_final.fit(
        X[list(annotations.keys())], [annotations[i] for i in annotations.keys()]
    )

    return clf_final, annotations


# ==============================================================================
# 4. MAIN
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active Learning (Word Shapes) + Debug."
    )
    parser.add_argument("input_file", type=str, help="Fichier source (.txt, .md)")
    parser.add_argument(
        "--max-queries", type=int, default=0, help="Nombre max de questions (0=infini)"
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    lines = [
        line.strip()
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    console.print(f"[bold green]✓ {len(lines)} lignes chargées.[/bold green]")

    X, feature_names = extract_features(lines)
    console.print(
        f"[dim]Matrice X construite : {X.shape[0]} lignes × {X.shape[1]} features de Shapes.[/dim]"
    )

    clf, annotations = active_learning(lines, X, feature_names, args.max_queries)

    if clf is None:
        console.print(
            "[bold yellow]Session interrompue. Aucun fichier généré.[/bold yellow]"
        )
        sys.exit(0)

    predictions = list(clf.predict(X))
    for idx, true_label in annotations.items():
        predictions[idx] = true_label

    output_file = input_file.with_suffix(".csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "class"])
        for i, (line, pred) in enumerate(zip(lines, predictions)):
            writer.writerow([i, line, pred])

    console.print(
        f"\n[bold green]✓ Fichier de prédictions enregistré : {output_file} ({len(annotations)} annotations manuelles)[/bold green]"
    )


if __name__ == "__main__":
    main()

"""Tire un échantillon de paires candidates « inversion » à étiqueter à la
main, pour régler la passe résiduelle de `align_directories_nw.py`.

    uv run tools/sample_alignment_gold.py annuaires/1807_AD75-PER292 annuaires/1808_AD75-PER292

Une inversion est une paire d'entrées laissées seules par Needleman-Wunsch
(ni l'une ni l'autre dans une paire ordonnée) qui **croise** au moins une
paire ordonnée : l'entrée a changé de rang d'une édition à l'autre. Les
estimations sans étiquettes de la probabilité qu'une telle paire soit la
même entrée, selon sa similarité et son déplacement, ont échoué sur
1807/1808. Deux raisons à cela : les entrées déplacées sont surtout celles
dont le nom a changé de graphie, que l'échantillon « SUBJ identique » ne voit
pas ; et laisser l'EM estimer la loi des paires différentes le fait
diverger. Ce jeu étiqueté doit permettre de les estimer.

Univers : pour chaque entrée de gauche laissée seule, sa meilleure
partenaire de droite parmi les entrées laissées seules qu'elle croise
(similarité ≥ `--min-similarity`), c'est-à-dire ce que proposerait une
affectation. Mêmes segments, similarité et Needleman-Wunsch que
`align_directories_nw.py`, patch des rubriques compris.

Plan stratifié : déplacement (nombre de paires ordonnées croisées : 1, 2,
3, 4–5, 6–10, 11–50, > 50) × similarité (0,6–0,75, 0,75–0,85, 0,85–0,9,
≥ 0,9). On tire `--per-stratum` paires par strate (toutes si moins),
chacune avec son poids = effectif de la strate / taille tirée. La colonne
`regle_actuelle` indique si la règle en place (affectation optimale, sim ≥
0,85) retient la paire.

Sortie : `data/alignement/<gauche>__<droite>.gold-inversions.csv`
(versionnée), à compléter à la main dans la colonne `meme_entree` (`oui` /
`non` / `?`) et `note`. Un fichier existant n'est jamais écrasé sans
`--force`, puisqu'il contient des étiquettes.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ et aux scripts depuis tools/

import numpy as np
from rich.console import Console
from rich.table import Table

from align_directories_nw import DEFAULT_PATCH_DIR, Params, residual_pairs, segments, similarity_matrix
from lib.alignment import load_volume
from lib.section_alignment import SECTION_PATCH_SUFFIX, load_section_alignment
from lib.sequence import needleman_wunsch

console = Console()

GOLD_SUFFIX = ".gold-inversions.csv"
DISPLACEMENT_BOUNDS = (1, 2, 3, 5, 10, 50)  # bornes hautes des classes de déplacement ; au-delà : dernière classe
SIMILARITY_BOUNDS = (0.75, 0.85, 0.9)  # bornes basses des classes de similarité au-dessus du minimum
CURRENT_RULE_THRESHOLD = 0.85
FIELDS = [
    "strate",
    "poids",
    "deplacement",
    "similarite",
    "regle_actuelle",
    "meme_entree",
    "note",
    "gauche_rubrique",
    "gauche_page",
    "gauche_texte",
    "droite_rubrique",
    "droite_page",
    "droite_texte",
    "left_uuid",
    "right_uuid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tire un échantillon stratifié de paires candidates « inversion » à étiqueter.")
    parser.add_argument("left", type=Path, help="Dossier de l'annuaire de gauche.")
    parser.add_argument("right", type=Path, help="Dossier de l'annuaire de droite.")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"CSV à étiqueter (défaut : {DEFAULT_PATCH_DIR}/<gauche>__<droite>{GOLD_SUFFIX}).")
    parser.add_argument("--per-stratum", type=int, default=8, help="Paires tirées par strate (défaut : 8).")
    parser.add_argument("--min-similarity", type=float, default=0.6, help="Similarité minimale d'une candidate (défaut : 0.6).")
    parser.add_argument("--seed", type=int, default=0, help="Graine du tirage (défaut : 0).")
    parser.add_argument("--force", action="store_true", help="Écraser un fichier existant (et ses étiquettes).")
    return parser.parse_args()


def displacement_label(crossed: int) -> str:
    low = 1
    for high in DISPLACEMENT_BOUNDS:
        if crossed <= high:
            return str(high) if low == high else f"{low}-{high}"
        low = high + 1
    return f">{DISPLACEMENT_BOUNDS[-1]}"


def similarity_label(similarity: float, minimum: float) -> str:
    bounds = [minimum, *SIMILARITY_BOUNDS]
    for low, high in zip(bounds, [*SIMILARITY_BOUNDS, None]):
        if high is None or similarity < high:
            return f"{low:.2f}+" if high is None else f"{low:.2f}-{high:.2f}"
    raise AssertionError


def candidates(left_records, right_records, sections, params: Params, minimum: float) -> list[dict]:
    """Meilleure partenaire croisée de chaque entrée de gauche laissée seule
    par Needleman-Wunsch."""
    result = []
    for segment_left, segment_right in segments(sections):
        left = [record for section in segment_left for record in section.records]
        right = [record for section in segment_right for record in section.records]
        similarity = similarity_matrix(left, right, params.subj_weight)
        ordered = needleman_wunsch(similarity, params.threshold)
        current = set(residual_pairs(similarity, ordered, CURRENT_RULE_THRESHOLD))
        rows = sorted(set(range(len(left))) - {i for i, _ in ordered})
        cols = sorted(set(range(len(right))) - {j for _, j in ordered})
        if not rows or not cols:
            continue
        # paires ordonnées croisées : écart entre les paires au-dessus de la ligne et celles à gauche de la colonne
        above = np.searchsorted([i for i, _ in ordered], rows)
        before = np.searchsorted([j for _, j in ordered], cols)
        crossed = np.abs(above[:, None] - before[None, :])
        sub = np.where(crossed > 0, similarity[np.ix_(rows, cols)], -1.0)
        for a, i in enumerate(rows):
            b = int(sub[a].argmax())
            if sub[a, b] >= minimum:
                j = cols[b]
                result.append(
                    {
                        "deplacement": int(crossed[a, b]),
                        "similarite": float(sub[a, b]),
                        "regle_actuelle": "oui" if (i, j) in current else "non",
                        "left": left[i],
                        "right": right[j],
                    }
                )
    return result


def main() -> None:
    args = parse_args()
    pair_name = f"{args.left.name}__{args.right.name}"
    output = args.output or DEFAULT_PATCH_DIR / f"{pair_name}{GOLD_SUFFIX}"
    if output.exists() and not args.force:
        console.print(f"[bold red]Erreur :[/bold red] {output} existe déjà (étiquettes ?) ; --force pour l'écraser.")
        sys.exit(1)

    params = Params()
    left, right = load_volume(args.left), load_volume(args.right)
    sections = load_section_alignment(left, right, DEFAULT_PATCH_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}", params.section_threshold)
    found = candidates(left, right, sections, params, args.min_similarity)

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in found:
        strata[(displacement_label(candidate["deplacement"]), similarity_label(candidate["similarite"], args.min_similarity))].append(candidate)
    rng = random.Random(args.seed)
    sample = []
    for key in sorted(strata):
        population = sorted(strata[key], key=lambda candidate: candidate["left"].order)
        drawn = rng.sample(population, min(args.per_stratum, len(population)))
        sample += [(key, len(population) / len(drawn), candidate) for candidate in drawn]
    sample.sort(key=lambda item: item[2]["left"].order)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for (displacement, similarity), weight, candidate in sample:
            left_record, right_record = candidate["left"], candidate["right"]
            writer.writerow(
                {
                    "strate": f"d{displacement}/s{similarity}",
                    "poids": f"{weight:.2f}",
                    "deplacement": candidate["deplacement"],
                    "similarite": f"{candidate['similarite']:.4f}",
                    "regle_actuelle": candidate["regle_actuelle"],
                    "meme_entree": "",
                    "note": "",
                    "gauche_rubrique": left_record.section_title,
                    "gauche_page": left_record.page,
                    "gauche_texte": left_record.text,
                    "droite_rubrique": right_record.section_title,
                    "droite_page": right_record.page,
                    "droite_texte": right_record.text,
                    "left_uuid": left_record.uuid,
                    "right_uuid": right_record.uuid,
                }
            )

    similarity_labels = sorted({key[1] for key in strata})
    table = Table(title=f"Candidates (tirées) par strate — {len(found)} candidates, {len(sample)} tirées")
    table.add_column("Déplacement")
    for label in similarity_labels:
        table.add_column(f"sim {label}", justify="right")
    order = [displacement_label(bound) for bound in DISPLACEMENT_BOUNDS] + [displacement_label(DISPLACEMENT_BOUNDS[-1] + 1)]
    for displacement in sorted({key[0] for key in strata}, key=order.index):
        cells = []
        for label in similarity_labels:
            population = strata.get((displacement, label), [])
            cells.append(f"{len(population)} ({min(args.per_stratum, len(population))})" if population else "")
        table.add_row(displacement, *cells)
    console.print(table)
    console.print(f"[bold green]✅ À étiqueter (colonne meme_entree : oui / non / ?) :[/bold green] [yellow]{output}[/yellow]")


if __name__ == "__main__":
    main()

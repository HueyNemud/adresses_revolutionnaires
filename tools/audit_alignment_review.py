"""Mesure, sur le gold des inversions d'une paire d'annuaires, ce que vaut la
règle actuelle de la passe résiduelle et ce qu'attrapent les motifs de
relecture (`lib/alignment_review.py`).

    uv run tools/audit_alignment_review.py data/alignement/<gauche>__<droite>.gold-inversions.csv

Le gold vient de `tools/sample_alignment_gold.py` : paires candidates
« inversion » tirées par strate (déplacement × similarité), avec leur poids
(effectif de la strate / taille tirée), étiquetées à la main dans
`meme_entree` (`OUI` / `NON` / `INCERTAIN` ; `UNSURE` est lu comme
`INCERTAIN`). La paire de volumes se déduit du nom du fichier ; les deux
annuaires sont relus, ainsi que la sortie d'alignement (`--alignment`, par
défaut `annuaires/alignements/<gauche>__<droite>.nw.csv`), sans patch des
entrées : on évalue l'automatique.

Rapport `rapports/audit_alignement/<gauche>__<droite>.md` :

1. règle actuelle (`regle_actuelle` du gold) : précision et gain plafond,
   pondérés ;
2. chaque paire du gold selon ce que la relecture en ferait : retenue sans
   motif (incertitude faible), retenue avec motif, candidate non appariée,
   ou ni l'une ni l'autre ; on attend des OUI en incertitude faible et des
   NON / INCERTAIN ailleurs, et une part de OUI qui baisse de l'incertitude
   faible à forte ;
3. charge de relecture sur toute la sortie, par motif et par niveau ;
4. rubriques sans correspondance et leurs entrées : on n'apparie qu'entre
   rubriques appariées, elles sont donc exclues de tout appariement.

Le gold ne contient que des candidates hors de l'ordre (inversions), tirées
dans les rubriques appariées : les paires décidées par le pair-HMM (motif
`déduite des voisines`) et la perte due aux rubriques sans correspondance
n'y figurent pas ; la section 4 chiffre cette perte à part.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ et aux scripts depuis tools/

from rich.console import Console

from align_directories_nw import Params
from lib.alignment import load_volume, read_links
from lib.alignment_review import DEFAULT_MARGIN, LEVEL_LABELS, review
from lib.reporting import md_table
from lib.section_alignment import SECTION_PATCH_SUFFIX, load_section_alignment

console = Console()

ANNUAIRES_DIR = Path("annuaires")
ALIGNMENTS_DIR = ANNUAIRES_DIR / "alignements"
PATCH_DIR = Path("data/alignement")
REPORT_DIR = Path("rapports/audit_alignement")
GOLD_SUFFIX = ".gold-inversions.csv"
LABELS = ("OUI", "NON", "INCERTAIN")
LABEL_ALIASES = {"UNSURE": "INCERTAIN"}


def label(row: dict) -> str:
    value = row["meme_entree"].strip().upper()
    return LABEL_ALIASES.get(value, value)


def counts_table(groups: dict[str, list[dict]], order: list[str]) -> str:
    """Effectifs bruts et pondérés de chaque étiquette, par groupe."""
    rows = []
    for key in order:
        members = groups.get(key, [])
        raw = Counter(label(row) for row in members)
        weighted = Counter()
        for row in members:
            weighted[label(row)] += float(row["poids"])
        total = sum(weighted[name] for name in LABELS)
        share = f"{weighted['OUI'] / total:.0%}" if total else "–"
        rows.append([key, *(raw[name] for name in LABELS), *(round(weighted[name]) for name in LABELS), share])
    return md_table(["", *LABELS, *(f"{name} pondéré" for name in LABELS), "part OUI (pondérée)"], rows)


def unmatched_section_table(sections, left_name: str, right_name: str) -> str:
    """Rubriques sans correspondance de chaque côté, avec leur nombre d'entrées."""
    rows = [
        [name, section.title or "(sans rubrique)", "déclarée au patch" if (side, section.uuid) in sections.declared else "non alignée", len(section.records)]
        for side, name, unmatched in (("left", left_name, sections.unmatched_left), ("right", right_name, sections.unmatched_right))
        for section in unmatched
    ]
    return md_table(["annuaire", "rubrique", "origine", "entrées"], rows, align="lllr") if rows else "Aucune."


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit de la passe résiduelle et des motifs de relecture sur un gold d'inversions.")
    parser.add_argument("gold", type=Path, help=f"Gold étiqueté (`*{GOLD_SUFFIX}`).")
    parser.add_argument("--alignment", type=Path, default=None, help="Sortie d'alignement (défaut : <paire>.nw.csv).")
    parser.add_argument("--ecart", type=float, default=DEFAULT_MARGIN, help=f"Écart « homonyme proche » (défaut : {DEFAULT_MARGIN}).")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"Rapport (défaut : {REPORT_DIR}/<paire>.md).")
    args = parser.parse_args()

    pair_name = args.gold.name.removesuffix(GOLD_SUFFIX)
    left_name, separator, right_name = pair_name.partition("__")
    if not separator:
        parser.error(f"nom de fichier sans paire `<gauche>__<droite>` : {args.gold.name}")
    with args.gold.open(encoding="utf-8", newline="") as handle:
        gold = [row for row in csv.DictReader(handle) if label(row) in LABELS]
    if not gold:
        parser.error(f"aucune ligne étiquetée ({' / '.join(LABELS)}) dans {args.gold}")

    left, right = load_volume(ANNUAIRES_DIR / left_name), load_volume(ANNUAIRES_DIR / right_name)
    alignment_path = args.alignment or ALIGNMENTS_DIR / f"{pair_name}.nw.csv"
    links = read_links(alignment_path)
    sections = load_section_alignment(left, right, PATCH_DIR / f"{pair_name}{SECTION_PATCH_SUFFIX}", rewrite=False)
    params = Params()
    found = review(links, left, right, sections, params.threshold, params.residual_threshold, params.subj_weight, args.ecart)
    retained = {(link.left_uuid, link.right_uuid) for link in links}
    candidate_pairs = {(link.left_uuid, link.right_uuid) for link in found.candidates}

    # 1. Règle actuelle
    rule = defaultdict(list)
    for row in gold:
        rule["retenue par la règle" if row["regle_actuelle"] == "oui" else "rejetée par la règle"].append(row)

    # 2. Ce que la relecture fait de chaque paire du gold
    fate = defaultdict(list)
    levels = defaultdict(list)
    for row in gold:
        key = (row["left_uuid"], row["right_uuid"])
        reviewed = found.reviews.get(key)
        if key in retained:
            group = f"retenue, motif : {' + '.join(reviewed.reasons)}" if reviewed else "retenue, sans motif"
            levels[reviewed.level if reviewed else 0].append(row)
        elif key in candidate_pairs:
            group = "candidate non appariée"
            levels[reviewed.level].append(row)
        else:
            group = "ni retenue ni candidate"
        fate[group].append(row)

    # 3. Charge de relecture sur toute la sortie
    load = Counter(" + ".join(item.reasons) for item in found.reviews.values())
    load_levels = Counter(item.level for item in found.reviews.values())

    report = [
        f"# Audit de la relecture — {left_name} ⟷ {right_name}",
        "",
        f"Gold : `{args.gold}` ({len(gold)} paires étiquetées). Alignement : `{alignment_path}` ({len(links)} liens, sans patch). "
        f"Seuils : candidates non appariées dans [{params.threshold} ; {params.residual_threshold}[, écart « homonyme proche » {args.ecart}.",
        "",
        "## 1. Règle actuelle de la passe résiduelle (sim ≥ 0,85, affectation optimale)",
        "",
        counts_table(rule, ["retenue par la règle", "rejetée par la règle"]),
        "",
        "Précision = part de OUI parmi les retenues ; gain plafond = OUI pondérés parmi les rejetées.",
        "",
        "## 2. Paires du gold selon la relecture",
        "",
        counts_table(fate, sorted(fate)),
        "",
        "Par niveau d'incertitude (paires retenues ou candidates) — la part de OUI doit baisser de faible à forte :",
        "",
        counts_table({LEVEL_LABELS[level]: rows for level, rows in levels.items()}, [LEVEL_LABELS[level] for level in sorted(levels)]),
        "",
        "## 3. Charge de relecture sur toute la sortie",
        "",
        md_table(["motifs", "paires"], [[reasons, count] for reasons, count in load.most_common()]),
        "",
        md_table(["incertitude", "paires"], [[LEVEL_LABELS[level], load_levels[level]] for level in sorted(load_levels)]),
        "",
        "Le gold ne contient que des inversions : le motif `déduite des voisines` (pair-HMM) n'y est pas évalué.",
        "",
        "## 4. Rubriques sans correspondance (entrées exclues de tout appariement)",
        "",
        unmatched_section_table(sections, left_name, right_name),
        "",
        f"Total : {sum(len(section.records) for section in sections.unmatched_left)} entrée(s) de {left_name}, "
        f"{sum(len(section.records) for section in sections.unmatched_right)} de {right_name}. "
        "On n'apparie qu'entre rubriques appariées : ces entrées restent sans correspondance tant que leur rubrique "
        "n'est pas liée dans le patch des rubriques. Le gold, tiré dans les rubriques appariées, ne mesure pas cette perte.",
        "",
    ]
    output = args.output or REPORT_DIR / f"{pair_name}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    console.print(f"[bold green]✅ Rapport :[/bold green] [yellow]{output}[/yellow]")


if __name__ == "__main__":
    main()

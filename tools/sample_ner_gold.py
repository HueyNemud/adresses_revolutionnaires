"""Tire le jeu gold NER (SUBJ/DESC/ADDR) à corriger dans Label Studio.

Univers : toutes les ENTRY des `*.merged.csv` de `annuaires/`, textes
dédoublonnés (une même personne est souvent listée sous plusieurs
rubriques), **moins les textes déjà vus à l'entraînement** (`--exclude` :
JSON Label Studio ou CSV d'échantillons, par défaut les jeux
`data/ner/train*.ls.json` ; l'exclusion se fait sur le texte normalisé, car
les uid ne sont pas comparables d'un fichier à l'autre).

Plan de sondage stratifié, chaque entrée dans une seule strate (par ordre
de priorité) :

- `désaccord` : le curateur a corrigé la sortie du modèle (cas difficiles
  connus) ;
- `signature rare` : suite de classes autre que SUBJ,ADDR / SUBJ,DESC,ADDR ;
- `forme rare` : forme typographique (`lib.ner.shapes.coarse_shape`) vue
  moins de `--rare-shape` fois ;
- `courant` : le reste (~90 % des entrées).

Une part `--random-share` de l'échantillon est répartie proportionnellement
aux strates (image fidèle du corpus), le reste à parts égales entre les
strates non courantes (sur-représentation des cas atypiques). Chaque tâche
porte son poids `weight` = effectif de la strate / taille tirée, qui rend
les métriques de `audit_ner.py` représentatives du corpus entier.

Découpage figé `dev` / `test` par page (≈ `--dev-share` des pages en dev) :
`dev` sert aux réglages (seuil, prompt LLM, exemples few-shot), `test` à la
décision finale uniquement.

Pré-annotation : sortie du modèle corrigée à la main quand elle existe,
placée en `predictions` ; Label
Studio exporte la version relue sous `annotations`. Le texte est normalisé
(emphase Markdown retirée) ; `raw_text` garde le texte d'origine.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from rich.console import Console
from rich.table import Table

from lib.ner.corpus import Entry, iter_corpus
from lib.ner.shapes import shape_profile
from lib.ner.spans import canonical_spans, ls_result, normalize_markdown, render_tagged_text, signature

console = Console()

DEFAULT_ROOT = Path("annuaires")
DEFAULT_OUTPUT = Path("data/ner/gold_v1.ls.json")
DEFAULT_EXCLUDE = sorted(Path("data/ner").glob("train*.ls.json"))
COMMON_SIGNATURES = {"SUBJ,ADDR", "SUBJ,DESC,ADDR"}
STRATA = ("désaccord", "signature rare", "forme rare", "courant")


def excluded_texts(paths: list[Path]) -> set[str]:
    texts: set[str] = set()
    for path in paths:
        if path.suffix == ".json":
            for task in json.loads(path.read_text(encoding="utf-8")):
                data = task.get("data", {})
                texts.add(normalize_markdown(data.get("markdown") or data.get("text", "")).text)
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                texts.update(normalize_markdown(row.get("markdown", "")).text for row in csv.DictReader(handle))
    texts.discard("")
    return texts


def pre_annotation(entry: Entry) -> list:
    return entry.annotations.get("ner_curated") or entry.annotations.get("ner") or []


def stratum(entry: Entry, shape_counts: Counter, rare_shape: int, pre_signature: str) -> str:
    raw, curated = entry.annotations.get("ner"), entry.annotations.get("ner_curated")
    corrected = raw is not None and curated is not None and canonical_spans(entry.text, raw) != canonical_spans(entry.text, curated)
    if corrected:
        return "désaccord"
    if pre_signature not in COMMON_SIGNATURES:
        return "signature rare"
    if shape_counts[entry.shape] < rare_shape:
        return "forme rare"
    return "courant"


def allocate(population: dict[str, int], size: int, random_share: float) -> dict[str, int]:
    """Taille tirée par strate : part proportionnelle + part égale entre
    strates non courantes, plafonnée à l'effectif (l'excédent est reversé
    aux autres strates)."""
    total = sum(population.values())
    allocation = {name: min(n, round(size * random_share * n / total)) for name, n in population.items()}
    remaining = size - sum(allocation.values())
    while remaining > 0:
        open_strata = [name for name in STRATA if name != "courant" and allocation.get(name, 0) < population.get(name, 0)]
        if not open_strata:
            open_strata = [name for name in population if allocation[name] < population[name]]
            if not open_strata:
                break
        share = max(1, remaining // len(open_strata))
        for name in open_strata:
            extra = min(share, population[name] - allocation[name], remaining)
            allocation[name] += extra
            remaining -= extra
    return allocation


def page_split(entry: Entry, dev_share: float) -> str:
    digest = hashlib.sha1(f"{entry.document}#{entry.page}".encode()).digest()
    return "dev" if digest[0] / 256 < dev_share else "test"


def build_task(entry: Entry, spans, stratum_name: str, weight: float, split: str) -> dict:
    tagged = {
        f"{name}_tagged": render_tagged_text(entry.text, annotation)
        for name, annotation in entry.annotations.items()
        if annotation is not None
    }
    return {
        "data": {
            "text": entry.text,
            "raw_text": entry.raw_text,
            "key": entry.key,
            "document": entry.document,
            "volume": entry.volume,
            "uid": entry.uid,
            "page": entry.page,
            "shape": entry.shape,
            "profile": shape_profile(entry.text),
            "stratum": stratum_name,
            "weight": weight,
            "split": split,
            **tagged,
        },
        "predictions": [{"model_version": "ner_curated", "result": ls_result(entry.text, spans)}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tire le jeu gold NER stratifié et pondéré (tâches Label Studio).")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dossier des volumes (défaut : annuaires).")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"JSON Label Studio (défaut : {DEFAULT_OUTPUT}).")
    parser.add_argument("--size", type=int, default=600, help="Nombre d'entrées tirées (défaut : 600).")
    parser.add_argument("--random-share", type=float, default=0.5, help="Part répartie proportionnellement aux strates (défaut : 0.5).")
    parser.add_argument("--dev-share", type=float, default=0.25, help="Part des pages en dev (défaut : 0.25).")
    parser.add_argument("--rare-shape", type=int, default=50, help="Seuil d'effectif sous lequel une forme est rare (défaut : 50).")
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=DEFAULT_EXCLUDE,
        help="Échantillons d'entraînement à exclure (JSON Label Studio ou CSV ; défaut : data/ner/train*.ls.json).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        console.print(f"[bold red]Erreur :[/bold red] {args.output} existe déjà ; le gold est figé, choisissez un autre nom.")
        return

    entries = list(iter_corpus(args.root))
    excluded = excluded_texts(args.exclude)
    console.print(f"{len(entries)} ENTRY lues, {len(excluded)} textes d'entraînement exclus ({len(args.exclude)} fichier(s)).")

    rng = random.Random(args.seed)
    by_text: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        if entry.text not in excluded:
            by_text[entry.text].append(entry)
    universe = [rng.choice(group) for _, group in sorted(by_text.items())]
    shape_counts = Counter(entry.shape for entry in universe)

    strata: dict[str, list[tuple[Entry, list]]] = defaultdict(list)
    for entry in universe:
        spans = pre_annotation(entry)
        strata[stratum(entry, shape_counts, args.rare_shape, signature(spans))].append((entry, spans))

    population = {name: len(strata[name]) for name in STRATA if strata[name]}
    allocation = allocate(population, args.size, args.random_share)

    tasks = []
    for name, n in allocation.items():
        weight = population[name] / n if n else math.nan
        for entry, spans in rng.sample(strata[name], n):
            tasks.append(build_task(entry, spans, name, weight, page_split(entry, args.dev_share)))
    rng.shuffle(tasks)  # ordre de relecture sans regroupement par strate

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")

    table = Table(title=f"Jeu gold : {len(tasks)} entrées sur {len(universe)} textes distincts")
    for column in ("strate", "effectif", "tirées", "poids", "dev", "test"):
        table.add_column(column, justify="right" if column != "strate" else "left")
    splits = Counter((task["data"]["stratum"], task["data"]["split"]) for task in tasks)
    for name, n in allocation.items():
        table.add_row(name, str(population[name]), str(n), f"{population[name] / n:.1f}", str(splits[name, "dev"]), str(splits[name, "test"]))
    console.print(table)
    volumes = Counter(task["data"]["volume"] for task in tasks)
    console.print("Par volume : " + ", ".join(f"{v} {n}" for v, n in sorted(volumes.items())))
    console.print(f"[bold green]✅ Gold à corriger :[/bold green] [yellow]{args.output}[/yellow]")


if __name__ == "__main__":
    main()

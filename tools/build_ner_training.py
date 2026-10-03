"""Construit un jeu d'entraînement NER (SUBJ/DESC/ADDR) au format Label
Studio attendu par `tools/train_gliner.py`.

Source : les ENTRY de `annuaires/` avec leur annotation NER (`*.ner.csv`,
où les lignes corrigées à la main portent `corrige = oui`), ramenée sur le
texte normalisé. Ces CSV doivent venir du modèle courant
(`infer_gliner.py`) et suivre `docs/guide_annotation_ner.md` : un modèle
réapprend les écarts de convention de ses données.

Filtres : textes du jeu gold exclus (dev **et** test, comparaison sur le
texte normalisé) ; textes dédoublonnés ; entrées sans empan, ou qui ne
commencent pas par un SUBJ, écartées.

Tirage par forme typographique (`lib.ner.shapes.coarse_shape`) : quota ∝
√effectif de la forme, plafonné à l'effectif. Le cas de base (`w , w , 9`,
un tiers du corpus) est réduit ; les formes rares sont prises en entier.
Sur le gold, ce tirage a nettement battu un tirage uniforme.

Sorties : `<sortie>.json` (tâches avec `annotations`, `data.text` normalisé
et `data.raw_text` d'origine) et `<sortie>.manifest.json` (paramètres,
effectifs par volume, signature, forme).
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # accès à lib/ depuis tools/

from rich.console import Console

from lib.ner.corpus import iter_corpus, ls_texts
from lib.ner.spans import Span, ls_result, signature

console = Console()

DEFAULT_ROOT = Path("annuaires")
DEFAULT_GOLD = Path("data/ner/gold.ls.json")
DEFAULT_OUTPUT = Path("data/ner/train.ls.json")
DEFAULT_SIZE = 15000


@dataclass
class Candidate:
    text: str
    raw_text: str
    spans: list[Span]
    volume: str
    key: str
    page: str
    shape: str


def gold_texts(path: Path) -> set[str]:
    if not path.exists():
        console.print(f"[yellow]⚠ gold '{path}' introuvable : aucune exclusion.[/yellow]")
        return set()
    return ls_texts(path)


def corpus_candidates(root: Path) -> list[Candidate]:
    candidates = []
    for entry in iter_corpus(root):
        spans = entry.annotations.get("ner_curated") or entry.annotations.get("ner")
        if spans:
            candidates.append(Candidate(entry.text, entry.raw_text, spans, entry.volume, entry.key, entry.page, entry.shape))
    return candidates


def is_usable(candidate: Candidate) -> bool:
    return bool(candidate.spans) and signature(candidate.spans).startswith("SUBJ")


def shape_quotas(counts: Counter, size: int) -> dict[str, int]:
    """Quota par forme = min(effectif, λ·√effectif), λ tel que la somme
    atteigne `size` (recherche par dichotomie)."""
    if sum(counts.values()) <= size:
        return dict(counts)
    low, high = 0.0, float(max(counts.values()))
    for _ in range(60):
        middle = (low + high) / 2
        if sum(min(n, middle * math.sqrt(n)) for n in counts.values()) < size:
            low = middle
        else:
            high = middle
    return {shape: min(n, max(1, round(high * math.sqrt(n)))) for shape, n in counts.items()}


def sample(candidates: list[Candidate], size: int, rng: random.Random) -> list[Candidate]:
    if len(candidates) <= size:
        return list(candidates)
    by_shape: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_shape[candidate.shape].append(candidate)
    quotas = shape_quotas(Counter({shape: len(group) for shape, group in by_shape.items()}), size)
    return [c for shape, group in sorted(by_shape.items()) for c in rng.sample(group, quotas[shape])]


def to_task(candidate: Candidate) -> dict:
    return {
        "data": {
            "text": candidate.text,
            "raw_text": candidate.raw_text,
            "key": candidate.key,
            "volume": candidate.volume,
            "page": candidate.page,
            "shape": candidate.shape,
        },
        "annotations": [{"result": ls_result(candidate.text, candidate.spans)}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit un jeu d'entraînement NER à partir des CSV NER de annuaires/.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dossier des volumes (défaut : annuaires).")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help=f"Gold dont les textes sont exclus (défaut : {DEFAULT_GOLD}).")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"Nombre d'exemples (défaut : {DEFAULT_SIZE}).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"JSON de sortie (défaut : {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    excluded = gold_texts(args.gold)

    seen: set[str] = set()
    candidates: list[Candidate] = []
    dropped = Counter()
    for candidate in corpus_candidates(args.root):
        if candidate.text in excluded:
            dropped["gold"] += 1
        elif candidate.text in seen:
            dropped["doublon"] += 1
        elif not is_usable(candidate):
            dropped["sans SUBJ initial"] += 1
        else:
            seen.add(candidate.text)
            candidates.append(candidate)

    chosen = sample(candidates, args.size, rng)
    rng.shuffle(chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON compact : le fichier est versionné pour la machine d'entraînement.
    args.output.write_text(json.dumps([to_task(c) for c in chosen], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "candidats": len(candidates),
        "écartés": dict(dropped),
        "exemples": len(chosen),
        "par_volume": Counter(c.volume for c in chosen),
        "par_signature": Counter(signature(c.spans) for c in chosen),
        "formes_distinctes": len({c.shape for c in chosen}),
        "20_formes_principales": Counter(c.shape for c in chosen).most_common(20),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    console.print(f"{len(candidates)} candidats ({dict(dropped)} écartés) → [cyan]{len(chosen)}[/cyan] exemples.")
    console.print("Par volume : " + ", ".join(f"{k} {v}" for k, v in manifest["par_volume"].most_common()))
    console.print("Par signature : " + ", ".join(f"{k} {v}" for k, v in manifest["par_signature"].most_common(6)))
    console.print(f"[bold green]✅ Jeu d'entraînement :[/bold green] [yellow]{args.output}[/yellow] (manifeste : {manifest_path})")


if __name__ == "__main__":
    main()

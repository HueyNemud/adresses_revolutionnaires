"""Construit un jeu d'entraînement NER (SUBJ/DESC/ADDR) au format Label
Studio attendu par `tools/train_gliner.py`.

Sources (toutes ramenées sur le texte normalisé) :

- `corpus` : les ENTRY de `annuaires/` avec leur annotation v1 curée
  (`*.ner.curated.csv`, à défaut `*.ner.csv`) ;
- `--extra` : des JSON Label Studio supplémentaires, par défaut les
  pré-annotations LLM qui ont servi à entraîner v1 (elles couvrent des
  volumes absents de `annuaires/` : 1797-1798, 1801-1802, 1809, 1824).

Filtres : textes du jeu gold exclus (dev **et** test, comparaison sur le
texte normalisé) ; textes dédoublonnés (le corpus est prioritaire sur les
sources supplémentaires) ; entrées sans empan, ou qui ne commencent pas par
un SUBJ, écartées.

Tirage (`--sampling`) :
- `shape` (défaut) : quota par forme typographique ∝ √effectif, plafonné à
  l'effectif. Le cas de base (`w , w , 9`, un tiers du corpus) est réduit ;
  les formes rares sont prises en entier ;
- `random` : tirage uniforme, pour comparaison.

Sorties : `<sortie>.json` (tâches avec `annotations`, `data.text` normalisé
et `data.raw_text` d'origine) et `<sortie>.manifest.json` (paramètres,
effectifs par source, volume, forme).
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

from lib.ner.corpus import iter_corpus
from lib.ner.shapes import coarse_shape
from lib.ner.spans import Span, ls_result, ls_task_spans, normalize_markdown, project_spans, signature

console = Console()

DEFAULT_ROOT = Path("annuaires")
DEFAULT_GOLD = Path("data/ner/gold_v1.ls.json")
DEFAULT_EXTRA = [Path("models/sample_entry_5000_20260918_111801.ls-annotations.json")]
DEFAULT_OUTPUT = Path("data/ner/train_v2.ls.json")
DEFAULT_SIZE = 15000


@dataclass
class Candidate:
    text: str
    raw_text: str
    spans: list[Span]
    source: str
    volume: str
    key: str
    page: str
    shape: str


def gold_texts(path: Path) -> set[str]:
    if not path.exists():
        console.print(f"[yellow]⚠ gold '{path}' introuvable : aucune exclusion.[/yellow]")
        return set()
    return {task["data"]["text"] for task in json.loads(path.read_text(encoding="utf-8"))}


def corpus_candidates(root: Path) -> list[Candidate]:
    candidates = []
    for entry in iter_corpus(root):
        spans = entry.annotations.get("v1_curated") or entry.annotations.get("v1")
        if spans:
            candidates.append(Candidate(entry.text, entry.raw_text, spans, "corpus", entry.volume, entry.key, entry.page, entry.shape))
    return candidates


def extra_candidates(path: Path) -> list[Candidate]:
    candidates = []
    for task in json.loads(path.read_text(encoding="utf-8")):
        data = task.get("data", {})
        raw_text = data.get("text", "").strip()
        spans = ls_task_spans(task)
        if not raw_text or spans is None:
            continue
        normalized = normalize_markdown(raw_text)
        filename = data.get("filename", path.stem)
        candidates.append(
            Candidate(
                normalized.text,
                raw_text,
                project_spans(spans, normalized),
                path.stem,
                filename.split(".", 1)[0],
                f"{filename}#{data.get('uid', '')}",
                str(data.get("page_index", "")),
                coarse_shape(normalized.text),
            )
        )
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


def sample(candidates: list[Candidate], size: int, mode: str, rng: random.Random) -> list[Candidate]:
    if mode == "random" or len(candidates) <= size:
        return rng.sample(candidates, min(size, len(candidates)))
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
            "source": candidate.source,
            "shape": candidate.shape,
        },
        "annotations": [{"result": ls_result(candidate.text, candidate.spans)}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit le jeu d'entraînement NER v2 (silver conforme au guide).")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dossier des volumes (défaut : annuaires).")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help=f"Gold dont les textes sont exclus (défaut : {DEFAULT_GOLD}).")
    parser.add_argument("--extra", type=Path, nargs="*", default=DEFAULT_EXTRA, help="JSON Label Studio supplémentaires (défaut : pré-annotations LLM de v1).")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"Nombre d'exemples (défaut : {DEFAULT_SIZE}).")
    parser.add_argument("--sampling", choices=("shape", "random"), default="shape", help="Tirage par forme (√) ou uniforme (défaut : shape).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"JSON de sortie (défaut : {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    excluded = gold_texts(args.gold)

    pools = [corpus_candidates(args.root)]
    for path in args.extra:
        pools.append(extra_candidates(path))

    seen: set[str] = set()
    candidates: list[Candidate] = []
    dropped = Counter()
    for pool in pools:
        for candidate in pool:
            if candidate.text in excluded:
                dropped["gold"] += 1
            elif candidate.text in seen:
                dropped["doublon"] += 1
            elif not is_usable(candidate):
                dropped["sans SUBJ initial"] += 1
            else:
                seen.add(candidate.text)
                candidates.append(candidate)

    chosen = sample(candidates, args.size, args.sampling, rng)
    rng.shuffle(chosen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON compact : le fichier est versionné pour la machine d'entraînement.
    args.output.write_text(json.dumps([to_task(c) for c in chosen], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v for k, v in vars(args).items()},
        "candidats": len(candidates),
        "écartés": dict(dropped),
        "exemples": len(chosen),
        "par_source": Counter(c.source for c in chosen),
        "par_volume": Counter(c.volume for c in chosen),
        "par_signature": Counter(signature(c.spans) for c in chosen),
        "formes_distinctes": len({c.shape for c in chosen}),
        "20_formes_principales": Counter(c.shape for c in chosen).most_common(20),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    console.print(f"{len(candidates)} candidats ({dict(dropped)} écartés) → [cyan]{len(chosen)}[/cyan] exemples, tirage {args.sampling}.")
    console.print("Par source : " + ", ".join(f"{k} {v}" for k, v in manifest["par_source"].most_common()))
    console.print("Par signature : " + ", ".join(f"{k} {v}" for k, v in manifest["par_signature"].most_common(6)))
    console.print(f"[bold green]✅ Jeu d'entraînement :[/bold green] [yellow]{args.output}[/yellow] (manifeste : {manifest_path})")


if __name__ == "__main__":
    main()

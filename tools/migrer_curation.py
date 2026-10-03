"""Migration unique des fichiers curés vers le protocole de lib/curation.py.

    uv run tools/migrer_curation.py [--annuaires annuaires] [--data data] [--ecrire]

Avant : pour chaque plage, une sortie machine et une copie éditée à la main
(`*.ocr.lines.annotated.csv` + `*.annotated.curated.csv`, colonne
`prediction_curated` ; `*.curated.merged.ner.csv` + `*.ner.curated.csv`),
des `uuid` d'entité fondés sur les `uid`. Après : un seul fichier par
étape, à la fois sortie machine et fichier édité, et des `uuid` fondés sur
la clé `cle` de la ligne racine. Pour chaque document :

1. les lignes du CSV curé sont rattachées à l'export machine du JSON annoté
   (par `uid`, puis par (page, texte) unique pour les blocs re-segmentés) ;
   une ligne curée sans correspondant est une **ligne ajoutée**, une ligne
   machine sans correspondant une **ligne supprimée** (classe `SUPPRIMÉE`) ;
   une ligne **déplacée** (hors de la plus longue sous-suite dans l'ordre de
   l'OCR) devient une ligne supprimée à sa place d'origine et une ligne
   ajoutée à sa nouvelle place, comme le veut le protocole ;
   une ligne dont la classe ou le texte diffère est **corrigée** ;
2. le patch `data/curation/<document>.lignes.patch.csv` est écrit et
   appliqué : `*.ocr.lines.annotated.csv` est réécrit au nouveau format
   (l'ancien est gardé en `*.ocr.lines.annotated.avant-migration.csv`) ;
   on vérifie que ses lignes non supprimées redonnent exactement les classes
   et les textes du CSV curé ;
3. `build_entity_tree.py` produit `*.annotated.merged.csv`, dont les entités
   sont rapprochées de l'ancien `*.curated.merged.csv` (même type et même
   texte, dans l'ordre ; à défaut même ligne racine, quand le CSV curé a été
   retouché après la dernière fusion) : d'où la correspondance ancien →
   nouvel uuid ;
4. le NER (curation pas commencée : `ner.curated.csv` = `ner.csv`) est
   recopié sur les nouvelles entités dans `*.annotated.merged.ner.csv` ; une
   entité dont le texte a changé depuis l'ancienne fusion n'a plus de NER
   valide : ses colonnes NER restent vides (à repasser au modèle).

Enfin les uuid sont remplacés dans les patchs d'alignement versionnés
(`data/alignement/*.csv`) et dans les sorties d'alignement
(`annuaires/alignements/*.csv`, ancienne version gardée en
`*.avant-migration.csv`). Les anciens fichiers `*.curated*` ne sont ni
modifiés ni supprimés.

Sans `--ecrire`, rien n'est écrit dans `annuaires/` ni `data/` : tout est
produit dans un dossier temporaire et seules les vérifications sont
affichées.
"""

import argparse
import difflib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from build_entity_tree import process_csv
from export_lines_csv import CLASS_COLUMN, CSV_FIELDS, KEY_COLUMN, LINES_STEP, iter_csv_rows
from infer_gliner import NER_STEP, NEW_COLUMNS
from lib.curation import (
    AFTER_COLUMN,
    CORRECTED_COLUMN,
    DELETED_CLASS,
    FINGERPRINT_COLUMN,
    INSERTED_SEPARATOR,
    Curation,
    fingerprint,
    normalize_text,
    read_csv,
    write_csv,
)

console = Console()

OLD_CURATED = ".ocr.lines.annotated.curated.csv"
OLD_MERGED = ".ocr.lines.annotated.curated.merged.csv"
OLD_NER = ".ocr.lines.annotated.curated.merged.ner.curated.csv"
LINES = ".ocr.lines.annotated.csv"
MERGED = ".ocr.lines.annotated.merged.csv"
NER = ".ocr.lines.annotated.merged.ner.csv"
BACKUP = ".avant-migration.csv"


class MigrationError(Exception):
    pass


def match_rows(machine: list[dict[str, str]], curated: list[dict[str, str]]) -> list[dict[str, str] | None]:
    """Pour chaque ligne curée (dans l'ordre), la ligne machine
    correspondante ou None (ligne ajoutée)."""
    by_uid = {row["uid"]: row for row in machine}
    used: set[int] = set()
    matched: list[dict[str, str] | None] = []
    for row in curated:
        candidate = by_uid.get(row["uid"])
        matched.append(candidate if candidate is not None and id(candidate) not in used else None)
        if matched[-1] is not None:
            used.add(id(candidate))
    # Blocs re-segmentés : même page, même texte, unique parmi les restantes.
    remaining: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in machine:
        if id(row) not in used:
            remaining[(row["page_index"], normalize_text(row["markdown"]))].append(row)
    wanted = Counter(
        (row["page_index"], normalize_text(row["markdown"])) for row, match in zip(curated, matched) if match is None
    )
    for index, (row, match) in enumerate(zip(curated, matched)):
        key = (row["page_index"], normalize_text(row["markdown"]))
        if match is None and len(remaining.get(key, [])) == 1 and wanted[key] == 1:
            matched[index] = remaining.pop(key)[0]
            used.add(id(matched[index]))
    return matched


def keep_in_order(machine: list[dict[str, str]], matched: list[dict[str, str] | None]) -> int:
    """Détache (None) les correspondances qui sortent de la plus longue
    sous-suite croissante dans l'ordre machine : lignes déplacées. Retourne
    leur nombre."""
    position = {id(row): index for index, row in enumerate(machine)}
    ranks = [(index, position[id(match)]) for index, match in enumerate(matched) if match is not None]
    tails: list[int] = []  # rang dans `ranks` de la fin de chaque sous-suite
    parent = [-1] * len(ranks)
    for k, (_, value) in enumerate(ranks):
        low, high = 0, len(tails)
        while low < high:
            middle = (low + high) // 2
            if ranks[tails[middle]][1] < value:
                low = middle + 1
            else:
                high = middle
        parent[k] = tails[low - 1] if low else -1
        tails[low:low + 1] = [k]
    kept: set[int] = set()
    k = tails[-1] if tails else -1
    while k >= 0:
        kept.add(k)
        k = parent[k]
    moved = 0
    for k, (index, _) in enumerate(ranks):
        if k not in kept:
            matched[index] = None
            moved += 1
    return moved


def line_corrections(machine: list[dict[str, str]], curated: list[dict[str, str]]) -> tuple[list[dict[str, str]], Counter]:
    """Patch des lignes, dans l'ordre final du fichier, et ses comptes."""
    matched = match_rows(machine, curated)
    moved = keep_in_order(machine, matched)
    position = {id(row): index for index, row in enumerate(machine)}
    # Ordre final : lignes machine dans leur ordre, chaque ligne ajoutée juste
    # après la ligne curée qui la précède.
    after: dict[int, list[dict[str, str]]] = defaultdict(list)
    previous_machine = -1
    taken = {row[KEY_COLUMN] for row in machine}
    previous_key = ""
    plan: list[tuple[dict[str, str], dict[str, str] | None]] = []  # (ligne curée, ligne machine)
    for row, match in zip(curated, matched):
        if match is None:
            if previous_machine < 0:
                raise MigrationError(f"ligne ajoutée avant toute ligne machine : {row['uid']}")
            after[previous_machine].append(row)
        else:
            if position[id(match)] < previous_machine:
                raise MigrationError(f"ordre des lignes modifié à la curation : {row['uid']}")
            previous_machine = position[id(match)]
    matched_of = {id(match): row for row, match in zip(curated, matched) if match is not None}

    corrections: list[dict[str, str]] = []
    counts: Counter = Counter()
    for index, machine_row in enumerate(machine):
        key = machine_row[KEY_COLUMN]
        row = matched_of.get(id(machine_row))
        if row is None:
            corrections.append({KEY_COLUMN: key, AFTER_COLUMN: "", "uid": machine_row["uid"], CLASS_COLUMN: DELETED_CLASS, "markdown": machine_row["markdown"]})
            counts["supprimées"] += 1
        elif (row["prediction_curated"], row["markdown"]) != (machine_row[CLASS_COLUMN], machine_row["markdown"]):
            corrections.append({KEY_COLUMN: key, AFTER_COLUMN: "", "uid": machine_row["uid"], CLASS_COLUMN: row["prediction_curated"], "markdown": row["markdown"]})
            counts["corrigées"] += 1
        previous_key = key
        for added in after[index]:
            base, number = previous_key.split(INSERTED_SEPARATOR, 1)[0], 1
            while f"{base}{INSERTED_SEPARATOR}{number}" in taken:
                number += 1
            new_key = f"{base}{INSERTED_SEPARATOR}{number}"
            taken.add(new_key)
            corrections.append({KEY_COLUMN: new_key, AFTER_COLUMN: previous_key, "uid": added["uid"], CLASS_COLUMN: added["prediction_curated"], "markdown": added["markdown"]})
            counts["ajoutées"] += 1
            previous_key = new_key
    counts["déplacées"] = moved
    counts["par texte"] = sum(1 for row, match in zip(curated, matched) if match is not None and match["uid"] != row["uid"])
    return corrections, counts


def entities(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(row["entity"], normalize_text(row["markdown"])) for row in rows]


def match_entities(old: list[dict[str, str]], new: list[dict[str, str]]) -> list[tuple[int, int, bool]]:
    """(rang ancien, rang nouveau, même texte) des entités rapprochées : même
    (type, texte) dans l'ordre, sinon même ligne racine (`uid`) dans un
    passage qui diffère."""
    pairs: list[tuple[int, int, bool]] = []
    matcher = difflib.SequenceMatcher(None, entities(old), entities(new), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs += [(i1 + k, j1 + k, True) for k in range(i2 - i1)]
            continue
        roots = Counter(new[j]["uid"].split(",")[0] for j in range(j1, j2))
        by_root = {new[j]["uid"].split(",")[0]: j for j in range(j1, j2) if roots[new[j]["uid"].split(",")[0]] == 1}
        for i in range(i1, i2):
            j = by_root.pop(old[i]["uid"].split(",")[0], None)
            if j is not None and old[i]["entity"] == new[j]["entity"]:
                pairs.append((i, j, False))
    return pairs


def migrate_document(range_dir: Path, name: str, out_dir: Path, patch_dir: Path) -> tuple[dict[str, str], Counter]:
    """Migre un document ; retourne (ancien uuid → nouveau, comptes)."""
    base = range_dir / name
    document = json.loads(Path(f"{base}.ocr.lines.annotated.json").read_text(encoding="utf-8"))
    machine = list(iter_csv_rows(document))
    _, curated = read_csv(Path(f"{base}{OLD_CURATED}"))
    corrections, counts = line_corrections(machine, curated)

    lines_path = out_dir / f"{name}{LINES}"
    patch_file = patch_dir / f"{name}.lignes.patch.csv"
    curation = Curation(LINES_STEP, lines_path, patch_file, capture=False)
    curation.corrections = corrections
    rows = curation.apply(machine)
    curation.save_patch()
    write_csv(lines_path, CSV_FIELDS, rows)

    kept = [(row[CLASS_COLUMN], row["markdown"]) for row in rows if row[CLASS_COLUMN] != DELETED_CLASS]
    expected = [(row["prediction_curated"], row["markdown"]) for row in curated]
    if kept != expected:
        first = next(i for i, (a, b) in enumerate(zip(kept, expected + [None] * len(kept))) if a != b)
        raise MigrationError(f"{name} : lignes migrées ≠ CSV curé (rang {first} : {kept[first]!r} ≠ {expected[first] if first < len(expected) else None!r})")

    merged_path = out_dir / f"{name}{MERGED}"
    process_csv(lines_path, merged_path, CLASS_COLUMN, "uid", KEY_COLUMN)
    _, new_merged = read_csv(merged_path)
    _, old_merged = read_csv(Path(f"{base}{OLD_MERGED}"))
    _, old_ner = read_csv(Path(f"{base}{OLD_NER}"))
    if entities(old_ner) != entities(old_merged):
        raise MigrationError(f"{name} : le CSV NER ne suit pas l'ancien fichier fusionné")
    pairs = match_entities(old_merged, new_merged)
    mapping = {old_merged[i]["uuid"]: new_merged[j]["uuid"] for i, j, _ in pairs}
    if len(set(mapping.values())) != len(mapping) or len({row["uuid"] for row in new_merged}) != len(new_merged):
        raise MigrationError(f"{name} : nouveaux uuid non uniques")

    ner_of = {j: old_ner[i] for i, j, same_text in pairs if same_text}
    merged_fields = list(new_merged[0].keys())
    position = merged_fields.index("entity") + 1
    ner_fields = [*merged_fields[:position], *NEW_COLUMNS, *merged_fields[position:], FINGERPRINT_COLUMN]
    ner_rows = []
    for j, new in enumerate(new_merged):
        old = ner_of.get(j, {})
        row = {**new, **{column: old.get(column, "") for column in NEW_COLUMNS}, CORRECTED_COLUMN: ""}
        row[FINGERPRINT_COLUMN] = fingerprint(row, NER_STEP.fields)
        ner_rows.append(row)
        if not old and new["entity"] == "ENTRY":
            counts["entrées à repasser au NER"] += 1
            console.print(f"  NER à refaire : {new['uid']} · {new['markdown'][:70]}", markup=False)
    write_csv(out_dir / f"{name}{NER}", ner_fields, ner_rows)
    counts["entités"] = len(new_merged)
    counts["anciens uuid sans successeur"] = len(old_merged) - len(mapping)
    return mapping, counts


def remap_csv(path: Path, mapping: dict[str, str], target: Path) -> int:
    fieldnames, rows = read_csv(path)
    changed = 0
    for row in rows:
        for column in fieldnames:
            if column.endswith("uuid") and row.get(column) in mapping:
                row[column] = mapping[row[column]]
                changed += 1
    write_csv(target, fieldnames, rows)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annuaires", type=Path, default=Path("annuaires"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--ecrire", action="store_true", help="Écrire le résultat en place (sinon : essai à blanc).")
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="migration-curation-"))
    mapping: dict[str, str] = {}
    produced: list[tuple[Path, Path]] = []  # (fichier produit, destination)
    try:
        for curated in sorted(args.annuaires.glob(f"*/*/*{OLD_CURATED}")):
            name = curated.name.removesuffix(OLD_CURATED)
            out_dir = scratch / curated.parent.relative_to(args.annuaires)
            out_dir.mkdir(parents=True, exist_ok=True)
            document_mapping, counts = migrate_document(curated.parent, name, out_dir, scratch / "curation")
            mapping |= document_mapping
            console.print(f"[green]✓[/green] {name} : " + ", ".join(f"{count} {label}" for label, count in counts.items()))
            for suffix in (LINES, MERGED, NER):
                produced.append((out_dir / f"{name}{suffix}", curated.parent / f"{name}{suffix}"))
            produced.append((scratch / "curation" / f"{name}.lignes.patch.csv", args.data / "curation" / f"{name}.lignes.patch.csv"))
        remapped = [
            *sorted((args.data / "alignement").glob("*.csv")),
            *sorted((args.annuaires / "alignements").glob("*.csv")),
        ]
        for path in remapped:
            if path.name.endswith(BACKUP):
                continue
            target = scratch / "uuid" / path.name
            target.parent.mkdir(exist_ok=True)
            changed = remap_csv(path, mapping, target)
            console.print(f"uuid remplacés dans {path} : {changed}")
            produced.append((target, path))
    except MigrationError as error:
        console.print(f"[bold red]Migration impossible :[/bold red] {error}")
        sys.exit(1)

    if not args.ecrire:
        console.print(f"\nEssai à blanc : résultat dans [yellow]{scratch}[/yellow] (relancer avec --ecrire).")
        return
    for source, destination in produced:
        if not source.exists():
            continue
        if destination.exists() and (destination.name.endswith(LINES) or destination.parent.name == "alignements"):
            backup = destination.with_name(destination.name.removesuffix(".csv") + BACKUP)
            if not backup.exists():
                shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    console.print(f"[bold green]✅ Migration écrite[/bold green] ({len(produced)} fichiers).")


if __name__ == "__main__":
    main()

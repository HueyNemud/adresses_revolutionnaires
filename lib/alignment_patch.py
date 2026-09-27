"""Corrections manuelles d'un alignement : le fichier patch.

`align_directories.py` peut être relancé à chaque amélioration des étapes
amont ; les décisions humaines vivent donc à part, dans
`data/alignement/<gauche>__<droite>.patch.csv` (versionné), et sont
réappliquées après chaque inférence. Une ligne du patch est :

- une **paire** (`left_uuid` et `right_uuid`) : ces deux entrées se
  correspondent ;
- un **uuid seul** (l'autre vide) : cette entrée n'a pas de correspondance.

Le patch gagne : tout lien Dedupe qui touche un uuid du patch est écarté,
puis les paires du patch sont ajoutées (source `manuel`). Valider une paire
trouvée par Dedupe la protège des relances.

Rubrique et texte balisé sont recopiés dans le patch pour la relecture et
pour le **réancrage** : si une étape amont re-segmente une entrée, son uuid
change ; on cherche alors l'entrée unique de même texte normalisé dans la
même rubrique. Faute de candidat unique, la ligne est **orpheline** :
signalée, jamais appliquée.
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from lib.alignment import SOURCE_MANUAL, Link, Record, clean_text, clean_title, display_text
from lib.ner.spans import normalize_markdown, parse_tagged_text

SIDES = ("left", "right")


@dataclass(frozen=True)
class PatchEntry:
    left_file: str = ""
    left_uuid: str = ""  # vide : pas de décision sur ce côté
    right_uuid: str = ""
    right_file: str = ""
    left_section: str = ""
    right_section: str = ""
    left_tagged_text: str = ""
    right_tagged_text: str = ""
    note: str = ""

    def uuids(self) -> list[tuple[str, str]]:
        """(côté, uuid) renseignés."""
        return [(side, getattr(self, f"{side}_uuid")) for side in SIDES if getattr(self, f"{side}_uuid")]

    @property
    def is_pair(self) -> bool:
        return bool(self.left_uuid and self.right_uuid)


PATCH_FIELDS = [f.name for f in fields(PatchEntry)]


@dataclass
class Resolution:
    entries: list[PatchEntry]  # lignes applicables (uuid présents dans les annuaires)
    reanchored: list[tuple[PatchEntry, PatchEntry]] = field(default_factory=list)  # (avant, après)
    orphans: list[PatchEntry] = field(default_factory=list)


@dataclass
class PatchStats:
    overridden: int = 0  # liens Dedupe écartés par le patch
    manual_pairs: int = 0
    unmatched_left: int = 0  # uuid de gauche déclarés sans correspondance
    unmatched_right: int = 0


# ----------------------------------------------------------------------
# Lecture / écriture
# ----------------------------------------------------------------------
def read_patch(path: Path) -> list[PatchEntry]:
    """Lignes du patch, dans l'ordre du fichier (vide s'il n'existe pas)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            PatchEntry(**{name: (row.get(name) or "").strip() for name in PATCH_FIELDS})
            for row in csv.DictReader(handle)
        ]


def write_patch(path: Path, entries: list[PatchEntry]) -> None:
    """Écriture atomique (fichier temporaire puis renommage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATCH_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({name: getattr(entry, name) for name in PATCH_FIELDS})
    os.replace(temporary, path)


def validate(entries: list[PatchEntry]) -> None:
    """Lève ValueError si une ligne n'a aucun uuid ou si un uuid apparaît
    dans plusieurs lignes (numéros de ligne du CSV, en-tête = 1)."""
    problems = [f"ligne {number} : aucun uuid" for number, entry in enumerate(entries, start=2) if not entry.uuids()]
    seen: dict[tuple[str, str], list[int]] = defaultdict(list)
    for number, entry in enumerate(entries, start=2):
        for key in entry.uuids():
            seen[key].append(number)
    problems += [
        f"{side} {uuid} présent aux lignes {', '.join(map(str, numbers))}"
        for (side, uuid), numbers in seen.items()
        if len(numbers) > 1
    ]
    if problems:
        raise ValueError("Patch incohérent :\n- " + "\n- ".join(problems))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def entry_from_records(left: Record | None, right: Record | None, note: str = "") -> PatchEntry:
    """Ligne de patch pour une paire (deux entrées) ou une entrée sans
    correspondance (l'autre à None), avec l'instantané texte + rubrique."""
    values = {"note": note}
    for side, record in (("left", left), ("right", right)):
        if record is not None:
            values |= {
                f"{side}_file": record.document,
                f"{side}_uuid": record.uuid,
                f"{side}_section": record.section_title,
                f"{side}_tagged_text": display_text(record),
            }
    return PatchEntry(**values)


def without_uuids(entries: list[PatchEntry], left_uuids: set[str], right_uuids: set[str]) -> list[PatchEntry]:
    """Retire les lignes qui touchent l'un de ces uuid (avant d'en ajouter une nouvelle)."""
    return [entry for entry in entries if entry.left_uuid not in left_uuids and entry.right_uuid not in right_uuids]


# ----------------------------------------------------------------------
# Réancrage
# ----------------------------------------------------------------------
def anchor_key(section: str, tagged_text: str) -> tuple[str, str]:
    """(rubrique, texte) normalisés comme dans `lib/alignment.py`."""
    try:
        text, _ = parse_tagged_text(tagged_text)
    except ValueError:
        text = tagged_text
    return clean_title(section), clean_text(normalize_markdown(text).text)


def _index(records: dict[str, Record]) -> dict[tuple[str, str], list[Record]]:
    index: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records.values():
        index[(record.section, record.text)].append(record)
    return index


def resolve(entries: list[PatchEntry], left: dict[str, Record], right: dict[str, Record]) -> Resolution:
    """Vérifie que les uuid du patch existent ; réancre par (rubrique, texte)
    ceux qui ont disparu ; met à jour l'instantané des autres."""
    records = {"left": left, "right": right}
    indexes: dict[str, dict] = {}
    used = {(side, uuid) for entry in entries for side, uuid in entry.uuids() if uuid in records[side]}
    resolution = Resolution(entries=[])
    for entry in entries:
        updated, moved, orphan = entry, False, False
        for side, uuid in entry.uuids():
            record = records[side].get(uuid)
            if record is None:
                if side not in indexes:
                    indexes[side] = _index(records[side])
                key = anchor_key(getattr(entry, f"{side}_section"), getattr(entry, f"{side}_tagged_text"))
                candidates = [c for c in indexes[side].get(key, []) if (side, c.uuid) not in used]
                if len(candidates) != 1:
                    orphan = True
                    break
                record, moved = candidates[0], True
                used.add((side, record.uuid))
            updated = replace(
                updated,
                **{
                    f"{side}_file": record.document,
                    f"{side}_uuid": record.uuid,
                    f"{side}_section": record.section_title,
                    f"{side}_tagged_text": display_text(record),
                },
            )
        if orphan:
            resolution.orphans.append(entry)
            continue
        if moved:
            resolution.reanchored.append((entry, updated))
        resolution.entries.append(updated)
    return resolution


def updated_patch(entries: list[PatchEntry], resolution: Resolution) -> list[PatchEntry]:
    """Le patch à réécrire : lignes réancrées remplacées, orphelines gardées
    telles quelles (à corriger à la main), ordre conservé."""
    after = {id(before): new for before, new in resolution.reanchored}
    return [after.get(id(entry), entry) for entry in entries]


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------
def apply_patch(links: list[Link], entries: list[PatchEntry]) -> tuple[list[Link], PatchStats]:
    """Liens Dedupe + patch (voir la docstring du module). Une paire validée
    telle que Dedupe l'avait trouvée garde son score."""
    stats = PatchStats()
    patched_left = {entry.left_uuid for entry in entries if entry.left_uuid}
    patched_right = {entry.right_uuid for entry in entries if entry.right_uuid}
    scores = {(link.left_uuid, link.right_uuid): link.score for link in links}
    manual = [
        Link(entry.left_uuid, entry.right_uuid, scores.get((entry.left_uuid, entry.right_uuid)), SOURCE_MANUAL)
        for entry in entries
        if entry.is_pair
    ]
    kept = [link for link in links if link.left_uuid not in patched_left and link.right_uuid not in patched_right]
    validated = sum(1 for link in manual if (link.left_uuid, link.right_uuid) in scores)
    stats.overridden = len(links) - len(kept) - validated  # une paire validée n'est pas écartée
    stats.manual_pairs = len(manual)
    stats.unmatched_left = sum(1 for entry in entries if entry.left_uuid and not entry.right_uuid)
    stats.unmatched_right = sum(1 for entry in entries if entry.right_uuid and not entry.left_uuid)
    return kept + manual, stats

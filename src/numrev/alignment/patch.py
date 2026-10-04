"""Corrections manuelles d'un alignement : le fichier patch.

`numrev align dedupe` peut être relancé à chaque amélioration des étapes
amont ; les décisions humaines vivent donc à part, dans
`data/alignment/<gauche>__<droite>.patch.csv` (versionné), et sont
réappliquées après chaque inférence. Une ligne du patch est :

- une **paire** (`left_uuid` et `right_uuid`) : ces deux entrées se
  correspondent ;
- un **uuid seul** (l'autre vide) : cette entrée n'a pas de correspondance.

Le patch gagne : tout lien Dedupe qui touche un uuid du patch est écarté,
puis les paires du patch sont ajoutées (source `manuel`). Valider une paire
trouvée par Dedupe la protège des relances.

La colonne facultative `certitude` vaut `incertaine` quand la relecture n'a
pas permis de trancher (homonymes, graphies trop éloignées…) : la paire est
appliquée, mais avec la source `manuel-incertain`, qui reste visible dans
l'export ; pour un uuid seul, « probablement sans correspondance »
(appliqué comme les autres). Un patch sans cette colonne reste valide.

Rubrique et texte balisé sont recopiés dans le patch pour la relecture et
pour le **réancrage** : si une étape amont re-segmente une entrée, son uuid
change ; on cherche alors l'entrée unique de même texte normalisé dans la
même rubrique. Faute de candidat unique, la ligne est **orpheline** :
jamais appliquée, et les scripts qui produisent un alignement paniquent
(`--force` pour l'ignorer), comme pour les autres étapes curées
(numrev/curation.py).
"""

import csv
import io
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from numrev.alignment.records import SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, Link, Record, clean_text, clean_title, display_text
from numrev.curation import read_dataclasses, write_csv
from numrev.ner.spans import normalize_markdown, parse_tagged_text

SIDES = ("left", "right")
UNCERTAIN = "incertaine"


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
    certitude: str = ""  # vide : décision sûre ; UNCERTAIN : relecture sans conclusion ferme

    def uuids(self) -> list[tuple[str, str]]:
        """(côté, uuid) renseignés."""
        return [(side, getattr(self, f"{side}_uuid")) for side in SIDES if getattr(self, f"{side}_uuid")]

    @property
    def is_pair(self) -> bool:
        return bool(self.left_uuid and self.right_uuid)

    @property
    def is_uncertain(self) -> bool:
        return self.certitude.lower() == UNCERTAIN


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
    return read_dataclasses(path, PatchEntry)


def write_patch(path: Path, entries: list[PatchEntry]) -> None:
    """Écriture atomique (numrev/curation.py)."""
    write_csv(path, PATCH_FIELDS, (asdict(entry) for entry in entries))


def patch_csv(entries: list[PatchEntry]) -> str:
    """Le patch en texte CSV, comme `write_patch` l'écrirait (téléchargement du viewer)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PATCH_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(asdict(entry) for entry in entries)
    return buffer.getvalue()


def validate(entries: list[PatchEntry]) -> None:
    """Lève ValueError si une ligne n'a aucun uuid, si sa `certitude` n'est
    ni vide ni `incertaine`, ou si un uuid apparaît dans plusieurs lignes
    (numéros de ligne du CSV, en-tête = 1)."""
    problems = [f"ligne {number} : aucun uuid" for number, entry in enumerate(entries, start=2) if not entry.uuids()]
    problems += [
        f"ligne {number} : certitude « {entry.certitude} » (attendu : vide ou « {UNCERTAIN} »)"
        for number, entry in enumerate(entries, start=2)
        if entry.certitude and not entry.is_uncertain
    ]
    seen: dict[tuple[str, str], list[int]] = defaultdict(list)
    for number, entry in enumerate(entries, start=2):
        for key in entry.uuids():
            seen[key].append(number)
    problems += [
        f"{side} {uuid} présent aux lignes {', '.join(map(str, numbers))}" for (side, uuid), numbers in seen.items() if len(numbers) > 1
    ]
    if problems:
        raise ValueError("Patch incohérent :\n- " + "\n- ".join(problems))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def entry_from_records(left: Record | None, right: Record | None, note: str = "", certitude: str = "") -> PatchEntry:
    """Ligne de patch pour une paire (deux entrées) ou une entrée sans
    correspondance (l'autre à None), avec l'instantané texte + rubrique."""
    values = {"note": note, "certitude": certitude}
    for side, record in (("left", left), ("right", right)):
        if record is not None:
            values |= {
                f"{side}_file": record.document,
                f"{side}_uuid": record.uuid,
                f"{side}_section": record.section_title,
                f"{side}_tagged_text": display_text(record),
            }
    return PatchEntry(**values)


# ----------------------------------------------------------------------
# Réancrage
# ----------------------------------------------------------------------
def anchor_key(section: str, tagged_text: str) -> tuple[str, str]:
    """(rubrique, texte) normalisés comme dans `numrev/alignment/records.py`."""
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


def load_patch(path: Path, left: dict[str, Record], right: dict[str, Record]) -> tuple[list[PatchEntry], Resolution]:
    """Lit, valide (ValueError) et résout le patch, avant tout calcul."""
    entries = read_patch(path)
    validate(entries)
    return entries, resolve(entries, left, right)


def declared_unmatched(resolution: Resolution) -> set[str]:
    """uuid déclarés sans correspondance par le patch (jamais proposés comme candidates)."""
    return {uuid for entry in resolution.entries if not entry.is_pair for _, uuid in entry.uuids()}


def orphan_descriptions(resolution: Resolution) -> list[str]:
    """Lignes orphelines, lisibles (côté, uuid, rubrique, texte)."""
    return [
        " ↔ ".join(
            f"{side} {getattr(entry, f'{side}_uuid')} · {getattr(entry, f'{side}_section')} · {getattr(entry, f'{side}_tagged_text')}"
            for side, _ in entry.uuids()
        )
        for entry in resolution.orphans
    ]


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
    telle que Dedupe l'avait trouvée garde son score ; une paire marquée
    incertaine prend la source `manuel-incertain`."""
    stats = PatchStats()
    patched_left = {entry.left_uuid for entry in entries if entry.left_uuid}
    patched_right = {entry.right_uuid for entry in entries if entry.right_uuid}
    scores = {(link.left_uuid, link.right_uuid): link.score for link in links}
    manual = [
        Link(
            entry.left_uuid,
            entry.right_uuid,
            scores.get((entry.left_uuid, entry.right_uuid)),
            SOURCE_MANUAL_UNCERTAIN if entry.is_uncertain else SOURCE_MANUAL,
        )
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

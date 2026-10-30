"""Corrections humaines rejouables : le protocole commun aux étapes curées
(guide pour les curateurs : `docs/guide_curation.md`).

La sortie d'une étape curée (`<document>.lines.csv`, `<document>.ner.csv`)
est à la fois la sortie machine et le fichier corrigé à la main. Chaque
ligne y porte une clé stable (`cle` d'une ligne, `uuid` d'une entité), des
champs éditables, `corrige` (`oui` : corrigée, ou validée telle quelle) et
`empreinte` (hash des champs éditables tels que la machine les a écrits).

Une exécution réussit entièrement ou n'écrit rien (`CurationConflict`) :

1. **capture** : les lignes `corrige = oui` du fichier existant réécrivent le
   patch versionné `data/curation/<document>.<étape>.patch.csv` ; une ligne
   modifiée sans `corrige` fait paniquer ;
2. **génération** de la sortie machine, par l'étape ;
3. **application** du patch, qui gagne ; une correction qui ne s'applique
   plus (clé disparue, vérification de l'étape refusée) fait paniquer.

`--force` prend la sortie machine pour les lignes en cause et retire du
patch les corrections inapplicables ; `--no-capture` repart du patch
versionné sans lire le fichier existant.

Étape des lignes seulement : une ligne ajoutée (sans clé, ou dupliquée)
reçoit la clé `<clé précédente>+n` (`debut+n` en tête de fichier), et le
patch retient sa place (`apres`) ; elle reprend les colonnes de sa voisine,
sauf `Step.machine_only`, laissées vides. Une ligne machine effacée ou
déplacée fait paniquer : on supprime une ligne en lui donnant la classe
`SUPPRIMÉE`, on la déplace par `SUPPRIMÉE` + une copie sans clé.
"""

import csv
import hashlib
import os
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path

from numrev.command import Writes, write_or_plan
from numrev.paths import curation_patch

CORRECTED_COLUMN = "corrige"
FINGERPRINT_COLUMN = "empreinte"
AFTER_COLUMN = "apres"
CORRECTED = "oui"
DELETED_CLASS = "SUPPRIMÉE"  # classe d'une ligne supprimée à la main (étape des lignes)
INSERTED_SEPARATOR = "+"
START_KEY = "debut"  # base de clé d'une ligne ajoutée avant la première ligne du fichier
OCCURRENCE_SEPARATOR = "~"
HASH_LENGTH = 12


class CurationConflict(Exception):
    """Corrections humaines inapplicables, ou modifications non marquées."""

    def __init__(self, problems: list[str], hint: str = "") -> None:
        self.problems = problems
        self.hint = hint
        super().__init__(f"{len(problems)} problème(s) de curation")


# ----------------------------------------------------------------------
# Clés et empreintes
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalisation figée : la changer changerait toutes les clés déjà
    produites (forme NFC, espaces insécables, blancs regroupés)."""
    return " ".join(unicodedata.normalize("NFC", text).replace("\xa0", " ").split())


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def line_keys(texts: Iterable[str]) -> list[str]:
    """Clé de chaque ligne OCR, dans l'ordre du document : hash du texte
    normalisé, suffixé `~n` à partir de sa 2ᵉ occurrence (les lignes vides,
    séparateurs et titres répétés sont fréquents). Ne dépend ni de la page ni
    du découpage en blocs."""
    seen: dict[str, int] = {}
    keys = []
    for text in texts:
        digest = short_hash(normalize_text(text))
        seen[digest] = seen.get(digest, 0) + 1
        keys.append(digest if seen[digest] == 1 else f"{digest}{OCCURRENCE_SEPARATOR}{seen[digest]}")
    return keys


def fingerprint(row: dict[str, str], fields: Iterable[str]) -> str:
    """Empreinte des champs éditables (blancs normalisés : un tableur qui
    retire une espace de fin n'est pas une correction)."""
    return short_hash("\x1f".join(normalize_text(row.get(name) or "") for name in fields))


def is_corrected(row: dict[str, str]) -> bool:
    return (row.get(CORRECTED_COLUMN) or "").strip().lower() == CORRECTED


def is_inserted(key: str) -> bool:
    return INSERTED_SEPARATOR in key


# ----------------------------------------------------------------------
# Fichiers
# ----------------------------------------------------------------------
def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_dataclasses(path: Path, cls: type) -> list:
    """Lignes d'un CSV en instances de la dataclass `cls` (une colonne par
    champ, blancs de bord retirés), dans l'ordre du fichier ; vide si le
    fichier n'existe pas. Sert aux patchs édités à la main."""
    if not path.exists():
        return []
    names = [field.name for field in fields(cls)]
    return [cls(**{name: (row.get(name) or "").strip() for name in names}) for row in read_csv(path)[1]]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    """Écriture atomique (fichier temporaire puis renommage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


# ----------------------------------------------------------------------
# Protocole
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """Ce qu'une étape déclare : nom (dans le nom du patch), colonne clé,
    champs éditables, colonnes recopiées dans le patch pour la relecture,
    si l'on peut y ajouter des lignes à la main (et donc détecter celles
    qu'on y a effacées), et les colonnes propres à chaque ligne machine,
    vides sur une ligne ajoutée (qui reprend les autres de sa voisine)."""

    name: str
    key: str
    fields: tuple[str, ...]
    context: tuple[str, ...] = ()
    insertions: bool = False
    machine_only: tuple[str, ...] = ()

    @property
    def patch_fields(self) -> list[str]:
        return [self.key, *([AFTER_COLUMN] if self.insertions else []), *self.context, *self.fields]


# Vérification propre à une étape : message si la correction ne s'applique
# pas à la nouvelle ligne machine (ou à la ligne ajoutée reconstruite), None sinon.
Check = Callable[[dict[str, str], dict[str, str]], str | None]


@dataclass
class Report:
    captured: bool = False  # le fichier existant a été lu
    corrections: int = 0
    inserted: int = 0
    unmarked_discarded: list[str] = field(default_factory=list)  # modifiées sans `corrige`, --force
    dropped: list[str] = field(default_factory=list)  # corrections inapplicables, --force
    removed: list[str] = field(default_factory=list)  # clés retirées du patch par rapport à l'ancien


class Curation:
    """Une exécution du protocole pour un fichier de sortie : `capture` (à
    la création), puis `apply` sur la sortie machine, puis `save_patch`."""

    def __init__(self, step: Step, output_path: Path, patch_file: Path | None = None, *, force: bool = False, capture: bool = True) -> None:
        """`patch_file` : par défaut, le patch versionné du document
        (`numrev.paths.curation_patch`)."""
        self.step = step
        self.output_path = output_path
        self.patch_file = patch_file or curation_patch(output_path, step.name)
        self.force = force
        self.report = Report()
        self.old_patch = read_csv(self.patch_file)[1] if self.patch_file.exists() else []
        self.corrections = self.old_patch
        self.previous_keys: set[str] | None = None
        self.previous_order: list[str] = []  # clés machine, dans l'ordre du fichier existant
        if capture and output_path.exists():
            self._capture(read_csv(output_path))

    # -- 1. capture ----------------------------------------------------
    def _capture(self, table: tuple[list[str], list[dict[str, str]]]) -> None:
        fieldnames, rows = table
        step = self.step
        missing = [name for name in (step.key, CORRECTED_COLUMN, FINGERPRINT_COLUMN, *step.fields) if name not in fieldnames]
        if missing:
            raise CurationConflict(
                [f"{self.output_path} : colonne(s) absente(s) : {', '.join(missing)}"],
                "ce fichier n'est pas une sortie de cette étape : relancez avec --no-capture pour repartir du patch (il sera écrasé).",
            )
        problems: list[str] = []
        all_keys = {(row.get(step.key) or "").strip() for row in rows}
        seen: set[str] = set()
        corrections: list[dict[str, str]] = []
        previous = ""
        for number, row in enumerate(rows, start=2):
            key = (row.get(step.key) or "").strip()
            if not key or key in seen:
                if not step.insertions:
                    problems.append(f"ligne {number} : clé {'vide' if not key else 'en double : ' + key}")
                    continue
                key = self._new_key(previous or START_KEY, all_keys)
                all_keys.add(key)
            seen.add(key)
            row[step.key] = key
            if step.insertions and is_inserted(key):
                row[CORRECTED_COLUMN] = CORRECTED
                corrections.append(self._patch_row(row, previous))
            elif is_corrected(row):
                corrections.append(self._patch_row(row, previous))
            elif fingerprint(row, step.fields) != (row.get(FINGERPRINT_COLUMN) or "").strip():
                if self.force:
                    self.report.unmarked_discarded.append(key)
                else:
                    problems.append(f"ligne {number} ({key}) : modifiée sans « {CORRECTED_COLUMN} = {CORRECTED} »")
            previous = key
        if problems:
            raise CurationConflict(
                problems,
                f"marquez ces lignes « {CORRECTED_COLUMN} = {CORRECTED} » pour garder vos corrections, "
                "ou relancez avec --force pour reprendre la sortie machine.",
            )
        self.report.captured = True
        self.corrections = corrections
        self.previous_keys = seen
        self.previous_order = [key for key in (row[step.key] for row in rows) if not is_inserted(key)]

    @staticmethod
    def _new_key(previous: str, taken: set[str]) -> str:
        base = previous.split(INSERTED_SEPARATOR, 1)[0]
        number = 1
        while f"{base}{INSERTED_SEPARATOR}{number}" in taken:
            number += 1
        return f"{base}{INSERTED_SEPARATOR}{number}"

    def _patch_row(self, row: dict[str, str], previous: str) -> dict[str, str]:
        values = {name: row.get(name) or "" for name in self.step.patch_fields}
        if self.step.insertions:
            values[AFTER_COLUMN] = previous if is_inserted(values[self.step.key]) else ""
        return values

    # -- 3. application ------------------------------------------------
    def apply(self, machine_rows: list[dict[str, str]], check: Check | None = None) -> list[dict[str, str]]:
        """Sortie machine + patch. Les lignes machine reçoivent leur
        empreinte ; les corrections leurs champs et `corrige = oui`."""
        step = self.step
        by_key: dict[str, dict[str, str]] = {}
        for row in machine_rows:
            row[FINGERPRINT_COLUMN] = fingerprint(row, step.fields)
            row[CORRECTED_COLUMN] = ""
            by_key[row[step.key]] = row

        problems: list[str] = []
        kept: list[dict[str, str]] = []
        inserted_after: dict[str, list[dict[str, str]]] = {}

        def refuse(message: str) -> None:
            """Problème qui fait paniquer, ou correction abandonnée avec --force."""
            (self.report.dropped if self.force else problems).append(message)

        for correction in self.corrections:
            key = correction.get(step.key, "")
            if step.insertions and is_inserted(key):
                after = correction.get(AFTER_COLUMN, "")
                # `apres` vide : en tête du fichier ; la ligne reprend alors les colonnes de la première ligne machine.
                anchor = by_key.get(after) if after else next(iter(machine_rows), None)
                if anchor is None:
                    refuse(
                        f"{key} : "
                        + (f"ligne ajoutée après {after!r}, introuvable" if after else "ligne ajoutée en tête d'un fichier vide")
                    )
                    continue
                row = {**anchor, **dict.fromkeys(step.machine_only, "")}
                row |= {name: correction.get(name, "") for name in (*step.context, *step.fields)}
                row |= {step.key: key, FINGERPRINT_COLUMN: "", CORRECTED_COLUMN: CORRECTED}
                message = check(correction, row) if check else None
                if message:
                    refuse(f"{key} : {message}")
                    continue
                inserted_after.setdefault(after, []).append(row)
                by_key[key] = row
                self.report.inserted += 1
            else:
                row = by_key.get(key)
                if row is None:
                    refuse(f"{key} : introuvable dans la nouvelle sortie")
                    continue
                message = check(correction, row) if check else None
                if message:
                    refuse(f"{key} : {message}")
                    continue
                row.update({name: correction.get(name, "") for name in step.fields})
                row[CORRECTED_COLUMN] = CORRECTED
            kept.append(correction)

        if step.insertions and self.previous_order:
            position = {row[step.key]: index for index, row in enumerate(machine_rows)}
            highest = -1
            for key in self.previous_order:
                index = position.get(key)
                if index is None:
                    continue
                if index < highest:
                    refuse(
                        f"{key} (uid {by_key[key].get('uid', '?')}) : ligne déplacée ; pour la déplacer, donnez la "
                        "classe SUPPRIMÉE à l'originale et ajoutez une copie sans clé à la bonne place"
                    )
                highest = max(highest, index)

        if step.insertions and self.previous_keys is not None:
            for row in machine_rows:
                if row[step.key] not in self.previous_keys:
                    refuse(
                        f"{row[step.key]} (uid {row.get('uid', '?')}) : absente du fichier édité ; "
                        "pour supprimer une ligne, donnez-lui la classe SUPPRIMÉE"
                    )

        if problems:
            raise CurationConflict(problems, "corrigez le fichier ou le patch, ou relancez avec --force pour reprendre la sortie machine.")

        self.report.corrections = len(kept)
        new_keys = {correction.get(step.key) for correction in kept}
        self.report.removed = [row.get(step.key, "") for row in self.old_patch if row.get(step.key) not in new_keys]
        self.corrections = kept

        result: list[dict[str, str]] = []

        def emit(row: dict[str, str]) -> None:
            result.append(row)
            for child in inserted_after.get(row[step.key], []):
                emit(child)

        for row in [*inserted_after.pop("", []), *machine_rows]:
            emit(row)
        return result

    def save_patch(self, writes: Writes | None = None) -> None:
        """Réécrit le patch (créé seulement s'il y a des corrections), selon
        `writes` (numrev/command.py ; défaut : écrit)."""
        if self.corrections or self.patch_file.exists():
            write_or_plan(writes, self.patch_file, lambda: write_csv(self.patch_file, self.step.patch_fields, self.corrections))


def refuse_orphans(orphans: list[str], force: bool, what: str) -> None:
    """Patchs d'alignement : une ligne orpheline (uuid disparu, sans
    réancrage possible) fait paniquer comme une correction inapplicable ;
    `--force` la laisse de côté (signalée, non appliquée)."""
    if orphans and not force:
        raise CurationConflict(
            [f"{what} : {orphan}" for orphan in orphans],
            "corrigez le patch à la main, ou relancez avec --force pour ignorer ces lignes.",
        )


def print_report(console, curation: Curation) -> None:
    """Bilan de la curation sur la console `rich`."""
    report = curation.report
    origin = "capturé depuis le fichier existant" if report.captured else "repris tel quel"
    console.print(
        f"Patch [yellow]{curation.patch_file}[/yellow] ({origin}) : {report.corrections} correction(s)"
        + (f", dont {report.inserted} ligne(s) ajoutée(s)" if report.inserted else "")
        + "."
    )
    if report.removed:
        console.print(f"[yellow]{len(report.removed)} correction(s) retirée(s) du patch (voir git diff).[/yellow]")
    for title, items in (
        ("modifiée(s) sans « corrige », remplacée(s) par la sortie machine (--force)", report.unmarked_discarded),
        ("abandonnée(s) (--force)", report.dropped),
    ):
        if items:
            console.print(f"[bold yellow]⚠ {len(items)} ligne(s) {title} :[/bold yellow]")
            for item in items:
                console.print(f"  {item}", markup=False)


def print_conflict(console, error: CurationConflict, limit: int = 30) -> None:
    """Message de panique : rien n'a été écrit."""
    console.print(f"[bold red]✋ Arrêt, rien n'a été écrit : {len(error.problems)} problème(s) de curation.[/bold red]")
    for problem in error.problems[:limit]:
        console.print(f"  {problem}", markup=False)
    if len(error.problems) > limit:
        console.print(f"  … et {len(error.problems) - limit} autre(s).")
    if error.hint:
        console.print(f"[bold]→[/bold] {error.hint}")

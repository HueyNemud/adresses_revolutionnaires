"""Corrections humaines rejouables : le protocole commun aux étapes curées.

Les sorties curées de la chaîne (`<document>.lines.csv`,
`<document>.ner.csv`) sont à la fois la sortie machine et le fichier qu'on
édite à la main (Excel, OpenRefine). Pour pouvoir relancer une étape sans
perdre ces éditions, chaque ligne porte :

- une **clé stable** (`cle` d'une ligne, `uuid` d'une entité) ;
- des **champs éditables** (classe et texte ; balisage NER et titre parent) ;
- `corrige` : `oui` sur une ligne corrigée, ou validée telle quelle (confirmer
  une prédiction est aussi une décision humaine) ;
- `empreinte` : hash des champs éditables tels que la machine les a écrits.

Une commande d'étape enchaîne, et réussit entièrement ou n'écrit rien :

1. **capture** : si sa sortie existe déjà, ses lignes `corrige = oui`
   réécrivent le patch versionné `data/curation/<document>.<étape>.patch.csv`
   (git sert de copie de sécurité). Une ligne modifiée (empreinte qui ne
   correspond plus) mais sans `corrige` fait **paniquer** : oubli, ou
   altération par le tableur. `--force` lui fait prendre la nouvelle sortie
   machine, ce qui est aussi le moyen d'annuler une correction (vider
   `corrige`, relancer avec `--force`) ;
2. **génération** de la sortie machine ;
3. **application** du patch, qui gagne. Toute correction qui ne s'applique
   pas mécaniquement fait paniquer (clé disparue, texte modifié en amont…) ;
   avec `--force`, elle est abandonnée et retirée du patch.

`CurationConflict` porte la liste des problèmes ; rien n'est écrit tant
qu'elle est levée. `--no-capture` ignore le fichier existant et repart du
patch versionné (après un `git pull` qui l'a modifié, par exemple).

Lignes ajoutées à la main (étape des lignes seulement) : une ligne sans clé,
ou qui reprend la clé de la ligne précédente (ligne dupliquée), reçoit à la
capture la clé `<clé de base>+n` ; le patch garde sa position (`apres` = clé
de la ligne qui la précède) et elle est réinsérée à cette place. Une ligne
machine effacée du fichier fait paniquer : on supprime une ligne en lui
donnant la classe `SUPPRIMÉE`, pour que ce soit rejouable. Une ligne machine
déplacée fait aussi paniquer (l'ordre des lignes machine est celui de
l'OCR) : on la déplace en donnant la classe `SUPPRIMÉE` à l'originale et en
ajoutant une copie, sans clé, à la bonne place.
"""

import csv
import hashlib
import os
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path

from numrev.command import Writes
from numrev.paths import curation_patch

CORRECTED_COLUMN = "corrige"
FINGERPRINT_COLUMN = "empreinte"
AFTER_COLUMN = "apres"
CORRECTED = "oui"
DELETED_CLASS = "SUPPRIMÉE"  # classe d'une ligne supprimée à la main (étape des lignes)
INSERTED_SEPARATOR = "+"
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
    champs éditables, colonnes recopiées dans le patch pour la relecture, et
    si l'on peut y ajouter des lignes à la main (et donc détecter celles
    qu'on y a effacées)."""

    name: str
    key: str
    fields: tuple[str, ...]
    context: tuple[str, ...] = ()
    insertions: bool = False

    @property
    def patch_fields(self) -> list[str]:
        return [self.key, *([AFTER_COLUMN] if self.insertions else []), *self.context, *self.fields]


# Vérification propre à une étape : message si la correction ne s'applique
# plus à la nouvelle ligne machine, None sinon.
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

    def __init__(
        self, step: Step, output_path: Path, patch_file: Path | None = None, *, force: bool = False, capture: bool = True
    ) -> None:
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
                if not previous:
                    problems.append(f"ligne {number} : ligne ajoutée avant la première ligne du fichier")
                    continue
                key = self._new_key(previous, all_keys)
                all_keys.add(key)
                row[step.key] = key
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

        def refuse(correction: dict[str, str], message: str) -> None:
            text = f"{correction.get(step.key)} : {message}"
            if self.force:
                self.report.dropped.append(text)
            else:
                problems.append(text)

        for correction in self.corrections:
            key = correction.get(step.key, "")
            if step.insertions and is_inserted(key):
                anchor = by_key.get(correction.get(AFTER_COLUMN, ""))
                if anchor is None:
                    refuse(correction, f"ligne ajoutée après {correction.get(AFTER_COLUMN)!r}, introuvable")
                    continue
                row = {**anchor, **{name: correction.get(name, "") for name in (*step.context, *step.fields)}}
                row |= {step.key: key, FINGERPRINT_COLUMN: "", CORRECTED_COLUMN: CORRECTED}
                inserted_after.setdefault(anchor[step.key], []).append(row)
                by_key[key] = row
                self.report.inserted += 1
            else:
                row = by_key.get(key)
                if row is None:
                    refuse(correction, "introuvable dans la nouvelle sortie")
                    continue
                message = check(correction, row) if check else None
                if message:
                    refuse(correction, message)
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
                    message = (
                        f"{key} (uid {by_key[key].get('uid', '?')}) : ligne déplacée ; pour la déplacer, donnez la "
                        "classe SUPPRIMÉE à l'originale et ajoutez une copie sans clé à la bonne place"
                    )
                    if self.force:
                        self.report.dropped.append(message)
                    else:
                        problems.append(message)
                highest = max(highest, index)

        if step.insertions and self.previous_keys is not None:
            for row in machine_rows:
                if row[step.key] not in self.previous_keys:
                    message = (
                        f"{row[step.key]} (uid {row.get('uid', '?')}) : absente du fichier édité ; "
                        "pour supprimer une ligne, donnez-lui la classe SUPPRIMÉE"
                    )
                    if self.force:
                        self.report.dropped.append(message)
                    else:
                        problems.append(message)

        if problems:
            raise CurationConflict(
                problems, "corrigez le fichier ou le patch, ou relancez avec --force pour reprendre la sortie machine."
            )

        self.report.corrections = len(kept)
        new_keys = {correction.get(step.key) for correction in kept}
        self.report.removed = [row.get(step.key, "") for row in self.old_patch if row.get(step.key) not in new_keys]
        self.corrections = kept

        result: list[dict[str, str]] = []

        def emit(row: dict[str, str]) -> None:
            result.append(row)
            for child in inserted_after.get(row[step.key], []):
                emit(child)

        for row in machine_rows:
            emit(row)
        return result

    def save_patch(self, writes: Writes | None = None) -> None:
        """Réécrit le patch (créé seulement s'il y a des corrections), selon
        `writes` (numrev/command.py ; défaut : écrit)."""
        if self.corrections or self.patch_file.exists():
            (writes or Writes()).add(
                self.patch_file, lambda: write_csv(self.patch_file, self.step.patch_fields, self.corrections)
            )


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

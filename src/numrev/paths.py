"""Noms et emplacements des fichiers : la convention, en un seul endroit.

Un **document** est une plage de pages d'un volume,
`annuaires/<volume>/<plage>/<volume>.<plage>` (les noms de volumes et de
plages n'ont pas de point). Chaque étape ajoute à ce nom le suffixe de sa
sortie :

    <document>.ocr.json            0 · ocr (latynna, depuis annuaires/<volume>/<volume>.pdf)
    <document>.lines.json          1 · extract
    <document>.labeled.json        2 · label (+ <document>.label-session.json)
    <document>.lines.csv           3 · tabulate (corrigé à la main)
    <document>.entities.csv        4 · assemble (+ <document>.entities.report.txt)
    <document>.ner.csv             5 · tag (corrigé à la main)

Un **volume** entier a ses livrables finaux dans son dossier,
`annuaires/<volume>/<volume><suffixe>` :

    <volume>.document.json         6 · export (couche documentaire)
    <volume>.structure.json        6 · export (couche structurée, + <volume>.export.report.txt)

Une **paire** d'annuaires `<gauche>__<droite>` (noms des dossiers de
volumes) a ses sorties dans `annuaires/alignments/` et ses décisions
humaines versionnées dans `data/alignment/`.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ANNUAIRES_DIR = Path("annuaires")
ALIGNMENTS_DIR = ANNUAIRES_DIR / "alignments"
DATA_DIR = Path("data")
CURATION_DIR = DATA_DIR / "curation"
ALIGNMENT_DATA_DIR = DATA_DIR / "alignment"
NER_DATA_DIR = DATA_DIR / "ner"
REPORTS_DIR = Path("reports")
NER_GOLD = NER_DATA_DIR / "gold.ls.json"  # gold NER relu (évaluation)
NER_TRAIN = NER_DATA_DIR / "train.ls.json"  # jeu d'entraînement NER
MODELS_DIR = Path("models")
NER_MODEL = MODELS_DIR / "ner.gliner-model"  # modèle NER de production, utilisé par défaut par numrev tag

OCR = ".ocr.json"
LINES = ".lines.json"
LABELED = ".labeled.json"
LABEL_SESSION = ".label-session.json"
LINES_CSV = ".lines.csv"
ENTITIES = ".entities.csv"
ENTITIES_REPORT = ".entities.report.txt"
NER = ".ner.csv"
STEP_SUFFIXES = (OCR, LINES, LABELED, LABEL_SESSION, LINES_CSV, ENTITIES, ENTITIES_REPORT, NER)
DOCUMENT_JSON = ".document.json"
STRUCTURE_JSON = ".structure.json"
EXPORT_REPORT = ".export.report.txt"

RANGE_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def document_name(path: Path) -> str:
    """« 1808_AD75-PER292.7-186.lines.csv » → « 1808_AD75-PER292.7-186 » :
    même nom à toutes les étapes. ValueError si le fichier ne porte aucun
    suffixe de la chaîne."""
    for suffix in STEP_SUFFIXES:
        if path.name.endswith(suffix) and len(path.name) > len(suffix):
            return path.name.removesuffix(suffix)
    raise ValueError(f"{path.name} : suffixe inconnu (attendu : {', '.join(STEP_SUFFIXES)}).")


def step_path(path: Path, suffix: str) -> Path:
    """Fichier d'une autre étape du même document, à côté de `path`."""
    return path.with_name(document_name(path) + suffix)


def volume_of(document: str) -> str:
    """« 1808_AD75-PER292.7-186 » → « 1808_AD75-PER292 »."""
    return document.split(".", 1)[0]


def page_range(range_dir: Path) -> tuple[int, int]:
    """« …/7-186 » → (7, 186) : première et dernière page de la plage, numéros
    1-based des pages du PDF du volume. ValueError si le nom n'est pas `a-b` avec a ≤ b."""
    match = RANGE_PATTERN.match(range_dir.name)
    if match is None or int(match.group(1)) > int(match.group(2)) or int(match.group(1)) < 1:
        raise ValueError(
            f"{range_dir} : le dossier d'une plage se nomme <première page>-<dernière page> (ex. 7-186, pages comptées à partir de 1)."
        )
    return int(match.group(1)), int(match.group(2))


def volume_pdf(volume_dir: Path) -> Path:
    """PDF complet d'un volume, lu par `numrev ocr` : `annuaires/<volume>/<volume>.pdf`."""
    return volume_dir / f"{volume_dir.name}.pdf"


def volume_file(volume_dir: Path, suffix: str) -> Path:
    """Livrable d'un volume entier (`numrev export`) : `annuaires/<volume>/<volume><suffixe>`."""
    return volume_dir / f"{volume_dir.name}{suffix}"


def range_dirs(volume_dir: Path) -> list[Path]:
    """Sous-dossiers de plages (`7-177`, `179-186`…), triés par première page."""
    ranges = [path for path in volume_dir.iterdir() if path.is_dir() and RANGE_PATTERN.match(path.name)]
    return sorted(ranges, key=lambda path: int(RANGE_PATTERN.match(path.name).group(1)))


def selected_ranges(folder: Path) -> list[Path]:
    """Plages désignées par un dossier : lui-même si c'est une plage
    (`<première>-<dernière>`), sinon toutes les plages du volume, triées par
    première page. ValueError si le dossier n'est ni l'un ni l'autre."""
    if RANGE_PATTERN.match(folder.name):
        return [folder]
    ranges = range_dirs(folder)
    if not ranges:
        raise ValueError(f"{folder} n'est ni une plage (<première page>-<dernière page>) ni un volume qui contient des plages.")
    return ranges


def range_document(range_dir: Path, suffix: str) -> Path:
    """Fichier d'étape d'une plage, nommé d'après son volume et sa plage."""
    return range_dir / f"{range_dir.parent.name}.{range_dir.name}{suffix}"


def discover(root: Path, suffix: str) -> list[Path]:
    """Fichiers d'étape existants de toutes les plages sous `root`
    (`<root>/<volume>/<plage>/<volume>.<plage><suffixe>`), et eux seuls :
    les autres fichiers des dossiers (copies, sauvegardes) sont ignorés.
    Triés par chemin : l'ordre, dont dépendent les tirages aléatoires (gold,
    jeu d'entraînement), ne change pas d'une exécution à l'autre."""
    candidates = (range_document(path, suffix) for path in root.glob("*/*/") if RANGE_PATTERN.match(path.name))
    return sorted(path for path in candidates if path.exists())


def curation_patch(output_path: Path, step: str) -> Path:
    """Patch versionné des corrections humaines d'une étape curée."""
    return CURATION_DIR / f"{document_name(output_path)}.{step}.patch.csv"


# ----------------------------------------------------------------------
# Paires d'annuaires
# ----------------------------------------------------------------------
PAIR_SEPARATOR = "__"
NW_SUFFIX = ".nw.csv"
DEDUPE_SUFFIX = ".dedupe.csv"
JOIN_SUFFIX = ".join.csv"
ALIGNMENT_SUFFIXES = (DEDUPE_SUFFIX, NW_SUFFIX)  # sorties brutes, lues par le viewer


@dataclass(frozen=True)
class Pair:
    """Deux annuaires alignés, et les fichiers de la paire."""

    left: str  # nom du dossier de volume
    right: str
    root: Path = ANNUAIRES_DIR  # dossier des volumes

    @classmethod
    def of_dirs(cls, left_dir: Path, right_dir: Path) -> "Pair":
        """ValueError si les deux volumes ne sont pas dans le même dossier."""
        if left_dir.parent.resolve() != right_dir.parent.resolve():
            raise ValueError(f"{left_dir} et {right_dir} : les deux volumes doivent être dans le même dossier.")
        return cls(left_dir.name, right_dir.name, left_dir.parent)

    @classmethod
    def of_file(cls, path: Path) -> "Pair":
        """Paire d'un fichier `<gauche>__<droite>.….csv` (sortie, patch,
        gold) : le nom jusqu'au premier point."""
        left, separator, right = path.name.split(".", 1)[0].partition(PAIR_SEPARATOR)
        if not separator or not left or not right:
            raise ValueError(f"nom de fichier sans paire `<gauche>{PAIR_SEPARATOR}<droite>` : {path.name}")
        return cls(left, right, ANNUAIRES_DIR)

    @property
    def name(self) -> str:
        return f"{self.left}{PAIR_SEPARATOR}{self.right}"

    @property
    def left_dir(self) -> Path:
        return self.root / self.left

    @property
    def right_dir(self) -> Path:
        return self.root / self.right

    # Décisions humaines, versionnées
    @property
    def entry_patch(self) -> Path:
        return ALIGNMENT_DATA_DIR / f"{self.name}.patch.csv"

    @property
    def section_patch(self) -> Path:
        return ALIGNMENT_DATA_DIR / f"{self.name}.sections.csv"

    @property
    def training(self) -> Path:
        return ALIGNMENT_DATA_DIR / f"{self.name}.training.json"

    @property
    def gold_inversions(self) -> Path:
        return ALIGNMENT_DATA_DIR / f"{self.name}.gold-inversions.csv"

    # Sorties
    def output(self, suffix: str = ".csv") -> Path:
        return ALIGNMENTS_DIR / f"{self.name}{suffix}"


def join_output(alignment_path: Path) -> Path:
    return alignment_path.with_name(alignment_path.name.removesuffix(".csv") + JOIN_SUFFIX)

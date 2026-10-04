"""Publie le viewer d'alignement (`numrev view alignment`) dans un dépôt de
déploiement pour Streamlit Community Cloud.

    uv run numrev publish-viewer ../alignment-viewer [--alignment <sortie>…] --apply

Le dépôt de déploiement est un **instantané** : le code du viewer et les
données d'une version donnée, sans les dépendances lourdes de `numrev`
(torch, GLiNER, Dedupe) ni les dossiers de travail. On y copie, tels quels :

- la fermeture des imports `numrev.*` de `numrev.viewers.alignment`
  (analyse statique) et `viewers/assets/` ; une bibliothèque tierce hors de
  `ALLOWED` est refusée : c'est la frontière qui garde le viewer
  déployable (`tests/test_publish_viewer.py`) ;
- pour chaque paire publiée : les `<volume>.<plage>.ner.csv` des deux
  volumes, ses sorties d'alignement, la correspondance des rubriques et le
  patch des entrées s'il existe (les relecteurs y rejouent leurs décisions) ;
- `viewers/assets/theme.toml`, le thème, en `.streamlit/config.toml` ;
- générés : `streamlit_app.py` (point d'entrée), `requirements.txt`
  (versions installées de `ALLOWED`), `README.md` (provenance) et le
  manifeste `.numrev-publish.json`.

Seuls les fichiers du manifeste précédent sont supprimés quand ils ne sont
plus publiés ; le reste du dossier (`.git/`, `.streamlit/`, fichiers
ajoutés à la main) n'est jamais touché, seulement signalé. Refus si le code
de `src/numrev` a des modifications non commitées (le README ne décrirait
pas le code publié) : `--force` passe outre. La commande ne commite pas.
"""

import argparse
import ast
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numrev
from numrev import paths
from numrev.command import CommandError, Writes, add_apply_argument, add_force_argument, console
from numrev.paths import ALIGNMENT_SUFFIXES, Pair

DESCRIPTION = "Publie le viewer d'alignement (code + données) dans un dépôt de déploiement Streamlit Cloud."
ROOT_MODULE = "numrev.viewers.alignment"
ASSETS = "numrev/viewers/assets"
ALLOWED = ("numpy", "pandas", "rapidfuzz", "rich", "scipy", "streamlit")  # bibliothèques tierces du viewer
MANIFEST = ".numrev-publish.json"
ENTRY_POINT = "streamlit_app.py"
STREAMLIT_CONFIG = ".streamlit/config.toml"  # seul fichier géré de .streamlit/ (secrets.toml n'est jamais touché)
NEVER_TOUCHED = {".git", ".streamlit", ".venv", "__pycache__", ".gitignore", MANIFEST}
# Disposition publiée = valeurs par défaut de `numrev.paths` (le viewer déployé les lit telles quelles).
PUBLISHED_ANNUAIRES = Path("annuaires")
PUBLISHED_ALIGNMENTS = PUBLISHED_ANNUAIRES / "alignments"
PUBLISHED_ALIGNMENT_DATA = Path("data") / "alignment"

ENTRY_POINT_SOURCE = '''"""Point d'entrée Streamlit Cloud, généré par `numrev publish-viewer` : ne pas modifier."""

from numrev.viewers.alignment import main

main()
'''


# ----------------------------------------------------------------------
# Code
# ----------------------------------------------------------------------
def package_dir() -> Path:
    return Path(numrev.__file__).parent


def module_file(module: str) -> Path | None:
    """Fichier source d'un module `numrev.…` (None s'il n'existe pas : nom
    importé depuis un module, par exemple `numrev.paths.Pair`)."""
    path = package_dir().parent.joinpath(*module.split("."))
    if (path / "__init__.py").exists():
        return path / "__init__.py"
    if path.with_suffix(".py").exists():
        return path.with_suffix(".py")
    return None


def module_closure(root: str = ROOT_MODULE) -> tuple[list[Path], set[str]]:
    """(fichiers des modules `numrev.…` importés, de proche en proche, par
    `root`, paquets parents compris ; bibliothèques tierces importées). Les
    imports à l'intérieur des fonctions comptent aussi."""
    seen: dict[str, Path] = {}
    third_party: set[str] = set()
    todo = [root]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        path = module_file(module)
        if path is None:
            continue
        seen[module] = path
        parts = module.split(".")
        todo += [".".join(parts[:index]) for index in range(1, len(parts))]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top == "numrev":
                    todo.append(name)
                elif top not in sys.stdlib_module_names:
                    third_party.add(top)
    return sorted(seen.values()), third_party


def code_files() -> dict[str, Path]:
    """Chemin publié → source, pour le code du viewer."""
    files, third_party = module_closure()
    forbidden = sorted(third_party - set(ALLOWED))
    if forbidden:
        raise CommandError(
            f"le viewer importe des bibliothèques hors de la liste publiable ({', '.join(forbidden)}) : "
            f"le dépôt de déploiement ne doit dépendre que de {', '.join(ALLOWED)}."
        )
    root = package_dir().parent
    published = {path.relative_to(root).as_posix(): path for path in files}
    for asset in sorted((package_dir() / "viewers" / "assets").iterdir()):
        if asset.is_file():
            published[f"{ASSETS}/{asset.name}"] = asset
    # Le thème (Dracula / Alucard) devient la configuration Streamlit du dépôt publié.
    published[STREAMLIT_CONFIG] = package_dir() / "viewers" / "assets" / "theme.toml"
    return published


# ----------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------
def default_alignments() -> list[Path]:
    return sorted(path for suffix in ALIGNMENT_SUFFIXES for path in paths.ALIGNMENTS_DIR.glob(f"*{suffix}"))


def data_files(alignments: list[Path]) -> dict[str, Path]:
    """Chemin publié → source, pour les données des paires de `alignments`."""
    from numrev.alignment.records import ner_csv

    published: dict[str, Path] = {}
    for alignment in alignments:
        if not alignment.exists():
            raise CommandError(f"alignement '{alignment}' introuvable.")
        pair = Pair.of_file(alignment)
        published[(PUBLISHED_ALIGNMENTS / alignment.name).as_posix()] = alignment
        for volume_dir in (pair.left_dir, pair.right_dir):
            ranges = paths.range_dirs(volume_dir)
            if not ranges:
                raise CommandError(f"aucune plage de pages dans '{volume_dir}'.")
            for range_dir in ranges:
                try:
                    source = ner_csv(range_dir)
                except FileNotFoundError as error:
                    raise CommandError(str(error)) from error
                published[(PUBLISHED_ANNUAIRES / volume_dir.name / range_dir.name / source.name).as_posix()] = source
        for source in (pair.section_patch, pair.entry_patch):
            if source.exists():
                published[(PUBLISHED_ALIGNMENT_DATA / source.name).as_posix()] = source
    return published


# ----------------------------------------------------------------------
# Fichiers générés
# ----------------------------------------------------------------------
def source_commit() -> tuple[str, bool]:
    """(commit court du dépôt courant, code de `src/numrev` modifié depuis) ;
    ("inconnu", False) hors d'un dépôt git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(package_dir())], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu", False
    return commit, bool(status)


def requirements() -> str:
    return "".join(f"{name}=={version(name)}\n" for name in ALLOWED)


def readme(commit: str, alignments: list[Path]) -> str:
    pairs = sorted({Pair.of_file(path).name for path in alignments})
    listed = "\n".join(f"- `{name}`" for name in pairs)
    return f"""# Viewer d'alignement des annuaires de Paris

Instantané publié du viewer `numrev view alignment` du dépôt
`numerotation_revolutionnaire` (commit `{commit}`), pour Streamlit Community
Cloud. **Généré par `numrev publish-viewer` : ne rien modifier ici**, corriger
dans le dépôt source puis republier.

Paires publiées :

{listed}

Relecture d'un alignement entre deux éditions : vues Relecture, Table et
Documents. Les décisions restent dans le navigateur ; **Télécharger le patch**
donne le fichier à déposer dans `data/alignment/` du dépôt source.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run {ENTRY_POINT}
```

Streamlit Cloud : fichier principal `{ENTRY_POINT}`, Python ≥ 3.12.
"""


# ----------------------------------------------------------------------
# Commande
# ----------------------------------------------------------------------
def read_manifest(output: Path) -> set[str]:
    path = output / MANIFEST
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8"))["files"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise CommandError(f"manifeste '{path}' illisible : {error}") from error


def unmanaged(output: Path, published: set[str], previous: set[str]) -> list[str]:
    """Fichiers du dossier ni publiés ni gérés (hors `NEVER_TOUCHED`),
    regroupés par dossier de premier niveau."""
    if not output.exists():
        return []
    found: dict[str, int] = {}
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if path.is_dir() or NEVER_TOUCHED.intersection(relative.parts):
            continue
        name = relative.as_posix()
        if name in published or name in previous:
            continue
        top = relative.parts[0] + ("/" if len(relative.parts) > 1 else "")
        found[top] = found.get(top, 0) + 1
    return [f"{top} ({count} fichiers)" if top.endswith("/") else top for top, count in sorted(found.items())]


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def copy_bytes(path: Path, source: Path) -> None:
    path.write_bytes(source.read_bytes())


def prune_empty_dirs(output: Path, removed: list[str]) -> None:
    """Dossiers vidés par les suppressions, jusqu'à `output` exclu."""
    for name in removed:
        directory = (output / name).parent
        while directory != output and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path, help="Dossier du dépôt de déploiement (ex. ../alignment-viewer).")
    parser.add_argument(
        "--alignment",
        type=Path,
        action="append",
        help=f"Sortie d'alignement à publier (répétable). "
        f"Par défaut : tous les *{'/*'.join(ALIGNMENT_SUFFIXES)} de {paths.ALIGNMENTS_DIR}/.",
    )
    add_force_argument(parser, "Publie même si le code de src/numrev a des modifications non commitées.")
    add_apply_argument(parser)


def run(args: argparse.Namespace) -> None:
    output: Path = args.output
    if output.resolve() == Path.cwd().resolve() or Path.cwd().resolve() in output.resolve().parents:
        raise CommandError("le dépôt de déploiement doit être hors du dépôt source.")
    alignments = args.alignment or default_alignments()
    if not alignments:
        raise CommandError(f"aucun alignement à publier sous {paths.ALIGNMENTS_DIR}/.")
    commit, dirty = source_commit()
    if dirty and not args.force:
        raise CommandError("src/numrev a des modifications non commitées : committer d'abord (ou --force).")

    sources = code_files() | data_files(alignments)
    generated = {ENTRY_POINT: ENTRY_POINT_SOURCE, "requirements.txt": requirements(), "README.md": readme(commit, alignments)}
    published = set(sources) | set(generated)
    previous = read_manifest(output)

    writes = Writes(args.apply)
    unchanged = 0
    for name, source in sorted(sources.items()):
        target = output / name
        if target.exists() and target.read_bytes() == source.read_bytes():
            unchanged += 1
            continue
        writes.add(target, lambda target=target, source=source: copy_bytes(target, source))
    for name, text in sorted(generated.items()):
        target = output / name
        if target.exists() and target.read_text(encoding="utf-8") == text:
            unchanged += 1
            continue
        writes.add(target, lambda target=target, text=text: write_text(target, text))
    removed = sorted(previous - published)
    for name in removed:
        writes.remove(output / name)
    manifest = json.dumps({"source_commit": commit, "files": sorted(published)}, ensure_ascii=False, indent=1) + "\n"
    manifest_path = output / MANIFEST
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != manifest:
        writes.add(manifest_path, lambda: write_text(manifest_path, manifest))
    if args.apply:
        prune_empty_dirs(output, removed)

    pairs = sorted({Pair.of_file(path).name for path in alignments})
    console.print(
        f"Viewer publié depuis le commit [bold]{commit}[/bold]{' (modifié)' if dirty else ''} : "
        f"{len(published)} fichiers ({len(pairs)} paire(s) : {', '.join(pairs)}), {unchanged} inchangé(s)."
    )
    foreign = unmanaged(output, published, previous)
    if foreign:
        console.print(
            "[yellow]Fichiers hors manifeste, laissés tels quels (à retirer à la main s'ils sont obsolètes) :[/yellow]",
            *(f"  {name}" for name in foreign),
            sep="\n",
        )
    writes.finish(console)
    if args.apply and writes.planned:
        console.print(f"Dans {output} : relire, committer et pousser (git add -A && git commit && git push).")

"""Viewers Streamlit (lecture seule), lancés par `numrev view <viewer>` :

- `directory` : un annuaire, sortie NER (`numrev/viewers/directory.py`) ;
- `alignment` : un alignement entre deux annuaires, relecture et décisions
  enregistrées dans le patch (`numrev/viewers/alignment.py`).

Ce module n'importe pas Streamlit : il lance `streamlit run` sur le script
du viewer, depuis la racine du dépôt.
"""

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

VIEWERS = ("directory", "alignment")
THEME = Path(__file__).with_name("assets") / "theme.toml"  # thème du viewer d'alignement (Dracula / Alucard)


def theme_options(path: Path = THEME) -> list[str]:
    """Le thème en options de `streamlit run` (`--theme.dark.primaryColor=…`)."""

    def flatten(prefix: str, table: dict) -> list[str]:
        options = []
        for key, value in table.items():
            if isinstance(value, dict):
                options += flatten(f"{prefix}.{key}", value)
            else:
                options.append(f"--{prefix}.{key}={str(value).lower() if isinstance(value, bool) else value}")
        return options

    return flatten("theme", tomllib.loads(path.read_text(encoding="utf-8"))["theme"])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("viewer", choices=VIEWERS, help="directory : un annuaire (NER) ; alignment : un alignement entre deux annuaires.")


def run(args: argparse.Namespace) -> None:
    script = Path(__file__).with_name(f"{args.viewer}.py")
    # En local, le viewer d'alignement enregistre le patch directement
    # (`numrev/viewers/alignment.py`) ; la copie hébergée ne fait que le télécharger.
    environment = os.environ | {"NUMREV_PATCH_WRITABLE": "1"}
    options = theme_options() if args.viewer == "alignment" else []
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", *options, str(script)], env=environment))

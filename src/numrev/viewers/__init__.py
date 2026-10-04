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
from pathlib import Path

VIEWERS = ("directory", "alignment")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("viewer", choices=VIEWERS, help="directory : un annuaire (NER) ; alignment : un alignement entre deux annuaires.")


def run(args: argparse.Namespace) -> None:
    script = Path(__file__).with_name(f"{args.viewer}.py")
    # En local, le viewer d'alignement enregistre le patch directement
    # (`numrev/viewers/alignment.py`) ; la copie hébergée ne fait que le télécharger.
    environment = os.environ | {"NUMREV_PATCH_WRITABLE": "1"}
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)], env=environment))

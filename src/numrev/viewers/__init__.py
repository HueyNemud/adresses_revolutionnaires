"""Viewers (lecture seule), lancés par `numrev view <viewer>` :

- `directory` : un annuaire, sortie NER (`numrev/viewers/directory.py`, Streamlit) ;
- `alignment` : un alignement entre deux annuaires, aide à la relecture
  (`numrev/viewers/alignment.py`, Streamlit) ;
- `ocr` : la sortie OCR d'une plage posée sur le PDF du volume (`numrev/viewers/ocr.html`,
  page autonome, aussi utilisable sans numrev).

Ce module n'importe pas Streamlit : il lance `streamlit run` sur le script
du viewer, depuis la racine du dépôt. Pour `ocr`, il sert la page, le JSON et
le PDF sur un serveur local (127.0.0.1) : un navigateur refuse à une page
ouverte depuis le disque de lire d'autres fichiers.
"""

import argparse
import contextlib
import http.server
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

from numrev.command import CommandError, console, require_file, require_input
from numrev.paths import OCR, page_range, volume_pdf

VIEWERS = ("directory", "alignment", "ocr")


DESCRIPTION = """Ouvre un viewer dans le navigateur (lecture seule) :
  directory  un annuaire : entrées dans leurs rubriques, empans SUBJ / DESC / ADDR, entrées à relire ;
  alignment  un alignement de deux éditions : paires, entrées seules, incertitudes, boutons pour les patchs ;
  ocr        l'OCR d'une plage posé sur le PDF du volume : lignes, blocs, mots douteux (donner la plage)."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "viewer",
        choices=VIEWERS,
        help="directory : un annuaire (NER) ; alignment : un alignement entre deux annuaires ; ocr : l'OCR d'une plage.",
    )
    parser.add_argument(
        "range_dir",
        type=Path,
        nargs="?",
        metavar="plage",
        help="Pour ocr seulement : dossier de la plage, annuaires/<volume>/<première page>-<dernière page>.",
    )


def run(args: argparse.Namespace) -> None:
    if args.viewer == "ocr":
        if args.range_dir is None:
            raise CommandError("numrev view ocr attend une plage : annuaires/<volume>/<première page>-<dernière page>.")
        serve_ocr(args.range_dir)
        return
    if args.range_dir is not None:
        raise CommandError(f"numrev view {args.viewer} ne prend pas de plage : le fichier se choisit dans le viewer.")
    script = Path(__file__).with_name(f"{args.viewer}.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)]))


def serve_ocr(range_dir: Path) -> None:
    """Sert `ocr.html`, le `.ocr.json` de la plage et le PDF du volume, et ouvre la page sur la première page de la plage."""
    try:
        first, _ = page_range(range_dir)
    except ValueError as error:
        raise CommandError(str(error)) from error
    json_path = require_input(range_dir, OCR, "ocr")
    pdf_path = require_file(volume_pdf(range_dir.parent))
    # Seuls ces trois fichiers sont servis, sous leur nom (la page l'affiche).
    files = {
        "/": (Path(__file__).with_name("ocr.html"), "text/html; charset=utf-8"),
        f"/{quote(json_path.name)}": (json_path, "application/json"),
        f"/{quote(pdf_path.name)}": (pdf_path, "application/pdf"),
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            entry = files.get(self.path.split("?", 1)[0])
            if entry is None:
                self.send_error(404)
                return
            path, content_type = entry
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as source, contextlib.suppress(BrokenPipeError, ConnectionResetError):
                shutil.copyfileobj(source, self.wfile)  # le navigateur peut couper un téléchargement en cours

        def log_message(self, format: str, *args) -> None:
            pass  # pas de journal des requêtes dans la console

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # port libre choisi par le système
    url = f"http://127.0.0.1:{server.server_port}/?json={quote(json_path.name)}&pdf={quote(pdf_path.name)}#p={first}"
    console.print(f"Viewer OCR de {range_dir} : {url}\n(Ctrl+C pour arrêter)", soft_wrap=True)  # URL d'un seul tenant
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

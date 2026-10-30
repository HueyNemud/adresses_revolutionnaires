"""Viewers (lecture seule), lancés par `numrev view <viewer>` :

- `directory` : le livrable d'un volume (`numrev export`) posé sur son PDF
  (`numrev/viewers/directory.html`, page autonome, aussi utilisable sans numrev) ;
- `alignment` : un alignement entre deux annuaires, aide à la relecture
  (`numrev/viewers/alignment.py`, Streamlit) ;
- `ocr` : la sortie OCR d'une plage posée sur le PDF du volume (`numrev/viewers/ocr.html`,
  page autonome, aussi utilisable sans numrev).

Ce module n'importe pas Streamlit : il lance `streamlit run` sur le script
du viewer, depuis la racine du dépôt. Pour `directory` et `ocr`, il sert la
page et ses fichiers (JSON, rapport, PDF) sur un serveur local (127.0.0.1) :
un navigateur refuse à une page ouverte depuis le disque de lire d'autres
fichiers.
"""

import argparse
import contextlib
import http.server
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote, urlencode

from numrev.command import CommandError, console, fine_tuning, require_file, require_input, require_volume
from numrev.paths import DOCUMENT_JSON, EXPORT_REPORT, OCR, STRUCTURE_JSON, page_range, range_dirs, volume_file, volume_pdf

VIEWERS = ("directory", "alignment", "ocr")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
}

DESCRIPTION = """Ouvre un viewer dans le navigateur (lecture seule) :
  directory  le livrable d'un volume (numrev export) posé sur son PDF : rubriques, entrées et empans SUBJ / DESC / ADDR,
             groupes à accolade, filtres de relecture, statistiques, rapport d'export (donner le volume) ;
  alignment  un alignement de deux éditions : paires, entrées seules, incertitudes, boutons pour les patchs ;
  ocr        l'OCR d'une plage posé sur le PDF du volume : lignes, blocs, mots douteux (donner la plage)."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "viewer",
        choices=VIEWERS,
        help="directory : le livrable d'un volume ; alignment : un alignement entre deux annuaires ; ocr : l'OCR d'une plage.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        metavar="dossier",
        help="directory : le volume, annuaires/<volume> ; ocr : la plage, annuaires/<volume>/<première page>-<dernière page> ; "
        "alignment : aucun (l'alignement se choisit dans le viewer).",
    )
    fine_tuning(parser).add_argument(
        "-p",
        "--port",
        type=port_number,
        metavar="PORT",
        help="Port local du viewer. Par défaut : directory et ocr, un port libre choisi par le système, différent à chaque "
        "lancement (un port fixe garde d'une session à l'autre les réglages que la page retient dans le navigateur) ; "
        "alignment, celui de Streamlit (8501).",
    )


def port_number(text: str) -> int:
    """Port TCP donné par `--port` : un entier de 1 à 65535."""
    if not text.isdigit() or not 1 <= int(text) <= 65535:
        raise argparse.ArgumentTypeError(f"port invalide : « {text} » (attendu : un entier de 1 à 65535)")
    return int(text)


def run(args: argparse.Namespace) -> None:
    if args.viewer == "alignment":
        if args.folder is not None:
            raise CommandError("numrev view alignment ne prend pas de dossier : l'alignement se choisit dans le viewer.")
        script = Path(__file__).with_name("alignment.py")
        port = [] if args.port is None else ["--server.port", str(args.port)]
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), *port]))
    if args.viewer == "ocr":
        if args.folder is None:
            raise CommandError("numrev view ocr attend une plage : annuaires/<volume>/<première page>-<dernière page>.")
        files, fragment = ocr_files(args.folder)
        serve(Path(__file__).with_name("ocr.html"), files, fragment, f"Viewer OCR de {args.folder}", args.port)
        return
    if args.folder is None:
        raise CommandError("numrev view directory attend un volume : annuaires/<volume>.")
    files = directory_files(args.folder)
    warn_directory(args.folder, files)
    serve(Path(__file__).with_name("directory.html"), files, "", f"Viewer du volume {args.folder}", args.port)


def ocr_files(range_dir: Path) -> tuple[dict[str, Path], str]:
    """Fichiers servis à `ocr.html` (paramètre d'URL → fichier) et page du PDF à ouvrir (`p=<première page>`)."""
    try:
        first, _ = page_range(range_dir)
    except ValueError as error:
        raise CommandError(str(error)) from error
    return {"json": require_input(range_dir, OCR, "ocr"), "pdf": require_file(volume_pdf(range_dir.parent))}, f"p={first}"


def directory_files(volume_dir: Path) -> dict[str, Path]:
    """Fichiers servis à `directory.html` (paramètre d'URL → fichier) : les deux JSON du livrable, et s'ils existent
    le rapport d'export et le PDF du volume."""
    require_volume(volume_dir, "view directory")
    files = {"document": volume_file(volume_dir, DOCUMENT_JSON), "structure": volume_file(volume_dir, STRUCTURE_JSON)}
    for path in files.values():
        if not path.exists():
            raise CommandError(f"livrable absent : {path} (lancez d'abord uv run numrev export {volume_dir} --apply).")
    for key, path in (("report", volume_file(volume_dir, EXPORT_REPORT)), ("pdf", volume_pdf(volume_dir))):
        if path.exists():
            files[key] = path
    return files


def warn_directory(volume_dir: Path, files: dict[str, Path]) -> None:
    """Signale un livrable plus ancien que les fichiers de plage qu'il lit, et un PDF absent."""
    from numrev.devtools.status import STALE, export_state

    if export_state(volume_dir, range_dirs(volume_dir)) == STALE:
        console.print(
            f"[yellow]Livrable périmé : une table des lignes ou un CSV NER a changé depuis l'export "
            f"(relancez uv run numrev export {volume_dir} --apply pour le voir à jour).[/yellow]",
            soft_wrap=True,
        )
    if "pdf" not in files:
        console.print(f"[yellow]PDF absent ({volume_pdf(volume_dir)}) : les lignes s'affichent sur une page blanche.[/yellow]")


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Octets demandés par un en-tête `Range: bytes=<début>-<fin>` (bornes incluses), bornés à la taille du
    fichier ; None si l'en-tête est absent, demande plusieurs plages ou tombe hors du fichier (fichier entier)."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip()) if header else None
    if match is None or match.groups() == ("", ""):
        return None
    first, last = match.groups()
    if first == "":  # « bytes=-500 » : les 500 derniers octets
        start, end = max(0, size - int(last)), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    return (start, end) if start <= end else None


def serve(page: Path, files: dict[str, Path], fragment: str, label: str, port: int | None = None) -> None:
    """Sert `page` et `files` (chacun sous son nom) sur 127.0.0.1:`port` (None : port libre choisi par le système),
    et ouvre la page avec `?<paramètre>=<nom>…#<fragment>`.
    Le serveur accepte les requêtes partielles (`Range`) : pdf.js lit alors un gros PDF par morceaux."""
    served = {"/": page} | {f"/{quote(path.name)}": path for path in files.values()}  # seuls ces fichiers sont servis

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = served.get(self.path.split("?", 1)[0])
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            wanted = byte_range(self.headers.get("Range"), size)
            start, end = wanted or (0, size - 1)
            self.send_response(206 if wanted else 200)
            self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if wanted:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source, contextlib.suppress(BrokenPipeError, ConnectionResetError):
                source.seek(start)  # le navigateur peut couper un téléchargement en cours
                remaining = end - start + 1
                while remaining > 0 and (chunk := source.read(min(remaining, 1 << 20))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, format: str, *args) -> None:
            pass  # pas de journal des requêtes dans la console

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port or 0), Handler)  # 0 : port libre choisi par le système
    except OSError as error:
        raise CommandError(f"port {port} sur 127.0.0.1 : {error.strerror} (choisissez un autre port avec --port).") from error
    query = urlencode({key: path.name for key, path in files.items()}, quote_via=quote)
    url = f"http://127.0.0.1:{server.server_port}/?{query}" + (f"#{fragment}" if fragment else "")
    console.print(f"{label} : {url}\n(Ctrl+C pour arrêter)", soft_wrap=True)  # URL d'un seul tenant
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

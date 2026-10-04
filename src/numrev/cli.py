"""Point d'entrée unique : `uv run numrev <commande> [<sous-commande>] …`.

Les commandes sont déclarées ici (nom, module, résumé) et leur module n'est
importé qu'à l'exécution : `numrev --help` reste instantané sans charger
torch, Dedupe ni Streamlit. Chaque module expose `add_arguments(parser)` et
`run(args)` (`numrev/command.py`).
"""

import argparse
import importlib
import sys
from dataclasses import dataclass

from numrev.command import CommandError, console
from numrev.curation import CurationConflict, print_conflict


@dataclass(frozen=True)
class Command:
    module: str
    summary: str


PIPELINE: dict[str, Command | dict[str, Command]] = {
    "extract": Command("numrev.pipeline.extract", "1 · lignes Markdown de la sortie OCR       <doc>.ocr.json → <doc>.lines.json"),
    "label": Command("numrev.pipeline.label", "2 · annotation interactive des lignes (CRF) → <doc>.labeled.json"),
    "tabulate": Command("numrev.pipeline.tabulate", "3 · table des lignes, corrigée à la main   → <doc>.lines.csv"),
    "assemble": Command("numrev.pipeline.assemble", "4 · entités et arbre des titres            → <doc>.entities.csv"),
    "tag": Command("numrev.pipeline.tag", "5 · empans SUBJ / DESC / ADDR (GLiNER)     → <doc>.ner.csv"),
    "align": {
        "nw": Command(
            "numrev.pipeline.align_nw", "6 · alignement ordonné (Needleman-Wunsch + pair-HMM) → annuaires/alignments/<paire>.nw.csv"
        ),
        "dedupe": Command("numrev.pipeline.align_dedupe", "6 · alignement par Dedupe (+ patch) → annuaires/alignments/<paire>.csv"),
    },
    "join": Command("numrev.pipeline.join", "7 · jointure lisible de deux annuaires alignés → <alignement>.join.csv"),
}
TOOLS: dict[str, Command | dict[str, Command]] = {
    "view": Command("numrev.viewers", "viewer Streamlit : directory (un annuaire, NER) | alignment (relecture d'un alignement)"),
    "gold": {
        "ner": Command("numrev.devtools.ner_gold", "tire le gold NER à relire (data/ner/gold.ls.json)"),
        "alignment": Command("numrev.devtools.alignment_gold", "tire le gold d'inversions à étiqueter (data/alignment/)"),
    },
    "train-set": Command("numrev.devtools.ner_dataset", "construit le jeu d'entraînement NER (data/ner/train.ls.json)"),
    "train": Command("numrev.devtools.ner_train", "entraîne un modèle GLiNER (machine GPU)"),
    "audit": {
        "crf": Command("numrev.devtools.crf_audit", "audit du CRF de lignes → reports/crf/"),
        "ner": Command("numrev.devtools.ner_audit", "audit de la NER sur le gold → reports/ner/"),
        "alignment": Command("numrev.devtools.alignment_audit", "audit de la relecture d'alignement sur le gold → reports/alignment/"),
    },
    "publish-viewer": Command("numrev.devtools.publish_viewer", "publie le viewer d'alignement (code + données) pour Streamlit Cloud"),
}
COMMANDS = PIPELINE | TOOLS


def usage() -> str:
    def lines(table: dict) -> list[str]:
        result = []
        for name, entry in table.items():
            if isinstance(entry, Command):
                result.append(f"  {name:<20} {entry.summary}")
            else:
                result += [f"  {name + ' ' + sub:<20} {command.summary}" for sub, command in entry.items()]
        return result

    return "\n".join(
        [
            "usage : numrev <commande> [<sous-commande>] [options]   (numrev <commande> --help pour le détail)",
            "",
            "Chaîne de traitement, par plage de pages (1-5) puis par paire d'annuaires (6-7) :",
            *lines(PIPELINE),
            "",
            "Consultation, modèles et évaluation :",
            *lines(TOOLS),
            "",
            "Les commandes qui écrivent des données simulent par défaut : --apply écrit.",
        ]
    )


def resolve(argv: list[str]) -> tuple[str, Command, list[str]]:
    """(nom complet, commande, arguments restants) ; SystemExit sur une commande inconnue."""
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        raise SystemExit(0 if argv else 2)
    entry = COMMANDS.get(argv[0])
    if entry is None:
        print(f"numrev : commande inconnue « {argv[0]} ».\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)
    if isinstance(entry, Command):
        return argv[0], entry, argv[1:]
    if len(argv) < 2 or argv[1] not in entry:
        choices = "\n".join(f"  {argv[0]} {sub:<12} {command.summary}" for sub, command in entry.items())
        print(f"usage : numrev {argv[0]} <{'|'.join(entry)}> [options]\n\n{choices}", file=sys.stderr)
        raise SystemExit(0 if argv[1:2] in (["-h"], ["--help"]) else 2)
    return f"{argv[0]} {argv[1]}", entry[argv[1]], argv[2:]


def main(argv: list[str] | None = None) -> None:
    name, command, rest = resolve(sys.argv[1:] if argv is None else argv)
    module = importlib.import_module(command.module)
    parser = argparse.ArgumentParser(
        prog=f"numrev {name}",
        description=getattr(module, "DESCRIPTION", command.summary),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    module.add_arguments(parser)
    args = parser.parse_args(rest)
    try:
        module.run(args)
    except CommandError as error:
        console.print(f"[bold red]Erreur :[/bold red] {error}", highlight=False)
        raise SystemExit(1)
    except CurationConflict as conflict:
        print_conflict(console, conflict)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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
    files: str = ""  # entrée → sortie, pour la liste des commandes

    @property
    def description(self) -> str:
        """Aide d'une commande dont le module n'a pas de `DESCRIPTION`."""
        return f"{self.summary} ({self.files})" if self.files else self.summary


PIPELINE: dict[str, Command | dict[str, Command]] = {
    "ocr": Command("numrev.pipeline.ocr", "0 · OCR d'une plage de pages (latynna)", "<vol>.pdf → <doc>.ocr.json"),
    "extract": Command("numrev.pipeline.extract", "1 · découpage de la sortie OCR en lignes", "→ <doc>.lines.json"),
    "label": Command("numrev.pipeline.label", "2 · annotation interactive des lignes", "→ <doc>.labeled.json"),
    "tabulate": Command("numrev.pipeline.tabulate", "3 · table des lignes, à corriger à la main", "→ <doc>.lines.csv"),
    "assemble": Command("numrev.pipeline.assemble", "4 · entrées et arbre des titres", "→ <doc>.entities.csv"),
    "tag": Command("numrev.pipeline.tag", "5 · découpage des entrées en SUBJ / DESC / ADDR", "→ <doc>.ner.csv"),
    "export": Command("numrev.pipeline.export", "6 · livrable final d'un volume (deux JSON)", "→ <vol>.document.json, .structure.json"),
    "align": {
        "nw": Command("numrev.pipeline.align_nw", "7 · alignement de deux éditions, par l'ordre des entrées", "→ <A>__<B>.nw.csv"),
        "dedupe": Command("numrev.pipeline.align_dedupe", "7 · alignement de deux éditions, par Dedupe", "→ <A>__<B>.csv"),
    },
    "join": Command("numrev.pipeline.join", "8 · jointure lisible de deux éditions alignées", "→ <alignement>.join.csv"),
}
TOOLS: dict[str, Command | dict[str, Command]] = {
    "status": Command("numrev.devtools.status", "état de chaque plage et prochaine commande à lancer"),
    "view": Command(
        "numrev.viewers",
        "viewer : directory (livrable d'un volume sur son PDF), alignment (relecture d'un alignement) ou ocr (OCR d'une plage sur le PDF)",
    ),
    "gold": {
        "ner": Command("numrev.devtools.ner_gold", "tire l'échantillon NER à relire, pour l'évaluation", "→ data/ner/gold.ls.json"),
        "alignment": Command("numrev.devtools.alignment_gold", "tire les paires d'inversions à étiqueter", "→ data/alignment/"),
    },
    "train-set": Command("numrev.devtools.ner_dataset", "construit le jeu d'entraînement NER", "→ data/ner/train.ls.json"),
    "train": Command("numrev.devtools.ner_train", "entraîne un modèle NER GLiNER (machine GPU)", "→ models/"),
    "audit": {
        "crf": Command("numrev.devtools.crf_audit", "audit du modèle d'annotation des lignes (CRF)", "→ reports/crf/"),
        "ner": Command("numrev.devtools.ner_audit", "audit du modèle NER sur l'échantillon relu", "→ reports/ner/"),
        "alignment": Command(
            "numrev.devtools.alignment_audit", "audit de l'alignement sur les inversions étiquetées", "→ reports/alignment/"
        ),
    },
}
COMMANDS = PIPELINE | TOOLS


def usage() -> str:
    def entries(table: dict) -> list[tuple[str, Command]]:
        result = []
        for name, entry in table.items():
            if isinstance(entry, Command):
                result.append((name, entry))
            else:
                result += [(f"{name} {sub}", command) for sub, command in entry.items()]
        return result

    width = max(len(command.summary) for _, command in entries(COMMANDS) if command.files)

    def lines(table: dict) -> list[str]:
        return [f"  {name:<16} {command.summary:<{width}}  {command.files}".rstrip() for name, command in entries(table)]

    return "\n".join(
        [
            "usage : numrev <commande> [<sous-commande>] [options]   (numrev <commande> --help pour le détail)",
            "",
            "Chaîne de traitement : étapes 0-5 sur une plage annuaires/<volume>/<plage> ou un volume annuaires/<volume>,"
            "\nétape 6 sur un volume, étapes 7-8 sur une paire d'annuaires (sorties dans annuaires/alignments/) :",
            *lines(PIPELINE),
            "",
            "Consultation, modèles et évaluation :",
            *lines(TOOLS),
            "",
            "Les commandes qui écrivent des données simulent par défaut : --apply écrit.",
            "Pour savoir où en est chaque plage et quoi lancer ensuite : numrev status.",
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


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "usage : " if prefix is None else prefix)


def build_parser(name: str, command: Command, module) -> argparse.ArgumentParser:
    """Analyseur d'une commande, avec une aide en français (argparse écrit
    sinon « usage: », « positional arguments », « show this help message »)."""
    parser = argparse.ArgumentParser(
        prog=f"numrev {name}",
        description=getattr(module, "DESCRIPTION", command.description),
        formatter_class=HelpFormatter,
        add_help=False,
    )
    parser._positionals.title = "arguments"  # titres des groupes par défaut, sans autre accès public
    parser._optionals.title = "options"
    parser.add_argument("-h", "--help", action="help", help="Affiche cette aide.")
    module.add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    name, command, rest = resolve(sys.argv[1:] if argv is None else argv)
    module = importlib.import_module(command.module)
    args = build_parser(name, command, module).parse_args(rest)
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

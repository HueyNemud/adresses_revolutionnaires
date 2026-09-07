import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pycrfsuite
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# EXAMPLE OF REAL INPUT DATA:
# {140}------------------------------------------------
# ## Chapeliers, Chevaux. ( et Mds. de )
# 27
# **Hurvoix-Launay**, R. S. Honoré, 124. — Tuil.
# **Joly** ( Mlle. ), R. de la Loi, 882. — B. des Moul.
# **Lacrolrière**, R. Française, 1. — Bon C.
# **Laloge**, R. Neuve des Petits Champs, 1291. — B. des Moul.
# **Landry** ( Me. ) R. du Bacq, 145. — Font. de Gren.
# **Laprelle**, Pont S. Michel, 24. — P. Neuf.
# **Lebel**, R. du Caire. V — B. Nouv.
# **Lefebure** ( Ve. ), R. S. Antoine, 40. — Dr. de l'Homme.
# **Legendre**, R. Croix des Petits Champs, — 87, H. au Bl.
# **Lemoine** ( Ve. ), R. de l'Echelle, 560. — Tuil.
# **Lorbette**, R. Neuve des Petits Champs, 1280, — B. des Moul.
# **Magy**, R. et Div. des Gravilliers, 36.
# **Maneglier**, R. Barbette, 476. — Indivis.
# **Maneglier** R. de la Loi, 323. — Lepell.


# ----------------------------------------------------------------------
# 1. Extraction des Features & Heuristique Métier
# ----------------------------------------------------------------------
def get_shape(token: str) -> str:
    if token in (",", ".", "—", "-", "_"):
        return "PUNCT"
    if token.isdigit():
        return "NUM"
    if token.isupper():
        return "UPPER"
    if token.istitle():
        return "TITLE"
    if token.islower():
        return "LOWER"
    if token in ("*", "~_", "$", "#"):
        return "SYM"
    return "UNK"  # unknown / other


def get_heuristic_label(line: str, prev_line: str = "") -> str:
    """Retourne un indice de sélection, jamais une vérité de référence."""
    stripped = line.strip()
    if not stripped:
        return "BRUIT"
    if stripped.startswith("#"):
        return "TITRE"
    if re.fullmatch(r"\{\d+\}[-—_]*", stripped) or re.fullmatch(r"[-—_]{3,}", stripped):
        return "BRUIT"
    if stripped[0].islower() or stripped.startswith(("-", "—")):
        return "SUITE_ENTREE"
    if prev_line and prev_line.rstrip().endswith(("-", "—")):
        return "SUITE_ENTREE"
    return "DEBUT_ENTREE"


def extract_features(
    lines: list[str], source_line_numbers: list[int] | None = None
) -> list[dict[str, str]]:
    seq_features = []
    n = len(lines)

    for t, line in enumerate(lines):
        tokens = re.findall(r"\w+|[^\w\s]", line)
        shapes = [get_shape(tok) for tok in tokens]

        is_heading = line.startswith("#")
        heading_level = len(line) - len(line.lstrip("#")) if is_heading else 0

        feat = {
            "bias": "1.0",
            "is_heading": str(is_heading),
            "heading_level": str(heading_level),
            "starts_lower": str(shapes[0] == "LOWER") if shapes else "False",
            "ends_punct": str(shapes[-1] == "PUNCT") if shapes else "False",
            "token_count": str(min(len(tokens), 12)),
            # "has_markdown_bold": str("**" in line),
            "is_page_marker": str(bool(re.fullmatch(r"\{\d+\}[-—_]*", line))),
            "BOS": str(t == 0),
            "EOS": str(t == n - 1),
        }

        for i in range(min(4, len(shapes))):
            feat[f"shape_start_{i}"] = shapes[i]
            feat[f"shape_end_{i}"] = shapes[len(shapes) - 1 - i]

        if t > 0:
            feat["prev_is_heading"] = str(lines[t - 1].startswith("#"))
            feat["prev_ends_dash"] = str(lines[t - 1].rstrip().endswith(("-", "—")))
            if source_line_numbers is not None:
                feat["previous_source_gap"] = str(
                    source_line_numbers[t] - source_line_numbers[t - 1] > 1
                )

        seq_features.append(feat)

    return seq_features


@dataclass(frozen=True)
class SourceLine:
    """Une ligne annotable, liée à sa position dans le Markdown source."""

    source_line: int
    text: str


def load_source_lines(input_path: Path) -> tuple[list[SourceLine], str]:
    """Charge les lignes non vides sans perdre leur position physique d'origine."""
    raw_text = input_path.read_text(encoding="utf-8")
    records = [
        SourceLine(source_line=index, text=line.strip())
        for index, line in enumerate(raw_text.splitlines(), start=1)
        if line.strip()
    ]
    return records, hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 2. Moteur CRF Active Learning
# ----------------------------------------------------------------------
class ActiveCRF:
    CLASSES = ["DEBUT_ENTREE", "SUITE_ENTREE", "TITRE", "BRUIT"]

    def __init__(
        self,
        records: list[SourceLine],
        seed_size: int = 12,
    ):
        self.records = records
        self.lines = [record.text for record in records]
        self.source_line_numbers = [record.source_line for record in records]
        self.features = extract_features(self.lines, self.source_line_numbers)
        self.labels: list[str | None] = [None] * len(records)
        self.heuristic_labels = [
            get_heuristic_label(line, self.lines[index - 1] if index else "")
            for index, line in enumerate(self.lines)
        ]
        self.seed_size = seed_size
        self.model_path = os.path.join(
            tempfile.gettempdir(), f"crf_{uuid.uuid4().hex}.crfsuite"
        )
        # python-crfsuite ne publie pas de stubs complets pour Pylance.
        self.tagger: Any | None = None
        self.known_classes: set[str] = set()

    @property
    def annotated_count(self) -> int:
        return sum(label is not None for label in self.labels)

    def restore_labels(self, labels: list[str | None]) -> None:
        if len(labels) != len(self.labels):
            raise ValueError(
                "Le nombre de labels de session ne correspond pas au document."
            )
        invalid_labels = {
            label for label in labels if label is not None and label not in self.CLASSES
        }
        if invalid_labels:
            raise ValueError(f"Labels de session invalides : {sorted(invalid_labels)}")
        self.labels = labels
        self.retrain()

    def set_labels(self, annotations: dict[int, str]) -> None:
        """Enregistre atomiquement les labels humains d'un bloc d'annotation."""
        if not annotations:
            raise ValueError("Un bloc d'annotation ne peut pas être vide.")
        for index, label in annotations.items():
            if not 0 <= index < len(self.labels):
                raise IndexError(f"Indice de ligne invalide : {index}")
            if label not in self.CLASSES:
                raise ValueError(f"Classe inconnue : {label}")
        for index, label in annotations.items():
            self.labels[index] = label
        self.retrain()

    def retrain(self):
        """Entraîne le CRF sur les segments contigus validés par un humain."""
        self.tagger = None
        self.known_classes = {label for label in self.labels if label is not None}
        if len(self.known_classes) < 2:
            return

        trainer: Any = getattr(pycrfsuite, "Trainer")(verbose=False)
        trainer.set_params(
            {
                "c1": 0.1,
                "c2": 0.01,
                "max_iterations": 50,
                "feature.possible_transitions": True,
            }
        )

        segment_features: list[dict[str, str]] = []
        segment_labels: list[str] = []
        for feature, label in zip(self.features, self.labels):
            if label is None:
                if segment_labels:
                    trainer.append(segment_features, segment_labels)
                    segment_features = []
                    segment_labels = []
                continue
            segment_features.append(feature)
            segment_labels.append(label)
        if segment_labels:
            trainer.append(segment_features, segment_labels)
        trainer.train(self.model_path)

        tagger: Any = getattr(pycrfsuite, "Tagger")()
        tagger.open(self.model_path)
        tagger.set(self.features)
        self.tagger = tagger

    def _next_exploration_index(self, unannotated: list[int]) -> int:
        """Échantillonne des strates heuristiques sous-représentées au démarrage."""
        observed = Counter(
            self.heuristic_labels[index]
            for index, label in enumerate(self.labels)
            if label is not None
        )
        return min(
            unannotated,
            key=lambda index: (
                observed[self.heuristic_labels[index]],
                self.heuristic_labels[index],
                index,
            ),
        )

    def _annotation_block(self, target_index: int, unannotated: list[int]) -> list[int]:
        """Retourne le candidat et un voisin non annoté, si disponible."""
        available = set(unannotated)
        neighbours = [
            index
            for index in (target_index - 1, target_index + 1)
            if index in available
        ]
        if not neighbours:
            return [target_index]
        # Préférer la ligne précédente : le candidat a ainsi un contexte CRF humain.
        neighbour = (
            target_index - 1 if target_index - 1 in neighbours else neighbours[0]
        )
        return sorted([target_index, neighbour])

    def next_block(self) -> tuple[int, list[int]] | None:
        unannotated = [i for i, l in enumerate(self.labels) if l is None]
        if not unannotated:
            return None

        if self.annotated_count < self.seed_size or not self.tagger:
            target_index = self._next_exploration_index(unannotated)
        else:

            def margin(idx: int) -> float:
                probs = sorted(self.get_marginals(idx).values(), reverse=True)
                return probs[0] - probs[1]

            target_index = min(unannotated, key=margin)
        return target_index, self._annotation_block(target_index, unannotated)

    def get_marginals(self, idx: int) -> dict[str, float]:
        """Retourne des marginales normalisées, toujours comprises entre 0 et 1."""
        if not self.tagger:
            return {label: 0.0 for label in self.CLASSES}
        scores = {
            label: max(0.0, float(self.tagger.marginal(label, idx)))
            for label in self.known_classes
        }
        total = sum(scores.values())
        if total <= 0.0:
            return {label: 0.0 for label in self.CLASSES}
        return {label: scores.get(label, 0.0) / total for label in self.CLASSES}

    def export_csv(self, output_path: str):
        # `tag()` doit recevoir la séquence. Appeler `tag()` sans argument après
        # `set()` corrompt les marginales dans python-crfsuite.
        preds = (
            self.tagger.tag(self.features) if self.tagger else [None] * len(self.lines)
        )

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_line", "ligne", "classe", "provenance", "proba"])
            for i, line in enumerate(self.lines):
                human_label = self.labels[i]
                label = human_label or preds[i] or ""
                provenance = (
                    "human"
                    if human_label
                    else ("model" if preds[i] else "unclassified")
                )
                marginals = self.get_marginals(i)
                probability = 1.0 if human_label else marginals.get(label, 0.0)
                writer.writerow(
                    [
                        self.source_line_numbers[i],
                        line,
                        label,
                        provenance,
                        f"{probability:.4f}",
                    ]
                )

        console.print(
            "\n[bold green]✅ Export CSV réussi :[/bold green] "
            f"[yellow]{output_path}[/yellow] ({len(self.lines)} lignes)"
        )


# ----------------------------------------------------------------------
# 3. Dashboard Rich Debug
# ----------------------------------------------------------------------
def display_dashboard(crf: ActiveCRF, target_idx: int, block_indices: list[int]):
    console.clear()

    annotated_cnt = crf.annotated_count
    total_cnt = len(crf.lines)

    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Key", style="bold cyan")
    metrics_table.add_column("Val", style="yellow")
    metrics_table.add_row(
        "Avancement :",
        f"{annotated_cnt}/{total_cnt} ({annotated_cnt/total_cnt*100:.1f}%)",
    )

    console.print(
        Panel(
            metrics_table,
            title="📈 [bold yellow]ÉTAT DE L'ANNOTATION CRF[/bold yellow]",
        )
    )

    if crf.tagger is not None:
        info = crf.tagger.info()
        trans_table = Table(
            title="🔗 Transitions apprises dans les blocs annotés",
            header_style="bold magenta",
        )
        trans_table.add_column("Transition (Ligne N-1 ➔ Ligne N)")
        trans_table.add_column("Poids CRF", justify="right")

        sorted_trans = sorted(
            info.transitions.items(), key=lambda x: x[1], reverse=True
        )
        for (from_l, to_l), w in sorted_trans[:5]:
            color = "green" if w > 0 else "red"
            trans_table.add_row(f"{from_l} ➔ {to_l}", f"[{color}]{w:+.3f}[/]")

        console.print(trans_table)

    probs = crf.get_marginals(target_idx)
    prob_table = Table(
        title=f"🎯 Probas marginales [source {crf.source_line_numbers[target_idx]}]",
        header_style="bold green",
    )
    prob_table.add_column("Classe")
    prob_table.add_column("Probabilité", justify="right")

    for c, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        support = "" if c in crf.known_classes else " (classe non observée)"
        prob_table.add_row(c, f"{p*100:.1f}%{support}")

    console.print(prob_table)

    block_table = Table(
        title="📝 Bloc à annoter",
        header_style="bold yellow",
    )
    block_table.add_column("Source", justify="right")
    block_table.add_column("Texte")
    for index in block_indices:
        marker = " → cible" if index == target_idx else ""
        block_table.add_row(
            f"{crf.source_line_numbers[index]}{marker}",
            crf.lines[index],
        )
    console.print(block_table)

    feat_table = Table(
        title=f"🔍 Features actives [source {crf.source_line_numbers[target_idx]}]",
        header_style="bold cyan",
        padding=(0, 1),
    )
    feat_table.add_column("Feature / Transition Interne", style="yellow")
    feat_table.add_column("Valeur", style="green")

    for k, v in crf.features[target_idx].items():
        if v not in ("False", "0"):
            feat_table.add_row(k, str(v))

    console.print(feat_table)

    console.print("\n[bold yellow]📄 Contexte du document :[/bold yellow]")
    start = max(0, target_idx - 2)
    end = min(len(crf.lines), target_idx + 3)

    for i in range(start, end):
        prefix = "▶ " if i in block_indices else "  "
        lbl_str = f"[{crf.labels[i]}]" if crf.labels[i] else "[NO_LABEL]"
        style = "bold reverse green" if i in block_indices else "dim"
        console.print(
            f"{prefix}[{crf.source_line_numbers[i]:4d}] {lbl_str:15} {crf.lines[i]}",
            style=style,
        )


# ----------------------------------------------------------------------
# 4. Point d'entrée CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning CRF pour classification de lignes."
    )
    parser.add_argument("file", type=Path, help="Fichier .md ou .txt")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions_crf.csv",
        help="Nom du CSV de sortie",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=12,
        help="Nombre d'annotations diversifiées avant l'échantillonnage par incertitude.",
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Fichier JSON de session (défaut : suffixe .crf-session.json).",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Ignore une session existante et démarre une nouvelle annotation.",
    )
    return parser.parse_args()


def load_session(session_path: Path, document_hash: str) -> list[str | None] | None:
    if not session_path.exists():
        return None
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if session.get("document_hash") != document_hash:
        raise ValueError("La session concerne une autre version du document source.")
    labels = session.get("labels")
    if not isinstance(labels, list):
        raise ValueError("La session ne contient pas de liste de labels valide.")
    return labels


def save_session(session_path: Path, document_hash: str, crf: ActiveCRF) -> None:
    session_path.write_text(
        json.dumps(
            {
                "version": 1,
                "document_hash": document_hash,
                "source_lines": crf.source_line_numbers,
                "labels": crf.labels,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()

    if not args.file.exists():
        console.print(
            f"[bold red]Erreur :[/bold red] Fichier '{args.file}' introuvable."
        )
        return

    if args.seed_size < 1:
        console.print(
            "[bold red]Erreur :[/bold red] --seed-size doit être supérieur à zéro."
        )
        return

    records, document_hash = load_source_lines(args.file)
    if not records:
        console.print("[yellow]Le fichier est vide.[/yellow]")
        return

    session_path = args.session or args.file.with_suffix(".crf-session.json")
    crf = ActiveCRF(records, seed_size=args.seed_size)
    if not args.reset_session:
        try:
            saved_labels = load_session(session_path, document_hash)
            if saved_labels is not None:
                crf.restore_labels(saved_labels)
                console.print(
                    f"[green]Session reprise : {crf.annotated_count} annotations chargées.[/green]"
                )
        except (json.JSONDecodeError, ValueError) as error:
            console.print(f"[bold red]Erreur de session :[/bold red] {error}")
            return
    mapping = {
        "d": "DEBUT_ENTREE",
        "s": "SUITE_ENTREE",
        "t": "TITRE",
        "b": "BRUIT",
    }

    while True:
        selection = crf.next_block()
        if selection is None:
            console.print(
                "\n[bold green]🎉 Annotation terminée pour tout le document ![/bold green]"
            )
            break
        target_idx, block_indices = selection

        display_dashboard(crf, target_idx, block_indices)
        if len(block_indices) == 1:
            console.print(
                "\n[dim]Dernière ligne non annotée : bloc réduit à une ligne.[/dim]"
            )

        pending_labels: dict[int, str] = {}
        cancelled = False
        for index in block_indices:
            console.print(
                f"\n[bold]Label source {crf.source_line_numbers[index]} ?[/bold] "
                "(d=DEBUT, s=SUITE, t=TITRE, b=BRUIT, q=Quitter) : ",
                end="",
            )
            choice = input().strip().lower()
            if choice == "q":
                cancelled = True
                break
            if choice not in mapping:
                console.print("[yellow]Choix invalide : bloc non enregistré.[/yellow]")
                cancelled = True
                break
            pending_labels[index] = mapping[choice]

        if cancelled:
            break
        crf.set_labels(pending_labels)
        save_session(session_path, document_hash, crf)

    crf.export_csv(args.output)


if __name__ == "__main__":
    main()

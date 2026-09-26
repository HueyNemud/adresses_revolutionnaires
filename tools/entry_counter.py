import argparse
import csv
import os
import re
import sys


class TitleNode:
    def __init__(self, text, depth, raw_markdown=""):
        self.text = text
        self.depth = depth
        self.raw_markdown = raw_markdown
        self.direct_entries = 0
        self.children = []

    @property
    def total_entries(self):
        """Calcule le nombre total d'entrées (directes + enfants récursifs)."""
        return self.direct_entries + sum(child.total_entries for child in self.children)

    def print_tree(self, indent_level=0):
        indent = "  " * indent_level
        hashes = "#" * self.depth
        print(
            f"{indent}{hashes} {self.text} -> {self.total_entries} ENTRY(s) (dont {self.direct_entries} directs)"
        )
        for child in self.children:
            child.print_tree(indent_level + 1)

    def flatten(self):
        """Retourne la liste à plat de ce nœud et de tous ses descendants."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes


def parse_markdown_depth(markdown_str):
    """Extrait le niveau de titre (nombre de #) et le texte propre."""
    if not markdown_str:
        return 1, ""
    match = re.match(r"^(#+)\s*(.*)", markdown_str.strip())
    if match:
        depth = len(match.group(1))
        text = match.group(2) if match.group(2) else markdown_str.strip()
        return depth, text
    return 1, markdown_str.strip()


def process_csv(file_path, type_col=None, md_col=None, output_csv_path=None):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier '{file_path}' n'existe pas.")
        sys.exit(1)

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Détection automatique des colonnes si non spécifiées
        if not type_col:
            type_col = next(
                (
                    c
                    for c in fieldnames
                    if c.lower()
                    in ["entity", "type", "label", "ner", "ner_label", "category"]
                ),
                None,
            )
        if not md_col:
            md_col = next(
                (
                    c
                    for c in fieldnames
                    if c.lower() in ["markdown", "text", "content", "line"]
                ),
                None,
            )

        if not type_col or not md_col:
            print(
                f"Erreur : Impossible de trouver les colonnes de type ('{type_col}') et/ou markdown ('{md_col}')."
            )
            print(f"Colonnes disponibles : {fieldnames}")
            sys.exit(1)

        root = TitleNode("Racine Document", 0, "")
        stack = [root]

        for row in reader:
            row_type = str(row.get(type_col, "")).strip().upper()

            if row_type == "TITLE":
                md_val = row.get(md_col, "")
                depth, text = parse_markdown_depth(md_val)

                # Dépiler jusqu'à trouver un parent de niveau strictement inférieur
                while len(stack) > 1 and stack[-1].depth >= depth:
                    stack.pop()

                node = TitleNode(text, depth, md_val)
                stack[-1].children.append(node)
                stack.append(node)

            elif row_type == "ENTRY":
                # L'entrée est attribuée au titre actif
                stack[-1].direct_entries += 1

    # 1. Affichage dans la console
    print(f"=== Compte récursif des éléments ENTRY par TITLE ===\n")
    if root.direct_entries > 0:
        print(f"ENTRY orphelins (avant le premier titre) : {root.direct_entries}\n")
    for child in root.children:
        child.print_tree(indent_level=0)

    # 2. Génération du fichier CSV de sortie (.count.csv)
    if not output_csv_path:
        base, _ = os.path.splitext(file_path)
        output_csv_path = f"{base}.count.csv"

    all_titles = root.flatten()[1:]  # Exclut le nœud racine virtuel

    with open(output_csv_path, mode="w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["depth", "markdown", "title", "direct_entries", "total_entries_recursive"]
        )
        for node in all_titles:
            writer.writerow(
                [
                    node.depth,
                    node.raw_markdown,
                    node.text,
                    node.direct_entries,
                    node.total_entries,
                ]
            )

    print(f"\n[OK] Fichier CSV de comptage créé : {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcule et exporte le comptage direct et récursif de ENTRY par TITLE dans un fichier CSV."
    )
    parser.add_argument("csv_file", help="Chemin du fichier CSV d'entrée")
    parser.add_argument(
        "--type-col", help="Nom de la colonne indiquant le type (ex: entity, type)"
    )
    parser.add_argument(
        "--md-col", help="Nom de la colonne contenant la chaîne Markdown des titres"
    )
    parser.add_argument("--out", help="Chemin explicite du CSV de sortie")

    args = parser.parse_args()
    process_csv(args.csv_file, args.type_col, args.md_col, args.out)

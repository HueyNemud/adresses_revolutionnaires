import glob
import csv
import random
import os
from datetime import datetime

K = 5000
DIR_A = "./annuaires"
SEED = 42


def main():
    random.seed(SEED)
    files = glob.glob(os.path.join(DIR_A, "**", "*.merged.csv"), recursive=True)

    fieldnames = None
    rows = []

    for filepath in files:
        print(f"Traitement du fichier : {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            target_col = next(
                (
                    col
                    for col in reader.fieldnames
                    if col in ("entity")
                ),
                None,
            )
            if not target_col:
                continue

            if fieldnames is None:
                fieldnames = list(reader.fieldnames) + ["filename"]

            filename = os.path.basename(filepath)
            for row in reader:
                if row.get(target_col) == "ENTRY":
                    row["filename"] = filename
                    rows.append(row)

    if not rows:
        print("Aucune ligne 'ENTRY' trouvée.")
        return

    sampled_rows = random.sample(rows, min(K, len(rows)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(DIR_A, f"sample_entry_{K}_{timestamp}.csv")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"Fichier généré : {output_path} ({len(sampled_rows)} lignes)")


if __name__ == "__main__":
    main()

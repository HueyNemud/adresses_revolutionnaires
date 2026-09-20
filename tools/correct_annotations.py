import json
import re
import sys


def clean_labelstudio_annotations(data):
    """Corrige en masse les annotations Label Studio selon les 3 règles spécifiées."""
    for item in data:
        full_text = item.get("data", {}).get("text", "")
        if not full_text:
            continue

        predictions = item.get("predictions", [])
        for pred in predictions:
            results = pred.get("result", [])
            if not results:
                continue

            # Tri des entités par ordre d'apparition dans le texte
            results.sort(key=lambda x: x["value"]["start"])

            # -----------------------------------------------------------------
            # RÈGLE 2 : Parenthèses courtes en fin de SUBJ
            # -----------------------------------------------------------------
            i = 0
            while i < len(results):
                res = results[i]
                labels = res["value"].get("labels", [])

                if "SUBJ" in labels:
                    subj_text = full_text[res["value"]["start"] : res["value"]["end"]]
                    # Détection d'une parenthèse en fin d'entité SUBJ
                    match = re.search(r"(\s*\(\s*([^)]+)\s*\))\s*$", subj_text)

                    if match:
                        inner_content = match.group(2)
                        # Vérifier le premier caractère alphabétique
                        alpha_match = re.search(r"[a-zA-ZÀ-ÿ]", inner_content)

                        if alpha_match and alpha_match.group(0).islower():
                            # La parenthèse ne commence pas par une majuscule -> Extrait vers DESC
                            paren_start_rel = match.start(1)
                            abs_paren_start = res["value"]["start"] + paren_start_rel
                            abs_paren_end = res["value"]["start"] + match.end(1)

                            # Raccourcir l'entité SUBJ
                            res["value"]["end"] = abs_paren_start
                            res["value"]["text"] = full_text[
                                res["value"]["start"] : res["value"]["end"]
                            ].rstrip()
                            res["value"]["end"] = res["value"]["start"] + len(
                                res["value"]["text"]
                            )

                            # Regarder si l'entité suivante est déjà un DESC
                            next_is_desc = i + 1 < len(results) and "DESC" in results[
                                i + 1
                            ]["value"].get("labels", [])

                            if next_is_desc:
                                # Fusionner la parenthèse au début du DESC existant
                                desc_res = results[i + 1]
                                desc_res["value"]["start"] = abs_paren_start
                                desc_res["value"]["text"] = full_text[
                                    abs_paren_start : desc_res["value"]["end"]
                                ]
                            else:
                                # Créer une nouvelle entité DESC
                                new_desc = {
                                    "from_name": res.get("from_name", "label"),
                                    "to_name": res.get("to_name", "text"),
                                    "type": res.get("type", "labels"),
                                    "value": {
                                        "start": abs_paren_start,
                                        "end": abs_paren_end,
                                        "text": full_text[
                                            abs_paren_start:abs_paren_end
                                        ],
                                        "labels": ["DESC"],
                                    },
                                }
                                results.insert(i + 1, new_desc)
                i += 1

            # -----------------------------------------------------------------
            # RÈGLE 1 : ADDR suivi d'un DESC commençant par un tiret
            # -----------------------------------------------------------------
            merged_results = []
            i = 0
            while i < len(results):
                res = results[i]
                labels = res["value"].get("labels", [])

                if "ADDR" in labels and i + 1 < len(results):
                    next_res = results[i + 1]
                    next_labels = next_res["value"].get("labels", [])

                    if "DESC" in next_labels:
                        desc_text = next_res["value"].get("text", "")
                        between_and_desc = full_text[
                            res["value"]["end"] : next_res["value"]["end"]
                        ]

                        # Motif : ponctuation/espaces suivis d'un tiret (cadratin, demi-cadratin ou trait d'union)
                        if re.match(r"^[\s\.,;:!\?\-—–]*[—–\-]", desc_text) or re.match(
                            r"^[\s\.,;:!\?]*[—–\-]", between_and_desc
                        ):
                            # Fusionner DESC dans ADDR (inclut le texte non annoté entre les deux)
                            res["value"]["end"] = next_res["value"]["end"]
                            res["value"]["text"] = full_text[
                                res["value"]["start"] : res["value"]["end"]
                            ]
                            merged_results.append(res)
                            i += 2  # Sauter le DESC qui a été fusionné
                            continue

                merged_results.append(res)
                i += 1

            results = merged_results

            # -----------------------------------------------------------------
            # RÈGLE 3 : Suppression de la ponctuation en début d'entité
            # -----------------------------------------------------------------
            for res in results:
                val = res["value"]
                curr_text = full_text[val["start"] : val["end"]]

                # Détecte les espaces et séparateurs (virgules, points, tirets, etc.) en début d'entité
                lead_match = re.match(r"^[\s,;.:!\-—–]+", curr_text)
                if lead_match:
                    trim_len = len(lead_match.group(0))
                    if trim_len < len(curr_text):
                        val["start"] += trim_len
                        val["text"] = full_text[val["start"] : val["end"]]

            pred["result"] = results

    return data


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python correct_annotations.py <fichier_input.json> <fichier_output.json>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_corrected = clean_labelstudio_annotations(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_corrected, f, ensure_ascii=False, indent=2)

    print(f"Correction terminée. Fichier sauvegardé sous : {output_file}")


if __name__ == "__main__":
    main()

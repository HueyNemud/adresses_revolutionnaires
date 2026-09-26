import unittest

import numpy as np

from lib.ner.metrics import METRICS, Scored, compare, page_bootstrap, paired_delta, summarize
from lib.ner.rules import SectionLexicon, apply_rules, is_identity_group
from lib.ner.shapes import coarse_shape
from lib.ner.spans import (
    Span,
    canonical_spans,
    char_span_to_word_span,
    ls_result,
    ls_task_spans,
    normalize_markdown,
    parse_tagged_text,
    project_spans,
    render_tagged_text,
    tokenize_with_offsets,
    unproject_spans,
)


def tagged(text_with_tags: str) -> tuple[str, list[Span]]:
    return parse_tagged_text(text_with_tags)


def rules(text_with_tags: str, lexicon: SectionLexicon | None = None) -> str:
    text, spans = tagged(text_with_tags)
    result = apply_rules(text, spans, lexicon) if lexicon else apply_rules(text, spans)
    return render_tagged_text(text, result.spans)


class SpansTests(unittest.TestCase):
    def test_tagged_text_round_trip_with_escaping(self):
        source = "<SUBJ>A &lt;b&gt;</SUBJ>, <ADDR>R. X, 1.</ADDR>"
        text, spans = parse_tagged_text(source)
        self.assertEqual(text, "A <b>, R. X, 1.")
        self.assertEqual([s[:3] for s in spans], [(0, 5, "SUBJ"), (7, 15, "ADDR")])
        self.assertEqual(render_tagged_text(text, spans), source)

    def test_malformed_tags_are_rejected(self):
        for bad in ("<SUBJ>a<DESC>b</DESC></SUBJ>", "<SUBJ>a</DESC>", "<DESC>a <DESC>b</DESC>"):
            with self.assertRaises(ValueError):
                parse_tagged_text(bad)

    def test_emphasis_split_by_a_boundary_is_projected_cleanly(self):
        raw, spans = tagged("<SUBJ>**Cetto</SUBJ> <DESC>(minist.)**</DESC>, <ADDR>R. de Clichy, 356.</ADDR>")
        normalized = normalize_markdown(raw)
        self.assertEqual(normalized.text, "Cetto (minist.), R. de Clichy, 356.")
        projected = project_spans(spans, normalized)
        self.assertEqual(render_tagged_text(normalized.text, projected), "<SUBJ>Cetto</SUBJ> <DESC>(minist.)</DESC>, <ADDR>R. de Clichy, 356.</ADDR>")
        back = unproject_spans(projected, normalized)
        self.assertEqual([raw[s.start : s.end] for s in back], ["Cetto", "(minist.)", "R. de Clichy, 356."])

    def test_canonical_form_ignores_linking_punctuation(self):
        text = "Dupont, R. Vivienne, 44."
        with_period = [Span(0, 6, "SUBJ"), Span(8, 24, "ADDR")]
        without = [Span(0, 7, "SUBJ"), Span(8, 23, "ADDR")]
        self.assertEqual(canonical_spans(text, with_period), canonical_spans(text, without))

    def test_label_studio_round_trip_prefers_reviewed_annotation(self):
        text = "Dupont, rue A, 1."
        predicted = [Span(0, 6, "SUBJ"), Span(8, 17, "ADDR")]
        reviewed = [Span(0, 6, "SUBJ"), Span(8, 17, "DESC")]
        task = {"predictions": [{"result": ls_result(text, predicted)}], "annotations": [{"result": ls_result(text, reviewed)}]}
        self.assertEqual(ls_task_spans(task), reviewed)
        self.assertEqual(ls_task_spans(task, prefer="predictions"), predicted)

    def test_label_studio_export_format_keeps_prediction_inside_annotation(self):
        # Export JSON de Label Studio : `predictions` ne contient que des identifiants.
        text = "Dupont, rue A, 1."
        predicted = [Span(0, 6, "SUBJ"), Span(8, 17, "ADDR")]
        reviewed = [Span(0, 6, "SUBJ"), Span(8, 17, "DESC")]
        task = {
            "predictions": [8957],
            "annotations": [{"result": ls_result(text, reviewed), "prediction": {"id": 8957, "result": ls_result(text, predicted)}}],
        }
        self.assertEqual(ls_task_spans(task), reviewed)
        self.assertEqual(ls_task_spans(task, prefer="predictions"), predicted)

    def test_char_to_word_span_snaps_to_overlapping_words(self):
        tokens = tokenize_with_offsets("Dupont, rue A, 1.")
        self.assertEqual(char_span_to_word_span(tokens, 0, 6), (0, 0))
        self.assertEqual(char_span_to_word_span(tokens, 8, 17), (1, 3))
        self.assertIsNone(char_span_to_word_span(tokens, 30, 40))


class ShapeTests(unittest.TestCase):
    def test_base_cases(self):
        self.assertEqual(coarse_shape("Fouteau-Beauregard, R. de Grenelle S. Honoré, 54."), "w , w , 9")
        self.assertEqual(coarse_shape("Bourdois, R. S.-Honoré, 87. — Pl. Vendôme."), "w , w , 9 — w")
        self.assertEqual(coarse_shape("Andry ( Ch. L. Fr. ), R. des Ecouffes, 8."), "w ( w ) , w , 9")


class RulesTests(unittest.TestCase):
    def test_identity_groups(self):
        for inner in (" Ch. L. Fr. ", "le jeune", "Mad.", "Ve. de", "frères", " lomb. Serilly ", "Moysse et Sollivet", "de"):
            self.assertTrue(is_identity_group(inner), inner)
        for inner in (" histoire ", "de Malte", " du grand hospice ", "fourbiss.", "bureau de correspondance", "général"):
            self.assertFalse(is_identity_group(inner), inner)

    def test_lafitte_first_names_and_qualifier_go_to_subj(self):
        self.assertEqual(
            rules("<SUBJ>Lafitte</SUBJ> <DESC>(le jeune), ( J. Bapt. )</DESC>, <ADDR>R. Favart, 425. — Lepelletier.</ADDR>"),
            "<SUBJ>Lafitte (le jeune), ( J. Bapt. )</SUBJ>, <ADDR>R. Favart, 425. — Lepelletier.</ADDR>",
        )

    def test_identity_then_activity_is_split(self):
        self.assertEqual(
            rules("<SUBJ>Ronder</SUBJ> <DESC>(veuve) (en gros)</DESC>, <ADDR>Q. de la Tournelle, 113.</ADDR>"),
            "<SUBJ>Ronder (veuve)</SUBJ> <DESC>(en gros)</DESC>, <ADDR>Q. de la Tournelle, 113.</ADDR>",
        )
        self.assertEqual(
            rules("<SUBJ>Chevallier</SUBJ>, <DESC>aîné ( en épicerie )</DESC>, <ADDR>R. S. Martin, 56.</ADDR>"),
            "<SUBJ>Chevallier, aîné</SUBJ> <DESC>( en épicerie )</DESC>, <ADDR>R. S. Martin, 56.</ADDR>",
        )

    def test_activity_parenthesis_stays_desc(self):
        source = "<SUBJ>Bresler</SUBJ> <DESC>(piano)</DESC>, <ADDR>R. Ste. Avoie, 155. — Réunion.</ADDR>"
        self.assertEqual(rules(source), source)

    def test_trailing_section_and_landmark_go_to_addr(self):
        self.assertEqual(
            rules("<SUBJ>Pajot</SUBJ>, <ADDR>R. S. Denis, 19</ADDR>. <DESC>Amis de la Patrie.</DESC>"),
            "<SUBJ>Pajot</SUBJ>, <ADDR>R. S. Denis, 19. Amis de la Patrie.</ADDR>",
        )
        self.assertEqual(
            rules("<SUBJ>Carpentier</SUBJ>, <ADDR>rue du Harlay, 7.</ADDR> <DESC>(Indivisibil.)</DESC>"),
            "<SUBJ>Carpentier</SUBJ>, <ADDR>rue du Harlay, 7. (Indivisibil.)</ADDR>",
        )
        self.assertEqual(
            rules("<SUBJ>Baudry</SUBJ>, <ADDR>R. S. Denis</ADDR>, <DESC>près le marché.</DESC>"),
            "<SUBJ>Baudry</SUBJ>, <ADDR>R. S. Denis, près le marché.</ADDR>",
        )

    def test_dash_section_merged_but_not_prices(self):
        self.assertEqual(
            rules("<SUBJ>Morlay</SUBJ>, <ADDR>R. d'Anjou, 970.</ADDR> <DESC>— Roule.</DESC>"),
            "<SUBJ>Morlay</SUBJ>, <ADDR>R. d'Anjou, 970. — Roule.</ADDR>",
        )
        source = "<SUBJ>Affiches</SUBJ>, <ADDR>R. N. S. Augustin, 582.</ADDR> — <DESC>12 f. pour 3 mois.</DESC>"
        self.assertEqual(rules(source), source)

    def test_titles_and_signboards_are_not_identity(self):
        source = "<SUBJ>Maloet</SUBJ> <DESC>(D. R.)</DESC>, <ADDR>R. N. S. Augustin, 930.</ADDR>"
        self.assertEqual(rules(source), source)
        source = "<SUBJ>Lebrun</SUBJ> <DESC>(Café du Berceau Lyrique)</DESC>, <ADDR>Palais du Tribunal, 103.</ADDR>"
        self.assertEqual(rules(source), source)
        self.assertEqual(
            rules("<SUBJ>Noleau ( veuve )</SUBJ> <DESC>et fils ( flaconn. )</DESC> , <ADDR>R. Bourg l'Abbé , 17.</ADDR>"),
            "<SUBJ>Noleau ( veuve ) et fils</SUBJ> <DESC>( flaconn. )</DESC> , <ADDR>R. Bourg l'Abbé , 17.</ADDR>",
        )

    def test_orphan_text_is_filled(self):
        self.assertEqual(
            rules("<SUBJ>Bary</SUBJ>, marché Boulainvilliers, 13."),
            "<SUBJ>Bary</SUBJ>, <ADDR>marché Boulainvilliers, 13.</ADDR>",
        )
        self.assertEqual(rules("<SUBJ>Geoffroy</SUBJ> (Réné Cl.)."), "<SUBJ>Geoffroy (Réné Cl.)</SUBJ>.")
        self.assertEqual(
            rules("Caille <DESC>(A. F.)</DESC> , <ADDR>rue Hautefeuille , 22.</ADDR>"),
            "<SUBJ>Caille (A. F.)</SUBJ> , <ADDR>rue Hautefeuille , 22.</ADDR>",
        )
        self.assertEqual(
            rules("<SUBJ>Le Maire</SUBJ>, mécaniciens, fabriquent les peignes; <ADDR>rue St.-Denis, 315</ADDR>."),
            "<SUBJ>Le Maire</SUBJ>, <DESC>mécaniciens, fabriquent les peignes</DESC>; <ADDR>rue St.-Denis, 315</ADDR>.",
        )
        self.assertEqual(
            rules("<SUBJ>Boquet</SUBJ> <DESC>(paysages)</DESC>, à l'Abbaye S. Germain, R. Childebert, 909."),
            "<SUBJ>Boquet</SUBJ> <DESC>(paysages)</DESC>, <ADDR>à l'Abbaye S. Germain, R. Childebert, 909.</ADDR>",
        )
        self.assertEqual(rules("Correspondance des Professeurs."), "Correspondance des Professeurs.")

    def test_address_like_desc_and_adjacent_merge(self):
        self.assertEqual(
            rules("<SUBJ>Briard (Me.)</SUBJ>, <DESC>pal. du Trib.</DESC>, passage du Perron, 94."),
            "<SUBJ>Briard (Me.)</SUBJ>, <ADDR>pal. du Trib., passage du Perron, 94.</ADDR>",
        )
        for source in (
            "<SUBJ>Douay</SUBJ>, <DESC>petite mercerie</DESC>, <ADDR>rue Transnonain, 37.</ADDR>",
            "<SUBJ>Delapierre (Mlle.)</SUBJ> <DESC>(port.)</DESC>, <ADDR>R. S. Thomas du Louvre, 242.</ADDR>",
        ):
            self.assertEqual(rules(source), source)
        self.assertEqual(
            rules("<SUBJ>Walblet</SUBJ>, <DESC>drapier</DESC>, <DESC>nouveautés</DESC>, <ADDR>rue Vivienne, 4.</ADDR>"),
            "<SUBJ>Walblet</SUBJ>, <DESC>drapier, nouveautés</DESC>, <ADDR>rue Vivienne, 4.</ADDR>",
        )

    def test_lexicon_learns_sections_after_dash(self):
        corpus = [tagged(f"<SUBJ>X{i}</SUBJ>, <ADDR>R. Y, {i}. — Quartier Neuf.</ADDR>") for i in range(3)]
        lexicon = SectionLexicon.from_annotations(corpus)
        self.assertTrue(lexicon.matches("Quart. Neuf"))
        self.assertEqual(
            rules("<SUBJ>Z</SUBJ>, <ADDR>R. Y, 2</ADDR>. <DESC>Quartier Neuf.</DESC>", lexicon),
            "<SUBJ>Z</SUBJ>, <ADDR>R. Y, 2. Quartier Neuf.</ADDR>",
        )


class MetricsTests(unittest.TestCase):
    def test_compare_error_kinds(self):
        text, gold = tagged("<SUBJ>Lafitte (le jeune)</SUBJ>, <ADDR>R. Favart, 425.</ADDR>")
        _, same = tagged("<SUBJ>Lafitte (le jeune),</SUBJ> <ADDR>R. Favart, 425</ADDR>.")
        _, signature_error = tagged("<SUBJ>Lafitte</SUBJ> <DESC>(le jeune)</DESC>, <ADDR>R. Favart, 425.</ADDR>")
        _, boundary_error = tagged("<SUBJ>Lafitte</SUBJ> (le jeune), <ADDR>R. Favart, 425.</ADDR>")
        self.assertEqual(compare(text, gold, same).error_kind, "ok")
        self.assertEqual(compare(text, gold, signature_error).error_kind, "signature")
        comparison = compare(text, gold, boundary_error)
        self.assertEqual(comparison.error_kind, "frontière")
        # Tokens alphanumériques : Lafitte (le jeune), R. Favart, 425. → 2 sur 6 faux.
        self.assertEqual(tuple(comparison.vector[-2:]), (4, 6))

    def test_weighted_exactitude_and_paired_delta(self):
        text, gold = tagged("<SUBJ>A</SUBJ>, <ADDR>R. B, 1.</ADDR>")
        _, wrong = tagged("<SUBJ>A, R. B, 1.</SUBJ>")
        right_vector = compare(text, gold, gold).vector
        wrong_vector = compare(text, gold, wrong).vector
        weights = np.array([1.0, 3.0])
        clusters = np.array([0, 1])
        reference = Scored(np.stack([right_vector, wrong_vector]), weights, clusters)
        candidate = Scored(np.stack([right_vector, right_vector]), weights, clusters)
        self.assertAlmostEqual(reference.point()[0], 0.25)
        bootstrap = page_bootstrap(2, 200, seed=1)
        self.assertEqual(len(summarize(reference, bootstrap)), len(METRICS))
        delta, (low, high) = paired_delta(reference, candidate, bootstrap)[0]
        self.assertAlmostEqual(delta, 0.75)
        self.assertGreaterEqual(low, 0.0)


if __name__ == "__main__":
    unittest.main()

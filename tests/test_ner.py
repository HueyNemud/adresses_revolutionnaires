import unittest

import numpy as np

from numrev.ner.metrics import METRICS, Scored, compare, page_bootstrap, paired_delta, summarize
from numrev.ner.shapes import coarse_shape
from numrev.ner.spans import (
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


class SuspicionTests(unittest.TestCase):
    def reasons(self, tagged_text: str, scores=None, min_score: float = 0.9):
        from numrev.ner.suspicion import suspicion_reasons

        text, spans = parse_tagged_text(tagged_text)
        if scores:
            spans = [span._replace(score=score) for span, score in zip(spans, scores)]
        return suspicion_reasons(text, spans, min_score)

    def test_clean_entry_is_not_suspect(self):
        self.assertEqual(self.reasons("<SUBJ>Dupont</SUBJ>, <ADDR>rue A, 1.</ADDR>", [0.99, 0.98]), [])

    def test_each_reason(self):
        self.assertEqual(self.reasons("Dupont, rue A, 1."), ["aucun empan"])
        self.assertEqual(self.reasons("<SUBJ>Dupont</SUBJ>, <ADDR>rue A, 1.</ADDR>", [0.99, 0.5]), ["score bas"])
        self.assertEqual(self.reasons("<SUBJ>Dupont</SUBJ>, rue A, 1."), ["texte non couvert"])
        self.assertEqual(self.reasons("<DESC>Dupont</DESC>, <ADDR>rue A, 1.</ADDR>"), ["SUBJ absent en tête"])
        self.assertEqual(
            self.reasons("<SUBJ>Lebon</SUBJ>, <ADDR>rue B, 4.</ADDR> <SUBJ>Lejeune</SUBJ>, <ADDR>rue C, 5.</ADDR>"),
            ["signature inhabituelle"],
        )

    def test_confidence_is_minimal_span_score(self):
        from numrev.ner.suspicion import confidence

        self.assertEqual(confidence([Span(0, 1, "SUBJ", 0.9), Span(2, 3, "ADDR", 0.7)]), 0.7)
        self.assertEqual(confidence([]), 0.0)
        self.assertIsNone(confidence([Span(0, 1, "SUBJ")]))

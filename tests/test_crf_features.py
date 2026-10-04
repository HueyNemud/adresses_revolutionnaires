import unittest

from numrev.crf.features import (
    CANDIDATE_GROUPS,
    LEGACY_GROUPS,
    PLACEBO_GROUPS,
    PRODUCTION_GROUPS,
    SequenceContext,
    alphabetical_key,
    assemble_features,
    extract_features,
    extract_features_from_context,
    extract_grouped_features,
    first_word,
    normalize_line,
)


def features_of(lines, groups, **context):
    return extract_features_from_context(SequenceContext(lines, **context), groups)


class LegacyFeaturesTests(unittest.TestCase):
    def test_v1_groups_keep_historical_keys_order_and_values(self):
        features = features_of(
            ["## TITRE", "**Dupont**, rue A, 1.", "suite -"],
            LEGACY_GROUPS,
            source_line_numbers=[1, 2, 3],
            ocr_labels=["Section-Header", "Text", "Text"],
            page_positions=[0, 0, 0],
        )
        self.assertEqual(
            list(features[1]),
            [
                "bias",
                "is_heading",
                "heading_level",
                "starts_lower",
                "ends_punct",
                "token_count",
                "is_page_marker",
                "is_page_start",
                "ocr_data_block_label",
                "BOS",
                "EOS",
                "shape_start_0",
                "shape_end_0",
                "shape_start_1",
                "shape_end_1",
                "shape_start_2",
                "shape_end_2",
                "shape_start_3",
                "shape_end_3",
                "prev_is_heading",
                "prev_ends_dash",
                "previous_source_gap",
            ],
        )
        # v1 : texte brut, le gras est vu comme un symbole.
        self.assertEqual(features[1]["shape_start_0"], "SYM")
        self.assertEqual(features[1]["prev_is_heading"], "True")


class ProductionFeaturesTests(unittest.TestCase):
    def test_production_keys(self):
        features = extract_features(
            ["## TITRE", "Dupont, rue A, 1."],
            source_line_numbers=[1, 2],
            ocr_labels=["Section-Header", "Text"],
            page_positions=[0, 0],
        )
        self.assertEqual(
            list(features[1]),
            [
                "bias",
                "is_heading",
                "heading_level",
                "starts_lower",
                "ends_punct",
                "token_count",
                "is_page_start",
                "ocr_data_block_label",
                "BOS",
                "EOS",
                "shape_start_0",
                "shape_end_0",
                "shape_start_1",
                "shape_end_1",
                "shape_start_2",
                "shape_end_2",
                "shape_start_3",
                "shape_end_3",
                "italic",
                "prev_is_heading",
                "prev_end",
                "end_start",
            ],
        )
        self.assertEqual(features[0]["heading_level"], "2")
        self.assertEqual(features[0]["shape_start_0"], "UPPER")  # « ## » ignoré par les formes

    def test_markdown_emphasis_is_ignored_by_shapes(self):
        feat = features_of(["**Bluget**, rue X, 4."], ("starts_lower", "token_shapes"))[0]
        self.assertEqual(feat["shape_start_0"], "TITLE")
        self.assertEqual(feat["shape_end_0"], "PUNCT")

    def test_italic_buckets(self):
        lines = ["Dupont, rue X, 1.", "*Marchands de laines.*", "Lopineau (V.e), marchand de draps, *menue*, rue", "*ouvert sans fin"]
        values = [f["italic"] for f in features_of(lines, ("italic",))]
        self.assertEqual(values, ["none", "full", "partial", "none"])

    def test_previous_line_ending_and_transition(self):
        lines = ["Trudon, ( manufact. d'Anto-", "ine), rue X, 2.", "Guerard (suite de la maison", "**Th. Thierry**), R. T., 18.  "]
        features = features_of(lines, ("prev_line",))
        self.assertEqual(features[0], {})
        self.assertEqual((features[1]["prev_end"], features[1]["end_start"]), ("hyphen", "hyphen→lower"))
        self.assertEqual(features[2]["end_start"], "period→upper")
        self.assertEqual(features[3]["end_start"], "lower→upper")

    def test_extra_context_does_not_change_production_features(self):
        lines = ["**Dupont**, rue A, 1.", "rue B, 2."]
        minimal = SequenceContext(lines, [1, 2], ["Text", "Text"], [0, 1])
        full = SequenceContext(
            lines,
            [1, 2],
            ["Text", "Text"],
            [0, 1],
            block_keys=[(0, 0), (1, 0)],
            block_bboxes=[(0, 0, 10, 10), None],
            follows_blank=[False, True],
        )
        self.assertEqual(extract_features_from_context(minimal), extract_features_from_context(full))

    def test_grouped_features_reassemble_to_the_same_sequence(self):
        ctx = SequenceContext(["A, rue B, 1.", "suite, 2."], page_positions=[0, 0])
        groups = PRODUCTION_GROUPS + CANDIDATE_GROUPS + PLACEBO_GROUPS
        grouped = extract_grouped_features(ctx, groups)
        self.assertEqual(assemble_features(grouped, groups), extract_features_from_context(ctx, groups))

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_features_from_context(SequenceContext(["x"]), ("inexistant",))


class NormalizationTests(unittest.TestCase):
    def test_normalize_line(self):
        line = normalize_line("###\xa0*Anciens greffiers.*  ")
        self.assertEqual(line.plain, "Anciens greffiers.")
        self.assertEqual(line.italic_ratio, 1.0)
        bold = normalize_line("**Bluget**, R. S. Sauveur")
        self.assertEqual((bold.plain, bold.italic_ratio, bold.starts_bold), ("Bluget, R. S. Sauveur", 0.0, True))

    def test_alphabetical_key_and_first_word(self):
        self.assertEqual(alphabetical_key("Épée (Mme), rue"), "epee")
        self.assertEqual(alphabetical_key("12, 14."), "")
        self.assertEqual(first_word("D'Arcy, rue"), "d'")
        self.assertEqual(first_word("du Département"), "du")


class CandidateFeaturesTests(unittest.TestCase):
    def test_title_candidates(self):
        features = features_of(
            ["DOCTEURS EN MÉDECINE", "Du Département de la Seine", "Dupont, rue X, 1."],
            ("small_word_start", "block_continuation", "uppercase"),
            ocr_labels=["Section-Header", "Section-Header", "Text"],
            block_keys=[0, 0, 1],
        )
        self.assertEqual([f["first_is_small_word"] for f in features], ["False", "True", "False"])
        self.assertEqual([f["same_block_as_prev"] for f in features], ["new_block", "section_header", "new_block"])
        self.assertEqual([f["uppercase"] for f in features], ["all", "low", "mixed"])

    def test_alpha_sequence_flags_a_continuation_breaking_the_order(self):
        features = features_of(["Lopineau, rue", "S.-Martin, 27.", "Lopinot, rue S.-Denis, 208."], ("alpha_sequence",))
        self.assertEqual(features[0]["alpha_order"], "in_order")
        self.assertEqual(features[1]["alpha_order"], "after_next")

    def test_placebos_are_deterministic(self):
        ctx = SequenceContext(["a", "b", "c"])
        first = extract_features_from_context(ctx, PLACEBO_GROUPS)
        self.assertEqual(first, extract_features_from_context(ctx, PLACEBO_GROUPS))
        self.assertTrue(all(f["placebo_constant"] == "1" for f in first))


if __name__ == "__main__":
    unittest.main()

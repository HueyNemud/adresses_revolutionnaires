import unittest

from lib.crf.features import (
    CANDIDATE_GROUPS,
    PLACEBO_GROUPS,
    PRODUCTION_GROUPS,
    SequenceContext,
    alphabetical_key,
    extract_features,
    extract_features_from_context,
    extract_grouped_features,
    assemble_features,
    strip_emphasis,
)


class ProductionFeaturesTests(unittest.TestCase):
    def test_production_features_keep_historical_keys_and_order(self):
        features = extract_features(
            ["## TITRE", "Dupont, rue A, 1."],
            source_line_numbers=[1, 2],
            ocr_labels=["Section-Header", "Text"],
            page_positions=[0, 0],
        )
        self.assertEqual(
            list(features[1]),
            [
                "bias", "is_heading", "heading_level", "starts_lower", "ends_punct",
                "token_count", "is_page_marker", "is_page_start", "ocr_data_block_label",
                "BOS", "EOS",
                "shape_start_0", "shape_end_0", "shape_start_1", "shape_end_1",
                "shape_start_2", "shape_end_2", "shape_start_3", "shape_end_3",
                "prev_is_heading", "prev_ends_dash", "previous_source_gap",
            ],
        )
        self.assertEqual(features[0]["heading_level"], "2")
        self.assertEqual(features[1]["prev_is_heading"], "True")
        self.assertEqual(features[1]["ocr_data_block_label"], "text")

    def test_extra_context_does_not_change_production_features(self):
        lines = ["**Dupont**, rue A, 1.", "rue B, 2."]
        minimal = SequenceContext(lines, [1, 2], ["Text", "Text"], [0, 1])
        full = SequenceContext(
            lines, [1, 2], ["Text", "Text"], [0, 1],
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


class CandidateFeaturesTests(unittest.TestCase):
    def test_strip_emphasis_and_alphabetical_key(self):
        self.assertEqual(strip_emphasis("**Bluget**, R. S. Sauveur"), "Bluget, R. S. Sauveur")
        self.assertEqual(alphabetical_key("*Épée (Mme)*, rue"), "epee")
        self.assertEqual(alphabetical_key("12, 14."), "")

    def test_plain_shapes_ignore_bold_markers(self):
        ctx = SequenceContext(["**Bluget**, rue X, 4."])
        feat = extract_features_from_context(ctx, ("token_shapes", "token_shapes_plain"))[0]
        self.assertEqual(feat["shape_start_0"], "SYM")
        self.assertEqual(feat["plain_shape_start_0"], "TITLE")

    def test_alpha_sequence_flags_a_continuation_breaking_the_order(self):
        ctx = SequenceContext(["Lopineau, rue", "S.-Martin, 27.", "Lopinot, rue S.-Denis, 208."])
        features = extract_features_from_context(ctx, ("alpha_sequence",))
        self.assertEqual(features[0]["alpha_order"], "in_order")
        self.assertEqual(features[1]["alpha_order"], "after_next")

    def test_placebos_are_deterministic(self):
        ctx = SequenceContext(["a", "b", "c"])
        first = extract_features_from_context(ctx, PLACEBO_GROUPS)
        self.assertEqual(first, extract_features_from_context(ctx, PLACEBO_GROUPS))
        self.assertTrue(all(f["placebo_constant"] == "1" for f in first))


if __name__ == "__main__":
    unittest.main()

import unittest

from lib.chandra_document import iter_line_locations


class IterLineLocationsTests(unittest.TestCase):
    def test_yields_page_block_line_with_positions(self) -> None:
        document = [
            {
                "page_index": 0,
                "data_blocks": [
                    {
                        "index": 1,
                        "label": "Text",
                        "chunk_index": 0,
                        "lines": [{"uid": "0.1.0", "line_index": 0, "markdown": "A"}],
                    }
                ],
            }
        ]

        locations = list(iter_line_locations(document))

        self.assertEqual(len(locations), 1)
        location = locations[0]
        self.assertIs(location.page, document[0])
        self.assertIs(location.block, document[0]["data_blocks"][0])
        self.assertIs(location.line, document[0]["data_blocks"][0]["lines"][0])
        self.assertEqual((location.page_pos, location.block_pos, location.line_pos), (0, 0, 0))

    def test_rejects_a_page_missing_data_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "data_blocks"):
            list(iter_line_locations([{"page_index": 0}]))

    def test_rejects_a_block_missing_lines(self) -> None:
        document = [{"page_index": 0, "data_blocks": [{"index": 1}]}]

        with self.assertRaisesRegex(ValueError, "lines"):
            list(iter_line_locations(document))

    def test_rejects_a_non_dict_page(self) -> None:
        with self.assertRaisesRegex(ValueError, "page"):
            list(iter_line_locations(["not-a-page"]))


if __name__ == "__main__":
    unittest.main()

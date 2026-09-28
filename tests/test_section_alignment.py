import tempfile
import unittest
from pathlib import Path

from lib.alignment import SOURCE_MANUAL, Record, dedupe_records
from lib.section_alignment import (
    SOURCE_AUTO,
    SectionPatchEntry,
    align_sections,
    canonical_keys,
    canonical_training,
    corresponding,
    load_section_alignment,
    read_section_patch,
    sections,
    validate_section_patch,
    write_section_patch,
)


def records(side: str, keys: list[str], uuids: dict[str, str] | None = None) -> list[Record]:
    """Une entrée par clé de rubrique donnée (dans l'ordre) ; uuid de
    rubrique `<côté>:<clé>` sauf indication contraire."""
    uuids = uuids or {}
    return [
        Record(
            document=f"{side}.csv",
            uuid=f"{side}{order}",
            order=order,
            page="1",
            section=key,
            section_title=key.upper(),
            subj="",
            text=f"entrée {order}",
            markdown="",
            tagged_text="",
            section_uuid=uuids.get(key, f"{side}:{key}"),
        )
        for order, key in enumerate(keys)
    ]


def keys_of(group) -> tuple[list[str], list[str]]:
    return [section.key for section in group.left], [section.key for section in group.right]


LEFT = ["architectes", "architectes", "jardiniers fleuristes", "liste de non commercans", "vins"]
RIGHT = ["architectes", "listes de non commercans", "marchands d arbres", "vins"]


class SectionsTests(unittest.TestCase):
    def test_contiguous_runs(self):
        found = sections(records("g", LEFT))
        self.assertEqual([section.key for section in found], ["architectes", "jardiniers fleuristes", "liste de non commercans", "vins"])
        self.assertEqual(len(found[0].records), 2)
        self.assertEqual(found[0].uuid, "g:architectes")


class AlignSectionsTests(unittest.TestCase):
    def test_automatic_alignment_tolerates_renaming(self):
        alignment = align_sections(records("g", LEFT), records("d", RIGHT))
        self.assertEqual(
            [keys_of(group) for group in alignment.groups],
            [(["architectes"], ["architectes"]), (["liste de non commercans"], ["listes de non commercans"]), (["vins"], ["vins"])],
        )
        self.assertTrue(all(group.source == SOURCE_AUTO for group in alignment.groups))
        self.assertEqual([section.key for section in alignment.unmatched_left], ["jardiniers fleuristes"])

    def test_manual_pair_out_of_order(self):
        right = ["architectes", "listes de non commercans", "vins", "zz marchands d arbres"]
        patch = [SectionPatchEntry(left_uuid="g:jardiniers fleuristes", right_uuid="d:zz marchands d arbres")]
        alignment = align_sections(records("g", LEFT), records("d", right), patch)
        manual = [group for group in alignment.groups if group.source == SOURCE_MANUAL]
        self.assertEqual([keys_of(group) for group in manual], [(["jardiniers fleuristes"], ["zz marchands d arbres"])])
        self.assertEqual(len(alignment.groups), 4)
        self.assertEqual(alignment.unmatched_left, [])

    def test_many_to_one_group(self):
        left = ["fleuristes", "jardiniers", "vins"]
        right = ["jardiniers et fleuristes", "vins"]
        patch = [
            SectionPatchEntry(left_uuid="g:fleuristes", right_uuid="d:jardiniers et fleuristes"),
            SectionPatchEntry(left_uuid="g:jardiniers", right_uuid="d:jardiniers et fleuristes"),
        ]
        alignment = align_sections(records("g", left), records("d", right), patch)
        self.assertEqual(keys_of(alignment.groups[0]), (["fleuristes", "jardiniers"], ["jardiniers et fleuristes"]))
        self.assertEqual(
            corresponding(alignment),
            {("g:fleuristes", "d:jardiniers et fleuristes"), ("g:jardiniers", "d:jardiniers et fleuristes"), ("g:vins", "d:vins")},
        )

    def test_declared_unmatched_is_left_out_of_automatic_alignment(self):
        patch = [SectionPatchEntry(left_uuid="g:jardiniers fleuristes")]
        alignment = align_sections(records("g", LEFT), records("d", RIGHT), patch)
        self.assertNotIn("jardiniers fleuristes", [section.key for section in alignment.auto_left])
        self.assertEqual(alignment.declared, {("left", "g:jardiniers fleuristes")})
        self.assertEqual([section.key for section in alignment.unmatched_left], ["jardiniers fleuristes"])

    def test_lost_uuid_reanchored_by_title(self):
        patch = [SectionPatchEntry(left_uuid="ancien", right_uuid="d:marchands d arbres", left_title="Jardiniers - Fleuristes.")]
        alignment = align_sections(records("g", LEFT), records("d", RIGHT), patch)
        self.assertEqual(len(alignment.resolution.reanchored), 1)
        self.assertEqual(alignment.resolution.entries[0].left_uuid, "g:jardiniers fleuristes")
        orphan = [SectionPatchEntry(left_uuid="ancien", right_uuid="d:vins", left_title="Inconnue")]
        self.assertEqual(len(align_sections(records("g", LEFT), records("d", RIGHT), orphan).resolution.orphans), 1)

    def test_patch_rewritten_after_reanchoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paire.sections.csv"
            write_section_patch(path, [SectionPatchEntry(left_uuid="ancien", right_uuid="d:marchands d arbres", left_title="JARDINIERS FLEURISTES")])
            load_section_alignment(records("g", LEFT), records("d", RIGHT), path)
            self.assertEqual(read_section_patch(path)[0].left_uuid, "g:jardiniers fleuristes")


class ValidateTests(unittest.TestCase):
    def test_accepts_groups_and_singletons(self):
        validate_section_patch(
            [SectionPatchEntry(left_uuid="a", right_uuid="x"), SectionPatchEntry(left_uuid="b", right_uuid="x"), SectionPatchEntry(right_uuid="y")]
        )

    def test_rejects_inconsistencies(self):
        with self.assertRaisesRegex(ValueError, "aucun uuid"):
            validate_section_patch([SectionPatchEntry(note="rien")])
        with self.assertRaisesRegex(ValueError, "paire a ↔ x aux lignes 2, 3"):
            validate_section_patch([SectionPatchEntry(left_uuid="a", right_uuid="x")] * 2)
        with self.assertRaisesRegex(ValueError, "apparié"):
            validate_section_patch([SectionPatchEntry(left_uuid="a", right_uuid="x"), SectionPatchEntry(left_uuid="a")])


class CanonicalTests(unittest.TestCase):
    def setUp(self):
        self.alignment = align_sections(records("g", LEFT), records("d", RIGHT))
        self.left_keys, self.right_keys = canonical_keys(self.alignment)

    def test_groups_share_a_key_and_mapping_is_idempotent(self):
        self.assertEqual(self.right_keys["listes de non commercans"], "liste de non commercans")
        for keys in (self.left_keys, self.right_keys):
            for canonical in keys.values():
                self.assertEqual(self.left_keys.get(canonical, canonical), canonical)
        self.assertNotIn("jardiniers fleuristes", self.left_keys)

    def test_dedupe_records(self):
        right = records("d", RIGHT)
        self.assertEqual(dedupe_records(right, self.right_keys)["d1"]["section"], "liste de non commercans")
        self.assertEqual(dedupe_records(right)["d1"]["section"], "listes de non commercans")

    def test_training_pairs_rewritten_by_side(self):
        pair = [{"section": "liste de non commercans", "subj": "a", "text": "a"}, {"section": "listes de non commercans", "subj": "a", "text": "a"}]
        data = {"match": [{"__class__": "tuple", "__value__": pair}], "distinct": [[pair[0], {"section": None, "subj": "b", "text": "b"}]]}
        rewritten = canonical_training(data, self.left_keys, self.right_keys)
        self.assertEqual([record["section"] for record in rewritten["match"][0]["__value__"]], ["liste de non commercans"] * 2)
        self.assertEqual(rewritten["match"][0]["__class__"], "tuple")
        self.assertIsNone(rewritten["distinct"][0][1]["section"])


if __name__ == "__main__":
    unittest.main()

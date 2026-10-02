import tempfile
import unittest
from pathlib import Path

from lib.alignment import SOURCE_DEDUPE, SOURCE_MANUAL, SOURCE_MANUAL_UNCERTAIN, Link, Record, clean_text, clean_title
from lib.titles import title_text
from lib.alignment_patch import (
    PatchEntry,
    anchor_key,
    apply_patch,
    entry_from_records,
    read_patch,
    resolve,
    updated_patch,
    validate,
    write_patch,
)


def record(uuid: str, markdown: str, title: str = "## AGENS DE CHANGE.", document: str = "Vol.1-9.csv", order: int = 0) -> Record:
    return Record(
        document=document,
        uuid=uuid,
        order=order,
        page="1",
        section=clean_title(title),
        section_title=title_text(title),
        subj="",
        text=clean_text(markdown),
        markdown=markdown,
        tagged_text=f"<SUBJ>{markdown}</SUBJ>",
    )


def by_uuid(*records: Record) -> dict[str, Record]:
    return {r.uuid: r for r in records}


class ValidateTests(unittest.TestCase):
    def test_accepts_pairs_and_singletons(self):
        validate([PatchEntry(left_uuid="a", right_uuid="x"), PatchEntry(left_uuid="b"), PatchEntry(right_uuid="y")])

    def test_rejects_empty_line_and_duplicate_uuid(self):
        with self.assertRaisesRegex(ValueError, "ligne 2 : aucun uuid"):
            validate([PatchEntry(note="rien")])
        with self.assertRaisesRegex(ValueError, "left a présent aux lignes 2, 3"):
            validate([PatchEntry(left_uuid="a", right_uuid="x"), PatchEntry(left_uuid="a")])

    def test_same_uuid_on_both_sides_is_not_a_conflict(self):
        validate([PatchEntry(left_uuid="a"), PatchEntry(right_uuid="a")])

    def test_certitude_is_empty_or_uncertain(self):
        validate([PatchEntry(left_uuid="a", right_uuid="x", certitude="incertaine"), PatchEntry(left_uuid="b", certitude="Incertaine")])
        with self.assertRaisesRegex(ValueError, "ligne 2 : certitude « peut-être »"):
            validate([PatchEntry(left_uuid="a", certitude="peut-être")])


class ApplyPatchTests(unittest.TestCase):
    LINKS = [Link("a", "x", 0.9), Link("b", "y", 0.8), Link("c", "z", 0.7)]

    def test_manual_pair_overrides_both_dedupe_links(self):
        links, stats = apply_patch(self.LINKS, [PatchEntry(left_uuid="a", right_uuid="y")])
        self.assertEqual(set(links), {Link("c", "z", 0.7), Link("a", "y", None, SOURCE_MANUAL)})
        self.assertEqual((stats.overridden, stats.manual_pairs), (2, 1))

    def test_singleton_removes_its_link(self):
        links, stats = apply_patch(self.LINKS, [PatchEntry(right_uuid="z")])
        self.assertEqual(links, self.LINKS[:2])
        self.assertEqual((stats.overridden, stats.unmatched_right), (1, 1))

    def test_uncertain_pair_has_its_own_source(self):
        links, stats = apply_patch(self.LINKS, [PatchEntry(left_uuid="b", right_uuid="y", certitude="incertaine")])
        self.assertIn(Link("b", "y", 0.8, SOURCE_MANUAL_UNCERTAIN), links)
        self.assertEqual(stats.manual_pairs, 1)

    def test_validated_pair_keeps_its_score(self):
        links, stats = apply_patch(self.LINKS, [PatchEntry(left_uuid="b", right_uuid="y")])
        self.assertIn(Link("b", "y", 0.8, SOURCE_MANUAL), links)
        self.assertEqual(len(links), 3)
        self.assertEqual(stats.overridden, 0)
        self.assertTrue(all(link.source == SOURCE_DEDUPE for link in links if link.left_uuid != "b"))


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.left = by_uuid(record("a", "Dupont, rue A."), record("b", "Durand, rue B."))
        self.right = by_uuid(record("x", "Dupont, rue C."))

    def test_known_uuids_refresh_the_snapshot(self):
        resolution = resolve([PatchEntry(left_uuid="a", right_uuid="x")], self.left, self.right)
        self.assertEqual(resolution.orphans, [])
        self.assertEqual(resolution.reanchored, [])
        entry = resolution.entries[0]
        self.assertEqual((entry.left_section, entry.left_tagged_text), ("AGENS DE CHANGE.", "<SUBJ>Dupont, rue A.</SUBJ>"))

    def test_lost_uuid_is_reanchored_by_section_and_text(self):
        stale = PatchEntry(left_uuid="ancien", left_section="AGENS DE CHANGE", left_tagged_text="<SUBJ>**Durand**</SUBJ>, rue B.")
        resolution = resolve([stale], self.left, self.right)
        self.assertEqual(resolution.entries[0].left_uuid, "b")
        self.assertEqual(resolution.reanchored, [(stale, resolution.entries[0])])

    def test_ambiguous_or_moved_entry_becomes_orphan(self):
        left = self.left | by_uuid(record("c", "Durand, rue B.", order=2))
        stale = PatchEntry(left_uuid="ancien", right_uuid="x", left_section="AGENS DE CHANGE", left_tagged_text="Durand, rue B.")
        self.assertEqual(resolve([stale], left, self.right).orphans, [stale])
        other_section = PatchEntry(left_uuid="ancien", left_section="BANQUIERS", left_tagged_text="Durand, rue B.")
        self.assertEqual(resolve([other_section], self.left, self.right).orphans, [other_section])

    def test_updated_patch_keeps_order_and_orphans(self):
        orphan = PatchEntry(right_uuid="perdu", right_tagged_text="Inconnu")
        stale = PatchEntry(left_uuid="ancien", left_section="AGENS DE CHANGE", left_tagged_text="Durand, rue B.")
        entries = [orphan, stale]
        resolution = resolve(entries, self.left, self.right)
        self.assertEqual([e.left_uuid or e.right_uuid for e in updated_patch(entries, resolution)], ["perdu", "b"])

    def test_reanchoring_never_reuses_an_uuid_already_in_the_patch(self):
        taken = PatchEntry(left_uuid="b")
        stale = PatchEntry(left_uuid="ancien", left_section="AGENS DE CHANGE", left_tagged_text="Durand, rue B.")
        self.assertEqual(resolve([taken, stale], self.left, self.right).orphans, [stale])

    def test_anchor_key_matches_record_fields(self):
        r = record("a", "**Dupont**, rue A.", title="##\xa0**BOIS. ( MARCHANDS DE )**")
        self.assertEqual(anchor_key(r.section_title, r.tagged_text), (r.section, clean_text("Dupont, rue A.")))


class EditingTests(unittest.TestCase):
    def test_entry_from_records_and_roundtrip(self):
        left, right = record("a", "Dupont, rue A."), record("x", "Dupont, rue C.", document="Vol2.1-9.csv")
        entries = [entry_from_records(left, right, note="vérifié"), entry_from_records(None, right)]
        self.assertEqual(entries[1].left_uuid, "")
        self.assertEqual(entries[1].right_file, "Vol2.1-9.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sous-dossier" / "p.patch.csv"
            write_patch(path, entries)
            self.assertEqual(read_patch(path), entries)
            self.assertEqual(read_patch(Path(tmp) / "absent.csv"), [])

    def test_patch_without_certitude_column_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ancien.patch.csv"
            path.write_text("left_uuid,right_uuid,note\na,x,vu\n", encoding="utf-8")
            self.assertEqual(read_patch(path), [PatchEntry(left_uuid="a", right_uuid="x", note="vu")])
        entry = entry_from_records(record("a", "Dupont"), record("x", "Dupont"), certitude="incertaine")
        self.assertTrue(entry.is_uncertain)


if __name__ == "__main__":
    unittest.main()

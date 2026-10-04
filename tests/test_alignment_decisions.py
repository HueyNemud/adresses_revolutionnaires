import tempfile
import unittest
from pathlib import Path

from numrev.alignment.decisions import (
    ALONE,
    DIFFERENT,
    PROBABLE,
    SAME,
    UNDO,
    apply_decisions,
    decide,
    from_json,
    import_patch,
    patch_digest,
    to_json,
    touched_uuids,
)
from numrev.alignment.patch import UNCERTAIN, PatchEntry, validate
from numrev.alignment.records import Record


def record(uuid: str, text: str = "Martin, rue A, 1.") -> Record:
    return Record("Vol.1-9.ner.csv", uuid, 0, "1", "vins", "VINS", "Martin", text, text, f"<SUBJ>{text}</SUBJ>")


G1, G2, D1, D2 = record("g1"), record("g2"), record("d1"), record("d2")


def pairs(entries: list[PatchEntry]) -> list[tuple[str, str, str]]:
    return [(entry.left_uuid, entry.right_uuid, entry.certitude) for entry in entries]


class DecideTests(unittest.TestCase):
    def test_actions(self):
        self.assertEqual(pairs(decide(SAME, G1, D1).entries), [("g1", "d1", "")])
        self.assertEqual(pairs(decide(PROBABLE, G1, D1).entries), [("g1", "d1", UNCERTAIN)])
        self.assertEqual(pairs(decide(DIFFERENT, G1, D1).entries), [("g1", "", ""), ("", "d1", "")])
        self.assertEqual(pairs(decide(ALONE, None, D1).entries), [("", "d1", "")])
        self.assertEqual(decide(UNDO, G1, D1).entries, ())
        self.assertEqual(decide(SAME, G1, D1, note="vu").entries[0].note, "vu")

    def test_invalid_actions(self):
        with self.assertRaises(ValueError):
            decide(SAME, G1, None)
        with self.assertRaises(ValueError):
            decide(ALONE, G1, D1)
        with self.assertRaises(ValueError):
            decide(UNDO, None, None)


class ApplyTests(unittest.TestCase):
    def test_decision_replaces_lines_touching_its_uuids(self):
        base = [PatchEntry(left_uuid="g1", right_uuid="d2"), PatchEntry(left_uuid="g2")]
        effective = apply_decisions(base, [decide(SAME, G1, D1)])
        # g1–d2 retirée (g1 touché), g2 seule gardée, g1–d1 ajoutée
        self.assertEqual(pairs(effective), [("g2", "", ""), ("g1", "d1", "")])
        validate(effective)

    def test_different_then_repair(self):
        journal = [decide(DIFFERENT, G1, D1), decide(SAME, G1, D2)]
        effective = apply_decisions([], journal)
        self.assertEqual(pairs(effective), [("", "d1", ""), ("g1", "d2", "")])
        validate(effective)

    def test_undo_restores_automatic_alignment(self):
        base = [PatchEntry(left_uuid="g1", right_uuid="d1")]
        self.assertEqual(apply_decisions(base, [decide(UNDO, G1, D1)]), [])
        self.assertEqual(apply_decisions(base, []), base)  # retirer la dernière décision du journal = l'annuler

    def test_touched(self):
        self.assertEqual(touched_uuids([decide(SAME, G1, D1), decide(ALONE, G2, None)]), {"g1", "d1", "g2"})


class JournalTests(unittest.TestCase):
    def test_json_round_trip(self):
        journal = [decide(PROBABLE, G1, D1, note="homonymes"), decide(DIFFERENT, G2, D2), decide(UNDO, G1, None)]
        decisions, digest = from_json(to_json(journal, "abc"))
        self.assertEqual(decisions, journal)
        self.assertEqual(digest, "abc")

    def test_unreadable_journal(self):
        for text in ("", "{}", '{"version": 1}', '{"version": 99, "decisions": []}', "[1]"):
            with self.assertRaises(ValueError):
                from_json(text)

    def test_import_patch(self):
        entries = [PatchEntry(left_uuid="g1", right_uuid="d1", certitude=UNCERTAIN), PatchEntry(right_uuid="d2")]
        decisions = import_patch(entries)
        self.assertEqual([d.action for d in decisions], [PROBABLE, ALONE])
        self.assertEqual(apply_decisions([PatchEntry(left_uuid="g1", right_uuid="d2")], decisions), entries)

    def test_patch_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            self.assertEqual(patch_digest(path), "")
            path.write_text("x")
            self.assertEqual(len(patch_digest(path)), 40)


if __name__ == "__main__":
    unittest.main()

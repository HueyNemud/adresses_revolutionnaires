"""Décisions de relecture prises dans le viewer d'alignement, avant
enregistrement dans le patch (`numrev/alignment/patch.py`).

Le viewer tient un **journal** : une liste ordonnée de décisions, rejouée
sur le patch versionné pour obtenir le patch effectif. Chaque décision
retire les lignes du patch qui touchent ses uuid, puis ajoute ses propres
lignes ; le patch effectif reste donc valide (`validate` : un uuid par ligne
au plus) et annuler la dernière décision revient à la retirer du journal.

Actions (format du patch inchangé) :

- `SAME` : les deux entrées se correspondent (une paire) ;
- `PROBABLE` : idem, `certitude = incertaine` ;
- `DIFFERENT` : ce ne sont pas les mêmes entrées. Le patch ne sait pas dire
  « pas avec celle-là » : les deux entrées sont déclarées sans
  correspondance (deux lignes à un seul uuid) ; on ré-apparie ensuite celle
  qui a un autre partenaire ;
- `ALONE` : une entrée seule, confirmée sans correspondance ;
- `UNDO` : retire les lignes du patch qui touchent ces entrées
  (l'alignement automatique reprend la main).

Le journal se sérialise en JSON (stockage dans le navigateur) avec
l'empreinte du patch versionné sur lequel il a été construit.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from numrev.alignment.patch import UNCERTAIN, PatchEntry, entry_from_records
from numrev.alignment.records import Record

SAME = "meme"
PROBABLE = "probable"
DIFFERENT = "differente"
ALONE = "seule"
UNDO = "annuler"
ACTION_LABELS = {
    SAME: "même entrée",
    PROBABLE: "même entrée, incertaine",
    DIFFERENT: "pas la même entrée",
    ALONE: "sans correspondance",
    UNDO: "décision annulée",
}
JOURNAL_VERSION = 1


@dataclass(frozen=True)
class Decision:
    action: str  # clé de ACTION_LABELS
    touched: tuple[tuple[str, str], ...]  # (côté, uuid) dont les lignes du patch sont retirées
    entries: tuple[PatchEntry, ...]  # lignes ajoutées ensuite
    at: str = ""  # horodatage ISO, pour l'affichage du journal

    def summary(self) -> str:
        """Texte court pour la liste du journal."""
        texts = [getattr(entry, f"{side}_tagged_text") for entry in self.entries for side, _ in entry.uuids()]
        return f"{ACTION_LABELS.get(self.action, self.action)} : " + " ↔ ".join(texts or [uuid for _, uuid in self.touched])


def decide(action: str, left: Record | None, right: Record | None, note: str = "") -> Decision:
    """La décision `action` sur cette paire (ou cette entrée seule). Lève
    ValueError si l'action ne s'applique pas aux entrées données."""
    sides = [("left", left), ("right", right)]
    touched = tuple((side, record.uuid) for side, record in sides if record is not None)
    at = datetime.now().isoformat(timespec="seconds")
    if action in (SAME, PROBABLE, DIFFERENT) and (left is None or right is None):
        raise ValueError(f"« {ACTION_LABELS[action]} » demande une entrée de chaque côté")
    if action == ALONE and len(touched) != 1:
        raise ValueError("« sans correspondance » demande une seule entrée")
    if not touched:
        raise ValueError("aucune entrée")
    if action == SAME:
        entries = (entry_from_records(left, right, note=note),)
    elif action == PROBABLE:
        entries = (entry_from_records(left, right, note=note, certitude=UNCERTAIN),)
    elif action == DIFFERENT:
        entries = (entry_from_records(left, None, note=note), entry_from_records(None, right, note=note))
    elif action == ALONE:
        entries = (entry_from_records(left, right, note=note),)
    elif action == UNDO:
        entries = ()
    else:
        raise ValueError(f"action inconnue : {action}")
    return Decision(action, touched, entries, at)


def apply_decisions(base: list[PatchEntry], decisions: list[Decision]) -> list[PatchEntry]:
    """Patch effectif : `base` (dans son ordre) puis les décisions, rejouées
    dans l'ordre du journal."""
    entries = list(base)
    for decision in decisions:
        touched = set(decision.touched)
        entries = [entry for entry in entries if not touched.intersection(entry.uuids())]
        entries += decision.entries
    return entries


def touched_uuids(decisions: list[Decision]) -> set[str]:
    """uuid concernés par le journal (badge « décision locale »)."""
    return {uuid for decision in decisions for _, uuid in decision.touched}


# ----------------------------------------------------------------------
# Sérialisation
# ----------------------------------------------------------------------
def patch_digest(path: Path) -> str:
    """Empreinte du patch versionné ("" s'il n'existe pas)."""
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else ""


def to_json(decisions: list[Decision], digest: str) -> str:
    return json.dumps(
        {"version": JOURNAL_VERSION, "patch": digest, "decisions": [asdict(decision) for decision in decisions]},
        ensure_ascii=False,
        sort_keys=True,
    )


def from_json(text: str) -> tuple[list[Decision], str]:
    """(décisions, empreinte du patch de base). Lève ValueError si le texte
    n'est pas un journal de cette version."""
    try:
        data = json.loads(text)
        if data.get("version") != JOURNAL_VERSION:
            raise ValueError(f"version de journal inattendue : {data.get('version')}")
        decisions = [
            Decision(
                item["action"],
                tuple(tuple(pair) for pair in item["touched"]),
                tuple(PatchEntry(**entry) for entry in item["entries"]),
                item.get("at", ""),
            )
            for item in data["decisions"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"journal illisible : {error}") from error
    return decisions, data.get("patch", "")


def import_patch(entries: list[PatchEntry]) -> list[Decision]:
    """Décisions équivalentes à un patch importé (chaque ligne remplace celles
    qui touchent ses uuid)."""
    at = datetime.now().isoformat(timespec="seconds")
    return [
        Decision(
            (PROBABLE if entry.is_uncertain else SAME) if entry.is_pair else ALONE,
            tuple(entry.uuids()),
            (entry,),
            at,
        )
        for entry in entries
    ]

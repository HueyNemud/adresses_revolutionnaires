"""Chargement d'un modèle GLiNER entraîné et prédiction d'empans.

Libellés : GLiNER prédit les libellés descriptifs vus à l'entraînement, pas
les codes SUBJ/DESC/ADDR. `tools/train_gliner.py` enregistre la
correspondance dans `<modèle>/ner_config.json`, obligatoire au chargement :
sans lui, la conversion retour vers les codes échouerait en silence.

Le modèle travaille sur le texte normalisé (emphase Markdown retirée,
`normalize_markdown`) ; les empans prédits sont donc sur ce texte, référence
de l'évaluation, et `unproject_spans` les ramène sur le texte d'origine.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lib.ner.spans import Span, normalize_markdown, trim_spans

CONFIG_FILENAME = "ner_config.json"
# Libellés par défaut donnés au modèle à l'entraînement (l'encodeur de
# libellés est pré-entraîné en anglais).
DEFAULT_LABEL_TEXT = {
    "SUBJ": "person or business name",
    "DESC": "activity description",
    "ADDR": "postal address",
}


@dataclass(frozen=True)
class NerConfig:
    label_text: dict[str, str]

    @classmethod
    def load(cls, model_dir: Path) -> "NerConfig":
        path = model_dir / CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"{path} introuvable : modèle non entraîné par tools/train_gliner.py.")
        return cls(json.loads(path.read_text(encoding="utf-8"))["label_text"])

    def save(self, model_dir: Path) -> None:
        payload = {"label_text": self.label_text}
        (model_dir / CONFIG_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(model_dir: Path):
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()
    return model


def predict_spans(
    model,
    config: NerConfig,
    raw_texts: Sequence[str],
    threshold: float,
    batch_size: int = 16,
    on_batch: Callable[[int], None] | None = None,
) -> list[list[Span]]:
    """Empans (avec score) sur le **texte normalisé** de chaque entrée."""
    reverse = {text: code for code, text in config.label_text.items()}
    labels = list(config.label_text.values())
    inputs = [normalize_markdown(text).text for text in raw_texts]

    results: list[list[Span]] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        predictions = model.batch_predict_entities(chunk, labels, threshold=threshold)
        for text, entities in zip(chunk, predictions):
            spans = [Span(e["start"], e["end"], reverse.get(e["label"], e["label"]), float(e["score"])) for e in entities]
            results.append(trim_spans(text, spans))
        if on_batch:
            on_batch(len(chunk))
    return results

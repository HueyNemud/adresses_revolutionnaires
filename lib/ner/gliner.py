"""Chargement d'un modèle GLiNER entraîné et prédiction d'empans.

Libellés : GLiNER prédit les libellés descriptifs vus à l'entraînement, pas
les codes SUBJ/DESC/ADDR. Un modèle entraîné par une version récente de
`tools/train_gliner.py` enregistre la correspondance dans
`<modèle>/ner_config.json` ; à défaut (modèle v1), on retombe sur
`DEFAULT_LABEL_TEXT`, qui doit alors être celui de l'entraînement.

`input` indique sur quel texte le modèle a été entraîné : `raw` (colonne
`markdown`, emphase comprise, comme v1) ou `normalized` (emphase retirée).
Les empans prédits sont toujours rendus sur le texte normalisé, référence
de l'évaluation.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lib.ner.spans import NormalizedText, Span, normalize_markdown, project_spans, trim_spans

CONFIG_FILENAME = "ner_config.json"
DEFAULT_LABEL_TEXT = {
    "SUBJ": "person or business name",
    "DESC": "activity description",
    "ADDR": "postal address",
}


@dataclass(frozen=True)
class NerConfig:
    label_text: dict[str, str]
    input: str = "raw"  # "raw" ou "normalized"

    @classmethod
    def load(cls, model_dir: Path) -> "NerConfig":
        path = model_dir / CONFIG_FILENAME
        if not path.exists():
            return cls(dict(DEFAULT_LABEL_TEXT))
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["label_text"], data.get("input", "raw"))

    def save(self, model_dir: Path) -> None:
        payload = {"label_text": self.label_text, "input": self.input}
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
    normalized = [normalize_markdown(text) for text in raw_texts]
    inputs = [n.text if config.input == "normalized" else text for n, text in zip(normalized, raw_texts)]

    results: list[list[Span]] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        predictions = model.batch_predict_entities(chunk, labels, threshold=threshold)
        for index, entities in enumerate(predictions):
            spans = [Span(e["start"], e["end"], reverse.get(e["label"], e["label"]), float(e["score"])) for e in entities]
            results.append(_to_normalized(spans, normalized[start + index], config.input))
        if on_batch:
            on_batch(len(chunk))
    return results


def _to_normalized(spans: list[Span], normalized: NormalizedText, input_kind: str) -> list[Span]:
    if input_kind == "normalized":
        return trim_spans(normalized.text, spans)
    # Le texte brut donné au modèle est `markdown.strip()`, celui dont
    # `normalize_markdown` conserve les positions d'origine.
    return project_spans(spans, normalized)

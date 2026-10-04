"""Chargement d'un modèle GLiNER entraîné et prédiction d'empans.

Libellés : GLiNER prédit les libellés descriptifs vus à l'entraînement, pas
les codes SUBJ/DESC/ADDR. `numrev train` enregistre la
correspondance dans `<modèle>/ner_config.json`, obligatoire au chargement :
sans lui, la conversion retour vers les codes échouerait en silence.

Le modèle travaille sur le texte normalisé (emphase Markdown retirée,
`normalize_markdown`) ; les empans prédits sont donc sur ce texte, référence
de l'évaluation, et `unproject_spans` les ramène sur le texte d'origine.
"""

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from numrev.ner.spans import Span, normalize_markdown, trim_spans

CONFIG_FILENAME = "ner_config.json"
DEFAULT_THRESHOLD = 0.5  # score minimal d'un empan prédit (inférence, entraînement, audit)
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
            raise FileNotFoundError(f"{path} introuvable : modèle non entraîné par numrev train.")
        return cls(json.loads(path.read_text(encoding="utf-8"))["label_text"])

    def save(self, model_dir: Path) -> None:
        payload = {"label_text": self.label_text}
        (model_dir / CONFIG_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(model_dir: Path):
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()
    return model


def predict_entities(model, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[dict]]:
    """Entités GLiNER brutes d'un lot de textes. Un modèle relex (entités et
    relations) renvoie par défaut un couple (entités, relations) : on ne lui
    demande que les entités."""
    options = {"return_relations": False} if "return_relations" in inspect.signature(model.inference).parameters else {}
    return model.inference(list(texts), labels, threshold=threshold, **options)


def predict_spans(
    model,
    config: NerConfig,
    raw_texts: Sequence[str],
    threshold: float,
    batch_size: int = 16,
    on_batch: Callable[[int], None] | None = None,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[list[Span] | None]:
    """Empans (avec score) sur le **texte normalisé** de chaque entrée,
    prédits par lots. Si un lot échoue (un texte pathologique suffit), ses
    textes sont retentés un par un : un texte qui échoue encore vaut None et
    est signalé à `on_error(indice, erreur)`, sans perdre le reste du lot."""
    reverse = {text: code for code, text in config.label_text.items()}
    labels = list(config.label_text.values())
    inputs = [normalize_markdown(text).text for text in raw_texts]

    def to_spans(text: str, entities: list[dict]) -> list[Span]:
        spans = [Span(e["start"], e["end"], reverse.get(e["label"], e["label"]), float(e["score"])) for e in entities]
        return trim_spans(text, spans)

    results: list[list[Span] | None] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        try:
            predictions = predict_entities(model, chunk, labels, threshold)
            results += [to_spans(text, entities) for text, entities in zip(chunk, predictions)]
        except Exception:  # Surface large et imprévisible côté torch.
            for offset, text in enumerate(chunk):
                try:
                    results.append(to_spans(text, predict_entities(model, [text], labels, threshold)[0]))
                except Exception as error:
                    results.append(None)
                    if on_error:
                        on_error(start + offset, error)
        if on_batch:
            on_batch(len(chunk))
    return results

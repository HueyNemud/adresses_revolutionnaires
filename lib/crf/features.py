"""Extraction des features de lignes pour le CRF, organisée en groupes nommés.

Chaque groupe de features (`FeatureGroup`) calcule un petit dictionnaire
d'attributs pour une ligne, à partir du contexte complet de la séquence
(`SequenceContext`). Découper les features en groupes permet :

- de construire exactement le jeu de features utilisé en production
  (`PRODUCTION_GROUPS`, identique à l'historique de annotate_lines_crf.py,
  clés et ordre compris) ;
- d'en retirer ou d'en ajouter un groupe à la fois pour les auditer
  (voir audit_crf_features.py) ;
- d'expérimenter des groupes candidats (`CANDIDATE_GROUPS`) sans toucher au
  comportement de l'annotateur tant qu'ils n'ont pas été validés.
"""

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

from lib.crf.labels import AnnotationLabel

Feature: TypeAlias = dict[str, str]
FeatureSequence: TypeAlias = list[Feature]
BBox: TypeAlias = tuple[float, float, float, float]


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
PAGE_MARKER_PATTERN = re.compile(r"\{\d+\}[-—_]*")


def tokenize(line: str) -> list[str]:
    return TOKEN_PATTERN.findall(line)


def get_shape(token: str) -> str:
    if token in (",", ".", "—", "-", "_"):
        return "PUNCT"
    if token.isdigit():
        return "NUM"
    if token.isupper():
        return "UPPER"
    if token.istitle():
        return "TITLE"
    if token.islower():
        return "LOWER"
    if token in ("*", "~_", "$", "#"):
        return "SYM"
    return "UNK"  # unknown / other


def get_heuristic_label(line: str, prev_line: str = "") -> str:
    """Retourne un indice de sélection, jamais une vérité de référence."""
    stripped = line.strip()
    if not stripped:
        return AnnotationLabel.OOS.value
    if stripped.startswith("#"):
        return AnnotationLabel.BTITLE.value
    if re.fullmatch(r"\{\d+\}[-—_]*", stripped) or re.fullmatch(r"[-—_]{3,}", stripped):
        return AnnotationLabel.OOS.value
    if stripped[0].islower() or stripped.startswith(("-", "—")):
        return AnnotationLabel.IENTRY.value
    if prev_line and prev_line.rstrip().endswith(("-", "—")):
        return AnnotationLabel.IENTRY.value
    return AnnotationLabel.BENTRY.value


def normalize_ocr_label(label: str) -> str:
    """Normalize an OCR block class so equivalent spellings share one feature."""
    normalized = re.sub(r"[^\w]+", "_", label.strip().casefold())
    return normalized.strip("_") or "missing"


def parse_bbox(value: object) -> BBox | None:
    """Parse une bbox « [x0, y0, x1, y1] » (liste ou chaîne), None si absente."""
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    elif isinstance(value, (list, tuple)):
        numbers = list(value)
    else:
        return None
    if len(numbers) != 4:
        return None
    x0, y0, x1, y1 = (float(number) for number in numbers)
    return x0, y0, x1, y1


EMPHASIS_PATTERN = re.compile(r"\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)")


def strip_emphasis(line: str) -> str:
    """Retire les marqueurs d'emphase Markdown (gras, italique)."""
    return EMPHASIS_PATTERN.sub("", line)


def alphabetical_key(line: str) -> str:
    """Premier mot alphabétique, sans accents ni casse (ordre des annuaires)."""
    match = re.search(r"[^\W\d_]+", strip_emphasis(line))
    if not match:
        return ""
    decomposed = unicodedata.normalize("NFKD", match.group(0))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def line_ending_kind(line: str) -> str:
    stripped = strip_emphasis(line).rstrip()
    if not stripped:
        return "none"
    last = stripped[-1]
    if last in "-—":
        return "dash"
    if last == ",":
        return "comma"
    if last == ".":
        return "period"
    if last.isdigit():
        return "digit"
    if last.isalpha():
        return "letter"
    return "other"


# ----------------------------------------------------------------------
# Contexte de séquence
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SequenceContext:
    """Observations disponibles pour une séquence de lignes non vides.

    Seuls `lines` et les trois champs suivants sont utilisés par les
    features de production ; les champs restants ne servent qu'aux groupes
    candidats et peuvent rester à None.
    """

    lines: Sequence[str]
    source_line_numbers: Sequence[int] | None = None
    ocr_labels: Sequence[str] | None = None
    page_positions: Sequence[int] | None = None
    block_keys: Sequence[Hashable] | None = None
    block_bboxes: Sequence[BBox | None] | None = None
    follows_blank: Sequence[bool] | None = None

    def __post_init__(self) -> None:
        n = len(self.lines)
        if self.source_line_numbers is not None and len(self.source_line_numbers) != n:
            raise ValueError("Chaque ligne doit avoir un numéro de ligne source.")
        if self.ocr_labels is not None and len(self.ocr_labels) != n:
            raise ValueError("Chaque ligne doit avoir une classe de bloc OCR.")
        if self.page_positions is not None and len(self.page_positions) != n:
            raise ValueError("Chaque ligne doit avoir une position de page.")
        for name in ("block_keys", "block_bboxes", "follows_blank"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"'{name}' doit avoir une valeur par ligne.")

    def __len__(self) -> int:
        return len(self.lines)

    @cached_property
    def tokens(self) -> list[list[str]]:
        return [tokenize(line) for line in self.lines]

    @cached_property
    def shapes(self) -> list[list[str]]:
        return [[get_shape(token) for token in tokens] for tokens in self.tokens]

    @cached_property
    def plain_shapes(self) -> list[list[str]]:
        """Formes des tokens après retrait de l'emphase Markdown (*, **, _)."""
        return [[get_shape(token) for token in tokenize(strip_emphasis(line))] for line in self.lines]

    @cached_property
    def alpha_keys(self) -> list[str]:
        return [alphabetical_key(line) for line in self.lines]

    def is_page_start(self, t: int) -> bool:
        return self.page_positions is not None and (
            t == 0 or self.page_positions[t] != self.page_positions[t - 1]
        )

    @cached_property
    def page_horizontal_extent(self) -> dict[Hashable, tuple[float, float]]:
        """(x min, x max) des blocs de chaque page, pour normaliser l'indentation."""
        extents: dict[Hashable, tuple[float, float]] = {}
        if self.block_bboxes is None or self.page_positions is None:
            return extents
        for page, bbox in zip(self.page_positions, self.block_bboxes):
            if bbox is None:
                continue
            low, high = extents.get(page, (bbox[0], bbox[2]))
            extents[page] = (min(low, bbox[0]), max(high, bbox[2]))
        return extents


GroupFunction: TypeAlias = Callable[[SequenceContext, int], Feature]


PRODUCTION = "production"
CANDIDATE = "candidat"
PLACEBO = "placebo"


@dataclass(frozen=True)
class FeatureGroup:
    """Un groupe de features.

    `kind` : `production` (utilisé par l'annotateur), `candidat` (à évaluer)
    ou `placebo` (sans information, par construction : sert de témoin pour
    mesurer l'effet d'un simple changement de l'espace de features).
    """

    name: str
    description: str
    compute: GroupFunction
    kind: str = PRODUCTION


# ----------------------------------------------------------------------
# Groupes de production (ordre et clés identiques à l'historique)
# ----------------------------------------------------------------------
def _bias(ctx: SequenceContext, t: int) -> Feature:
    return {"bias": "1.0"}


def _heading(ctx: SequenceContext, t: int) -> Feature:
    line = ctx.lines[t]
    is_heading = line.startswith("#")
    heading_level = len(line) - len(line.lstrip("#")) if is_heading else 0
    return {"is_heading": str(is_heading), "heading_level": str(heading_level)}


def _starts_lower(ctx: SequenceContext, t: int) -> Feature:
    shapes = ctx.shapes[t]
    return {"starts_lower": str(shapes[0] == "LOWER") if shapes else "False"}


def _ends_punct(ctx: SequenceContext, t: int) -> Feature:
    shapes = ctx.shapes[t]
    return {"ends_punct": str(shapes[-1] == "PUNCT") if shapes else "False"}


def _token_count(ctx: SequenceContext, t: int) -> Feature:
    return {"token_count": str(min(len(ctx.tokens[t]), 12))}


def _page_marker(ctx: SequenceContext, t: int) -> Feature:
    return {"is_page_marker": str(bool(PAGE_MARKER_PATTERN.fullmatch(ctx.lines[t])))}


def _page_start(ctx: SequenceContext, t: int) -> Feature:
    return {"is_page_start": str(ctx.is_page_start(t))}


def _ocr_block(ctx: SequenceContext, t: int) -> Feature:
    return {
        "ocr_data_block_label": (
            normalize_ocr_label(ctx.ocr_labels[t])
            if ctx.ocr_labels is not None
            else "missing"
        )
    }


def _sequence_bounds(ctx: SequenceContext, t: int) -> Feature:
    return {"BOS": str(t == 0), "EOS": str(t == len(ctx) - 1)}


def _token_shapes(ctx: SequenceContext, t: int) -> Feature:
    shapes = ctx.shapes[t]
    feat: Feature = {}
    for i in range(min(4, len(shapes))):
        feat[f"shape_start_{i}"] = shapes[i]
        feat[f"shape_end_{i}"] = shapes[len(shapes) - 1 - i]
    return feat


def _prev_line(ctx: SequenceContext, t: int) -> Feature:
    if t == 0:
        return {}
    previous = ctx.lines[t - 1]
    return {
        "prev_is_heading": str(previous.startswith("#")),
        "prev_ends_dash": str(previous.rstrip().endswith(("-", "—"))),
    }


def _source_gap(ctx: SequenceContext, t: int) -> Feature:
    if t == 0 or ctx.source_line_numbers is None:
        return {}
    gap = ctx.source_line_numbers[t] - ctx.source_line_numbers[t - 1]
    return {"previous_source_gap": str(gap > 1)}


# ----------------------------------------------------------------------
# Groupes candidats (expérimentaux, absents de la production)
# ----------------------------------------------------------------------
ADDRESS_PATTERN = re.compile(
    r"\b(rue|r\.|quai|q\.|place|pl\.|faub|fg|f\.|boulevard|boul|bd|cour|passage|"
    r"pass\.|cloître|clo[iî]tre|marché|port|pont|carré|enclos|impasse|cul-de-sac|"
    r"barrière|chaussée|montagne|vieille|neuve)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[^\W\d_]{1,15}")


def _next_line(ctx: SequenceContext, t: int) -> Feature:
    if t == len(ctx) - 1:
        return {}
    following = ctx.lines[t + 1]
    shapes = ctx.shapes[t + 1]
    return {
        "next_starts_lower": str(bool(shapes) and shapes[0] == "LOWER"),
        "next_starts_dash": str(following.startswith(("-", "—"))),
        "next_is_heading": str(following.startswith("#")),
        "next_is_page_start": str(ctx.is_page_start(t + 1)),
    }


def _blank_before(ctx: SequenceContext, t: int) -> Feature:
    if ctx.follows_blank is None:
        return {}
    return {"follows_blank": str(bool(ctx.follows_blank[t]))}


def _block_position(ctx: SequenceContext, t: int) -> Feature:
    if ctx.block_keys is None:
        return {}
    keys = ctx.block_keys
    first = t == 0 or keys[t] != keys[t - 1]
    last = t == len(ctx) - 1 or keys[t] != keys[t + 1]
    return {"first_in_block": str(first), "last_in_block": str(last)}


def _line_ending(ctx: SequenceContext, t: int) -> Feature:
    return {"last_char": line_ending_kind(ctx.lines[t])}


def _address_lexicon(ctx: SequenceContext, t: int) -> Feature:
    line = ctx.lines[t]
    return {
        "has_street_word": str(bool(ADDRESS_PATTERN.search(line))),
        "ends_with_number": str(bool(re.search(r"\d+\s*[.,;]?\s*$", line))),
        "has_number": str(any(char.isdigit() for char in line)),
    }


def _char_length(ctx: SequenceContext, t: int) -> Feature:
    length = len(ctx.lines[t])
    feat = {"char_length_log2": str(min(int(math.log2(length + 1)), 8))}
    if t > 0:
        ratio = length / max(len(ctx.lines[t - 1]), 1)
        feat["length_vs_prev"] = (
            "shorter" if ratio < 0.6 else "longer" if ratio > 1.6 else "similar"
        )
    return feat


def _layout_indent(ctx: SequenceContext, t: int) -> Feature:
    if ctx.block_bboxes is None or ctx.page_positions is None:
        return {}
    bbox = ctx.block_bboxes[t]
    extent = ctx.page_horizontal_extent.get(ctx.page_positions[t])
    if bbox is None or extent is None or extent[1] <= extent[0]:
        return {"block_indent": "missing"}
    span = extent[1] - extent[0]
    indent = (bbox[0] - extent[0]) / span
    width = (bbox[2] - bbox[0]) / span
    return {
        "block_indent": str(min(int(indent * 5), 4)),
        "block_width": str(min(int(width * 4), 3)),
    }


def _lexical_words(ctx: SequenceContext, t: int) -> Feature:
    words = WORD_PATTERN.findall(ctx.lines[t])
    if not words:
        return {}
    return {"first_word": words[0].casefold(), "last_word": words[-1].casefold()}


def _prev_ending(ctx: SequenceContext, t: int) -> Feature:
    if t == 0:
        return {}
    previous = ctx.lines[t - 1]
    return {
        "prev_last_char": line_ending_kind(previous),
        "prev_ends_with_number": str(bool(re.search(r"\d+\s*[.,;]?\s*$", strip_emphasis(previous)))),
    }


def _alpha_sequence(ctx: SequenceContext, t: int) -> Feature:
    """Position alphabétique de la ligne par rapport à ses voisines : une
    continuation (« rue S.-Martin, 27. ») rompt l'ordre des noms d'entrées."""
    key = ctx.alpha_keys[t]
    if not key:
        return {"alpha_order": "no_word"}
    before = t > 0 and bool(ctx.alpha_keys[t - 1]) and key < ctx.alpha_keys[t - 1]
    after = t < len(ctx) - 1 and bool(ctx.alpha_keys[t + 1]) and key > ctx.alpha_keys[t + 1]
    kind = "out" if before and after else "before_prev" if before else "after_next" if after else "in_order"
    return {"alpha_order": kind}


def _token_shapes_plain(ctx: SequenceContext, t: int) -> Feature:
    """Comme starts_lower + token_shapes, mais après retrait de l'emphase Markdown."""
    shapes = ctx.plain_shapes[t]
    feat: Feature = {"plain_starts_lower": str(bool(shapes) and shapes[0] == "LOWER")}
    for i in range(min(4, len(shapes))):
        feat[f"plain_shape_start_{i}"] = shapes[i]
        feat[f"plain_shape_end_{i}"] = shapes[len(shapes) - 1 - i]
    return feat


def _markdown_emphasis(ctx: SequenceContext, t: int) -> Feature:
    line = ctx.lines[t].lstrip("# ")
    return {
        "starts_bold": str(line.startswith("**")),
        "starts_italic": str(line.startswith("*") and not line.startswith("**")),
        "has_emphasis": str("*" in line),
    }


def _placebo_constant(ctx: SequenceContext, t: int) -> Feature:
    return {"placebo_constant": "1"}


def _placebo_random(seed: int) -> GroupFunction:
    def compute(ctx: SequenceContext, t: int) -> Feature:
        digest = hashlib.blake2b(f"{seed}:{t}:{ctx.lines[t]}".encode(), digest_size=2).digest()
        return {f"placebo_random_{seed}": str(digest[0] & 1)}

    return compute


# ----------------------------------------------------------------------
# Registre
# ----------------------------------------------------------------------
_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("bias", "Terme constant", _bias),
    FeatureGroup("heading", "Ligne Markdown « # » et niveau de titre", _heading),
    FeatureGroup("starts_lower", "Premier token en minuscules", _starts_lower),
    FeatureGroup("ends_punct", "Dernier token = ponctuation , . — - _", _ends_punct),
    FeatureGroup("token_count", "Nombre de tokens (plafonné à 12)", _token_count),
    FeatureGroup("page_marker", "Ligne marqueur de page « {n}--- »", _page_marker),
    FeatureGroup("page_start", "Première ligne d'une nouvelle page", _page_start),
    FeatureGroup("ocr_block", "Classe du bloc OCR (Text, List-Group, ...)", _ocr_block),
    FeatureGroup("sequence_bounds", "Début / fin de séquence (BOS / EOS)", _sequence_bounds),
    FeatureGroup("token_shapes", "Formes des 4 premiers / 4 derniers tokens", _token_shapes),
    FeatureGroup("prev_line", "Ligne précédente : titre, finit par un tiret", _prev_line),
    FeatureGroup("source_gap", "Saut de numéro de ligne source > 1", _source_gap),
    FeatureGroup("next_line", "Ligne suivante : minuscule, tiret, titre, page", _next_line, CANDIDATE),
    FeatureGroup("prev_ending", "Ligne précédente : dernier caractère, finit par un nombre", _prev_ending, CANDIDATE),
    FeatureGroup("alpha_sequence", "Ordre alphabétique du 1er mot vs lignes voisines", _alpha_sequence, CANDIDATE),
    FeatureGroup("blank_before", "Ligne précédée d'une ligne vide", _blank_before, CANDIDATE),
    FeatureGroup("block_position", "Première / dernière ligne de son bloc OCR", _block_position, CANDIDATE),
    FeatureGroup("line_ending", "Nature du dernier caractère", _line_ending, CANDIDATE),
    FeatureGroup("address_lexicon", "Mots de voirie, nombres, nombre final", _address_lexicon, CANDIDATE),
    FeatureGroup("char_length", "Longueur en caractères, rapport à la précédente", _char_length, CANDIDATE),
    FeatureGroup("layout_indent", "Indentation / largeur relatives du bloc", _layout_indent, CANDIDATE),
    FeatureGroup("lexical_words", "Premier / dernier mot (forme minuscule)", _lexical_words, CANDIDATE),
    FeatureGroup("markdown_emphasis", "Gras / italique Markdown en début de ligne", _markdown_emphasis, CANDIDATE),
    FeatureGroup("token_shapes_plain", "starts_lower + token_shapes sans emphase Markdown", _token_shapes_plain, CANDIDATE),
    FeatureGroup("placebo_constant", "Témoin : attribut constant", _placebo_constant, PLACEBO),
    *(
        FeatureGroup(f"placebo_random_{seed}", f"Témoin : bit pseudo-aléatoire (graine {seed})", _placebo_random(seed), PLACEBO)
        for seed in (1, 2, 3)
    ),
)
FEATURE_GROUPS: dict[str, FeatureGroup] = {group.name: group for group in _GROUPS}
PRODUCTION_GROUPS: tuple[str, ...] = tuple(g.name for g in _GROUPS if g.kind == PRODUCTION)
CANDIDATE_GROUPS: tuple[str, ...] = tuple(g.name for g in _GROUPS if g.kind == CANDIDATE)
PLACEBO_GROUPS: tuple[str, ...] = tuple(g.name for g in _GROUPS if g.kind == PLACEBO)

GroupedFeatures: TypeAlias = list[dict[str, Feature]]


def extract_grouped_features(
    ctx: SequenceContext, groups: Sequence[str] = PRODUCTION_GROUPS
) -> GroupedFeatures:
    """Features de chaque ligne, séparées par groupe : {groupe: {clé: valeur}}."""
    unknown = [name for name in groups if name not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Groupes de features inconnus : {unknown}")
    functions = [(name, FEATURE_GROUPS[name].compute) for name in groups]
    return [{name: compute(ctx, t) for name, compute in functions} for t in range(len(ctx))]


def assemble_features(grouped: GroupedFeatures, groups: Sequence[str]) -> FeatureSequence:
    """Fusionne les groupes choisis, dans l'ordre donné, en une séquence CRF."""
    sequence: FeatureSequence = []
    for line_groups in grouped:
        feat: Feature = {}
        for name in groups:
            feat.update(line_groups[name])
        sequence.append(feat)
    return sequence


def extract_features_from_context(
    ctx: SequenceContext, groups: Sequence[str] = PRODUCTION_GROUPS
) -> FeatureSequence:
    return assemble_features(extract_grouped_features(ctx, groups), groups)


def extract_features(
    lines: list[str],
    source_line_numbers: list[int] | None = None,
    ocr_labels: list[str] | None = None,
    page_positions: list[int] | None = None,
) -> FeatureSequence:
    """Features de production (API historique de annotate_lines_crf.py)."""
    return extract_features_from_context(
        SequenceContext(lines, source_line_numbers, ocr_labels, page_positions)
    )

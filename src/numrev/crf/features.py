"""Extraction des features de lignes pour le CRF, organisée en groupes nommés.

Chaque groupe de features (`FeatureGroup`) calcule un petit dictionnaire
d'attributs pour une ligne, à partir du contexte complet de la séquence
(`SequenceContext`). Découper les features en groupes permet :

- de construire exactement le jeu de features utilisé en production
  (`PRODUCTION_GROUPS`) ;
- de reconstituer à l'identique le jeu historique (`LEGACY_GROUPS`,
  « production_v1 »), pour mesurer l'apport des évolutions ;
- d'en retirer ou d'en ajouter un groupe à la fois pour les auditer
  (voir numrev audit crf) ;
- d'expérimenter des groupes candidats (`CANDIDATE_GROUPS`) sans toucher au
  comportement de l'annotateur tant qu'ils n'ont pas été validés.

Typographie : les features de production sont calculées sur le texte
normalisé (`normalize_line`) — sans marqueurs d'emphase Markdown (`*`, `_`),
sans « # » de tête ni espaces de fin — car la mise en forme produite par
l'OCR varie d'un volume à l'autre. Seuls les « # » (groupe `heading`) et la
proportion d'italique (groupe `italic`, les sous-titres étant souvent en
italique sans « # ») sont conservés comme signaux.
"""

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

from numrev.crf.labels import AnnotationLabel

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
    if line.strip().startswith("#"):
        return AnnotationLabel.BTITLE.value
    stripped = normalize_line(line).plain.strip()
    if not stripped or re.fullmatch(r"[-—_]{3,}", stripped):
        return AnnotationLabel.OOS.value
    if stripped[0].islower() or stripped.startswith(("-", "—")):
        return AnnotationLabel.IENTRY.value
    if prev_line and normalize_line(prev_line).plain.endswith(("-", "—")):
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
HEADING_PREFIX_PATTERN = re.compile(r"^\s*#+\s*")


@dataclass(frozen=True)
class NormalizedLine:
    """Ligne débarrassée de sa typographie Markdown, et ce qu'on en garde."""

    plain: str
    italic_ratio: float
    starts_bold: bool


def normalize_line(line: str) -> NormalizedLine:
    """Retire « # » de tête, espaces insécables / de fin et marqueurs d'emphase.

    `italic_ratio` est la part des caractères non blancs du texte situés
    entre deux marqueurs d'italique appariés (`*…*`, `_…_`, `***…***`) ; un
    marqueur non apparié (italique commencé sur la ligne précédente, ou
    poursuivi sur la suivante) est ignoré.
    """
    text = HEADING_PREFIX_PATTERN.sub("", line.replace("\xa0", " "))
    starts_bold = text.startswith(("**", "__"))
    plain_parts: list[str] = []
    italic_markers: list[int] = []  # positions (dans le texte nettoyé) des marqueurs d'italique
    position = 0
    for match in EMPHASIS_PATTERN.finditer(text):
        chunk = text[position : match.start()]
        plain_parts.append(chunk)
        position = match.end()
        if len(match.group(0)) in (1, 3):
            italic_markers.append(sum(len(part) for part in plain_parts))
    plain_parts.append(text[position:])
    plain = "".join(plain_parts).rstrip()

    italic = [False] * len(plain)
    for start, end in zip(italic_markers[0::2], italic_markers[1::2]):
        for index in range(start, min(end, len(plain))):
            italic[index] = True
    visible = [index for index, char in enumerate(plain) if not char.isspace()]
    ratio = sum(italic[index] for index in visible) / len(visible) if visible else 0.0
    return NormalizedLine(plain, ratio, starts_bold)


def alphabetical_key(plain: str) -> str:
    """Premier mot alphabétique, sans accents ni casse (ordre des annuaires)."""
    match = re.search(r"[^\W\d_]+", plain)
    if not match:
        return ""
    decomposed = unicodedata.normalize("NFKD", match.group(0))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def end_kind(plain: str) -> str:
    """Nature du dernier caractère : une fin d'entrée claire est un point."""
    if not plain:
        return "none"
    last = plain[-1]
    if last == ".":
        return "period"
    if last == "-":
        return "hyphen"
    if last in "—–":
        return "dash"
    if last == ",":
        return "comma"
    if last.isdigit():
        return "digit"
    if last.isalpha():
        return "lower" if last.islower() else "upper"
    return "other"


def start_kind(plain: str) -> str:
    """Nature du premier caractère : minuscule ou chiffre évoquent une suite."""
    if not plain:
        return "none"
    first = plain[0]
    if first.isalpha():
        return "lower" if first.islower() else "upper"
    if first.isdigit():
        return "digit"
    return "punct"


def ratio_bucket(ratio: float, thresholds: Sequence[tuple[float, str]], zero: str) -> str:
    """Découpe une proportion en tranches : `zero` si nulle, sinon la
    dernière tranche dont le seuil est atteint."""
    if ratio <= 0.0:
        return zero
    label = thresholds[0][1]
    for threshold, name in thresholds:
        if ratio >= threshold:
            label = name
    return label


# ----------------------------------------------------------------------
# Contexte de séquence
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SequenceContext:
    """Observations disponibles pour une séquence de lignes non vides.

    Seuls `lines` et les trois champs suivants sont utilisés par les
    features de production ; les champs restants ne servent qu'aux groupes
    candidats et peuvent rester à None. `tokens` / `shapes` sont calculés sur
    le texte normalisé, `raw_tokens` / `raw_shapes` sur le texte brut (jeu v1).
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
    def normalized(self) -> list[NormalizedLine]:
        return [normalize_line(line) for line in self.lines]

    @cached_property
    def tokens(self) -> list[list[str]]:
        return [tokenize(line.plain) for line in self.normalized]

    @cached_property
    def shapes(self) -> list[list[str]]:
        return [[get_shape(token) for token in tokens] for tokens in self.tokens]

    @cached_property
    def raw_tokens(self) -> list[list[str]]:
        return [tokenize(line) for line in self.lines]

    @cached_property
    def raw_shapes(self) -> list[list[str]]:
        return [[get_shape(token) for token in tokens] for tokens in self.raw_tokens]

    @cached_property
    def alpha_keys(self) -> list[str]:
        return [alphabetical_key(line.plain) for line in self.normalized]

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
LEGACY = "v1"
CANDIDATE = "candidat"
PLACEBO = "placebo"


@dataclass(frozen=True)
class FeatureGroup:
    """Un groupe de features.

    `kind` : `production` (utilisé par l'annotateur), `v1` (jeu historique,
    conservé pour comparaison), `candidat` (à évaluer) ou `placebo` (sans
    information, par construction : sert de témoin pour mesurer l'effet d'un
    simple changement de l'espace de features).
    """

    name: str
    description: str
    compute: GroupFunction
    kind: str = PRODUCTION


# ----------------------------------------------------------------------
# Groupes de production
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


def _shape_features(shapes: list[str]) -> Feature:
    feat: Feature = {}
    for i in range(min(4, len(shapes))):
        feat[f"shape_start_{i}"] = shapes[i]
        feat[f"shape_end_{i}"] = shapes[len(shapes) - 1 - i]
    return feat


def _token_shapes(ctx: SequenceContext, t: int) -> Feature:
    return _shape_features(ctx.shapes[t])


ITALIC_BUCKETS = ((0.0, "partial"), (0.5, "mostly"), (0.9, "full"))


def _italic(ctx: SequenceContext, t: int) -> Feature:
    """Proportion d'italique en tranches : `full` repère les sous-titres
    entièrement en italique que l'OCR n'a pas marqués « # »."""
    return {"italic": ratio_bucket(ctx.normalized[t].italic_ratio, ITALIC_BUCKETS, "none")}


def _prev_line(ctx: SequenceContext, t: int) -> Feature:
    """Transition ligne précédente → ligne courante.

    `prev_end` : la ligne précédente se termine-t-elle clairement (point) ou
    reste-t-elle ouverte (tiret, minuscule, virgule...) ? `end_start` croise
    cette fin avec le début de la ligne courante (« hyphen→lower » =
    continuation quasi certaine) : un CRF linéaire ne forme pas cette
    conjonction de lui-même.
    """
    if t == 0:
        return {}
    prev_end = end_kind(ctx.normalized[t - 1].plain)
    cur_start = start_kind(ctx.normalized[t].plain)
    return {
        "prev_is_heading": str(ctx.lines[t - 1].startswith("#")),
        "prev_end": prev_end,
        "end_start": f"{prev_end}→{cur_start}",
    }


# ----------------------------------------------------------------------
# Jeu historique v1 (texte brut, features supprimées depuis) — mêmes clés
# ----------------------------------------------------------------------
def _starts_lower_v1(ctx: SequenceContext, t: int) -> Feature:
    shapes = ctx.raw_shapes[t]
    return {"starts_lower": str(shapes[0] == "LOWER") if shapes else "False"}


def _ends_punct_v1(ctx: SequenceContext, t: int) -> Feature:
    shapes = ctx.raw_shapes[t]
    return {"ends_punct": str(shapes[-1] == "PUNCT") if shapes else "False"}


def _token_count_v1(ctx: SequenceContext, t: int) -> Feature:
    return {"token_count": str(min(len(ctx.raw_tokens[t]), 12))}


def _page_marker_v1(ctx: SequenceContext, t: int) -> Feature:
    return {"is_page_marker": str(bool(PAGE_MARKER_PATTERN.fullmatch(ctx.lines[t])))}


def _token_shapes_v1(ctx: SequenceContext, t: int) -> Feature:
    return _shape_features(ctx.raw_shapes[t])


def _prev_line_v1(ctx: SequenceContext, t: int) -> Feature:
    if t == 0:
        return {}
    previous = ctx.lines[t - 1]
    return {
        "prev_is_heading": str(previous.startswith("#")),
        "prev_ends_dash": str(previous.rstrip().endswith(("-", "—"))),
    }


def _source_gap_v1(ctx: SequenceContext, t: int) -> Feature:
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


SMALL_WORDS = frozenset(
    {"du", "de", "des", "d'", "la", "le", "les", "l'", "et", "en", "au", "aux", "à", "pour", "sur", "par", "dans", "avec", "ou"}
)
FIRST_WORD_PATTERN = re.compile(r"[^\w]*([^\W\d_]+)(['’])?")
UPPERCASE_BUCKETS = ((0.0, "low"), (0.2, "mixed"), (0.5, "mostly"), (0.9, "all"))


def _next_line(ctx: SequenceContext, t: int) -> Feature:
    if t == len(ctx) - 1:
        return {}
    following = ctx.lines[t + 1]
    shapes = ctx.shapes[t + 1]
    return {
        "next_starts_lower": str(bool(shapes) and shapes[0] == "LOWER"),
        "next_starts_dash": str(ctx.normalized[t + 1].plain.startswith(("-", "—"))),
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
    return {"last_char": end_kind(ctx.normalized[t].plain)}


def _address_lexicon(ctx: SequenceContext, t: int) -> Feature:
    plain = ctx.normalized[t].plain
    return {
        "has_street_word": str(bool(ADDRESS_PATTERN.search(plain))),
        "ends_with_number": str(bool(re.search(r"\d+\s*[.,;]?$", plain))),
        "has_number": str(any(char.isdigit() for char in plain)),
    }


def _char_length(ctx: SequenceContext, t: int) -> Feature:
    length = len(ctx.normalized[t].plain)
    feat = {"char_length_log2": str(min(int(math.log2(length + 1)), 8))}
    if t > 0:
        ratio = length / max(len(ctx.normalized[t - 1].plain), 1)
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
    words = WORD_PATTERN.findall(ctx.normalized[t].plain)
    if not words:
        return {}
    return {"first_word": words[0].casefold(), "last_word": words[-1].casefold()}


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


def _bold_start(ctx: SequenceContext, t: int) -> Feature:
    return {"starts_bold": str(ctx.normalized[t].starts_bold)}


def first_word(plain: str) -> str:
    """Premier mot en minuscules, avec son apostrophe d'élision (« d' », « l' »)."""
    match = FIRST_WORD_PATTERN.match(plain)
    if not match:
        return ""
    return match.group(1).casefold() + ("'" if match.group(2) else "")


def _small_word_start(ctx: SequenceContext, t: int) -> Feature:
    """La ligne commence par un « petit mot » (du, de la, et...), quelle que
    soit sa casse : typique d'une suite de titre."""
    return {"first_is_small_word": str(first_word(ctx.normalized[t].plain) in SMALL_WORDS)}


def _block_continuation(ctx: SequenceContext, t: int) -> Feature:
    """Même bloc OCR que la ligne précédente, et type de ce bloc."""
    if ctx.block_keys is None:
        return {}
    if t == 0 or ctx.block_keys[t] != ctx.block_keys[t - 1]:
        return {"same_block_as_prev": "new_block"}
    label = normalize_ocr_label(ctx.ocr_labels[t]) if ctx.ocr_labels is not None else "missing"
    return {"same_block_as_prev": label}


def _uppercase(ctx: SequenceContext, t: int) -> Feature:
    letters = [char for char in ctx.normalized[t].plain if char.isalpha()]
    if not letters:
        return {"uppercase": "no_letters"}
    ratio = sum(char.isupper() for char in letters) / len(letters)
    return {"uppercase": ratio_bucket(ratio, UPPERCASE_BUCKETS, "low")}


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
    FeatureGroup("page_start", "Première ligne d'une nouvelle page", _page_start),
    FeatureGroup("ocr_block", "Classe du bloc OCR (Text, List-Group, ...)", _ocr_block),
    FeatureGroup("sequence_bounds", "Début / fin de séquence (BOS / EOS)", _sequence_bounds),
    FeatureGroup("token_shapes", "Formes des 4 premiers / 4 derniers tokens", _token_shapes),
    FeatureGroup("italic", "Proportion d'italique (none / partial / mostly / full)", _italic),
    FeatureGroup("prev_line", "Ligne précédente : titre, fin de ligne, transition fin → début", _prev_line),
    FeatureGroup("starts_lower_v1", "v1 : premier token en minuscules (texte brut)", _starts_lower_v1, LEGACY),
    FeatureGroup("ends_punct_v1", "v1 : dernier token = ponctuation (texte brut)", _ends_punct_v1, LEGACY),
    FeatureGroup("token_count_v1", "v1 : nombre de tokens (texte brut)", _token_count_v1, LEGACY),
    FeatureGroup("page_marker_v1", "v1 : ligne marqueur de page « {n}--- »", _page_marker_v1, LEGACY),
    FeatureGroup("token_shapes_v1", "v1 : formes de tokens (texte brut)", _token_shapes_v1, LEGACY),
    FeatureGroup("prev_line_v1", "v1 : ligne précédente titre / finit par un tiret", _prev_line_v1, LEGACY),
    FeatureGroup("source_gap_v1", "v1 : saut de numéro de ligne source > 1", _source_gap_v1, LEGACY),
    FeatureGroup("next_line", "Ligne suivante : minuscule, tiret, titre, page", _next_line, CANDIDATE),
    FeatureGroup("alpha_sequence", "Ordre alphabétique du 1er mot vs lignes voisines", _alpha_sequence, CANDIDATE),
    FeatureGroup("blank_before", "Ligne précédée d'une ligne vide", _blank_before, CANDIDATE),
    FeatureGroup("block_position", "Première / dernière ligne de son bloc OCR", _block_position, CANDIDATE),
    FeatureGroup("line_ending", "Nature du dernier caractère", _line_ending, CANDIDATE),
    FeatureGroup("address_lexicon", "Mots de voirie, nombres, nombre final", _address_lexicon, CANDIDATE),
    FeatureGroup("char_length", "Longueur en caractères, rapport à la précédente", _char_length, CANDIDATE),
    FeatureGroup("layout_indent", "Indentation / largeur relatives du bloc", _layout_indent, CANDIDATE),
    FeatureGroup("lexical_words", "Premier / dernier mot (forme minuscule)", _lexical_words, CANDIDATE),
    FeatureGroup("bold_start", "Ligne commençant en gras", _bold_start, CANDIDATE),
    FeatureGroup("small_word_start", "Premier mot = petit mot (du, de, et...), casse ignorée", _small_word_start, CANDIDATE),
    FeatureGroup("block_continuation", "Même bloc OCR que la ligne précédente, et son type", _block_continuation, CANDIDATE),
    FeatureGroup("uppercase", "Proportion de capitales (tranches)", _uppercase, CANDIDATE),
    FeatureGroup("placebo_constant", "Témoin : attribut constant", _placebo_constant, PLACEBO),
    *(
        FeatureGroup(f"placebo_random_{seed}", f"Témoin : bit pseudo-aléatoire (graine {seed})", _placebo_random(seed), PLACEBO)
        for seed in (1, 2, 3)
    ),
)
FEATURE_GROUPS: dict[str, FeatureGroup] = {group.name: group for group in _GROUPS}
PRODUCTION_GROUPS: tuple[str, ...] = tuple(g.name for g in _GROUPS if g.kind == PRODUCTION)
# Jeu historique (production avant les évolutions v2), clés et ordre identiques.
LEGACY_GROUPS: tuple[str, ...] = (
    "bias",
    "heading",
    "starts_lower_v1",
    "ends_punct_v1",
    "token_count_v1",
    "page_marker_v1",
    "page_start",
    "ocr_block",
    "sequence_bounds",
    "token_shapes_v1",
    "prev_line_v1",
    "source_gap_v1",
)
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
    """Features de production (API historique de numrev label)."""
    return extract_features_from_context(
        SequenceContext(lines, source_line_numbers, ocr_labels, page_positions)
    )

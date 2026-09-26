"""Règles de convention (voir `docs/guide_annotation_ner.md`).

Appliquées aux annotations existantes (silver issu de `tagged_text`, sorties
LLM) pour les mettre en conformité avec le guide, et évaluables en
post-traitement d'inférence par `audit_ner.py`. Chaque règle est une
fonction pure `(texte, empans) -> empans` ; `apply_rules` les enchaîne et
indique lesquelles ont modifié l'annotation.

Elles remplacent `tools/correct_annotations.py`, dont la règle 2 (une
parenthèse en minuscule en fin de SUBJ passe en DESC) est **abandonnée** :
elle produisait `<SUBJ>Lafitte</SUBJ> <DESC>(le jeune)</DESC>`, contraire au
guide.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from lib.ner.spans import Span, trim_spans

# Liaison tolérée entre deux empans pour les considérer comme adjacents.
LINK_PATTERN = re.compile(r"[\s,;:.]*")

# Mots d'identité : civilités, état civil, rang familial, raison sociale,
# titres de noblesse, particules. Les grades et fonctions (`général`,
# `sénateur`) relèvent de DESC. Comparés sans casse, sans accents ni points.
IDENTITY_WORDS = {
    "m", "me", "mad", "madame", "mme", "mlle", "mlles", "melle", "demoiselle", "demoiselles",
    "mm", "mrs", "messieurs", "mess", "sr", "sieur", "dame", "citoyen", "citoyenne", "cit", "citne",
    "ve", "veuve", "vve", "vc", "vs", "epouse", "femme",
    "aine", "ae", "jeune", "jne", "cadet", "fils", "pere", "frere", "freres", "fr", "soeur", "soeurs",
    "neveu", "neveux", "gendre", "oncle", "fille", "filles", "enfant", "enfants", "heritiers",
    "et", "comp", "compe", "cie", "compagnie", "associe", "associes", "dit", "dite", "ci-devant", "ci",
    "devant", "ancien", "baron", "baronne", "comte", "comtesse", "marquis", "marquise", "duc",
    "duchesse", "vicomte", "vicomtesse", "chevalier", "prince", "princesse",
    "le", "la", "les", "l", "de", "du", "des", "d", "etc", "lomb", "lombard",
}
PARTICLES = {"de", "du", "des", "d"}
# Titres en initiales (« D. R. » : docteur-régent, « D. M. » : docteur en
# médecine) : une fonction, donc DESC, malgré leurs majuscules.
TITLE_INITIALS = {"d r", "d m", "d m p", "dr"}
# Premier mot d'une enseigne (« (Café du Berceau Lyrique) ») : DESC. Comparé
# accents compris, pour ne pas confondre « à » avec l'initiale « A. ».
SIGNBOARD_WORDS = {"café", "cafe", "hôtel", "hotel", "maison", "magasin", "au", "aux", "à", "manufacture", "fabrique"}
# Premier mot d'un segment qui localise (voies, lieux, bâtiments).
ADDRESS_START_PATTERN = re.compile(
    r"^(r|rue|q|quai|pl|place|pal|palais|marche|passage|allee|boul|boulev|boulevard|bd|cour|cul|"
    r"faub|faubourg|fb|fg|enclos|pont|carrefour|cloitre|barriere|impasse|chaussee|halle|galerie|gal|"
    r"rotonde|parvis|terrasse|chemin|avenue|av|ile|exterieur|"
    r"interieur|pres|vis|foire|arcades|portail|square|ruelle|enceinte)$"
)
# Adjectifs qui ne comptent que devant un mot de voie (« petite R. de Reuilly »,
# mais pas « petite mercerie »).
STREET_ADJECTIVES = {"petite", "pet", "grande", "gr", "vieille", "neuve", "n"}
MAX_ADDRESS_DESC_WORDS = 10
# Voie suivie d'un numéro, n'importe où dans le segment (« …, R. Childebert, 909. »).
STREET_NUMBER_PATTERN = re.compile(r"\b(?:R|rue|Q|quai|[Pp]l|place|boul\w*|passage|cour|allée)\.?\s[^()]*?\d")
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]*)*\.?|\d+|\S")
PARENTHESIS_PATTERN = re.compile(r"\s*[,;]?\s*\(([^()]*)\)")
LOCATIVE_PATTERN = re.compile(
    r"^(pr[eè]s|en face|vis-?[àa]-?vis|au coin|[àa] c[ôo]t[ée]|derri[èe]re|attenant|proche|au-dessus|au-dessous)\b",
    re.IGNORECASE,
)
INITIALS_PATTERN = re.compile(r"^(?:[A-ZÉ]\.?\s*){1,3}$")
MAX_TRAILING_ADDR_WORDS = 8
MAX_DASH_SECTION_WORDS = 5

# Sections de Paris (1790-1811), complétées par le lexique observé dans les
# données (`SectionLexicon.from_annotations`).
PARIS_SECTIONS = (
    "Tuileries", "Champs-Élysées", "Roule", "Place Vendôme", "Piques", "Butte des Moulins", "Montagne",
    "Lepelletier", "Mont-Blanc", "Gardes-Françaises", "Halle au Blé", "Contrat-Social", "Mail",
    "Guillaume Tell", "Brutus", "Bonne-Nouvelle", "Amis de la Patrie", "Bondy", "Temple", "Popincourt",
    "Montreuil", "Quinze-Vingts", "Faubourg du Nord", "Poissonnière", "Faubourg Montmartre", "Réunion",
    "Homme Armé", "Droits de l'Homme", "Arcis", "Marchés", "Lombards", "Gravilliers", "Bon Conseil",
    "Cité", "Fraternité", "Pont-Neuf", "Muséum", "Invalides", "Fontaine de Grenelle", "Unité", "Ouest",
    "Luxembourg", "Thermes", "Panthéon", "Observatoire", "Jardin des Plantes", "Plantes", "Finistère",
    "Indivisibilité", "Arsenal", "Fidélité", "Théâtre Français", "Faubourg Saint-Germain",
    "Faubourg Saint-Antoine", "Faubourg Saint-Denis",
)


def fold(text: str) -> str:
    """Forme de comparaison : sans accents, casse, ponctuation ni emphase."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def _words(text: str) -> list[str]:
    return [token for token in WORD_PATTERN.findall(text) if token[0].isalnum()]


# ----------------------------------------------------------------------
# Identité (parenthèses de prénoms, civilités, rang familial)
# ----------------------------------------------------------------------
def is_identity_group(inner: str) -> bool:
    """Contenu de parenthèse relevant de l'identité du sujet :
    `( Ch. L. Fr. )`, `(le jeune)`, `(Mad.)`, `(Ve. de)`, `(frères)`,
    `( lomb. Serilly )`, `(Moysse et Sollivet)` ; mais pas `( histoire )`,
    `(de Malte)`, `( du grand hospice )`, `(fourbiss.)`, `(D. R.)`,
    `(Café du Berceau Lyrique)`."""
    words = _words(inner)
    if not words:
        return False
    folded = [fold(word) for word in words]
    if " ".join(folded) in TITLE_INITIALS or words[0].casefold() in SIGNBOARD_WORDS:
        return False
    if folded[0] in PARTICLES and len(words) > 1:
        return False  # « (de Malte) », « (de l'Impératrice) » : qualité, pas identité
    for word, key in zip(words, folded):
        if key in IDENTITY_WORDS or key.replace(" ", "-") in IDENTITY_WORDS:
            continue
        if word[0].isupper():
            continue
        return False
    return True


def identity_to_subj(text: str, spans: Sequence[Span]) -> list[Span]:
    """Parenthèses d'identité (et qualificatifs familiaux nus) placées en
    tête d'un DESC qui suit directement le SUBJ : rattachées au SUBJ."""
    spans = sorted(spans, key=lambda span: span.start)
    result = list(spans)
    for index in range(len(spans) - 1):
        subj, desc = spans[index], spans[index + 1]
        if subj.label != "SUBJ" or desc.label != "DESC":
            continue
        if LINK_PATTERN.fullmatch(text[subj.end : desc.start]) is None:
            continue
        cursor = desc.start
        while True:
            match = PARENTHESIS_PATTERN.match(text, cursor, desc.end)
            if match and is_identity_group(match.group(1)):
                cursor = match.end()
                continue
            bare = re.match(r"\s*[,;]?\s*(?:et\s+)?([^\W\d_]+\.?)(?=\s*[,;(]|\s*$)", text[cursor : desc.end])
            if bare and fold(bare.group(1)) in IDENTITY_WORDS - PARTICLES - {"et"}:
                cursor += bare.end()
                continue
            break
        if cursor == desc.start:
            continue
        rest = LINK_PATTERN.match(text, cursor, desc.end).end()
        result[index] = subj._replace(end=cursor)
        result[index + 1] = desc._replace(start=rest) if rest < desc.end else None
    return [span for span in result if span is not None]


# ----------------------------------------------------------------------
# Localisation (section après tiret, section ou repère après l'adresse)
# ----------------------------------------------------------------------
class SectionLexicon:
    """Noms de sections / quartiers, comparés mot à mot par préfixe pour
    reconnaître les abréviations (« Pl. Vendôme », « Indivisib. »)."""

    def __init__(self, names: Iterable[str]):
        self.entries = {fold(name) for name in names if len(fold(name)) >= 3}
        self.by_length: dict[int, list[list[str]]] = {}
        for entry in self.entries:
            words = entry.split()
            self.by_length.setdefault(len(words), []).append(words)

    @classmethod
    def from_annotations(cls, annotations: Iterable[tuple[str, Sequence[Span]]], min_count: int = 3) -> "SectionLexicon":
        """Lexique = sections intégrées + ce qui suit le dernier tiret long
        des ADDR du corpus, s'il est vu au moins `min_count` fois."""
        counts: Counter[str] = Counter()
        for text, spans in annotations:
            for span in spans:
                if span.label == "ADDR":
                    parts = re.split(r"[—–]", text[span.start : span.end])
                    if len(parts) > 1 and len(_words(parts[-1])) <= 5:
                        counts[parts[-1].strip(" .,;")] += 1
        return cls([*PARIS_SECTIONS, *(name for name, count in counts.items() if count >= min_count)])

    def matches(self, text: str) -> bool:
        words = fold(text).split()
        if not words:
            return False
        if " ".join(words) in self.entries:
            return True
        for candidate in self.by_length.get(len(words), []):
            if all(entry.startswith(word) for word, entry in zip(words, candidate)) and len("".join(words)) >= 3:
                return True
        return False


DEFAULT_LEXICON = SectionLexicon(PARIS_SECTIONS)


def dash_section_to_addr(text: str, spans: Sequence[Span]) -> list[Span]:
    """DESC qui suit un ADDR et commence (liaison comprise) par un tiret :
    section/quartier, fusionnée dans l'ADDR (ancienne règle 1). Limité aux
    DESC courts et sans chiffre, pour ne pas absorber un prix ou une notice
    (« — 12 f. pour 3 mois »)."""
    result: list[Span] = []
    for span in sorted(spans, key=lambda span: span.start):
        previous = result[-1] if result else None
        if (
            previous is not None
            and previous.label == "ADDR"
            and span.label == "DESC"
            and re.match(r"[\s.,;:]*[—–-]", text[previous.end : span.end])
            and len(_words(text[span.start : span.end])) <= MAX_DASH_SECTION_WORDS
            and not re.search(r"\d", text[span.start : span.end])
        ):
            result[-1] = previous._replace(end=span.end)
            continue
        result.append(span)
    return result


def trailing_location_to_addr(
    text: str, spans: Sequence[Span], lexicon: SectionLexicon = DEFAULT_LEXICON
) -> list[Span]:
    """DESC court qui suit un ADDR et n'est qu'une localisation : section
    connue (`Amis de la Patrie`, `(Indivisib.)`), initiales de section
    (`F. G.`) ou repère (`près l'opéra`) : fusionné dans l'ADDR."""
    result: list[Span] = []
    for span in sorted(spans, key=lambda span: span.start):
        previous = result[-1] if result else None
        if previous is not None and previous.label == "ADDR" and span.label == "DESC":
            content = text[span.start : span.end].strip(" .,;:()[]*")
            short = len(_words(content)) <= MAX_TRAILING_ADDR_WORDS
            if short and (
                INITIALS_PATTERN.match(content) or LOCATIVE_PATTERN.match(content) or lexicon.matches(content)
            ):
                result[-1] = previous._replace(end=span.end)
                continue
        result.append(span)
    return result


# ----------------------------------------------------------------------
# Texte orphelin, segments adjacents
# ----------------------------------------------------------------------
def is_address_like(fragment: str) -> bool:
    """Le segment commence par un mot de voie ou de lieu (`rue`, `Q.`,
    `marché`, `pal.`, `passage`…). Un segment entre parenthèses n'est
    jamais une adresse (`(port.)`, `(march. de …)`, `(hôtel de Sens)`)."""
    fragment = fragment.lstrip(" ,;:.—–-")
    if fragment.startswith("("):
        return False
    keys = [fold(word) for word in _words(fragment)[:2]]
    if keys and keys[0] in STREET_ADJECTIVES:
        keys = keys[1:]
    return bool(keys) and ADDRESS_START_PATTERN.match(keys[0].split(" ")[0] or "") is not None


def _identity_prefix_end(text: str, start: int, end: int) -> int:
    """Fin de la suite de parenthèses d'identité qui commence en `start`."""
    cursor = start
    while (match := PARENTHESIS_PATTERN.match(text, cursor, end)) and is_identity_group(match.group(1)):
        cursor = match.end()
    return cursor


def fill_gaps(text: str, spans: Sequence[Span]) -> list[Span]:
    """Aucun mot orphelin (règle générale 3 du guide) : un texte non annoté
    avant le premier empan rejoint le SUBJ ; après un SUBJ, ses parenthèses
    d'identité rejoignent le SUBJ ; le reste devient ADDR s'il commence par
    un mot de voie ou de lieu, DESC sinon. Une entrée sans aucun empan est
    laissée telle quelle (rien pour l'ancrer)."""
    spans = sorted(spans, key=lambda span: span.start)
    if not spans:
        return []
    result: list[Span] = []
    bounds = [0, *(position for span in spans for position in (span.start, span.end)), len(text)]
    for index in range(len(spans) + 1):
        gap_start, gap_end = bounds[2 * index], bounds[2 * index + 1]
        gap = text[gap_start:gap_end]
        if re.search(r"\w", gap):
            if index == 0:
                spans[0] = spans[0]._replace(start=gap_start) if spans[0].label == "SUBJ" else spans[0]
                if spans[0].label != "SUBJ":
                    result.append(Span(gap_start, gap_end, "SUBJ"))
            else:
                previous = result[-1]
                start = LINK_PATTERN.match(text, gap_start, gap_end).end()
                if previous.label == "SUBJ":
                    identity_end = _identity_prefix_end(text, gap_start, gap_end)
                    if identity_end > gap_start:
                        result[-1] = previous._replace(end=identity_end)
                        start = LINK_PATTERN.match(text, identity_end, gap_end).end()
                if re.search(r"\w", text[start:gap_end]):
                    fragment = text[start:gap_end]
                    last = index == len(spans)
                    address = is_address_like(fragment) or (last and STREET_NUMBER_PATTERN.search(fragment) is not None)
                    label = "ADDR" if address else "DESC"
                    result.append(Span(start, gap_end, label))
        if index < len(spans):
            result.append(spans[index])
    return trim_linking_punctuation(text, result)


def address_like_desc_to_addr(text: str, spans: Sequence[Span]) -> list[Span]:
    """DESC court collé à un ADDR (avant ou après) qui commence par un mot
    de voie ou de lieu (`pal. du Trib.`, `Marché Boulainvilliers`) : ADDR."""
    spans = sorted(spans, key=lambda span: span.start)
    result = list(spans)
    for index, span in enumerate(spans):
        fragment = text[span.start : span.end]
        if span.label != "DESC" or len(_words(fragment)) > MAX_ADDRESS_DESC_WORDS or not is_address_like(fragment):
            continue
        neighbours = [spans[i] for i in (index - 1, index + 1) if 0 <= i < len(spans)]
        if any(n.label == "ADDR" and LINK_PATTERN.fullmatch(text[min(n.end, span.end) : max(n.start, span.start)]) for n in neighbours):
            result[index] = span._replace(label="ADDR")
    return result


def merge_adjacent(text: str, spans: Sequence[Span]) -> list[Span]:
    """Deux empans de même classe séparés par de la seule ponctuation de
    liaison n'en font qu'un (`<DESC>drapier, nouveautés</DESC>`, adresses
    multiples dans un seul ADDR)."""
    result: list[Span] = []
    for span in sorted(spans, key=lambda span: span.start):
        previous = result[-1] if result else None
        if previous is not None and previous.label == span.label and LINK_PATTERN.fullmatch(text[previous.end : span.start]):
            result[-1] = previous._replace(end=span.end)
            continue
        result.append(span)
    return result


# ----------------------------------------------------------------------
# Bornes
# ----------------------------------------------------------------------
def trim_linking_punctuation(text: str, spans: Sequence[Span]) -> list[Span]:
    """Retire la ponctuation de liaison en bord d'empan : en tête tout
    séparateur, en fin virgule / point-virgule / deux-points / tiret (le
    point final, souvent celui d'une abréviation, est conservé)."""
    trimmed = []
    for span in trim_spans(text, spans):
        start, end = span.start, span.end
        while start < end and text[start] in " ,;:.—–-":
            start += 1
        while end > start and text[end - 1] in " ,;:—–-":
            end -= 1
        if end > start:
            trimmed.append(span._replace(start=start, end=end))
    return trimmed


# ----------------------------------------------------------------------
# Enchaînement
# ----------------------------------------------------------------------
class RuleResult(NamedTuple):
    spans: list[Span]
    fired: tuple[str, ...]  # règles ayant modifié l'annotation


def apply_rules(text: str, spans: Sequence[Span], lexicon: SectionLexicon = DEFAULT_LEXICON) -> RuleResult:
    rules: list[tuple[str, Callable[[str, Sequence[Span]], list[Span]]]] = [
        ("bornes", trim_linking_punctuation),
        ("identité→SUBJ", identity_to_subj),
        ("orphelins", fill_gaps),
        ("identité→SUBJ", identity_to_subj),
        ("voie→ADDR", address_like_desc_to_addr),
        ("tiret→ADDR", dash_section_to_addr),
        ("localisation→ADDR", lambda t, s: trailing_location_to_addr(t, s, lexicon)),
        ("fusion", merge_adjacent),
        ("bornes", trim_linking_punctuation),
    ]
    current = sorted(spans, key=lambda span: span.start)
    fired: list[str] = []
    for name, rule in rules:
        updated = rule(text, current)
        if [s[:3] for s in updated] != [s[:3] for s in current] and name not in fired:
            fired.append(name)
        current = updated
    return RuleResult(current, tuple(fired))

import re
import html
import string
import unicodedata
from typing import Set
import regex as reg
import pandas as pd
from ftfy import fix_text
from nltk.corpus import stopwords
from collections import Counter


BOILERPLATE_PHRASES = ["li li strong", "li li", "br br", "et de"]

MY_STOPWORDS: Set[str] = {"m?"}

# NLTK stopwords (French + English)
NLTK_STOPS: Set[str] = set(stopwords.words("french")) | set(stopwords.words("english"))

# Punctuation characters (standard + additional French/typographic)
PUNCT_CHARS = set(string.punctuation) | {
    "…", "'", '"', "«", "»", "•", "·", "–", "—", "‹", "›"
}


# =============================================================================
# Utility Functions
# =============================================================================

def is_single_letter(token: str) -> bool:
    """Check if token is a single alphabetic character."""
    return len(token) == 1 and token.isalpha()


def is_single_digit(token: str) -> bool:
    """Check if token is a single numeric digit."""
    return len(token) == 1 and token.isdigit()


def is_pure_punctuation(token: str) -> bool:
    """Check if token consists only of punctuation characters."""
    if not token:
        return False
    return all(ch in PUNCT_CHARS for ch in token)


# =============================================================================
# Text Processing Functions
# =============================================================================

def merge_x_dimensions(text):
    """
    Normalize dimension patterns by merging them into single tokens.

    Examples:
        - '22 x 11 x 2' -> '22x11x2'
        - '180 x 180'   -> '180x180'
        - 'L x H x L'   -> 'LxHxL'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # 1) numeric triplets: 22 x 11 x 2 → 22x11x2
    s = re.sub(r"\b(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\b", r"\1x\2x\3", s, flags=re.IGNORECASE)

    # 2) numeric pairs: 180 x 180 → 180x180
    s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1x\2", s, flags=re.IGNORECASE)

    # 3) letter triplets: L x H x L → LxHxL
    s = re.sub(r"\b([lh])\s*x\s*([lh])\s*x\s*([lh])\b", r"\1x\2x\3", s, flags=re.IGNORECASE)

    return s



def merge_numeric_units(text):
    """
    Merge common numeric + unit patterns into single tokens.

    Examples:
        '500 g'    -> '500g'
        '2 kg'     -> '2kg'
        '1 L'      -> '1l'
        '50 ml'    -> '50ml'
        '32 Go'    -> '32go'
        '8 GB'     -> '8gb'
        '100 %'    -> '100pct'
        '3000 mAh' -> '3000mah'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # Weight / volume
    s = re.sub(r"\b(\d+)\s*(kg|g|mg|ml|l)\b", r"\1\2", s, flags=re.IGNORECASE)

    # Length
    s = re.sub(r"\b(\d+)\s*(mm|cm|m)\b", r"\1\2", s, flags=re.IGNORECASE)

    # Storage / memory
    s = re.sub(r"\b(\d+)\s*(go|gb|mo|mb)\b", r"\1\2", s, flags=re.IGNORECASE)

    # Percentage
    s = re.sub(r"\b(\d+)\s*%\b", r"\1pct", s, flags=re.IGNORECASE)

    # Battery capacity
    s = re.sub(r"\b(\d+)\s*(mah|ah)\b", r"\1\2", s, flags=re.IGNORECASE)

    return s



def merge_durations(text):
    """
    Normalize duration expressions as single tokens.

    Examples:
        '24 h'       -> '24h'
        '48 heures'  -> '48h'
        '7 j / 7'    -> '7j7'
        '12 mois'    -> '12mois'
        '3 ans'      -> '3ans'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # Hours
    s = re.sub(r"\b(\d+)\s*(h|heures?)\b", r"\1h", s, flags=re.IGNORECASE)

    # Days
    s = re.sub(r"\b(\d+)\s*(j|jours?)\b", r"\1j", s, flags=re.IGNORECASE)

    # Months
    s = re.sub(r"\b(\d+)\s*mois\b", r"\1mois", s, flags=re.IGNORECASE)

    # Years
    s = re.sub(r"\b(\d+)\s*ans?\b", r"\1ans", s, flags=re.IGNORECASE)

    # 24h/24, 7j/7
    s = re.sub(r"\b24\s*h\s*/\s*24\b", "24h24", s, flags=re.IGNORECASE)
    s = re.sub(r"\b7\s*j\s*/\s*7\b", "7j7", s, flags=re.IGNORECASE)

    return s



def merge_age_ranges(text):
    """
    Normalize age range expressions.

    Examples:
        '0-3 ans'       -> '0_3ans'
        '3 - 5 ans'     -> '3_5ans'
        '3-5ans'        -> '3_5ans'
        '6 ans et plus' -> '6plus_ans'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    # Ranges like '0-3 ans', '3 - 5 ans', '3-5 ans'
    s = re.sub(
        r"\b(\d+)\s*-\s*(\d+)\s*ans\b",
        r"\1_\2ans",
        s,
        flags=re.IGNORECASE,
    )

    # Extra safety: '3-5ans' (no space before 'ans')
    s = re.sub(
        r"\b(\d+)\s*-\s*(\d+)ans\b",
        r"\1_\2ans",
        s,
        flags=re.IGNORECASE,
    )

    # "X ans et plus"
    s = re.sub(
        r"\b(\d+)\s*ans?\s*et\s*plus\b",
        r"\1plus_ans",
        s,
        flags=re.IGNORECASE,
    )

    return s



def tag_years(text):
    """
    Optionally tag 4-digit years to make them stand out.

    Example:
        'Paru en 1917' -> 'paru en year1917'
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)

    def repl(m):
        year = m.group(0)
        return f" year{year} "

    s = re.sub(r"\b(18|19|20)\d{2}\b", repl, s)
    return s





def nettoyer_texte(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = reg.sub(r"<[^>]+>", " ", s)          # Remove HTML tags
    s = html.unescape(s)                     # Decode HTML entities
    s = fix_text(s)                          # Fix broken text encoding
    s = unicodedata.normalize("NFC", s)      # Normalize Unicode
    s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)  # Remove non-numeric periods
    s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)   # Remove isolated hyphens
    s = reg.sub(r"(?<!\S):(?!\S)", " ", s)   # Remove isolated colons
    s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)   # Remove isolated middle dots
    s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)   # Remove isolated slashes
    s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)  # Remove isolated plus signs
    s = s.replace("////", " ")
    s = s.lower()
    s = reg.sub(r"\s+", " ", s).strip()      # Normalize whitespace
    return s


def nltk_stopwords(text: str, stopwords_set=None):
    if stopwords_set is None:
        stopwords_set = set()
    if not isinstance(text, str):
        return []

    tokens = []
    for w in text.split():
        w = w.lower()
        if w in stopwords_set:
            continue
        tokens.append(w)
    return tokens


def get_word_freq_with_nltk_stopwords(series, stopwords_set=None):
    all_tokens = []
    for text in series:
        all_tokens.extend(nltk_stopwords(text, stopwords_set))
    return Counter(all_tokens)


def global_text_cleaner(
    text,
    normalize_x_dimensions: bool = True,
    normalize_units: bool = True,
    normalize_durations: bool = True,
    normalize_age_ranges: bool = True,
    tag_year_numbers: bool = False,
    remove_boilerplate: bool = True,
    remove_nltk_stops: bool = True,
    remove_custom_stops: bool = True,
    remove_single_digit: bool = True,
    remove_single_letter: bool = True,
    lowercase: bool = True,
):
    # pour eviter les erreurs avec des NaN
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    # ------------------------------------------------------------------
    # 1) Structurel merges
    # ------------------------------------------------------------------
    if normalize_x_dimensions:
        s = merge_x_dimensions(s)

    if normalize_units:
        s = merge_numeric_units(s)

    if normalize_durations:
        s = merge_durations(s)

    if normalize_age_ranges:
        s = merge_age_ranges(s)

    if tag_year_numbers:
        s = tag_years(s)

    # ------------------------------------------------------------------
    # 2) nettoyage basique: HTML, unicode, punctuation, lowercase
    # ------------------------------------------------------------------
    # Remove HTML tags
    s = reg.sub(r"<[^>]+>", " ", s)

    # Decode HTML entities
    s = html.unescape(s)

    # Fix broken encodings
    s = fix_text(s)

    # Normalize Unicode
    s = unicodedata.normalize("NFC", s)

    # Keep decimal points in numbers, remove others like ". xxx"
    s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)

    # Remove isolated hyphens / colons / middle dots / slashes / plus signs
    s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
    s = s.replace("////", " ")

    # Lowercase ou pas
    if lowercase:
        s = s.lower()

    # ------------------------------------------------------------------
    # 3) boilerplate phrases
    # ------------------------------------------------------------------
    if remove_boilerplate and BOILERPLATE_PHRASES:
        for phrase in BOILERPLATE_PHRASES:
            if not phrase:
                continue
            pattern = r"\b" + re.escape(phrase) + r"\b"
            s = re.sub(pattern, " ", s, flags=re.IGNORECASE)


    # ------------------------------------------------------------------
    # 4) Stopwords + single-letter/digit + pure punctuation filtering
    # ------------------------------------------------------------------
    if remove_nltk_stops or remove_custom_stops or remove_single_digit or remove_single_letter:
        tokens = s.split()

        # Build stopword set
        stops_to_exclude = set()
        if remove_nltk_stops:
            stops_to_exclude.update(NLTK_STOPS)
        if remove_custom_stops:
            stops_to_exclude.update(MY_STOPWORDS)

        filtered = []
        for w in tokens:
            # Skip if in stopword list
            if w in stops_to_exclude:
                continue

            # Skip pure punctuation tokens
            if is_pure_punctuation(w):
                continue

            # Optional: skip single letters
            if remove_single_letter and is_single_letter(w):
                continue

            # Optional: skip single digits
            if remove_single_digit and is_single_digit(w):
                continue

            filtered.append(w)

        s = " ".join(filtered)

    # ------------------------------------------------------------------
    # 5) Final whitespace normalization
    # ------------------------------------------------------------------
    s = reg.sub(r"\s+", " ", s).strip()
    return s

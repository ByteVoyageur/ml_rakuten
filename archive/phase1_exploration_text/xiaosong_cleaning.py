"""
Modular text cleaning utilities for Rakuten product classification.

This module provides a single configurable cleaning function where each
preprocessing step can be independently enabled/disabled for ablation studies.

Design Principle:
- All cleaning options are OFF by default (preserve raw data)
- Each option can be enabled explicitly via boolean flags
- Logical execution order: encoding fixes → structural merges → filtering
"""

import re
import html
import string
import unicodedata
from typing import Set
import regex as reg
import pandas as pd
from ftfy import fix_text
from nltk.corpus import stopwords


# =============================================================================
# Constants
# =============================================================================

# French + English stopwords
NLTK_STOPWORDS = set(stopwords.words("french")) | set(stopwords.words("english"))

# Extended punctuation set
PUNCTUATION = set(string.punctuation) | {
    "…", "'", '"', "«", "»", "•", "·", "–", "—", "‹", "›"
}

# Common boilerplate phrases from HTML templates
BOILERPLATE_PHRASES = ["li li strong", "li li", "br br", "et de"]


# =============================================================================
# Core Cleaning Function
# =============================================================================

def clean_text(
    text,
    # Encoding & Unicode
    fix_encoding: bool = False,
    unescape_html: bool = False,
    normalize_unicode: bool = False,
    # HTML & Structure
    remove_html_tags: bool = False,
    remove_boilerplate: bool = False,
    # Case Transformation
    lowercase: bool = False,
    # Structural Merges (preserve semantic units)
    merge_dimensions: bool = False,      # "22 x 11 x 2" → "22x11x2"
    merge_units: bool = False,           # "500 g" → "500g"
    merge_durations: bool = False,       # "24 h" → "24h"
    merge_age_ranges: bool = False,      # "3-5 ans" → "3_5ans"
    tag_years: bool = False,             # "1917" → "year1917"
    # Punctuation & Special Chars
    remove_punctuation: bool = False,    # Remove isolated punctuation
    # Token Filtering
    remove_stopwords: bool = False,
    remove_single_letters: bool = False,
    remove_single_digits: bool = False,
    remove_pure_punct_tokens: bool = False,
):
    """
    Modular text cleaning function with granular control over each step.

    Args:
        text: Input text (str or None/NaN)

        Encoding & Unicode:
            fix_encoding: Fix broken text encoding using ftfy
            unescape_html: Decode HTML entities (&amp; → &)
            normalize_unicode: Apply Unicode NFC normalization

        HTML & Structure:
            remove_html_tags: Remove HTML tags <tag>content</tag>
            remove_boilerplate: Remove common template phrases

        Case Transformation:
            lowercase: Convert to lowercase

        Structural Merges:
            merge_dimensions: "22 x 11" → "22x11"
            merge_units: "500 g" → "500g", "32 Go" → "32go"
            merge_durations: "24 h" → "24h", "7 j" → "7j"
            merge_age_ranges: "3-5 ans" → "3_5ans"
            tag_years: "1917" → "year1917"

        Punctuation:
            remove_punctuation: Remove isolated punctuation marks

        Token Filtering:
            remove_stopwords: Remove French/English stopwords
            remove_single_letters: Remove single alphabetic characters
            remove_single_digits: Remove single numeric digits
            remove_pure_punct_tokens: Remove tokens consisting only of punctuation

    Returns:
        Cleaned text string

    Examples:
        >>> # Baseline: no cleaning
        >>> clean_text("Hello <b>World</b>")
        'Hello <b>World</b>'

        >>> # Only lowercase
        >>> clean_text("Hello <b>World</b>", lowercase=True)
        'hello <b>world</b>'

        >>> # Only remove HTML
        >>> clean_text("Hello <b>World</b>", remove_html_tags=True)
        'Hello  World'
    """
    # Handle missing values
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    # =========================================================================
    # Phase 1: Encoding Fixes (do this first to ensure proper character handling)
    # =========================================================================

    if fix_encoding:
        s = fix_text(s)

    if unescape_html:
        s = html.unescape(s)

    if normalize_unicode:
        s = unicodedata.normalize("NFC", s)

    # =========================================================================
    # Phase 2: HTML & Structural Cleanup
    # =========================================================================

    if remove_html_tags:
        s = reg.sub(r"<[^>]+>", " ", s)

    # =========================================================================
    # Phase 3: Case Transformation
    # =========================================================================

    if lowercase:
        s = s.lower()

    # =========================================================================
    # Phase 4: Structural Merges (preserve semantic units as single tokens)
    # =========================================================================

    if merge_dimensions:
        # Triplets: "22 x 11 x 2" → "22x11x2"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\b", r"\1x\2x\3", s, flags=re.IGNORECASE)
        # Pairs: "180 x 180" → "180x180"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1x\2", s, flags=re.IGNORECASE)
        # Letter triplets: "L x H x L" → "LxHxL"
        s = re.sub(r"\b([lh])\s*x\s*([lh])\s*x\s*([lh])\b", r"\1x\2x\3", s, flags=re.IGNORECASE)

    if merge_units:
        # Weight/volume: "500 g" → "500g"
        s = re.sub(r"\b(\d+)\s*(kg|g|mg|ml|l)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Length: "50 cm" → "50cm"
        s = re.sub(r"\b(\d+)\s*(mm|cm|m)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Storage: "32 Go" → "32go"
        s = re.sub(r"\b(\d+)\s*(go|gb|mo|mb)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Percentage: "100 %" → "100pct"
        s = re.sub(r"\b(\d+)\s*%\b", r"\1pct", s, flags=re.IGNORECASE)
        # Battery: "3000 mAh" → "3000mah"
        s = re.sub(r"\b(\d+)\s*(mah|ah)\b", r"\1\2", s, flags=re.IGNORECASE)

    if merge_durations:
        # Hours: "24 h" → "24h"
        s = re.sub(r"\b(\d+)\s*(h|heures?)\b", r"\1h", s, flags=re.IGNORECASE)
        # Days: "7 j" → "7j"
        s = re.sub(r"\b(\d+)\s*(j|jours?)\b", r"\1j", s, flags=re.IGNORECASE)
        # Months: "12 mois" → "12mois"
        s = re.sub(r"\b(\d+)\s*mois\b", r"\1mois", s, flags=re.IGNORECASE)
        # Years: "3 ans" → "3ans"
        s = re.sub(r"\b(\d+)\s*ans?\b", r"\1ans", s, flags=re.IGNORECASE)
        # Special: "24h/24" → "24h24"
        s = re.sub(r"\b24\s*h\s*/\s*24\b", "24h24", s, flags=re.IGNORECASE)
        s = re.sub(r"\b7\s*j\s*/\s*7\b", "7j7", s, flags=re.IGNORECASE)

    if merge_age_ranges:
        # "0-3 ans" → "0_3ans"
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)\s*ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "3-5ans" → "3_5ans" (no space before "ans")
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "6 ans et plus" → "6plus_ans"
        s = re.sub(r"\b(\d+)\s*ans?\s*et\s*plus\b", r"\1plus_ans", s, flags=re.IGNORECASE)

    if tag_years:
        # "1917" → "year1917" (only 4-digit years 18xx, 19xx, 20xx)
        s = re.sub(r"\b(18|19|20)\d{2}\b", lambda m: f" year{m.group(0)} ", s)

    # =========================================================================
    # Phase 5: Punctuation Removal (isolated punctuation only)
    # =========================================================================

    if remove_punctuation:
        # Remove periods not in numbers: "Hello. World" → "Hello  World" (but keep "3.14")
        s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)
        # Remove isolated hyphens/colons/etc (but keep "well-known")
        s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
        s = s.replace("////", " ")

    # =========================================================================
    # Phase 6: Boilerplate Removal
    # =========================================================================

    if remove_boilerplate:
        for phrase in BOILERPLATE_PHRASES:
            if phrase:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                s = re.sub(pattern, " ", s, flags=re.IGNORECASE)

    # =========================================================================
    # Phase 7: Token-Level Filtering
    # =========================================================================

    # If any token filtering is enabled, we need to split into tokens
    if (remove_stopwords or remove_single_letters or
        remove_single_digits or remove_pure_punct_tokens):

        tokens = s.split()
        filtered = []

        for token in tokens:
            # Filter: stopwords
            if remove_stopwords and token.lower() in NLTK_STOPWORDS:
                continue

            # Filter: single letters
            if remove_single_letters and len(token) == 1 and token.isalpha():
                continue

            # Filter: single digits
            if remove_single_digits and len(token) == 1 and token.isdigit():
                continue

            # Filter: pure punctuation tokens
            if remove_pure_punct_tokens and token and all(ch in PUNCTUATION for ch in token):
                continue

            filtered.append(token)

        s = " ".join(filtered)

    # =========================================================================
    # Phase 8: Final Whitespace Normalization (always do this)
    # =========================================================================

    s = reg.sub(r"\s+", " ", s).strip()

    return s


# =============================================================================
# Utility Functions for Inspection
# =============================================================================

def get_available_options():
    """
    Returns a dictionary of all available cleaning options with descriptions.

    Returns:
        dict: {option_name: description}
    """
    return {
        # Encoding & Unicode
        "fix_encoding": "Fix broken text encoding using ftfy",
        "unescape_html": "Decode HTML entities (&amp; → &)",
        "normalize_unicode": "Apply Unicode NFC normalization",

        # HTML & Structure
        "remove_html_tags": "Remove HTML tags <tag>content</tag>",
        "remove_boilerplate": "Remove common template phrases",

        # Case Transformation
        "lowercase": "Convert to lowercase",

        # Structural Merges
        "merge_dimensions": "Merge dimension patterns (22 x 11 → 22x11)",
        "merge_units": "Merge numeric units (500 g → 500g)",
        "merge_durations": "Merge durations (24 h → 24h)",
        "merge_age_ranges": "Merge age ranges (3-5 ans → 3_5ans)",
        "tag_years": "Tag 4-digit years (1917 → year1917)",

        # Punctuation
        "remove_punctuation": "Remove isolated punctuation marks",

        # Token Filtering
        "remove_stopwords": "Remove French/English stopwords",
        "remove_single_letters": "Remove single alphabetic characters",
        "remove_single_digits": "Remove single numeric digits",
        "remove_pure_punct_tokens": "Remove tokens consisting only of punctuation",
    }


def print_available_options():
    """Print all available cleaning options with descriptions."""
    options = get_available_options()
    print("Available Cleaning Options:")
    print("=" * 80)
    for option, description in options.items():
        print(f"  {option:25s} : {description}")
    print("=" * 80)


# =============================================================================
# Production-Ready Final Cleaner (Winner Strategy from Benchmark Phase 2)
# =============================================================================

def final_text_cleaner(text):
    """
    Production-ready text cleaning function based on benchmark winning strategy.

    This function implements the 'optimized_traditional' configuration that
    achieved the best performance (F1: 0.8024, +1.32% vs baseline) in the
    comprehensive benchmark study (Phase 2: Champion Combination Test).

    Cleaning Pipeline (strict order):
    1. Handle NaN/None values
    2. Fix broken text encoding (ftfy)
    3. Decode HTML entities (&amp; → &)
    4. Normalize Unicode (NFC)
    5. Remove HTML tags (<tag>content</tag> → content)
    6. Convert to lowercase
    7. Remove isolated punctuation (preserve punctuation in numbers)
    8. Remove French/English stopwords
    9. Normalize whitespace

    Args:
        text (str or None): Input text to clean

    Returns:
        str: Cleaned text ready for TF-IDF vectorization

    Examples:
        >>> text = "<p>Ordinateur <strong>portable</strong> 15.6 pouces</p>"
        >>> final_text_cleaner(text)
        'ordinateur portable 15.6 pouces'

        >>> text = "Prix: 299,99&nbsp;€ - Livraison gratuite!"
        >>> final_text_cleaner(text)
        'prix 299,99 € livraison gratuite'

    Notes:
        - This configuration was validated on 84,916 Rakuten product samples
        - Outperformed 22 alternative preprocessing strategies
        - Optimized for e-commerce product classification with TF-IDF features
        - Benchmark details: notebooks/xiaosong_01_benchmark.ipynb

    References:
        Winning configuration: optimized_traditional
        F1 Score (weighted): 0.8024
        Improvement vs baseline: +1.32%
        Test date: 2025-12-07
    """
    # =========================================================================
    # Step 1: Handle missing values
    # =========================================================================
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    # =========================================================================
    # Step 2: Fix broken text encoding
    # =========================================================================
    s = fix_text(s)

    # =========================================================================
    # Step 3: Decode HTML entities
    # =========================================================================
    s = html.unescape(s)

    # =========================================================================
    # Step 4: Normalize Unicode
    # =========================================================================
    s = unicodedata.normalize("NFC", s)

    # =========================================================================
    # Step 5: Remove HTML tags
    # =========================================================================
    s = reg.sub(r"<[^>]+>", " ", s)

    # =========================================================================
    # Step 6: Convert to lowercase
    # =========================================================================
    s = s.lower()

    # =========================================================================
    # Step 7: Remove isolated punctuation
    # =========================================================================
    # Remove periods not in numbers: "Hello. World" → "Hello  World" (keep "3.14")
    s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)

    # Remove isolated hyphens/colons/etc (but keep "well-known", "3-5")
    s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
    s = s.replace("////", " ")

    # =========================================================================
    # Step 8: Remove stopwords (French + English)
    # =========================================================================
    tokens = s.split()
    filtered = []

    for token in tokens:
        # Skip if token is a stopword
        if token in NLTK_STOPWORDS:
            continue

        # Keep all other tokens
        filtered.append(token)

    s = " ".join(filtered)

    # =========================================================================
    # Step 9: Normalize whitespace
    # =========================================================================
    s = reg.sub(r"\s+", " ", s).strip()

    return s

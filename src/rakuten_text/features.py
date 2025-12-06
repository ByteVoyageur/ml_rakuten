"""
Extraction de caractéristiques structurelles pour les textes Rakuten.

Ce module fournit des fonctions pour extraire des features basées sur la structure
du texte (longueurs, présence de chiffres, unités de mesure, etc.) plutôt que sur
le contenu sémantique.
"""

import re
from typing import List, Tuple

import pandas as pd


# =============================================================================
# Patterns Regex pour l'Extraction de Features
# =============================================================================

# Pattern pour détecter les unités de mesure (cm, mm, kg, etc.)
UNIT_PATTERN = re.compile(r"\b\d+\s*(cm|mm|kg|g|ml|l|m)\b", flags=re.IGNORECASE)

# Pattern pour détecter les expressions multiplicatives (x2, 3x, etc.)
MULT_PATTERN = re.compile(r"\bx\s*\d+\b|\b\d+\s*x\b", flags=re.IGNORECASE)

# Pattern pour détecter les chiffres
DIGIT_PATTERN = re.compile(r"\d")


# =============================================================================
# Fonctions Utilitaires
# =============================================================================

def safe_str(x):
    """
    Convertit une valeur en chaîne de caractères de manière sécurisée.

    Gère les valeurs manquantes (NaN) et assure que la sortie est toujours
    une chaîne de caractères.

    Args:
        x: Valeur à convertir (str, float, NaN, etc.)

    Returns:
        str: Chaîne de caractères (vide si x est NaN)
    """
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


# =============================================================================
# Extraction de Features Structurelles
# =============================================================================

def structural_stats(s: str) -> dict:
    """
    Calcule des indicateurs simples de structure textuelle.

    Cette fonction analyse un texte et extrait plusieurs caractéristiques
    structurelles qui peuvent être utiles pour la classification :
    - Longueur en caractères
    - Longueur en tokens (mots)
    - Nombre de chiffres
    - Nombre d'unités de mesure (cm, kg, etc.)
    - Nombre de patterns multiplicatifs (x2, 3x, etc.)

    Args:
        s: Texte à analyser

    Returns:
        dict: Dictionnaire contenant les statistiques structurelles
            - len_char: nombre de caractères
            - len_tokens: nombre de tokens (mots)
            - num_digits: nombre de chiffres
            - num_units: nombre d'unités de mesure détectées
            - num_mult_pattern: nombre de patterns multiplicatifs

    Examples:
        >>> structural_stats("Livre 21 x 15 cm, 200 pages")
        {
            'len_char': 27,
            'len_tokens': 5,
            'num_digits': 5,
            'num_units': 1,
            'num_mult_pattern': 1
        }
    """
    s = safe_str(s)
    tokens = s.split()
    length_char = len(s)
    length_tokens = len(tokens)

    num_digits = len(DIGIT_PATTERN.findall(s))
    num_units = len(UNIT_PATTERN.findall(s))
    num_mult = len(MULT_PATTERN.findall(s))

    return {
        "len_char": length_char,
        "len_tokens": length_tokens,
        "num_digits": num_digits,
        "num_units": num_units,
        "num_mult_pattern": num_mult,
    }


def add_structural_features(
    df: pd.DataFrame,
    text_cols: Tuple[str, ...] = ("designation_cleaned", "description_cleaned")
) -> pd.DataFrame:
    """
    Ajoute des features structurelles au DataFrame pour chaque colonne texte.

    Cette fonction calcule automatiquement les statistiques structurelles
    pour chaque colonne de texte spécifiée et ajoute les résultats comme
    nouvelles colonnes dans le DataFrame.

    Pour chaque colonne texte (ex: "designation_cleaned"), les colonnes
    suivantes sont créées :
    - {col}_len_char: longueur en caractères
    - {col}_len_tokens: longueur en tokens
    - {col}_num_digits: nombre de chiffres
    - {col}_num_units: nombre d'unités de mesure
    - {col}_num_mult_pattern: nombre de patterns multiplicatifs

    Args:
        df: DataFrame contenant les données textuelles
        text_cols: Tuple des noms de colonnes à analyser
            Par défaut: ("designation_cleaned", "description_cleaned")

    Returns:
        pd.DataFrame: DataFrame original avec les nouvelles colonnes de features
            ajoutées (modifie le DataFrame en place ET le retourne)

    Examples:
        >>> df = pd.DataFrame({
        ...     'designation_cleaned': ['Livre 200 pages', 'Stylo bleu'],
        ...     'description_cleaned': ['Roman 21x15 cm', '']
        ... })
        >>> df = add_structural_features(df)
        >>> 'designation_cleaned_len_char' in df.columns
        True
        >>> 'description_cleaned_num_units' in df.columns
        True

    Notes:
        - Les valeurs manquantes (NaN) sont automatiquement gérées
        - Le DataFrame est modifié en place (les colonnes sont ajoutées)
        - La fonction retourne aussi le DataFrame pour permettre le chaînage
    """
    # Normalisation des colonnes texte : s'assurer qu'elles sont des strings
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(safe_str)

    # Application de structural_stats sur chaque colonne texte
    for col in text_cols:
        if col not in df.columns:
            continue

        # Calculer les stats pour chaque ligne
        stats_series = df[col].apply(structural_stats)

        # Convertir la série de dicts en DataFrame
        stats_df = pd.DataFrame(list(stats_series))

        # Ajouter chaque statistique comme nouvelle colonne
        for stat_name in stats_df.columns:
            df[f"{col}_{stat_name}"] = stats_df[stat_name]

    return df


def get_meta_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifie et retourne les colonnes de features structurelles (meta-features).

    Cette fonction recherche automatiquement dans le DataFrame toutes les colonnes
    qui correspondent à des features structurelles générées par add_structural_features().

    Les colonnes reconnues sont celles qui commencent par :
    - "designation_cleaned_" (features du titre)
    - "description_cleaned_" (features de la description)

    Args:
        df: DataFrame contenant les colonnes de features

    Returns:
        list[str]: Liste des noms de colonnes correspondant aux features structurelles,
            triées par ordre alphabétique

    Examples:
        >>> df = pd.DataFrame({
        ...     'designation_cleaned_len_char': [10, 20],
        ...     'designation_cleaned_len_tokens': [2, 3],
        ...     'description_cleaned_num_digits': [0, 5],
        ...     'other_column': [1, 2]
        ... })
        >>> get_meta_feature_columns(df)
        ['designation_cleaned_len_char', 'designation_cleaned_len_tokens',
         'description_cleaned_num_digits']

    Notes:
        - Retourne une liste vide si aucune colonne ne correspond
        - Les colonnes sont triées alphabétiquement pour assurer la cohérence
        - Utile pour identifier automatiquement les features numériques
          à utiliser dans un pipeline scikit-learn
    """
    meta_cols = [
        c for c in df.columns
        if c.startswith("designation_cleaned_")
        or c.startswith("description_cleaned_")
    ]
    return sorted(meta_cols)

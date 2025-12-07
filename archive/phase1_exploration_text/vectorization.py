"""
Construction de vectoriseurs TF-IDF et préprocesseurs pour les données Rakuten.

Ce module fournit des fonctions pour créer des TfidfVectorizer et ColumnTransformer
configurés selon les meilleures pratiques identifiées dans les notebooks d'expérimentation.
"""

from typing import List, Optional, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Constantes de Configuration par Défaut
# =============================================================================

# Configuration commune à tous les TF-IDF
DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_MIN_DF = 5
DEFAULT_MAX_DF = 0.8
DEFAULT_LOWERCASE = False

# Max features spécifiques
DEFAULT_MAX_FEATURES_TITLE = 20000
DEFAULT_MAX_FEATURES_DESC = 30000
DEFAULT_MAX_FEATURES_ALL = 40000

# Pondérations par défaut pour la stratégie séparée
DEFAULT_WEIGHTS_SPLIT = {
    "tfidf_title": 2.0,
    "tfidf_desc": 1.0,
    "num": 1.0,
}

# Pondérations par défaut pour la stratégie fusionnée
DEFAULT_WEIGHTS_MERGED = {
    "tfidf_all": 1.0,
    "num": 1.0,
}


# =============================================================================
# Fonctions de Construction des Vectoriseurs TF-IDF
# =============================================================================

def build_tfidf_title(
    max_features: int = DEFAULT_MAX_FEATURES_TITLE,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    lowercase: bool = DEFAULT_LOWERCASE,
    **kwargs
) -> TfidfVectorizer:
    """
    Construit un vectoriseur TF-IDF configuré pour les titres de produits.

    Cette fonction crée un TfidfVectorizer avec les paramètres optimisés pour
    la colonne 'designation_cleaned' (titre du produit). Les paramètres par défaut
    sont basés sur les expérimentations du notebook 02_xiaosong_text_preprocessing_v2.

    Args:
        max_features: Nombre maximum de features à extraire (défaut: 20000)
        ngram_range: Plage des n-grams à considérer (défaut: (1, 3))
        min_df: Fréquence minimale de document (défaut: 5)
        max_df: Fréquence maximale de document (défaut: 0.8)
        lowercase: Si True, convertit en minuscules (défaut: False, car déjà fait)
        **kwargs: Arguments supplémentaires pour TfidfVectorizer

    Returns:
        TfidfVectorizer: Vectoriseur configuré pour les titres

    Examples:
        >>> tfidf = build_tfidf_title()
        >>> tfidf = build_tfidf_title(max_features=15000, ngram_range=(1, 2))

    Notes:
        - tokenizer=str.split est utilisé car le texte est déjà nettoyé
        - lowercase=False car la normalisation est déjà faite en amont
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        tokenizer=str.split,
        **kwargs
    )


def build_tfidf_desc(
    max_features: int = DEFAULT_MAX_FEATURES_DESC,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    lowercase: bool = DEFAULT_LOWERCASE,
    **kwargs
) -> TfidfVectorizer:
    """
    Construit un vectoriseur TF-IDF configuré pour les descriptions de produits.

    Cette fonction crée un TfidfVectorizer avec les paramètres optimisés pour
    la colonne 'description_cleaned' (description du produit). Les descriptions
    étant généralement plus longues, max_features est plus élevé (30000) que
    pour les titres.

    Args:
        max_features: Nombre maximum de features à extraire (défaut: 30000)
        ngram_range: Plage des n-grams à considérer (défaut: (1, 3))
        min_df: Fréquence minimale de document (défaut: 5)
        max_df: Fréquence maximale de document (défaut: 0.8)
        lowercase: Si True, convertit en minuscules (défaut: False)
        **kwargs: Arguments supplémentaires pour TfidfVectorizer

    Returns:
        TfidfVectorizer: Vectoriseur configuré pour les descriptions

    Examples:
        >>> tfidf = build_tfidf_desc()
        >>> tfidf = build_tfidf_desc(max_features=25000)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        tokenizer=str.split,
        **kwargs
    )


def build_tfidf_all(
    max_features: int = DEFAULT_MAX_FEATURES_ALL,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    lowercase: bool = DEFAULT_LOWERCASE,
    **kwargs
) -> TfidfVectorizer:
    """
    Construit un vectoriseur TF-IDF pour le texte fusionné (titre + description).

    Cette fonction crée un TfidfVectorizer pour la stratégie où le titre et la
    description sont concaténés en une seule colonne 'text_all'. Le nombre de
    features est le plus élevé (40000) pour capturer la richesse des deux champs.

    Args:
        max_features: Nombre maximum de features à extraire (défaut: 40000)
        ngram_range: Plage des n-grams à considérer (défaut: (1, 3))
        min_df: Fréquence minimale de document (défaut: 5)
        max_df: Fréquence maximale de document (défaut: 0.8)
        lowercase: Si True, convertit en minuscules (défaut: False)
        **kwargs: Arguments supplémentaires pour TfidfVectorizer

    Returns:
        TfidfVectorizer: Vectoriseur configuré pour le texte fusionné

    Examples:
        >>> tfidf = build_tfidf_all()
        >>> tfidf = build_tfidf_all(max_features=50000, ngram_range=(1, 4))

    Notes:
        - Cette approche est généralement moins performante que la stratégie
          séparée avec pondération du titre (voir notebook d'expérimentation)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        tokenizer=str.split,
        **kwargs
    )


# =============================================================================
# Fonctions de Construction des Préprocesseurs
# =============================================================================

def build_preprocess_split(
    tfidf_title: TfidfVectorizer,
    tfidf_desc: TfidfVectorizer,
    meta_cols: List[str],
    weights: Optional[Dict[str, float]] = None,
    with_mean: bool = False
) -> ColumnTransformer:
    """
    Construit un ColumnTransformer avec stratégie séparée (titre/description).

    Cette fonction crée un préprocesseur qui traite séparément le titre et la
    description, permettant d'appliquer des pondérations différentes à chaque
    champ. Cette approche s'est révélée supérieure à la fusion simple dans les
    expérimentations (F1 score de 0.8067 vs 0.7526).

    Args:
        tfidf_title: Vectoriseur TF-IDF pour les titres (designation_cleaned)
        tfidf_desc: Vectoriseur TF-IDF pour les descriptions (description_cleaned)
        meta_cols: Liste des colonnes de features structurelles numériques
        weights: Dictionnaire de pondérations pour transformer_weights.
            Si None, utilise les poids par défaut: titre=2.0, desc=1.0, num=1.0
        with_mean: Si True, centre les features numériques (défaut: False pour
            compatibilité avec les matrices creuses)

    Returns:
        ColumnTransformer: Préprocesseur configuré avec stratégie séparée

    Examples:
        >>> from rakuten_text import build_tfidf_title, build_tfidf_desc
        >>> tfidf_t = build_tfidf_title()
        >>> tfidf_d = build_tfidf_desc()
        >>> meta_cols = ['designation_cleaned_len_char', ...]
        >>> preprocessor = build_preprocess_split(tfidf_t, tfidf_d, meta_cols)
        >>>
        >>> # Avec pondérations personnalisées
        >>> custom_weights = {'tfidf_title': 3.0, 'tfidf_desc': 1.0, 'num': 1.0}
        >>> preprocessor = build_preprocess_split(
        ...     tfidf_t, tfidf_d, meta_cols, weights=custom_weights
        ... )

    Notes:
        - Les colonnes traitées sont : 'designation_cleaned', 'description_cleaned', et meta_cols
        - La pondération par défaut privilégie le titre (2.0) sur la description (1.0)
        - remainder="drop" : toutes les autres colonnes sont ignorées
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS_SPLIT.copy()

    # Créer le scaler pour les features numériques
    num_scaler = StandardScaler(with_mean=with_mean)

    # Construire le ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf_title", tfidf_title, "designation_cleaned"),
            ("tfidf_desc", tfidf_desc, "description_cleaned"),
            ("num", num_scaler, meta_cols),
        ],
        remainder="drop",
    )

    # Appliquer les pondérations si fournies
    if weights:
        preprocessor.set_params(transformer_weights=weights)

    return preprocessor


def build_preprocess_merged(
    tfidf_all: TfidfVectorizer,
    meta_cols: List[str],
    weights: Optional[Dict[str, float]] = None,
    with_mean: bool = False
) -> ColumnTransformer:
    """
    Construit un ColumnTransformer avec stratégie fusionnée (texte concaténé).

    Cette fonction crée un préprocesseur qui traite le texte fusionné (titre +
    description concaténés dans 'text_all'). Cette approche est plus simple mais
    généralement moins performante que la stratégie séparée.

    Args:
        tfidf_all: Vectoriseur TF-IDF pour le texte fusionné (text_all)
        meta_cols: Liste des colonnes de features structurelles numériques
        weights: Dictionnaire de pondérations pour transformer_weights.
            Si None, utilise les poids par défaut: tfidf_all=1.0, num=1.0
        with_mean: Si True, centre les features numériques (défaut: False)

    Returns:
        ColumnTransformer: Préprocesseur configuré avec stratégie fusionnée

    Examples:
        >>> from rakuten_text import build_tfidf_all
        >>> tfidf = build_tfidf_all()
        >>> meta_cols = ['designation_cleaned_len_char', ...]
        >>> preprocessor = build_preprocess_merged(tfidf, meta_cols)
        >>>
        >>> # Avec pondérations personnalisées
        >>> custom_weights = {'tfidf_all': 2.0, 'num': 0.5}
        >>> preprocessor = build_preprocess_merged(
        ...     tfidf, meta_cols, weights=custom_weights
        ... )

    Notes:
        - La colonne traitée est : 'text_all' (titre + description fusionnés)
        - Cette stratégie obtient généralement un F1 score inférieur (~0.75)
          comparé à la stratégie séparée (~0.81)
        - Utile comme baseline ou pour des expérimentations rapides
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS_MERGED.copy()

    # Créer le scaler pour les features numériques
    num_scaler = StandardScaler(with_mean=with_mean)

    # Construire le ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf_all", tfidf_all, "text_all"),
            ("num", num_scaler, meta_cols),
        ],
        remainder="drop",
    )

    # Appliquer les pondérations si fournies
    if weights:
        preprocessor.set_params(transformer_weights=weights)

    return preprocessor


# =============================================================================
# Fonctions Utilitaires de Haut Niveau
# =============================================================================

def build_split_pipeline_components(
    meta_cols: List[str],
    title_max_features: int = DEFAULT_MAX_FEATURES_TITLE,
    desc_max_features: int = DEFAULT_MAX_FEATURES_DESC,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[TfidfVectorizer, TfidfVectorizer, ColumnTransformer]:
    """
    Construit tous les composants pour un pipeline avec stratégie séparée.

    Cette fonction utilitaire crée en une seule fois tous les composants
    nécessaires pour un pipeline avec titre et description séparés.

    Args:
        meta_cols: Liste des colonnes de features numériques
        title_max_features: Max features pour le titre (défaut: 20000)
        desc_max_features: Max features pour la description (défaut: 30000)
        weights: Pondérations personnalisées (défaut: titre=2.0, desc=1.0, num=1.0)

    Returns:
        Tuple contenant (tfidf_title, tfidf_desc, preprocessor)

    Examples:
        >>> meta_cols = get_meta_feature_columns(df)
        >>> tfidf_t, tfidf_d, preprocessor = build_split_pipeline_components(meta_cols)
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = Pipeline([
        ...     ('preprocess', preprocessor),
        ...     ('model', LogisticRegression())
        ... ])
    """
    tfidf_title = build_tfidf_title(max_features=title_max_features)
    tfidf_desc = build_tfidf_desc(max_features=desc_max_features)
    preprocessor = build_preprocess_split(tfidf_title, tfidf_desc, meta_cols, weights)

    return tfidf_title, tfidf_desc, preprocessor


def build_merged_pipeline_components(
    meta_cols: List[str],
    all_max_features: int = DEFAULT_MAX_FEATURES_ALL,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[TfidfVectorizer, ColumnTransformer]:
    """
    Construit tous les composants pour un pipeline avec stratégie fusionnée.

    Cette fonction utilitaire crée en une seule fois tous les composants
    nécessaires pour un pipeline avec texte fusionné.

    Args:
        meta_cols: Liste des colonnes de features numériques
        all_max_features: Max features pour le texte fusionné (défaut: 40000)
        weights: Pondérations personnalisées (défaut: tfidf_all=1.0, num=1.0)

    Returns:
        Tuple contenant (tfidf_all, preprocessor)

    Examples:
        >>> meta_cols = get_meta_feature_columns(df)
        >>> tfidf, preprocessor = build_merged_pipeline_components(meta_cols)
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LogisticRegression
        >>> pipeline = Pipeline([
        ...     ('preprocess', preprocessor),
        ...     ('model', LogisticRegression())
        ... ])
    """
    tfidf_all = build_tfidf_all(max_features=all_max_features)
    preprocessor = build_preprocess_merged(tfidf_all, meta_cols, weights)

    return tfidf_all, preprocessor

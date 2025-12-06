"""
Rakuten text processing utilities.
"""

from .cleaning import (
    # Main functions
    global_text_cleaner,
    nettoyer_texte,
    merge_x_dimensions,
    nltk_stopwords,
    get_word_freq_with_nltk_stopwords,

    # Utility functions
    is_single_letter,
    is_single_digit,
    is_pure_punctuation,

    # Constants
    BOILERPLATE_PHRASES,
    MY_STOPWORDS,
    NLTK_STOPS,
    PUNCT_CHARS,
)

from .features import (
    # Feature extraction functions
    add_structural_features,
    get_meta_feature_columns,
    structural_stats,
    safe_str,

    # Pattern constants
    UNIT_PATTERN,
    MULT_PATTERN,
    DIGIT_PATTERN,
)

from .vectorization import (
    # TF-IDF builders
    build_tfidf_title,
    build_tfidf_desc,
    build_tfidf_all,

    # Preprocessor builders
    build_preprocess_split,
    build_preprocess_merged,

    # High-level utilities
    build_split_pipeline_components,
    build_merged_pipeline_components,

    # Default configuration constants
    DEFAULT_NGRAM_RANGE,
    DEFAULT_MIN_DF,
    DEFAULT_MAX_DF,
    DEFAULT_MAX_FEATURES_TITLE,
    DEFAULT_MAX_FEATURES_DESC,
    DEFAULT_MAX_FEATURES_ALL,
    DEFAULT_WEIGHTS_SPLIT,
    DEFAULT_WEIGHTS_MERGED,
)

from .modeling import (
    # Model builders
    build_logreg_model,
    build_pipeline,

    # Evaluation functions
    evaluate_pipeline,
    evaluate_weight_grid,

    # Analysis utilities
    print_weight_grid_summary,
    compare_strategies,

    # Default model constants
    DEFAULT_C,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_SOLVER,
    DEFAULT_MAX_ITER,
    DEFAULT_N_JOBS,
)

__all__ = [
    # Cleaning functions
    "global_text_cleaner",
    "nettoyer_texte",
    "merge_x_dimensions",
    "nltk_stopwords",
    "get_word_freq_with_nltk_stopwords",

    # Cleaning utilities
    "is_single_letter",
    "is_single_digit",
    "is_pure_punctuation",

    # Cleaning constants
    "BOILERPLATE_PHRASES",
    "MY_STOPWORDS",
    "NLTK_STOPS",
    "PUNCT_CHARS",

    # Feature extraction
    "add_structural_features",
    "get_meta_feature_columns",
    "structural_stats",
    "safe_str",

    # Feature patterns
    "UNIT_PATTERN",
    "MULT_PATTERN",
    "DIGIT_PATTERN",

    # Vectorization - TF-IDF builders
    "build_tfidf_title",
    "build_tfidf_desc",
    "build_tfidf_all",

    # Vectorization - Preprocessor builders
    "build_preprocess_split",
    "build_preprocess_merged",

    # Vectorization - High-level utilities
    "build_split_pipeline_components",
    "build_merged_pipeline_components",

    # Vectorization - Constants
    "DEFAULT_NGRAM_RANGE",
    "DEFAULT_MIN_DF",
    "DEFAULT_MAX_DF",
    "DEFAULT_MAX_FEATURES_TITLE",
    "DEFAULT_MAX_FEATURES_DESC",
    "DEFAULT_MAX_FEATURES_ALL",
    "DEFAULT_WEIGHTS_SPLIT",
    "DEFAULT_WEIGHTS_MERGED",

    # Modeling - Model builders
    "build_logreg_model",
    "build_pipeline",

    # Modeling - Evaluation
    "evaluate_pipeline",
    "evaluate_weight_grid",

    # Modeling - Analysis utilities
    "print_weight_grid_summary",
    "compare_strategies",

    # Modeling - Constants
    "DEFAULT_C",
    "DEFAULT_CLASS_WEIGHT",
    "DEFAULT_SOLVER",
    "DEFAULT_MAX_ITER",
    "DEFAULT_N_JOBS",
]

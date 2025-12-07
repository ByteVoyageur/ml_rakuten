"""
Exemples d'utilisation du module vectorization.

Ce script démontre comment utiliser les fonctions de vectorization pour
construire des pipelines scikit-learn complets.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rakuten_text import (
    # Features
    add_structural_features,
    get_meta_feature_columns,

    # Vectorization
    build_tfidf_title,
    build_tfidf_desc,
    build_tfidf_all,
    build_preprocess_split,
    build_preprocess_merged,
    build_split_pipeline_components,
    build_merged_pipeline_components,

    # Constants
    DEFAULT_WEIGHTS_SPLIT,
)


def example_basic_tfidf():
    """Exemple 1 : Construction de vectoriseurs TF-IDF individuels."""
    print("=" * 70)
    print("EXEMPLE 1 : Construction de vectoriseurs TF-IDF")
    print("=" * 70)

    # Construire les vectoriseurs avec configuration par défaut
    tfidf_title = build_tfidf_title()
    tfidf_desc = build_tfidf_desc()
    tfidf_all = build_tfidf_all()

    print("\n1. Vectoriseur pour les titres :")
    print(f"   - max_features: {tfidf_title.max_features}")
    print(f"   - ngram_range: {tfidf_title.ngram_range}")
    print(f"   - min_df: {tfidf_title.min_df}")

    print("\n2. Vectoriseur pour les descriptions :")
    print(f"   - max_features: {tfidf_desc.max_features}")
    print(f"   - ngram_range: {tfidf_desc.ngram_range}")

    print("\n3. Vectoriseur pour le texte fusionné :")
    print(f"   - max_features: {tfidf_all.max_features}")

    # Personnalisation
    print("\n4. Vectoriseur personnalisé :")
    tfidf_custom = build_tfidf_title(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=10
    )
    print(f"   - max_features: {tfidf_custom.max_features}")
    print(f"   - ngram_range: {tfidf_custom.ngram_range}")
    print(f"   - min_df: {tfidf_custom.min_df}")
    print()


def example_preprocessor_split():
    """Exemple 2 : Préprocesseur avec stratégie séparée."""
    print("=" * 70)
    print("EXEMPLE 2 : Préprocesseur avec stratégie séparée (titre/desc)")
    print("=" * 70)

    # Simuler des colonnes de features
    meta_cols = [
        'designation_cleaned_len_char',
        'designation_cleaned_len_tokens',
        'designation_cleaned_num_digits',
        'description_cleaned_len_char',
        'description_cleaned_len_tokens',
    ]

    # Construire les composants
    tfidf_title = build_tfidf_title()
    tfidf_desc = build_tfidf_desc()

    # Préprocesseur avec pondérations par défaut
    print("\n1. Avec pondérations par défaut :")
    preprocessor = build_preprocess_split(tfidf_title, tfidf_desc, meta_cols)
    print(f"   Transformers : {len(preprocessor.transformers)}")
    print(f"   - tfidf_title sur 'designation_cleaned'")
    print(f"   - tfidf_desc sur 'description_cleaned'")
    print(f"   - num sur {len(meta_cols)} colonnes numériques")
    print(f"   Pondérations : {DEFAULT_WEIGHTS_SPLIT}")

    # Préprocesseur avec pondérations personnalisées
    print("\n2. Avec pondérations personnalisées :")
    custom_weights = {
        'tfidf_title': 3.0,  # Augmenter l'importance du titre
        'tfidf_desc': 1.0,
        'num': 0.5,
    }
    preprocessor_custom = build_preprocess_split(
        tfidf_title, tfidf_desc, meta_cols, weights=custom_weights
    )
    print(f"   Pondérations : {custom_weights}")
    print()


def example_preprocessor_merged():
    """Exemple 3 : Préprocesseur avec stratégie fusionnée."""
    print("=" * 70)
    print("EXEMPLE 3 : Préprocesseur avec stratégie fusionnée (text_all)")
    print("=" * 70)

    meta_cols = [
        'designation_cleaned_len_char',
        'designation_cleaned_len_tokens',
        'description_cleaned_len_char',
    ]

    # Construire les composants
    tfidf_all = build_tfidf_all()
    preprocessor = build_preprocess_merged(tfidf_all, meta_cols)

    print("\n1. Préprocesseur fusionné :")
    print(f"   Transformers : {len(preprocessor.transformers)}")
    print(f"   - tfidf_all sur 'text_all' (titre + description)")
    print(f"   - num sur {len(meta_cols)} colonnes numériques")
    print()


def example_high_level_utilities():
    """Exemple 4 : Utilisation des fonctions utilitaires haut niveau."""
    print("=" * 70)
    print("EXEMPLE 4 : Fonctions utilitaires haut niveau")
    print("=" * 70)

    meta_cols = ['designation_cleaned_len_char', 'description_cleaned_len_char']

    # Stratégie séparée en une seule fonction
    print("\n1. build_split_pipeline_components() :")
    tfidf_t, tfidf_d, preprocessor = build_split_pipeline_components(meta_cols)
    print(f"   ✓ tfidf_title créé (max_features={tfidf_t.max_features})")
    print(f"   ✓ tfidf_desc créé (max_features={tfidf_d.max_features})")
    print(f"   ✓ preprocessor créé avec {len(preprocessor.transformers)} transformers")

    # Avec personnalisation
    print("\n2. Avec paramètres personnalisés :")
    custom_weights = {'tfidf_title': 3.0, 'tfidf_desc': 1.0, 'num': 1.0}
    tfidf_t, tfidf_d, preprocessor = build_split_pipeline_components(
        meta_cols,
        title_max_features=15000,
        desc_max_features=25000,
        weights=custom_weights
    )
    print(f"   ✓ tfidf_title (max_features={tfidf_t.max_features})")
    print(f"   ✓ tfidf_desc (max_features={tfidf_d.max_features})")
    print(f"   ✓ Pondérations : {custom_weights}")

    # Stratégie fusionnée en une seule fonction
    print("\n3. build_merged_pipeline_components() :")
    tfidf_all, preprocessor = build_merged_pipeline_components(meta_cols)
    print(f"   ✓ tfidf_all créé (max_features={tfidf_all.max_features})")
    print(f"   ✓ preprocessor créé avec {len(preprocessor.transformers)} transformers")
    print()


def example_complete_pipeline():
    """Exemple 5 : Pipeline complet avec modèle."""
    print("=" * 70)
    print("EXEMPLE 5 : Pipeline complet scikit-learn")
    print("=" * 70)

    # Note: Ceci est un exemple de code, ne sera pas exécuté
    code_example = """
# Étape 1 : Préparer les données
import pandas as pd
from rakuten_text import (
    add_structural_features,
    get_meta_feature_columns,
    build_split_pipeline_components
)

df = pd.read_csv("rakuten_text_train_v1.csv")
df = add_structural_features(df)
meta_cols = get_meta_feature_columns(df)

# Étape 2 : Construire le préprocesseur
tfidf_t, tfidf_d, preprocessor = build_split_pipeline_components(
    meta_cols,
    weights={'tfidf_title': 2.0, 'tfidf_desc': 1.0, 'num': 1.0}
)

# Étape 3 : Créer le pipeline complet
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(C=2.0, max_iter=1000, solver='saga'))
])

# Étape 4 : Entraîner et évaluer
from sklearn.model_selection import train_test_split

X = df[['designation_cleaned', 'description_cleaned'] + meta_cols]
y = df['prdtypecode']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
score = pipeline.score(X_valid, y_valid)
print(f"Accuracy: {score:.4f}")

# Étape 5 : Prédictions
y_pred = pipeline.predict(X_valid)
"""

    print("\nCode exemple pour un pipeline complet :")
    print(code_example)


def example_comparison():
    """Exemple 6 : Comparaison des deux stratégies."""
    print("=" * 70)
    print("EXEMPLE 6 : Comparaison stratégie séparée vs fusionnée")
    print("=" * 70)

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    COMPARAISON DES STRATÉGIES                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. STRATÉGIE SÉPARÉE (titre/description séparés)                     ║
║     • Utilise : build_split_pipeline_components()                     ║
║     • Colonnes : designation_cleaned + description_cleaned            ║
║     • Pondération : titre=2.0, desc=1.0 (titre plus important)        ║
║     • Performance : F1 score ≈ 0.8067                                 ║
║     • Avantage : Meilleure performance, contrôle fin                  ║
║                                                                       ║
║  2. STRATÉGIE FUSIONNÉE (texte concaténé)                             ║
║     • Utilise : build_merged_pipeline_components()                    ║
║     • Colonnes : text_all (titre + description)                       ║
║     • Pondération : tfidf_all=1.0, num=1.0                            ║
║     • Performance : F1 score ≈ 0.7526                                 ║
║     • Avantage : Plus simple, baseline rapide                         ║
║                                                                       ║
║  RECOMMANDATION : Utiliser la stratégie séparée pour la production    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    example_basic_tfidf()
    example_preprocessor_split()
    example_preprocessor_merged()
    example_high_level_utilities()
    example_complete_pipeline()
    example_comparison()

    print("=" * 70)
    print("✓ Tous les exemples ont été affichés avec succès !")
    print("=" * 70)

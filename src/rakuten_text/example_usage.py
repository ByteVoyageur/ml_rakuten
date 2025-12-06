"""
Exemples d'utilisation du module rakuten_text.

Ce script démontre comment utiliser les fonctions de nettoyage et d'extraction
de features pour les données textuelles Rakuten.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rakuten_text import (
    # Cleaning
    global_text_cleaner,

    # Features
    add_structural_features,
    get_meta_feature_columns,
    structural_stats,
)


def example_cleaning():
    """Exemple de nettoyage de texte."""
    print("=" * 60)
    print("EXEMPLE 1 : Nettoyage de texte")
    print("=" * 60)

    raw_text = """
    <strong>Produit de haute qualité</strong><br/>
    Dimensions : 22 x 11 x 5 cm
    Poids : 500 g
    """

    cleaned = global_text_cleaner(
        raw_text,
        use_basic_cleaning=True,
        normalize_x_dimensions=True,
        remove_boilerplate=True,
        remove_nltk_stops=True
    )

    print(f"Texte original :\n{raw_text}")
    print(f"\nTexte nettoyé :\n{cleaned}")
    print()


def example_structural_features():
    """Exemple d'extraction de features structurelles."""
    print("=" * 60)
    print("EXEMPLE 2 : Features structurelles d'un texte")
    print("=" * 60)

    text = "Livre 21 x 15 cm, poids 200 g, 300 pages"
    stats = structural_stats(text)

    print(f"Texte : {text}")
    print(f"\nStatistiques :")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    print()


def example_dataframe_features():
    """Exemple d'ajout de features à un DataFrame."""
    print("=" * 60)
    print("EXEMPLE 3 : Ajout de features à un DataFrame")
    print("=" * 60)

    # Créer un DataFrame exemple
    df = pd.DataFrame({
        'designation_cleaned': [
            'Livre pour enfants 200 pages',
            'Stylo bleu x10',
            'Cahier 21 x 29.7 cm'
        ],
        'description_cleaned': [
            'Roman fantastique 15 x 21 cm, 350 g',
            '',
            'Cahier spirale 100 pages, poids 200 g'
        ]
    })

    print("DataFrame original :")
    print(df)
    print()

    # Ajouter les features structurelles
    df = add_structural_features(df)

    print("Colonnes de features ajoutées :")
    meta_cols = get_meta_feature_columns(df)
    print(f"  {len(meta_cols)} colonnes : {meta_cols}")
    print()

    print("Aperçu des features :")
    print(df[meta_cols].head())
    print()


def example_pipeline_ready():
    """Exemple de préparation pour un pipeline scikit-learn."""
    print("=" * 60)
    print("EXEMPLE 4 : Préparation pour pipeline scikit-learn")
    print("=" * 60)

    # Simuler des données
    df = pd.DataFrame({
        'designation_cleaned': ['Produit A 10 cm', 'Produit B'],
        'description_cleaned': ['Description longue avec unités 5 kg', ''],
        'prdtypecode': [10, 20]
    })

    # Ajouter features
    df = add_structural_features(df)

    # Récupérer les colonnes de features numériques
    meta_cols = get_meta_feature_columns(df)

    print("Colonnes pour le pipeline :")
    print(f"  - Texte (titre) : designation_cleaned")
    print(f"  - Texte (desc)  : description_cleaned")
    print(f"  - Features num  : {meta_cols}")
    print()

    print("Exemple de ColumnTransformer scikit-learn :")
    print("""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf_title', TfidfVectorizer(), 'designation_cleaned'),
            ('tfidf_desc', TfidfVectorizer(), 'description_cleaned'),
            ('num', StandardScaler(with_mean=False), meta_cols),
        ]
    )
    """)


if __name__ == "__main__":
    example_cleaning()
    example_structural_features()
    example_dataframe_features()
    example_pipeline_ready()

    print("=" * 60)
    print("Tous les exemples ont été exécutés avec succès !")
    print("=" * 60)

#!/usr/bin/env python3
"""
Script de création du dataset texte nettoyé et enrichi pour Rakuten.

Ce script charge les données brutes, nettoie les textes, ajoute les features
structurelles, et sauvegarde le dataset final prêt pour la modélisation.

Usage:
    python scripts/make_text_dataset.py
    python scripts/make_text_dataset.py --input-x data/raw/X_train.csv
    python scripts/make_text_dataset.py --output data/processed/my_dataset.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

# Ajouter le répertoire src au path pour importer rakuten_text
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rakuten_text.cleaning import global_text_cleaner
from rakuten_text.features import add_structural_features


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Crée le dataset texte nettoyé et enrichi pour Rakuten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments d'entrée
    parser.add_argument(
        "--input-x",
        type=str,
        default="data/raw/X_train_update.csv",
        help="Chemin vers le fichier X_train (features)"
    )

    parser.add_argument(
        "--input-y",
        type=str,
        default="data/raw/Y_train_CVw08PX.csv",
        help="Chemin vers le fichier Y_train (labels)"
    )

    # Arguments de sortie
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/rakuten_text_train_v1.csv",
        help="Chemin de sortie pour le dataset final"
    )

    # Options de nettoyage
    parser.add_argument(
        "--no-basic-cleaning",
        action="store_true",
        help="Désactive le nettoyage de base (HTML, ponctuation, etc.)"
    )

    parser.add_argument(
        "--no-boilerplate",
        action="store_true",
        help="Désactive la suppression des phrases boilerplate"
    )

    parser.add_argument(
        "--no-nltk-stops",
        action="store_true",
        help="Désactive la suppression des stopwords NLTK"
    )

    parser.add_argument(
        "--no-custom-stops",
        action="store_true",
        help="Désactive la suppression des stopwords personnalisés"
    )

    # Options diverses
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Active le mode verbeux"
    )

    return parser.parse_args()


def load_data(x_path: str, y_path: str, verbose: bool = False):
    """
    Charge les données brutes X et Y.

    Args:
        x_path: Chemin vers X_train_update.csv
        y_path: Chemin vers Y_train_CVw08PX.csv
        verbose: Si True, affiche des informations

    Returns:
        tuple: (X_train DataFrame, Y_train DataFrame)
    """
    if verbose:
        print(f"Chargement de {x_path}...")
    X_train = pd.read_csv(x_path, index_col=0)

    if verbose:
        print(f"Chargement de {y_path}...")
    Y_train = pd.read_csv(y_path, index_col=0)

    if verbose:
        print(f"  X_train shape: {X_train.shape}")
        print(f"  Y_train shape: {Y_train.shape}")
        print(f"  X_train columns: {X_train.columns.tolist()}")
        print(f"  Y_train columns: {Y_train.columns.tolist()}")

    return X_train, Y_train


def merge_data(X_train, Y_train, verbose: bool = False):
    """
    Fusionne X_train et Y_train sur l'index (productid).

    Args:
        X_train: DataFrame des features
        Y_train: DataFrame des labels
        verbose: Si True, affiche des informations

    Returns:
        DataFrame: Données fusionnées
    """
    if verbose:
        print("\nFusion des données X et Y...")

    df = X_train.join(Y_train, how="inner")

    if verbose:
        print(f"  Données fusionnées shape: {df.shape}")
        print(f"  Colonnes: {df.columns.tolist()}")
        print(f"  Nombre de prdtypecode uniques: {df['prdtypecode'].nunique()}")

    return df


def clean_texts(df, args, verbose: bool = False):
    """
    Nettoie les colonnes de texte designation et description.

    Args:
        df: DataFrame contenant les données
        args: Arguments parsés de la ligne de commande
        verbose: Si True, affiche la progression

    Returns:
        DataFrame: DataFrame avec colonnes nettoyées ajoutées
    """
    if verbose:
        print("\nNettoyage des textes...")

    # Configuration du nettoyage basée sur les arguments
    cleaning_config = {
        "use_basic_cleaning": not args.no_basic_cleaning,
        "normalize_x_dimensions": True,
        "remove_boilerplate": not args.no_boilerplate,
        "remove_nltk_stops": not args.no_nltk_stops,
        "remove_custom_stops": not args.no_custom_stops,
        "remove_single_digit": True,
        "remove_single_letter": True,
    }

    if verbose:
        print("  Configuration de nettoyage:")
        for key, value in cleaning_config.items():
            print(f"    {key}: {value}")

    # Nettoyer designation
    if verbose:
        print("\n  Nettoyage de 'designation'...")
    df["designation_cleaned"] = df["designation"].fillna("").apply(
        lambda x: global_text_cleaner(x, **cleaning_config)
    )

    # Nettoyer description
    if verbose:
        print("  Nettoyage de 'description'...")
    df["description_cleaned"] = df["description"].fillna("").apply(
        lambda x: global_text_cleaner(x, **cleaning_config)
    )

    if verbose:
        print("  ✓ Nettoyage terminé")
        print(f"    designation_cleaned: {df['designation_cleaned'].str.len().mean():.1f} chars moyenne")
        print(f"    description_cleaned: {df['description_cleaned'].str.len().mean():.1f} chars moyenne")

    return df


def add_features(df, verbose: bool = False):
    """
    Ajoute les features structurelles au DataFrame.

    Args:
        df: DataFrame contenant designation_cleaned et description_cleaned
        verbose: Si True, affiche la progression

    Returns:
        DataFrame: DataFrame avec features structurelles ajoutées
    """
    if verbose:
        print("\nAjout des features structurelles...")

    # Ajouter les features
    df = add_structural_features(
        df,
        text_cols=("designation_cleaned", "description_cleaned")
    )

    if verbose:
        # Compter les nouvelles colonnes ajoutées
        feature_cols = [
            c for c in df.columns
            if c.startswith("designation_cleaned_")
            or c.startswith("description_cleaned_")
        ]
        print(f"  ✓ {len(feature_cols)} features structurelles ajoutées")
        print(f"    Colonnes: {feature_cols}")

    return df


def save_data(df, output_path: str, verbose: bool = False):
    """
    Sauvegarde le DataFrame final.

    Args:
        df: DataFrame à sauvegarder
        output_path: Chemin de sortie
        verbose: Si True, affiche des informations
    """
    # Créer le répertoire de sortie si nécessaire
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSauvegarde du dataset final dans {output_path}...")
        print(f"  Shape finale: {df.shape}")
        print(f"  Colonnes: {df.columns.tolist()}")

    # Sauvegarder
    df.to_csv(output_path, index=False)

    if verbose:
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  ✓ Dataset sauvegardé ({file_size:.2f} MB)")


def main():
    """Fonction principale du script."""
    args = parse_args()

    print("=" * 70)
    print("CRÉATION DU DATASET TEXTE RAKUTEN")
    print("=" * 70)

    # 1. Charger les données
    X_train, Y_train = load_data(args.input_x, args.input_y, args.verbose)

    # 2. Fusionner X et Y
    df = merge_data(X_train, Y_train, args.verbose)

    # 3. Nettoyer les textes
    df = clean_texts(df, args, args.verbose)

    # 4. Ajouter les features structurelles
    df = add_features(df, args.verbose)

    # 5. Sauvegarder
    save_data(df, args.output, args.verbose)

    print("\n" + "=" * 70)
    print("✓ TRAITEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)
    print(f"\nDataset final disponible : {args.output}")
    print(f"  - Nombre d'exemples : {len(df)}")
    print(f"  - Nombre de colonnes : {len(df.columns)}")
    print(f"  - Nombre de catégories : {df['prdtypecode'].nunique()}")

    # Afficher un aperçu des colonnes
    print(f"\nColonnes disponibles :")
    for col in df.columns:
        print(f"  - {col}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

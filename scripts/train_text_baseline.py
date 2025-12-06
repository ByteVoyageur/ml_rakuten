#!/usr/bin/env python3
"""
Script d'entraînement des modèles de classification texte baseline pour Rakuten.

Ce script entraîne et compare deux stratégies de modélisation :
1. Stratégie fusionnée : texte concaténé (titre + description)
2. Stratégie séparée : titre et description traités séparément avec pondérations

Le meilleur modèle est sauvegardé automatiquement.

Usage:
    python scripts/train_text_baseline.py
    python scripts/train_text_baseline.py --input data/processed/custom.csv
    python scripts/train_text_baseline.py --output models/my_model.joblib
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Ajouter le répertoire src au path pour importer rakuten_text
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rakuten_text import (
    # Features
    get_meta_feature_columns,

    # Vectorization
    build_tfidf_title,
    build_tfidf_desc,
    build_tfidf_all,
    build_preprocess_split,
    build_preprocess_merged,

    # Modeling
    build_logreg_model,
    build_pipeline,
    evaluate_pipeline,
    evaluate_weight_grid,
)


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Entraîne et compare les modèles de classification texte",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/rakuten_text_train_v1.csv",
        help="Chemin vers le dataset traité"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/text_logreg_best.joblib",
        help="Chemin de sortie pour le meilleur modèle"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de validation"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine aléatoire pour reproductibilité"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Active le mode verbeux"
    )

    return parser.parse_args()


def load_dataset(input_path: str, verbose: bool = False):
    """
    Charge le dataset traité.

    Args:
        input_path: Chemin vers le CSV
        verbose: Si True, affiche des informations

    Returns:
        DataFrame: Dataset chargé
    """
    if verbose:
        print(f"Chargement de {input_path}...")

    df = pd.read_csv(input_path)

    if verbose:
        print(f"  Dataset shape: {df.shape}")
        print(f"  Colonnes: {df.columns.tolist()}")
        print(f"  Catégories: {df['prdtypecode'].nunique()}")

    return df


def prepare_data(df, test_size=0.2, random_state=42, verbose=False):
    """
    Prépare les données pour l'entraînement.

    Args:
        df: DataFrame avec données nettoyées
        test_size: Proportion du jeu de validation
        random_state: Graine aléatoire
        verbose: Si True, affiche des informations

    Returns:
        tuple: (X_train, X_valid, y_train, y_valid, meta_cols)
    """
    if verbose:
        print("\nPréparation des données...")

    # Récupérer les colonnes de features structurelles
    meta_cols = get_meta_feature_columns(df)

    if verbose:
        print(f"  Features structurelles: {len(meta_cols)}")
        print(f"  Colonnes: {meta_cols}")

    # Préparer X et y
    X = df[['designation_cleaned', 'description_cleaned'] + meta_cols].copy()
    y = df['prdtypecode'].values

    if verbose:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

    # Split train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    if verbose:
        print(f"  X_train: {X_train.shape}")
        print(f"  X_valid: {X_valid.shape}")

    return X_train, X_valid, y_train, y_valid, meta_cols


def train_merged_strategy(
    X_train, X_valid, y_train, y_valid, meta_cols, verbose=False
):
    """
    Entraîne le modèle avec stratégie fusionnée (texte concaténé).

    Args:
        X_train, X_valid, y_train, y_valid: Données d'entraînement/validation
        meta_cols: Colonnes de features numériques
        verbose: Si True, affiche les résultats

    Returns:
        tuple: (pipeline, results)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STRATÉGIE FUSIONNÉE (texte concaténé)")
        print("=" * 70)

    # Créer la colonne text_all (titre + description)
    X_train_merged = X_train.copy()
    X_valid_merged = X_valid.copy()

    X_train_merged["text_all"] = (
        X_train_merged["designation_cleaned"].str.strip() + " " +
        X_train_merged["description_cleaned"].str.strip()
    ).str.strip()

    X_valid_merged["text_all"] = (
        X_valid_merged["designation_cleaned"].str.strip() + " " +
        X_valid_merged["description_cleaned"].str.strip()
    ).str.strip()

    # Construire le pipeline
    tfidf_all = build_tfidf_all()
    preprocessor = build_preprocess_merged(tfidf_all, meta_cols)
    model = build_logreg_model()
    pipeline = build_pipeline(preprocessor, model)

    # Entraîner et évaluer
    results = evaluate_pipeline(
        pipeline,
        X_train_merged,
        y_train,
        X_valid_merged,
        y_valid,
        verbose=verbose
    )

    return pipeline, results


def train_split_strategy(
    X_train, X_valid, y_train, y_valid, meta_cols, verbose=False
):
    """
    Entraîne le modèle avec stratégie séparée (titre/description).

    Teste plusieurs configurations de pondération et retourne la meilleure.

    Args:
        X_train, X_valid, y_train, y_valid: Données d'entraînement/validation
        meta_cols: Colonnes de features numériques
        verbose: Si True, affiche les résultats

    Returns:
        tuple: (best_pipeline, best_results, all_results)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STRATÉGIE SÉPARÉE (titre/description)")
        print("=" * 70)

    # Construire les composants de base
    tfidf_title = build_tfidf_title()
    tfidf_desc = build_tfidf_desc()
    preprocessor = build_preprocess_split(tfidf_title, tfidf_desc, meta_cols)
    model = build_logreg_model()

    # Grille de pondérations à tester (comme dans le notebook)
    weight_grid = [
        (1.0, 1.0),  # Titre et description égaux
        (2.0, 1.0),  # Titre 2x plus important
        (3.0, 1.0),  # Titre 3x plus important
        (1.0, 2.0),  # Description 2x plus importante
    ]

    # Évaluer toutes les configurations
    weight_results = evaluate_weight_grid(
        X_train, y_train, X_valid, y_valid,
        preprocessor, model, weight_grid,
        verbose=verbose
    )

    # Trouver la meilleure configuration
    best_result = max(weight_results, key=lambda x: x['f1_weighted'])

    if verbose:
        print("\n" + "=" * 70)
        print("MEILLEURE CONFIGURATION TROUVÉE")
        print("=" * 70)
        print(f"Pondérations : titre={best_result['w_title']}, "
              f"desc={best_result['w_desc']}")
        print(f"F1 Score     : {best_result['f1_weighted']:.4f}")
        print("=" * 70)

    # Reconstruire le pipeline avec les meilleurs poids
    preprocessor_best = build_preprocess_split(
        tfidf_title, tfidf_desc, meta_cols,
        weights={
            'tfidf_title': best_result['w_title'],
            'tfidf_desc': best_result['w_desc'],
            'num': 1.0
        }
    )
    model_best = build_logreg_model()
    best_pipeline = build_pipeline(preprocessor_best, model_best)

    # Entraîner le meilleur pipeline et obtenir les résultats détaillés
    print("\nEntraînement du meilleur modèle...")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_valid)

    from sklearn.metrics import f1_score, classification_report

    best_results = {
        'f1_weighted': f1_score(y_valid, y_pred, average='weighted'),
        'classification_report': classification_report(y_valid, y_pred),
        'y_pred': y_pred,
        'train_score': best_pipeline.score(X_train, y_train),
        'valid_score': best_pipeline.score(X_valid, y_valid),
        'w_title': best_result['w_title'],
        'w_desc': best_result['w_desc'],
    }

    if verbose:
        print(f"\nF1 Score final: {best_results['f1_weighted']:.4f}")
        print("\nClassification Report:")
        print(best_results['classification_report'])

    return best_pipeline, best_results, weight_results


def compare_and_save(
    pipeline_merged, results_merged,
    pipeline_split, results_split,
    output_path, verbose=False
):
    """
    Compare les deux stratégies et sauvegarde le meilleur modèle.

    Args:
        pipeline_merged: Pipeline de la stratégie fusionnée
        results_merged: Résultats de la stratégie fusionnée
        pipeline_split: Pipeline de la stratégie séparée
        results_split: Résultats de la stratégie séparée
        output_path: Chemin de sortie pour le modèle
        verbose: Si True, affiche la comparaison
    """
    f1_merged = results_merged['f1_weighted']
    f1_split = results_split['f1_weighted']

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARAISON FINALE")
        print("=" * 70)
        print(f"Stratégie FUSIONNÉE  : F1 = {f1_merged:.4f}")
        print(f"Stratégie SÉPARÉE    : F1 = {f1_split:.4f}")
        print(f"Différence (séparée - fusionnée) : {f1_split - f1_merged:+.4f}")
        print("=" * 70)

    # Sélectionner le meilleur modèle
    if f1_split >= f1_merged:
        best_pipeline = pipeline_split
        best_f1 = f1_split
        best_strategy = "SÉPARÉE"
        if 'w_title' in results_split:
            strategy_info = f" (titre={results_split['w_title']}, desc={results_split['w_desc']})"
        else:
            strategy_info = ""
    else:
        best_pipeline = pipeline_merged
        best_f1 = f1_merged
        best_strategy = "FUSIONNÉE"
        strategy_info = ""

    if verbose:
        print(f"\nMEILLEUR MODÈLE : Stratégie {best_strategy}{strategy_info}")
        print(f"F1 Score        : {best_f1:.4f}")

    # Créer le répertoire de sortie si nécessaire
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    if verbose:
        print(f"\nSauvegarde du modèle dans {output_path}...")

    # Sauvegarder avec métadonnées
    model_data = {
        'pipeline': best_pipeline,
        'strategy': best_strategy,
        'f1_score': best_f1,
        'results': results_split if f1_split >= f1_merged else results_merged,
    }

    joblib.dump(model_data, output_path)

    if verbose:
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  ✓ Modèle sauvegardé ({file_size:.2f} MB)")

    return best_strategy, best_f1


def main():
    """Fonction principale du script."""
    args = parse_args()

    print("=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES BASELINE TEXTE RAKUTEN")
    print("=" * 70)

    # 1. Charger le dataset
    df = load_dataset(args.input, args.verbose)

    # 2. Préparer les données
    X_train, X_valid, y_train, y_valid, meta_cols = prepare_data(
        df, args.test_size, args.random_state, args.verbose
    )

    # 3. Entraîner stratégie fusionnée
    pipeline_merged, results_merged = train_merged_strategy(
        X_train, X_valid, y_train, y_valid, meta_cols, args.verbose
    )

    # 4. Entraîner stratégie séparée (avec grid search de pondérations)
    pipeline_split, results_split, weight_results = train_split_strategy(
        X_train, X_valid, y_train, y_valid, meta_cols, args.verbose
    )

    # 5. Comparer et sauvegarder le meilleur
    best_strategy, best_f1 = compare_and_save(
        pipeline_merged, results_merged,
        pipeline_split, results_split,
        args.output, args.verbose
    )

    # 6. Résumé final
    print("\n" + "=" * 70)
    print("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)
    print(f"\nMeilleur modèle : Stratégie {best_strategy}")
    print(f"F1 Score        : {best_f1:.4f}")
    print(f"Modèle sauvegardé : {args.output}")
    print("\nPour charger le modèle :")
    print(f"  import joblib")
    print(f"  model_data = joblib.load('{args.output}')")
    print(f"  pipeline = model_data['pipeline']")
    print(f"  predictions = pipeline.predict(X_new)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

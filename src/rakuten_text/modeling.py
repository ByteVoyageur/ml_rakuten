"""
Construction et évaluation de modèles pour la classification Rakuten.

Ce module fournit des fonctions pour construire des modèles de classification,
créer des pipelines scikit-learn, et évaluer les performances.
"""

from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer


# =============================================================================
# Constantes de Configuration par Défaut
# =============================================================================

DEFAULT_C = 2.0
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_SOLVER = "saga"
DEFAULT_MAX_ITER = 1000
DEFAULT_N_JOBS = -1


# =============================================================================
# Construction de Modèles
# =============================================================================

def build_logreg_model(
    C: float = DEFAULT_C,
    class_weight: str = DEFAULT_CLASS_WEIGHT,
    solver: str = DEFAULT_SOLVER,
    max_iter: int = DEFAULT_MAX_ITER,
    n_jobs: int = DEFAULT_N_JOBS,
    **kwargs
) -> LogisticRegression:
    """
    Construit un modèle de régression logistique avec les paramètres optimisés.

    Cette fonction crée un modèle LogisticRegression avec les hyperparamètres
    identifiés comme optimaux dans les expérimentations du notebook
    02_xiaosong_text_preprocessing_v2.

    Args:
        C: Inverse de la force de régularisation (défaut: 2.0)
            Plus C est grand, moins la régularisation est forte
        class_weight: Pondération des classes (défaut: "balanced")
            "balanced" ajuste automatiquement les poids inversement proportionnels
            aux fréquences des classes
        solver: Algorithme d'optimisation (défaut: "saga")
            "saga" est recommandé pour les grands datasets avec features creuses
        max_iter: Nombre maximum d'itérations (défaut: 1000)
        n_jobs: Nombre de CPU à utiliser (défaut: -1, utilise tous les CPU)
        **kwargs: Arguments supplémentaires pour LogisticRegression

    Returns:
        LogisticRegression: Modèle configuré et prêt à être entraîné

    Examples:
        >>> model = build_logreg_model()
        >>> model = build_logreg_model(C=1.0, max_iter=500)

    Notes:
        - Le solveur "saga" supporte la pénalité L1, L2 et elastic-net
        - class_weight="balanced" aide pour les datasets déséquilibrés
        - n_jobs=-1 utilise tous les CPU disponibles pour accélérer l'entraînement
    """
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        n_jobs=n_jobs,
        **kwargs
    )


def build_pipeline(
    preprocess: ColumnTransformer,
    model: Any,
    step_names: Optional[Tuple[str, str]] = None
) -> Pipeline:
    """
    Construit un pipeline scikit-learn avec préprocesseur et modèle.

    Cette fonction crée un Pipeline qui enchaîne le prétraitement des données
    (TF-IDF + features numériques) et la classification.

    Args:
        preprocess: Préprocesseur ColumnTransformer (issu de build_preprocess_split
            ou build_preprocess_merged)
        model: Modèle de classification (généralement LogisticRegression)
        step_names: Tuple optionnel de (nom_preprocess, nom_model)
            Défaut: ("preprocess", "model")

    Returns:
        Pipeline: Pipeline scikit-learn prêt à être entraîné

    Examples:
        >>> from rakuten_text import build_split_pipeline_components, build_logreg_model
        >>> meta_cols = get_meta_feature_columns(df)
        >>> _, _, preprocessor = build_split_pipeline_components(meta_cols)
        >>> model = build_logreg_model()
        >>> pipeline = build_pipeline(preprocessor, model)
        >>> pipeline.fit(X_train, y_train)

    Notes:
        - Le pipeline peut être sauvegardé avec joblib.dump()
        - Tous les paramètres peuvent être ajustés via set_params()
        - Compatible avec GridSearchCV et autres outils scikit-learn
    """
    if step_names is None:
        step_names = ("preprocess", "model")

    return Pipeline([
        (step_names[0], preprocess),
        (step_names[1], model),
    ])


# =============================================================================
# Évaluation de Modèles
# =============================================================================

def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    average: str = "weighted",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Entraîne et évalue un pipeline sur les données train/validation.

    Cette fonction effectue l'entraînement complet du pipeline, génère les
    prédictions, et calcule les métriques de performance.

    Args:
        pipeline: Pipeline scikit-learn à entraîner et évaluer
        X_train: Features d'entraînement (DataFrame)
        y_train: Labels d'entraînement (Series ou array)
        X_valid: Features de validation (DataFrame)
        y_valid: Labels de validation (Series ou array)
        average: Type de moyenne pour le F1 score (défaut: "weighted")
            Options: "micro", "macro", "weighted", "samples"
        verbose: Si True, affiche les résultats (défaut: True)

    Returns:
        dict: Dictionnaire contenant les résultats d'évaluation
            - "f1_weighted": Score F1 pondéré (float)
            - "classification_report": Rapport de classification détaillé (str)
            - "y_pred": Prédictions sur l'ensemble de validation (array)
            - "train_score": Score d'accuracy sur l'ensemble d'entraînement (float)
            - "valid_score": Score d'accuracy sur l'ensemble de validation (float)

    Examples:
        >>> results = evaluate_pipeline(pipeline, X_train, y_train, X_valid, y_valid)
        >>> print(f"F1 Score: {results['f1_weighted']:.4f}")
        >>> print(results['classification_report'])

    Notes:
        - Le pipeline est modifié en place (entraîné)
        - Les prédictions sont stockées dans le dictionnaire retourné
        - Le rapport de classification inclut precision, recall, f1 par classe
    """
    # Entraîner le pipeline
    if verbose:
        print("Entraînement du pipeline...")

    pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline.predict(X_valid)

    # Calculer les métriques
    f1 = f1_score(y_valid, y_pred, average=average)
    train_score = pipeline.score(X_train, y_train)
    valid_score = pipeline.score(X_valid, y_valid)
    report = classification_report(y_valid, y_pred)

    # Afficher les résultats si verbose
    if verbose:
        print(f"\nWeighted F1 Score (validation): {f1:.4f}")
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Valid Accuracy: {valid_score:.4f}")
        print("\nClassification Report:")
        print(report)

    # Retourner les résultats
    return {
        "f1_weighted": f1,
        "classification_report": report,
        "y_pred": y_pred,
        "train_score": train_score,
        "valid_score": valid_score,
    }


def evaluate_weight_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    preprocess_split: ColumnTransformer,
    model: Any,
    weight_grid: List[Tuple[float, float]],
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Évalue différentes pondérations (titre, description) pour la stratégie séparée.

    Cette fonction teste systématiquement différentes configurations de pondération
    entre le titre et la description pour identifier la combinaison optimale.
    Elle reproduit la logique de grid search du notebook d'expérimentation.

    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_valid: Features de validation
        y_valid: Labels de validation
        preprocess_split: Préprocesseur de stratégie séparée (build_preprocess_split)
        model: Modèle de classification (généralement LogisticRegression)
        weight_grid: Liste de tuples (w_title, w_desc) à tester
            Exemple: [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
        verbose: Si True, affiche les résultats pour chaque configuration

    Returns:
        list[dict]: Liste de résultats pour chaque configuration
            Chaque dict contient:
                - "w_title": Pondération du titre (float)
                - "w_desc": Pondération de la description (float)
                - "f1_weighted": Score F1 pondéré (float)
                - "train_score": Accuracy sur train (float)
                - "valid_score": Accuracy sur validation (float)

    Examples:
        >>> weight_grid = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0)]
        >>> results = evaluate_weight_grid(
        ...     X_train, y_train, X_valid, y_valid,
        ...     preprocessor, model, weight_grid
        ... )
        >>> # Trouver la meilleure configuration
        >>> best = max(results, key=lambda x: x['f1_weighted'])
        >>> print(f"Meilleure config: titre={best['w_title']}, F1={best['f1_weighted']:.4f}")

    Notes:
        - Chaque configuration entraîne un nouveau modèle (peut être long)
        - Les résultats sont triés par ordre d'exécution (pas par performance)
        - Le préprocesseur original n'est pas modifié (utilise deepcopy)
        - Les pondérations sont appliquées via transformer_weights
    """
    results = []

    for w_title, w_desc in weight_grid:
        if verbose:
            print("\n" + "=" * 70)
            print(f"Pondérations : titre={w_title}, description={w_desc}")
            print("=" * 70)

        # Créer une copie du préprocesseur pour ne pas modifier l'original
        preprocess_w = deepcopy(preprocess_split)

        # Appliquer les nouvelles pondérations
        preprocess_w.set_params(
            transformer_weights={
                "tfidf_title": w_title,
                "tfidf_desc": w_desc,
                "num": 1.0,
            }
        )

        # Créer le pipeline avec les nouvelles pondérations
        clf_w = build_pipeline(preprocess_w, model)

        # Entraîner et évaluer
        clf_w.fit(X_train, y_train)
        y_pred = clf_w.predict(X_valid)

        # Calculer les métriques
        f1_w = f1_score(y_valid, y_pred, average="weighted")
        train_score = clf_w.score(X_train, y_train)
        valid_score = clf_w.score(X_valid, y_valid)

        if verbose:
            print(f"Weighted F1 (validation): {f1_w:.4f}")
            print(f"Train Accuracy: {train_score:.4f}")
            print(f"Valid Accuracy: {valid_score:.4f}")

        # Stocker les résultats
        results.append({
            "w_title": w_title,
            "w_desc": w_desc,
            "f1_weighted": f1_w,
            "train_score": train_score,
            "valid_score": valid_score,
        })

    # Afficher le résumé si verbose
    if verbose:
        print("\n" + "=" * 70)
        print("RÉSUMÉ DES PONDÉRATIONS / SCORES")
        print("=" * 70)
        for r in results:
            print(f" - (titre={r['w_title']}, desc={r['w_desc']}) "
                  f"-> F1={r['f1_weighted']:.4f}")

        # Trouver la meilleure configuration
        best = max(results, key=lambda x: x['f1_weighted'])
        print("\n" + "=" * 70)
        print("MEILLEURE CONFIGURATION")
        print("=" * 70)
        print(f"Pondérations : titre={best['w_title']}, description={best['w_desc']}")
        print(f"F1 Score     : {best['f1_weighted']:.4f}")
        print("=" * 70)

    return results


# =============================================================================
# Utilitaires d'Analyse
# =============================================================================

def print_weight_grid_summary(results: List[Dict[str, Any]]) -> None:
    """
    Affiche un résumé formaté des résultats de evaluate_weight_grid.

    Args:
        results: Liste de dictionnaires retournée par evaluate_weight_grid

    Examples:
        >>> results = evaluate_weight_grid(...)
        >>> print_weight_grid_summary(results)
    """
    if not results:
        print("Aucun résultat à afficher.")
        return

    print("\n" + "=" * 70)
    print("RÉSUMÉ DES RÉSULTATS PAR PONDÉRATION")
    print("=" * 70)
    print(f"{'Titre':>8} {'Description':>12} {'F1 Score':>12} {'Train Acc':>12} {'Valid Acc':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['w_title']:>8.1f} {r['w_desc']:>12.1f} "
              f"{r['f1_weighted']:>12.4f} {r['train_score']:>12.4f} "
              f"{r['valid_score']:>12.4f}")

    # Trouver la meilleure configuration
    best = max(results, key=lambda x: x['f1_weighted'])
    print("-" * 70)
    print(f"Meilleure : titre={best['w_title']}, desc={best['w_desc']}, "
          f"F1={best['f1_weighted']:.4f}")
    print("=" * 70)


def compare_strategies(
    results_split: Dict[str, Any],
    results_merged: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare les résultats de deux stratégies (séparée vs fusionnée).

    Args:
        results_split: Résultats de evaluate_pipeline pour stratégie séparée
        results_merged: Résultats de evaluate_pipeline pour stratégie fusionnée
        verbose: Si True, affiche la comparaison

    Returns:
        dict: Dictionnaire contenant la comparaison
            - "split_f1": F1 score stratégie séparée
            - "merged_f1": F1 score stratégie fusionnée
            - "difference": Différence (split - merged)
            - "better": "split" ou "merged"

    Examples:
        >>> results_split = evaluate_pipeline(pipeline_split, ...)
        >>> results_merged = evaluate_pipeline(pipeline_merged, ...)
        >>> comparison = compare_strategies(results_split, results_merged)
    """
    split_f1 = results_split["f1_weighted"]
    merged_f1 = results_merged["f1_weighted"]
    diff = split_f1 - merged_f1
    better = "split" if diff > 0 else "merged"

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARAISON DES STRATÉGIES")
        print("=" * 70)
        print(f"Stratégie SÉPARÉE   (titre/desc) : F1 = {split_f1:.4f}")
        print(f"Stratégie FUSIONNÉE (text_all)   : F1 = {merged_f1:.4f}")
        print(f"Différence (séparée - fusionnée) : {diff:+.4f}")
        print(f"\nMeilleure stratégie : {better.upper()}")
        print("=" * 70)

    return {
        "split_f1": split_f1,
        "merged_f1": merged_f1,
        "difference": diff,
        "better": better,
    }

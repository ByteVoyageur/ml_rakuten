"""
Exemples d'utilisation du module modeling.

Ce script démontre comment utiliser les fonctions de modeling pour
construire, entraîner et évaluer des modèles de classification.
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Note: Ces imports nécessitent que les données soient disponibles
# Les exemples montrent la structure du code, mais ne s'exécuteront
# que si les données sont présentes


def example_basic_model():
    """Exemple 1 : Construction d'un modèle de base."""
    print("=" * 70)
    print("EXEMPLE 1 : Construction de modèles")
    print("=" * 70)

    from rakuten_text import build_logreg_model

    # Modèle avec paramètres par défaut
    print("\n1. Modèle avec paramètres par défaut :")
    model = build_logreg_model()
    print(f"   C: {model.C}")
    print(f"   class_weight: {model.class_weight}")
    print(f"   solver: {model.solver}")
    print(f"   max_iter: {model.max_iter}")
    print(f"   n_jobs: {model.n_jobs}")

    # Modèle personnalisé
    print("\n2. Modèle personnalisé :")
    model_custom = build_logreg_model(
        C=1.0,
        max_iter=500,
        solver="lbfgs"
    )
    print(f"   C: {model_custom.C}")
    print(f"   max_iter: {model_custom.max_iter}")
    print(f"   solver: {model_custom.solver}")
    print()


def example_build_pipeline():
    """Exemple 2 : Construction d'un pipeline."""
    print("=" * 70)
    print("EXEMPLE 2 : Construction d'un pipeline")
    print("=" * 70)

    from rakuten_text import (
        build_split_pipeline_components,
        build_logreg_model,
        build_pipeline,
    )

    # Simuler des colonnes de features
    meta_cols = [
        'designation_cleaned_len_char',
        'designation_cleaned_len_tokens',
        'description_cleaned_len_char',
    ]

    print("\n1. Construction des composants :")
    _, _, preprocessor = build_split_pipeline_components(meta_cols)
    model = build_logreg_model()
    print(f"   ✓ Preprocessor créé")
    print(f"   ✓ Model créé (LogisticRegression)")

    print("\n2. Assemblage du pipeline :")
    pipeline = build_pipeline(preprocessor, model)
    print(f"   ✓ Pipeline créé avec {len(pipeline.steps)} étapes")
    print(f"   Étapes : {[name for name, _ in pipeline.steps]}")
    print()


def example_workflow_complet():
    """Exemple 3 : Workflow complet (pseudo-code)."""
    print("=" * 70)
    print("EXEMPLE 3 : Workflow complet (pseudo-code)")
    print("=" * 70)

    code_example = """
# ============================================================
# WORKFLOW COMPLET : Du nettoyage à l'évaluation
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from rakuten_text import (
    # Préparation
    add_structural_features,
    get_meta_feature_columns,

    # Construction
    build_split_pipeline_components,
    build_logreg_model,
    build_pipeline,

    # Évaluation
    evaluate_pipeline,
)

# Étape 1 : Charger et préparer les données
df = pd.read_csv("rakuten_text_train_v1.csv")
df = add_structural_features(df)
meta_cols = get_meta_feature_columns(df)

# Étape 2 : Split train/validation
X = df[['designation_cleaned', 'description_cleaned'] + meta_cols]
y = df['prdtypecode']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Étape 3 : Construire le pipeline
_, _, preprocessor = build_split_pipeline_components(meta_cols)
model = build_logreg_model()
pipeline = build_pipeline(preprocessor, model)

# Étape 4 : Entraîner et évaluer
results = evaluate_pipeline(
    pipeline, X_train, y_train, X_valid, y_valid
)

print(f"F1 Score: {results['f1_weighted']:.4f}")
print(results['classification_report'])
"""

    print(code_example)


def example_weight_grid():
    """Exemple 4 : Évaluation de grille de pondérations."""
    print("=" * 70)
    print("EXEMPLE 4 : Évaluation de grille de pondérations")
    print("=" * 70)

    code_example = """
# ============================================================
# RECHERCHE DE PONDÉRATIONS OPTIMALES
# ============================================================

from rakuten_text import (
    build_split_pipeline_components,
    build_logreg_model,
    evaluate_weight_grid,
    print_weight_grid_summary,
)

# Préparer les données (voir exemple 3)
# X_train, y_train, X_valid, y_valid, meta_cols...

# Construire les composants
tfidf_t, tfidf_d, preprocessor = build_split_pipeline_components(meta_cols)
model = build_logreg_model()

# Définir la grille de pondérations à tester
weight_grid = [
    (1.0, 1.0),  # Titre et description égaux
    (2.0, 1.0),  # Titre 2x plus important
    (3.0, 1.0),  # Titre 3x plus important
    (1.0, 2.0),  # Description 2x plus importante
]

# Évaluer toutes les configurations
results = evaluate_weight_grid(
    X_train, y_train, X_valid, y_valid,
    preprocessor, model, weight_grid,
    verbose=True
)

# Afficher un résumé formaté
print_weight_grid_summary(results)

# Trouver la meilleure configuration
best = max(results, key=lambda x: x['f1_weighted'])
print(f"\\nMeilleure : titre={best['w_title']}, F1={best['f1_weighted']:.4f}")
"""

    print(code_example)


def example_compare_strategies():
    """Exemple 5 : Comparaison de stratégies."""
    print("=" * 70)
    print("EXEMPLE 5 : Comparaison stratégie séparée vs fusionnée")
    print("=" * 70)

    code_example = """
# ============================================================
# COMPARAISON STRATÉGIE SÉPARÉE VS FUSIONNÉE
# ============================================================

from rakuten_text import (
    build_split_pipeline_components,
    build_merged_pipeline_components,
    build_logreg_model,
    build_pipeline,
    evaluate_pipeline,
    compare_strategies,
)

# Préparer les données
# ...

# 1. STRATÉGIE SÉPARÉE (titre/description séparés)
_, _, preprocessor_split = build_split_pipeline_components(meta_cols)
model_split = build_logreg_model()
pipeline_split = build_pipeline(preprocessor_split, model_split)

results_split = evaluate_pipeline(
    pipeline_split, X_train, y_train, X_valid, y_valid
)

# 2. STRATÉGIE FUSIONNÉE (texte concaténé)
# D'abord créer la colonne text_all
df["text_all"] = (
    df["designation_cleaned"] + " " + df["description_cleaned"]
).str.strip()

X_merged = df[['text_all'] + meta_cols]
X_train_m, X_valid_m, y_train_m, y_valid_m = train_test_split(
    X_merged, y, test_size=0.2, random_state=42, stratify=y
)

_, preprocessor_merged = build_merged_pipeline_components(meta_cols)
model_merged = build_logreg_model()
pipeline_merged = build_pipeline(preprocessor_merged, model_merged)

results_merged = evaluate_pipeline(
    pipeline_merged, X_train_m, y_train_m, X_valid_m, y_valid_m
)

# 3. COMPARER LES DEUX STRATÉGIES
comparison = compare_strategies(results_split, results_merged)

# Résultat attendu :
# Stratégie SÉPARÉE   : F1 ≈ 0.8067
# Stratégie FUSIONNÉE : F1 ≈ 0.7526
# Différence          : +0.0541
# Meilleure stratégie : SPLIT
"""

    print(code_example)


def example_complete_workflow():
    """Exemple 6 : Exemple complet réel (commenté)."""
    print("=" * 70)
    print("EXEMPLE 6 : Workflow complet (prêt à l'emploi)")
    print("=" * 70)

    code_example = """
# ============================================================
# EXEMPLE COMPLET PRÊT À UTILISER DANS UN NOTEBOOK
# ============================================================

import sys
sys.path.insert(0, 'src')

import pandas as pd
from sklearn.model_selection import train_test_split

from rakuten_text import (
    add_structural_features,
    get_meta_feature_columns,
    build_split_pipeline_components,
    build_logreg_model,
    build_pipeline,
    evaluate_pipeline,
    evaluate_weight_grid,
)

# ============================================================
# 1. PRÉPARATION DES DONNÉES
# ============================================================

# Charger les données nettoyées
df = pd.read_csv("rakuten_text_train_v1.csv")

# Ajouter les features structurelles
print("Ajout des features structurelles...")
df = add_structural_features(df)

# Récupérer les colonnes de features
meta_cols = get_meta_feature_columns(df)
print(f"Features créées : {len(meta_cols)}")

# Préparer X et y
X = df[['designation_cleaned', 'description_cleaned'] + meta_cols]
y = df['prdtypecode']

# Split train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {X_train.shape}, Valid : {X_valid.shape}")

# ============================================================
# 2. CONSTRUCTION DU PIPELINE
# ============================================================

# Construire les composants avec pondérations par défaut
_, _, preprocessor = build_split_pipeline_components(
    meta_cols,
    weights={'tfidf_title': 2.0, 'tfidf_desc': 1.0, 'num': 1.0}
)

# Construire le modèle
model = build_logreg_model(C=2.0, max_iter=1000)

# Assembler le pipeline
pipeline = build_pipeline(preprocessor, model)

# ============================================================
# 3. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================

print("\\nEntraînement du modèle...")
results = evaluate_pipeline(
    pipeline, X_train, y_train, X_valid, y_valid,
    verbose=True
)

# ============================================================
# 4. OPTIMISATION DES PONDÉRATIONS (OPTIONNEL)
# ============================================================

print("\\nRecherche des pondérations optimales...")
weight_grid = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]

weight_results = evaluate_weight_grid(
    X_train, y_train, X_valid, y_valid,
    preprocessor, model, weight_grid,
    verbose=True
)

# ============================================================
# 5. SAUVEGARDE DU MODÈLE
# ============================================================

import joblib
joblib.dump(pipeline, 'model_rakuten_final.pkl')
print("\\nModèle sauvegardé : model_rakuten_final.pkl")

# Pour charger le modèle plus tard :
# pipeline_loaded = joblib.load('model_rakuten_final.pkl')
# predictions = pipeline_loaded.predict(X_new)
"""

    print(code_example)


if __name__ == "__main__":
    example_basic_model()
    example_build_pipeline()
    example_workflow_complet()
    example_weight_grid()
    example_compare_strategies()
    example_complete_workflow()

    print("\n" + "=" * 70)
    print("✓ Tous les exemples ont été affichés !")
    print("=" * 70)
    print("\nPour exécuter ces exemples avec de vraies données,")
    print("copiez-collez le code dans un notebook Jupyter.")
    print("=" * 70)

# Utilisation dans un Notebook

Ce guide montre comment utiliser le module `rakuten_text` dans un notebook Jupyter.

## Installation et Import

```python
# Ajouter le chemin du module (à faire une seule fois au début du notebook)
import sys
sys.path.insert(0, 'src')

# Importer les fonctions nécessaires
from rakuten_text import (
    # Nettoyage
    global_text_cleaner,

    # Features
    add_structural_features,
    get_meta_feature_columns,

    # Vectorization
    build_split_pipeline_components,
    build_merged_pipeline_components,
)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
```

## Workflow Complet : Étape par Étape

### 1. Charger et Nettoyer les Données

```python
# Charger les données
df = pd.read_csv("rakuten_text_train_v1.csv")

# Si les données ne sont pas déjà nettoyées, nettoyer :
# df["designation_cleaned"] = df["designation"].apply(global_text_cleaner)
# df["description_cleaned"] = df["description"].apply(global_text_cleaner)

# Vérifier
print(df[['designation_cleaned', 'description_cleaned']].head())
```

### 2. Ajouter les Features Structurelles

```python
# Ajouter automatiquement toutes les features structurelles
df = add_structural_features(df)

# Récupérer les colonnes de features créées
meta_cols = get_meta_feature_columns(df)

print(f"Features structurelles créées : {len(meta_cols)}")
print(meta_cols)
# Résultat : 10 colonnes (5 pour designation + 5 pour description)
```

### 3. Préparer les Données pour le Modèle

```python
# Préparer X et y
X = df[['designation_cleaned', 'description_cleaned'] + meta_cols]
y = df['prdtypecode']

# Split train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
```

### 4. Construire le Pipeline - STRATÉGIE SÉPARÉE (Recommandée)

```python
# Construire tous les composants en une ligne !
_, _, preprocessor = build_split_pipeline_components(
    meta_cols,
    weights={'tfidf_title': 2.0, 'tfidf_desc': 1.0, 'num': 1.0}
)

# Créer le pipeline complet
clf_split = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight='balanced',
        solver='saga',
        n_jobs=-1
    ))
])

print("Pipeline créé avec stratégie séparée (titre/description)")
```

### 5. Entraîner et Évaluer

```python
# Entraîner
print("Entraînement du modèle...")
clf_split.fit(X_train, y_train)

# Prédire
y_pred = clf_split.predict(X_valid)

# Évaluer
f1 = f1_score(y_valid, y_pred, average='weighted')
print(f"\nWeighted F1 Score: {f1:.4f}")

# Rapport détaillé
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))
```

### 6. Comparer avec la Stratégie Fusionnée (Optionnel)

```python
# D'abord, créer la colonne text_all
df["text_all"] = (
    df["designation_cleaned"].str.strip() + " " +
    df["description_cleaned"].str.strip()
).str.strip()

# Préparer X avec text_all
X_merged = df[['text_all'] + meta_cols]
X_train_m, X_valid_m, y_train_m, y_valid_m = train_test_split(
    X_merged, y, test_size=0.2, random_state=42, stratify=y
)

# Construire le pipeline fusionné
_, preprocessor_merged = build_merged_pipeline_components(meta_cols)

clf_merged = Pipeline([
    ('preprocess', preprocessor_merged),
    ('model', LogisticRegression(
        C=2.0, max_iter=1000, class_weight='balanced',
        solver='saga', n_jobs=-1
    ))
])

# Entraîner et évaluer
clf_merged.fit(X_train_m, y_train_m)
y_pred_m = clf_merged.predict(X_valid_m)
f1_merged = f1_score(y_valid_m, y_pred_m, average='weighted')

# Comparaison
print("\n" + "="*60)
print("COMPARAISON DES STRATÉGIES")
print("="*60)
print(f"Stratégie SÉPARÉE   : F1 = {f1:.4f}")
print(f"Stratégie FUSIONNÉE : F1 = {f1_merged:.4f}")
print(f"Différence          : {f1 - f1_merged:.4f}")
print("="*60)
```

## Personnalisation Avancée

### Tester Différentes Pondérations

```python
# Tester différentes pondérations pour le titre
weight_configs = [
    {'tfidf_title': 1.0, 'tfidf_desc': 1.0, 'num': 1.0},
    {'tfidf_title': 2.0, 'tfidf_desc': 1.0, 'num': 1.0},
    {'tfidf_title': 3.0, 'tfidf_desc': 1.0, 'num': 1.0},
]

results = []
for weights in weight_configs:
    # Construire le pipeline
    _, _, preprocessor = build_split_pipeline_components(
        meta_cols, weights=weights
    )

    clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(C=2.0, max_iter=1000, solver='saga'))
    ])

    # Entraîner et évaluer
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average='weighted')

    results.append({
        'weights': weights,
        'f1_score': f1
    })

    print(f"Weights {weights} -> F1: {f1:.4f}")

# Afficher le meilleur
best = max(results, key=lambda x: x['f1_score'])
print(f"\nMeilleure configuration : {best['weights']}")
print(f"F1 Score : {best['f1_score']:.4f}")
```

### Personnaliser les Paramètres TF-IDF

```python
from rakuten_text import build_tfidf_title, build_tfidf_desc, build_preprocess_split

# Construire des vectoriseurs personnalisés
tfidf_title = build_tfidf_title(
    max_features=15000,  # Réduire le nombre de features
    ngram_range=(1, 2),  # Seulement unigrams et bigrams
    min_df=10            # Fréquence minimale plus élevée
)

tfidf_desc = build_tfidf_desc(
    max_features=25000,
    ngram_range=(1, 3)
)

# Construire le préprocesseur avec ces vectoriseurs personnalisés
preprocessor = build_preprocess_split(
    tfidf_title,
    tfidf_desc,
    meta_cols,
    weights={'tfidf_title': 2.5, 'tfidf_desc': 1.0, 'num': 0.5}
)

# Utiliser dans un pipeline
clf = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(C=2.0, max_iter=1000, solver='saga'))
])
```

## Conseils et Bonnes Pratiques

1. **Toujours utiliser la stratégie séparée en production** : F1 score supérieur de ~5 points

2. **Pondération du titre** : Commencer avec `titre=2.0, description=1.0`, puis ajuster

3. **Features numériques** : Ne pas oublier d'inclure `meta_cols` dans le pipeline

4. **Sauvegarde du modèle** :
   ```python
   import joblib
   joblib.dump(clf_split, 'model_rakuten_v1.pkl')
   ```

5. **Chargement du modèle** :
   ```python
   clf_loaded = joblib.load('model_rakuten_v1.pkl')
   predictions = clf_loaded.predict(X_new)
   ```

## Résolution de Problèmes

**Erreur : "No module named 'rakuten_text'"**
```python
# Vérifier que le path est correct
import sys
print(sys.path)
# Ajouter le bon chemin
sys.path.insert(0, '/chemin/vers/src')
```

**Performance inférieure à attendue**
- Vérifier que `add_structural_features()` a été appelé
- Vérifier que les textes sont bien nettoyés
- Essayer différentes pondérations
- Augmenter `max_features` ou `C` du modèle

**Temps d'entraînement trop long**
- Réduire `max_features` (15000 au lieu de 20000)
- Utiliser `ngram_range=(1, 2)` au lieu de `(1, 3)`
- Réduire `max_iter` du modèle

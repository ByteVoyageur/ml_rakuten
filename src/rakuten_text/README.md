# Rakuten Text Processing

Module de traitement de texte pour les données produits Rakuten.

## Structure

```
src/rakuten_text/
├── __init__.py                # Exports principaux du module
├── cleaning.py                # Fonctions de nettoyage de texte
├── features.py                # Extraction de features structurelles
├── vectorization.py           # Construction de vectoriseurs et pipelines
├── modeling.py                # Modèles, évaluation et comparaison
├── example_usage.py           # Exemples cleaning + features
├── example_vectorization.py   # Exemples vectorization + pipelines
├── example_modeling.py        # Exemples modeling + évaluation
└── README.md                  # Cette documentation
```

## Installation

Assurer que les dépendances sont installées :

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Nettoyage de texte

```python
from rakuten_text import global_text_cleaner

text = "<strong>Produit de haute qualité</strong> 22 x 11 cm"
cleaned = global_text_cleaner(
    text,
    use_basic_cleaning=True,
    normalize_x_dimensions=True,
    remove_boilerplate=True,
    remove_nltk_stops=True
)
# Résultat: "produit 22x11 cm"
```

### 2. Extraction de features structurelles

```python
import pandas as pd
from rakuten_text import add_structural_features, get_meta_feature_columns

# Charger les données
df = pd.read_csv("rakuten_text_train_v1.csv")

# Ajouter les features structurelles
df = add_structural_features(df)

# Obtenir les noms des colonnes de features
meta_cols = get_meta_feature_columns(df)
# Résultat: ['designation_cleaned_len_char', 'designation_cleaned_len_tokens', ...]
```

### 3. Pipeline scikit-learn

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Créer le préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf_title', TfidfVectorizer(max_features=20000), 'designation_cleaned'),
        ('tfidf_desc', TfidfVectorizer(max_features=30000), 'description_cleaned'),
        ('num', StandardScaler(with_mean=False), meta_cols),
    ],
    transformer_weights={
        'tfidf_title': 2.0,  # Pondération du titre
        'tfidf_desc': 1.0,
        'num': 1.0,
    }
)

# Pipeline complet
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression())
])

# Entraînement
pipeline.fit(X_train, y_train)
```

## Fonctions principales

### Module `cleaning`

- **`global_text_cleaner()`** : Fonction maîtresse de nettoyage avec options configurables
- **`nettoyer_texte()`** : Nettoyage de base (HTML, ponctuation, casse)
- **`merge_x_dimensions()`** : Normalise les dimensions (ex: "22 x 11" → "22x11")

### Module `features`

- **`add_structural_features(df, text_cols)`** : Ajoute les features structurelles au DataFrame
- **`get_meta_feature_columns(df)`** : Retourne la liste des colonnes de features
- **`structural_stats(text)`** : Calcule les statistiques d'un texte

### Module `vectorization`

#### Construction de vectoriseurs TF-IDF
- **`build_tfidf_title()`** : Crée un TfidfVectorizer pour les titres (max_features=20000)
- **`build_tfidf_desc()`** : Crée un TfidfVectorizer pour les descriptions (max_features=30000)
- **`build_tfidf_all()`** : Crée un TfidfVectorizer pour le texte fusionné (max_features=40000)

#### Construction de préprocesseurs
- **`build_preprocess_split(tfidf_title, tfidf_desc, meta_cols, weights)`** : ColumnTransformer avec stratégie séparée
- **`build_preprocess_merged(tfidf_all, meta_cols, weights)`** : ColumnTransformer avec stratégie fusionnée

#### Utilitaires haut niveau
- **`build_split_pipeline_components(meta_cols, ...)`** : Construit tous les composants en une fois (stratégie séparée)
- **`build_merged_pipeline_components(meta_cols, ...)`** : Construit tous les composants en une fois (stratégie fusionnée)

### Module `modeling`

#### Construction de modèles
- **`build_logreg_model(C=2.0, ...)`** : Crée un modèle LogisticRegression avec paramètres optimaux
- **`build_pipeline(preprocess, model)`** : Assemble préprocesseur et modèle en Pipeline

#### Évaluation
- **`evaluate_pipeline(pipeline, X_train, y_train, X_valid, y_valid)`** : Entraîne et évalue un pipeline
- **`evaluate_weight_grid(X_train, y_train, X_valid, y_valid, preprocess, model, weight_grid)`** : Teste différentes pondérations

#### Analyse
- **`compare_strategies(results_split, results_merged)`** : Compare les stratégies séparée vs fusionnée
- **`print_weight_grid_summary(results)`** : Affiche un résumé formaté des résultats

## Features extraites

Pour chaque colonne de texte (`designation_cleaned`, `description_cleaned`), les features suivantes sont créées :

- `{col}_len_char` : Nombre de caractères
- `{col}_len_tokens` : Nombre de tokens (mots)
- `{col}_num_digits` : Nombre de chiffres
- `{col}_num_units` : Nombre d'unités de mesure (cm, kg, etc.)
- `{col}_num_mult_pattern` : Nombre de patterns multiplicatifs (x2, 3x, etc.)

## Exemples

Voir `example_usage.py` pour des exemples complets d'utilisation.

```bash
python src/rakuten_text/example_usage.py
```

## Workflow typique

### Workflow complet : du nettoyage à la modélisation

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from rakuten_text import (
    global_text_cleaner,
    add_structural_features,
    get_meta_feature_columns,
    build_split_pipeline_components,
    build_logreg_model,
    build_pipeline,
    evaluate_pipeline,
)

# 1. Charger les données brutes
df = pd.read_csv("X_train_update.csv")
y_df = pd.read_csv("Y_train_CVw08PX.csv")

# 2. Nettoyer les textes
df["designation_cleaned"] = df["designation"].apply(global_text_cleaner)
df["description_cleaned"] = df["description"].apply(global_text_cleaner)

# 3. Ajouter les features structurelles
df = add_structural_features(df)

# 4. Récupérer les colonnes de features
meta_cols = get_meta_feature_columns(df)

# 5. Construire le pipeline (approche simplifiée)
_, _, preprocessor = build_split_pipeline_components(meta_cols)
model = build_logreg_model()
pipeline = build_pipeline(preprocessor, model)

# 6. Préparer les données
X = df[['designation_cleaned', 'description_cleaned'] + meta_cols]
y = y_df['prdtypecode']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Entraîner et évaluer (une seule fonction !)
results = evaluate_pipeline(pipeline, X_train, y_train, X_valid, y_valid)
print(f"F1 Score: {results['f1_weighted']:.4f}")

# 8. Sauvegarder le modèle
import joblib
joblib.dump(pipeline, 'model_rakuten.pkl')
```

## Comparaison des stratégies

| Stratégie | Fonction | Colonnes | F1 Score | Avantages |
|-----------|----------|----------|----------|-----------|
| **Séparée** | `build_split_pipeline_components()` | designation + description | ~0.8067 | Meilleure performance, contrôle fin |
| **Fusionnée** | `build_merged_pipeline_components()` | text_all | ~0.7526 | Plus simple, baseline rapide |

**Recommandation** : Utiliser la stratégie séparée pour la production.

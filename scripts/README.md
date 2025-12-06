# Scripts Rakuten

Ce répertoire contient des scripts utilitaires pour le projet Rakuten.

## Table des matières

1. [make_text_dataset.py](#make_text_datasetpy) - Création du dataset texte nettoyé
2. [train_text_baseline.py](#train_text_baselinepy) - Entraînement des modèles baseline

---

## make_text_dataset.py

Script de création du dataset texte nettoyé et enrichi.

### Description

Ce script effectue les opérations suivantes :
1. Charge les données brutes X_train et Y_train
2. Fusionne les données sur l'index (productid)
3. Nettoie les colonnes `designation` et `description` avec `global_text_cleaner`
4. Ajoute les features structurelles avec `add_structural_features`
5. Sauvegarde le dataset final

### Usage de base

```bash
# Utilisation par défaut (chemins standards)
python scripts/make_text_dataset.py

# Avec mode verbeux
python scripts/make_text_dataset.py -v
```

### Options

```bash
# Voir toutes les options
python scripts/make_text_dataset.py --help

# Personnaliser les chemins d'entrée
python scripts/make_text_dataset.py \
  --input-x data/raw/X_train_update.csv \
  --input-y data/raw/Y_train_CVw08PX.csv

# Personnaliser le chemin de sortie
python scripts/make_text_dataset.py \
  --output data/processed/my_dataset.csv

# Désactiver certaines étapes de nettoyage
python scripts/make_text_dataset.py \
  --no-nltk-stops \
  --no-boilerplate
```

### Arguments disponibles

#### Entrées/Sorties
- `--input-x` : Chemin vers X_train (défaut: `data/raw/X_train_update.csv`)
- `--input-y` : Chemin vers Y_train (défaut: `data/raw/Y_train_CVw08PX.csv`)
- `--output` : Chemin de sortie (défaut: `data/processed/rakuten_text_train_v1.csv`)

#### Options de nettoyage
- `--no-basic-cleaning` : Désactive le nettoyage de base (HTML, ponctuation)
- `--no-boilerplate` : Désactive la suppression des phrases boilerplate
- `--no-nltk-stops` : Désactive la suppression des stopwords NLTK
- `--no-custom-stops` : Désactive la suppression des stopwords personnalisés

#### Autres
- `--verbose`, `-v` : Active le mode verbeux

### Exemple complet

```bash
python scripts/make_text_dataset.py \
  --input-x data/raw/X_train_update.csv \
  --input-y data/raw/Y_train_CVw08PX.csv \
  --output data/processed/rakuten_text_train_v1.csv \
  --verbose
```

### Sortie attendue

```
======================================================================
CRÉATION DU DATASET TEXTE RAKUTEN
======================================================================
Chargement de data/raw/X_train_update.csv...
Chargement de data/raw/Y_train_CVw08PX.csv...
  X_train shape: (84916, 4)
  Y_train shape: (84916, 1)

Fusion des données X et Y...
  Données fusionnées shape: (84916, 5)

Nettoyage des textes...
  Configuration de nettoyage:
    use_basic_cleaning: True
    normalize_x_dimensions: True
    remove_boilerplate: True
    remove_nltk_stops: True
    remove_custom_stops: True
    remove_single_digit: True
    remove_single_letter: True

  Nettoyage de 'designation'...
  Nettoyage de 'description'...
  ✓ Nettoyage terminé

Ajout des features structurelles...
  ✓ 10 features structurelles ajoutées

Sauvegarde du dataset final dans data/processed/rakuten_text_train_v1.csv...
  Shape finale: (84916, 17)
  ✓ Dataset sauvegardé (15.23 MB)

======================================================================
✓ TRAITEMENT TERMINÉ AVEC SUCCÈS !
======================================================================

Dataset final disponible : data/processed/rakuten_text_train_v1.csv
  - Nombre d'exemples : 84916
  - Nombre de colonnes : 17
  - Nombre de catégories : 27

Colonnes disponibles :
  - designation
  - description
  - productid
  - imageid
  - prdtypecode
  - designation_cleaned
  - description_cleaned
  - designation_cleaned_len_char
  - designation_cleaned_len_tokens
  - designation_cleaned_num_digits
  - designation_cleaned_num_units
  - designation_cleaned_num_mult_pattern
  - description_cleaned_len_char
  - description_cleaned_len_tokens
  - description_cleaned_num_digits
  - description_cleaned_num_units
  - description_cleaned_num_mult_pattern
```

### Colonnes du dataset final

Le dataset final contient les colonnes suivantes :

**Colonnes originales :**
- `designation` : Titre original du produit
- `description` : Description originale du produit
- `productid` : ID unique du produit
- `imageid` : ID de l'image
- `prdtypecode` : Code de la catégorie (label)

**Colonnes nettoyées :**
- `designation_cleaned` : Titre nettoyé
- `description_cleaned` : Description nettoyée

**Features structurelles (designation) :**
- `designation_cleaned_len_char` : Longueur en caractères
- `designation_cleaned_len_tokens` : Longueur en tokens
- `designation_cleaned_num_digits` : Nombre de chiffres
- `designation_cleaned_num_units` : Nombre d'unités de mesure
- `designation_cleaned_num_mult_pattern` : Nombre de patterns multiplicatifs

**Features structurelles (description) :**
- `description_cleaned_len_char` : Longueur en caractères
- `description_cleaned_len_tokens` : Longueur en tokens
- `description_cleaned_num_digits` : Nombre de chiffres
- `description_cleaned_num_units` : Nombre d'unités de mesure
- `description_cleaned_num_mult_pattern` : Nombre de patterns multiplicatifs

### Utilisation dans un notebook

Après avoir exécuté le script, vous pouvez charger le dataset dans un notebook :

```python
import pandas as pd

# Charger le dataset traité
df = pd.read_csv("data/processed/rakuten_text_train_v1.csv")

# Vérifier
print(df.shape)
print(df.columns.tolist())
print(df.head())
```

### Dépendances

Le script nécessite :
- Python 3.7+
- pandas
- Le module `rakuten_text` (doit être dans `src/`)

Ces dépendances sont déjà installées dans l'environnement Jupyter du projet.

### Notes

- Le script crée automatiquement le répertoire de sortie si nécessaire
- Les données manquantes (NaN) sont gérées automatiquement
- Le script utilise les paramètres optimaux identifiés dans les notebooks
- Le temps d'exécution est d'environ 2-3 minutes pour le dataset complet

---

## train_text_baseline.py

Script d'entraînement des modèles de classification texte baseline.

### Description

Ce script entraîne et compare deux stratégies de modélisation :
1. **Stratégie fusionnée** : Texte concaténé (titre + description)
2. **Stratégie séparée** : Titre et description traités séparément avec pondérations

Pour la stratégie séparée, le script teste automatiquement plusieurs configurations de pondération [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0)] et conserve la meilleure.

Le meilleur modèle (basé sur le F1 score) est automatiquement sauvegardé.

### Usage de base

```bash
# Utilisation par défaut
python scripts/train_text_baseline.py

# Avec mode verbeux (recommandé)
python scripts/train_text_baseline.py -v
```

### Options

```bash
# Voir toutes les options
python scripts/train_text_baseline.py --help

# Personnaliser le dataset d'entrée
python scripts/train_text_baseline.py \
  --input data/processed/my_dataset.csv

# Personnaliser le chemin de sortie du modèle
python scripts/train_text_baseline.py \
  --output models/my_model.joblib

# Modifier les paramètres de split
python scripts/train_text_baseline.py \
  --test-size 0.3 \
  --random-state 123
```

### Arguments disponibles

- `--input` : Chemin vers le dataset traité (défaut: `data/processed/rakuten_text_train_v1.csv`)
- `--output` : Chemin de sortie pour le modèle (défaut: `models/text_logreg_best.joblib`)
- `--test-size` : Proportion du jeu de validation (défaut: `0.2`)
- `--random-state` : Graine aléatoire (défaut: `42`)
- `--verbose`, `-v` : Active le mode verbeux

### Sortie attendue

```
======================================================================
ENTRAÎNEMENT DES MODÈLES BASELINE TEXTE RAKUTEN
======================================================================
Chargement de data/processed/rakuten_text_train_v1.csv...
  Dataset shape: (84916, 17)

Préparation des données...
  Features structurelles: 10
  X_train: (67932, 12)
  X_valid: (16984, 12)

======================================================================
STRATÉGIE FUSIONNÉE (texte concaténé)
======================================================================
Entraînement du pipeline...

Weighted F1 Score (validation): 0.7526
Train Accuracy: 0.8234
Valid Accuracy: 0.7589

======================================================================
STRATÉGIE SÉPARÉE (titre/description)
======================================================================

======================================================================
Pondérations : titre=1.0, description=1.0
======================================================================
Weighted F1 (validation): 0.7723

======================================================================
Pondérations : titre=2.0, description=1.0
======================================================================
Weighted F1 (validation): 0.8067

======================================================================
Pondérations : titre=3.0, description=1.0
======================================================================
Weighted F1 (validation): 0.8159

======================================================================
Pondérations : titre=1.0, description=2.0
======================================================================
Weighted F1 (validation): 0.7808

======================================================================
MEILLEURE CONFIGURATION TROUVÉE
======================================================================
Pondérations : titre=3.0, desc=1.0
F1 Score     : 0.8159
======================================================================

======================================================================
COMPARAISON FINALE
======================================================================
Stratégie FUSIONNÉE  : F1 = 0.7526
Stratégie SÉPARÉE    : F1 = 0.8159
Différence (séparée - fusionnée) : +0.0633
======================================================================

MEILLEUR MODÈLE : Stratégie SÉPARÉE (titre=3.0, desc=1.0)
F1 Score        : 0.8159

Sauvegarde du modèle dans models/text_logreg_best.joblib...
  ✓ Modèle sauvegardé (125.45 MB)

======================================================================
✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !
======================================================================

Meilleur modèle : Stratégie SÉPARÉE
F1 Score        : 0.8159
Modèle sauvegardé : models/text_logreg_best.joblib
```

### Utilisation du modèle sauvegardé

```python
import joblib
import pandas as pd

# Charger le modèle
model_data = joblib.load('models/text_logreg_best.joblib')

# Accéder aux composants
pipeline = model_data['pipeline']
strategy = model_data['strategy']
f1_score = model_data['f1_score']

print(f"Stratégie: {strategy}")
print(f"F1 Score: {f1_score:.4f}")

# Faire des prédictions
X_new = pd.read_csv('data/new_data.csv')
predictions = pipeline.predict(X_new)
```

### Performances attendues

Basé sur les expérimentations du notebook `02_xiaosong_text_preprocessing_v2.ipynb` :

| Stratégie | Configuration | F1 Score attendu |
|-----------|---------------|------------------|
| Fusionnée | text_all | ~0.7526 |
| Séparée | titre=1.0, desc=1.0 | ~0.7723 |
| Séparée | titre=2.0, desc=1.0 | ~0.8067 |
| Séparée | titre=3.0, desc=1.0 | **~0.8159** ⭐ |
| Séparée | titre=1.0, desc=2.0 | ~0.7808 |

### Structure du modèle sauvegardé

Le fichier `.joblib` contient un dictionnaire avec :
- `pipeline` : Le pipeline scikit-learn complet (préprocesseur + modèle)
- `strategy` : Nom de la stratégie ("FUSIONNÉE" ou "SÉPARÉE")
- `f1_score` : Score F1 sur le jeu de validation
- `results` : Dictionnaire complet des résultats d'évaluation

### Workflow complet

```bash
# 1. Créer le dataset
python scripts/make_text_dataset.py -v

# 2. Entraîner les modèles
python scripts/train_text_baseline.py -v

# 3. Utiliser le modèle dans un notebook
# Voir l'exemple ci-dessus
```

### Dépendances

Le script nécessite :
- Python 3.7+
- pandas
- scikit-learn
- joblib
- Le module `rakuten_text` (doit être dans `src/`)

### Notes

- Le script teste automatiquement 4 configurations de pondération pour la stratégie séparée
- Les paramètres sont identiques à ceux du notebook d'expérimentation
- Le temps d'exécution est d'environ 15-20 minutes pour le dataset complet
- Le modèle final occupe environ 125 MB sur disque
- La stratégie séparée avec titre=3.0 donne généralement les meilleurs résultats

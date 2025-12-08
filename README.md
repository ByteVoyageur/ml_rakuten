Projet Rakuten – Classification de Produits

Ce dépôt rassemble le travail réalisé autour de la classification de produits Rakuten, en utilisant les informations texte (désignation + description) et, dans un second temps, les images.
L’objectif est surtout pratique : comprendre le jeu de données, tester différentes idées de prétraitement et construire un pipeline clair et reproductible.

📁 Structure du projet
rakuten/
├── archive/
│   └── phase1_exploration_text/     # Code exploratoire archivé (Phase 1)
├── data/                            # Datasets Rakuten
│   ├── X_train_update.csv
│   └── Y_train_CVw08PX.csv
├── notebooks/
│   ├── 00_text_exploration.ipynb             # Exploration initiale
│   ├── 01_Text_Preprocessing_Benchmark.ipynb # Phase 1: Preprocessing
│   ├── 02_Vectorization_Strategies.ipynb     # Phase 2: Vectorization
│   ├── 03_Model_Selection.ipynb              # Phase 2: Model Selection
│   └── archive/                              # Anciens notebooks
├── src/
│   └── rakuten_text/               # Bibliothèque modulaire de ML texte
│       ├── __init__.py
│       ├── preprocessing.py        # ✅ Nettoyage de texte (Phase 1)
│       ├── benchmark.py            # ✅ Benchmark preprocessing (Phase 1)
│       ├── features.py             # ✅ Features manuelles (Phase 2)
│       ├── vectorization.py        # ✅ TF-IDF/Count + weighting (Phase 2)
│       ├── experiments.py          # ✅ Expérimentations systématiques (Phase 2)
│       ├── models.py               # ✅ Pipelines ML (Phase 2)
│       ├── categories.py           # ✅ Mapping 27 catégories (Phase 2)
│       
├── results/
│   ├── configs/                    # Configurations optimales
│   └── models/                     # Modèles entraînés
└── README.md                       # Ce fichier
```

## 🎯 Objectif

Classifier automatiquement les produits Rakuten dans **27 catégories** en utilisant :
- **Texte** : Désignation + Description des produits
- **Images** : Photos des produits (phase en cours)

## 📊 État d'Avancement

### ✅ Phase 1 : Prétraitement de Texte (TERMINÉE)

**Résultats clés :**
- **Baseline raw** : F1 = 0.7919
- **Meilleure stratégie** : `final_text_cleaner()` → **F1 = 0.8024** (+1.32%)
- **22 stratégies** de nettoyage comparées sur 84,916 échantillons

**Fonction de production :** `final_text_cleaner()` dans `src/rakuten_text/preprocessing.py`

**Notebook :** `notebooks/01_Text_Preprocessing_Benchmark.ipynb`

### ✅ Phase 2 : Vectorization & Modèles Texte (TERMINÉE)

**Résultats clés :**
- **Configuration optimale** : TF-IDF Split + features manuelles + title weighting
- **Performance** : F1 = 0.8420 (+6.33% vs baseline)
- **Hyperparamètres** : max_features=20k, ngram_range=(1,2), split_size=0.15

**Expérimentations réalisées :**
1. Count vs TF-IDF vectorization
2. Split vs Merged text strategies
3. Manual features extraction (24 features)
4. **Title weighting** (1x-3x importance)
5. Hyperparameter grid search
6. Model comparison (LogReg, SVM, XGBoost, RF)

**Modules créés :**
- `vectorization.py` : TF-IDF/Count + FeatureWeighter (title weighting)
- `features.py` : 24 features manuelles textuelles
- `experiments.py` : Framework complet d'expérimentation + tracking + reporting
- `models.py` : Pipelines ML (LogReg, SVM, XGBoost, RF)
- `categories.py` : Mapping 27 catégories + noms courts

**Fonctionnalités clés :**
- ✅ Title weighting automatique (1x-3x)
- ✅ Tracking global des scores (tous les modèles)
- ✅ Vérification d'optimalité automatique
- ✅ Génération de rapports formatés

**Notebooks :**
- `02_Vectorization_Strategies.ipynb`
- `03_Model_Selection.ipynb`

### 🔄 Phase 3 : Traitement d'Images (EN COURS)

Exploration des features visuelles et architectures CNN/Transfer Learning.

### 📋 Phase 4 : Fusion Multimodale (PLANIFIÉE)

- Ensembles multi-modaux (texte + image)
- Fine-tuning de modèles transformer
- Optimisation hyperparamètres


Les notebooks trop anciens ou expérimentaux sont déplacés dans archive/ pour garder une arborescence propre.

🎯 Objectif général

Le dataset contient environ 85k produits répartis dans 27 catégories.
La première étape a été de mettre en place un prétraitement du texte stable et facilement testable.
Les expérimentations sur les images sont en cours et seront ajoutées au fur et à mesure.

📊 Phase 1 — Prétraitement du texte (terminée)

Le travail a porté principalement sur :

la correction des problèmes d’encodage,

la gestion des balises HTML,

la normalisation Unicode,

la suppression de ponctuation bruitée,

les stopwords français et anglais.

Au total, 22 configurations de nettoyage ont été comparées.

Quelques repères :

Baseline (texte brut) : F1 = 0.7921

Meilleure stratégie : F1 = 0.8024

Le pipeline final est implémenté dans
src/rakuten_text/preprocessing.py → final_text_cleaner()

Notebook de référence :
notebooks/01_Text_Preprocessing_Benchmark.ipynb

🖼️ Phase 2 — Images (en cours)

Exploration des premières features visuelles (HOG, couleurs, downsampling)
et tests préliminaires avec quelques architectures CNN.
Rien n’est encore figé : c’est une phase de repérage.

🔧 Phase 3 — Modélisation (à venir)

Combinaison texte + image

Tests de modèles plus modernes

Ajustement des hyperparamètres

Éventuels ensembles multi-modaux

🚀 Installation
git clone <url>
cd rakuten
pip install -r requirements.txt

# Stopwords pour NLTK
python -c "import nltk; nltk.download('stopwords')"

📝 Exemple d’utilisation du nettoyage
from src.rakuten_text.preprocessing import final_text_cleaner

txt = "<p>Ordinateur portable HP 15.6 pouces - 299,99&nbsp;€</p>"
print(final_text_cleaner(txt))

⚙️ Lancer un benchmark
from src.rakuten_text.benchmark import load_dataset, run_benchmark

df = load_dataset("data")
results = run_benchmark(df)
print(results.head())

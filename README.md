# Projet Rakuten - Classification de Produits E-commerce

Système de classification automatique de produits Rakuten utilisant des techniques de Machine Learning sur texte et images.

## 📁 Structure du Projet

```
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

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner le repo
git clone <url>
cd rakuten

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données NLTK (pour les stopwords)
python -c "import nltk; nltk.download('stopwords')"
```

### Utilisation de la Bibliothèque de Texte

```python
from src.rakuten_text.preprocessing import final_text_cleaner

# Nettoyer un texte produit
text = "<p>Ordinateur <strong>portable</strong> HP 15.6 pouces - 299,99&nbsp;€</p>"
cleaned = final_text_cleaner(text)
print(cleaned)
# Output: "ordinateur portable hp 15.6 pouces 299,99 €"
```

### Exécuter le Benchmark

```python
from src.rakuten_text.benchmark import load_dataset, run_benchmark, analyze_results

# Charger les données
df = load_dataset(data_dir="data")

# Exécuter le benchmark
results_df = run_benchmark(df, verbose=True)

# Analyser les résultats
analyze_results(results_df, top_n=10)
```

**Note :** Voir le notebook `notebooks/01_Text_Preprocessing_Benchmark.ipynb` pour un exemple complet.

## 📚 Documentation

### Modules Principaux

#### `src/rakuten_text/preprocessing.py`
- `clean_text()` : Fonction modulaire avec options configurables (pour expérimentations)
- `final_text_cleaner()` : Pipeline optimisé pour production (configuration gagnante)
- `get_available_options()` : Liste toutes les options de nettoyage disponibles

#### `src/rakuten_text/benchmark.py`
- `load_dataset()` : Charge les données Rakuten
- `define_experiments()` : Définit les configurations d'expériences
- `run_benchmark()` : Exécute le benchmark complet
- `analyze_results()` : Analyse et visualise les résultats
- `save_results()` : Sauvegarde les résultats en CSV

## 🧪 Tests et Expérimentations

Pour tester différentes stratégies de prétraitement :

```python
from src.rakuten_text.preprocessing import clean_text

# Tester une configuration custom
text = "Votre texte ici"
cleaned = clean_text(
    text,
    fix_encoding=True,
    remove_html_tags=True,
    lowercase=True,
    remove_stopwords=True
)
```

## 📈 Résultats de Benchmark

| Stratégie | F1 Score | Amélioration vs Baseline |
|-----------|----------|-------------------------|
| baseline_raw | 0.7921 | - |
| traditional_cleaning | **0.8024** | **+1.32%** |
| conservative_cleaning | 0.7985 | +0.81% |
| all_encoding_fixes | 0.7931 | +0.13% |

**Détails complets :** Voir `results/benchmark_results.csv` ou le notebook de démonstration.

## 🗂️ Archives

Les fichiers exploratoires de la Phase 1 sont archivés dans `archive/phase1_exploration_text/` :
- Notebooks d'exploration
- Scripts de tests
- Anciennes versions de code

## 👥 Contributeurs

- **Xiaosong** : Développement et expérimentations

## 📝 Notes Importantes

- **Langue** : Tous les commentaires et docstrings dans `src/` sont en **français** pour faciliter la collaboration
- **Reproductibilité** : Tous les benchmarks utilisent `random_state=42` pour garantir la reproductibilité
- **Performance** : Le pipeline de production est optimisé pour le e-commerce français (mots vides FR + EN)

## 📄 Licence

Ce projet est destiné à des fins éducatives et de recherche.

---

**Dernière mise à jour** : 2025-12-08
**Version** : 2.0 (Phase 2 terminée - Text ML Pipeline complet)

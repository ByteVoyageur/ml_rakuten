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
│   ├── 01_Text_Preprocessing_Benchmark.ipynb  # Notebook de démonstration (Phase 1)
│   └── archive/                     # Anciens notebooks exploratoires
├── src/
│   └── rakuten_text/               # Bibliothèque de prétraitement de texte
│       ├── __init__.py
│       ├── preprocessing.py        # ✅ Fonctions de nettoyage (Phase 1 - STABLE)
│       ├── benchmark.py            # ✅ Outils de benchmark (Phase 1 - STABLE)
│       ├── README.md               # Documentation du module
│       └── ...
├── results/                        # Résultats des expériences
├── scripts/                        # Scripts utilitaires
├── models/                         # Modèles sauvegardés
└── README.md                       # Ce fichier
```

## 🎯 Objectif

Classifier automatiquement les produits Rakuten dans **27 catégories** en utilisant :
- **Texte** : Désignation + Description des produits
- **Images** : Photos des produits (phase en cours)

## 📊 État d'Avancement

### ✅ Phase 1 : Prétraitement de Texte (TERMINÉE)

**Résultats clés :**
- **Baseline** : F1 = 0.7921
- **Meilleure stratégie** : `optimized_traditional` → **F1 = 0.8024** (+1.32%)
- **22 stratégies** de nettoyage comparées sur 84,916 échantillons

**Pipeline gagnant :**
1. Correction d'encodage (ftfy)
2. Décodage entités HTML
3. Normalisation Unicode
4. Suppression balises HTML
5. Conversion en minuscules
6. Suppression ponctuation isolée
7. Suppression mots vides (FR + EN)

**Fonction de production :** `final_text_cleaner()` dans `src/rakuten_text/preprocessing.py`

**Notebook de référence :** `notebooks/01_Text_Preprocessing_Benchmark.ipynb`

### 🔄 Phase 2 : Traitement d'Images (EN COURS)

Exploration des features visuelles et architectures CNN/Transfer Learning.

### 📋 Phase 3 : Modélisation Avancée (PLANIFIÉE)

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

**Dernière mise à jour** : 2025-12-07
**Version** : 1.0 (Phase 1 terminée)

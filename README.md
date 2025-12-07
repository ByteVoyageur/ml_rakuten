Projet Rakuten – Classification de Produits

Ce dépôt rassemble le travail réalisé autour de la classification de produits Rakuten, en utilisant les informations texte (désignation + description) et, dans un second temps, les images.
L’objectif est surtout pratique : comprendre le jeu de données, tester différentes idées de prétraitement et construire un pipeline clair et reproductible.

📁 Structure du projet
rakuten/
├── data/                   # Fichiers d’entraînement
├── notebooks/              # Notebooks d’exploration et de tests
├── src/
│   └── rakuten_text/       # Code de nettoyage et utilitaires
├── models/                 # Modèles entraînés
├── results/                # Résultats et tableaux de benchmark
└── README.md


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


Les résultats complets sont enregistrés dans results/benchmark_results.csv.

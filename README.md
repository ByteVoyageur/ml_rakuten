Projet Rakuten – Classification de Produits

Ce dépôt rassemble le travail mené autour de la classification automatique des produits Rakuten, en utilisant les informations texte (désignation + description) et, dans un second temps, les images.
L’objectif est avant tout pratique : comprendre le jeu de données, construire des pipelines propres et comparer différentes stratégies de prétraitement et de modélisation.

Structure du projet
rakuten/
├── data/                    # Fichiers d’entraînement
├── notebooks/               # Notebooks d’exploration et de tests
├── src/
│   └── rakuten_text/        # Code de nettoyage et fonctions utilitaires
├── models/                  # Modèles entraînés
├── results/                 # Résultats des benchmarks
└── README.md


Les notebooks plus anciens ou trop expérimentaux sont déplacés dans archive/ pour ne pas polluer le dossier principal.

Objectif général

Le jeu de données contient environ 85k produits appartenant à 27 catégories.
Le premier axe de travail a consisté à nettoyer et structurer le texte pour obtenir une baseline stable.
Les expérimentations sur les images sont en cours et seront intégrées progressivement.

État d’avancement
Phase 1 — Prétraitement du texte (terminée)

L’essentiel du travail a porté sur :

la correction des problèmes d’encodage,

la gestion des balises HTML,

la normalisation Unicode,

les stopwords français/anglais,

plusieurs stratégies de nettoyage plus ou moins agressives.

Au total, 22 configurations ont été comparées de manière systématique.

Quelques repères :

Baseline (texte brut) : F1 = 0.7921

Stratégie retenue : 0.8024

Le pipeline final est disponible via
src/rakuten_text/preprocessing.py → final_text_cleaner()

Le notebook qui résume les tests :
notebooks/01_Text_Preprocessing_Benchmark.ipynb

Phase 2 — Images (en cours)

Exploration des caractéristiques visuelles, tests HOG / CNN légers, comparaison de tailles d’images, etc.
Rien de figé pour l’instant : c’est en construction.

Phase 3 — Modélisation (à venir)

Combiner texte + image, tester quelques modèles plus modernes, comparer différentes approches d’ensemble.
On avancera selon les besoins et le temps disponible.

Prise en main rapide
Installation
git clone <url>
cd rakuten
pip install -r requirements.txt

# Stopwords pour NLTK
python -c "import nltk; nltk.download('stopwords')"

Nettoyer un texte
from src.rakuten_text.preprocessing import final_text_cleaner

txt = "<p>Ordinateur portable HP 15.6 pouces - 299,99&nbsp;€</p>"
print(final_text_cleaner(txt))

Lancer un benchmark complet
from src.rakuten_text.benchmark import load_dataset, run_benchmark

df = load_dataset("data")
results = run_benchmark(df)
print(results.head())

Le code principal
preprocessing.py

Contient :

clean_text() : fonction flexible, utile pour tester de nouvelles options,

final_text_cleaner() : la version “production”, issue des comparaisons,

quelques helpers pour l'encodage, le HTML, la ponctuation, etc.

benchmark.py

Permet :

de charger le dataset dans un format propre,

de définir les expériences,

d’exécuter toutes les variantes,

d’enregistrer les résultats.

Quelques résultats (texte)
Stratégie	F1
Texte brut	0.7921
Nettoyage “traditionnel”	0.8024
Nettoyage conservateur	0.7985

Les détails complets sont stockés dans results/benchmark_results.csv.

Remarques

Tous les commentaires dans src/ sont en français pour garder une cohérence.

Les expériences utilisent un random_state=42 pour limiter les surprises.

Le but n’est pas de battre un leaderboard, mais de construire un pipeline clair, reproductible et adaptable.

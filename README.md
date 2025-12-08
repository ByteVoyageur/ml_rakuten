# Projet Rakuten – Classification Multimodale de Produits (WIP)

Ce dépôt contient le code pour la classification automatique de produits Rakuten dans 27 catégories. Le projet vise à exploiter à la fois les **informations textuelles** (désignation et description) et les **images** des produits.

⚠️ **État du projet** : *En cours de développement*.
Actuellement, nous nous concentrons sur l'optimisation de la partie **Text Mining** (Nettoyage, Vectorisation, Modélisation). L'intégration des images (Computer Vision) et la fusion multimodale interviendront dans une seconde phase.

## 🛠 Installation et Environnement (Docker)

Le projet est entièrement conteneurisé pour garantir la reproductibilité, notamment pour la gestion des dépendances GPU (CUDA 12.1).

### Prérequis
* Docker & Docker Compose
* Drivers NVIDIA et NVIDIA Container Toolkit

### Démarrage rapide
L'environnement utilise une image Python 3.11 personnalisée avec PyTorch, JupyterLab et les outils de Data Science.


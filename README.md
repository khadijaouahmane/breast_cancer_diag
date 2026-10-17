# 🏥 Breast Cancer Diagnostic AI

> **Système d'aide au diagnostic du cancer du sein basé sur le Machine Learning.**
> Prédiction de la malignité d'une tumeur à partir de mesures cellulaires.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--ready-success.svg)

---

## 📋 Table des matières

- [Contexte médical](#-contexte-médical)
- [Objectifs du projet](#-objectifs-du-projet)
- [Dataset](#-dataset)
- [Architecture du projet](#-architecture-du-projet)
- [Résultats du modèle](#-résultats-du-modèle)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Technologies utilisées](#-technologies-utilisées)
- [Améliorations futures](#-améliorations-futures)
- [Avertissement médical](#️-avertissement-médical)
- [Auteur](#-auteur)

---

## 🩺 Contexte médical

Le **cancer du sein** est le cancer le plus fréquent chez la femme dans le monde. Selon l'OMS, il représente **2,3 millions de nouveaux cas par an**. 

Le diagnostic précoce est **crucial** : détecté à un stade précoce, le taux de survie à 5 ans dépasse **99%**, contre **30%** à un stade avancé.

Ce projet propose un **outil d'aide à la décision** permettant d'analyser automatiquement les caractéristiques d'une tumeur (mesures cellulaires issues d'une biopsie) et d'estimer la probabilité qu'elle soit maligne.

---

## 🎯 Objectifs du projet

- ✅ Construire un pipeline de Machine Learning **complet et reproductible**
- ✅ Appliquer les **bonnes pratiques** de l'architecture logicielle (séparation des responsabilités)
- ✅ Fournir une **interface clinique** simple d'utilisation
- ✅ Démontrer l'usage du **polymorphisme** (interchangeabilité des modèles)

---

## 📊 Dataset

Le projet utilise le célèbre dataset **Breast Cancer Wisconsin (Diagnostic)**, disponible publiquement dans `scikit-learn` et issu du **UCI Machine Learning Repository**.

| Caractéristique | Valeur |
|-----------------|--------|
| **Patients** | 569 |
| **Features** | 30 caractéristiques cellulaires |
| **Classes** | Malin (212) / Bénin (357) |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |

### Features utilisées (5 principales)

Pour garder le modèle interprétable, nous utilisons 5 features clés :
- `mean radius` — Rayon moyen de la tumeur
- `mean texture` — Texture moyenne
- `mean perimeter` — Périmètre moyen
- `mean area` — Surface moyenne
- `mean smoothness` — Régularité moyenne des contours

---

## 🏗️ Architecture du projet

Le projet suit une architecture **modulaire** inspirée des principes de **Clean Architecture** :

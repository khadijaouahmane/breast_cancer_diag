# 🏥 Breast Cancer Diagnostic AI

Système d'aide au diagnostic du cancer du sein basé sur le Machine Learning. Ce projet prédit si une tumeur est **maligne** (cancéreuse) ou **bénigne** à partir de mesures cellulaires.

## 🎯 Objectif

Fournir un outil d'aide à la décision médicale qui analyse les caractéristiques d'une tumeur (rayon, texture, périmètre, etc.) et estime la probabilité de malignité.

## 📊 Dataset

- **Source** : Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
- **Patients** : 569
- **Features** : 30 caractéristiques cellulaires
- **Classes** : Malin (212) / Bénin (357)

## 🏗️ Architecture
breast_cancer_diag/
├── data/ # Datasets (générés automatiquement)
├── core/ # Cœur métier (modèle IA + dataset)
│ ├── dataset.py
│ └── model.py
├── pipeline/ # Entraînement du modèle
│ └── trainer.py
├── utils/ # Prétraitement des données
│ └── preprocessing.py
└── app/ # Interface clinique
└── interface_clinique.py

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/breast_cancer_diag.git
cd breast_cancer_diag

# Installer les dépendances
pip install -r requirements.txt
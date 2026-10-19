# pipeline/evaluator.py
import pandas as pd
from utils.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    compute_classification_report,
)
from core.model import FEATURES


class Evaluator:
    def __init__(self, model):
        self.model = model

    def _predict_all(self, df):
        """Prédit sur toutes les lignes du dataset."""
        y_pred = []
        for _, row in df.iterrows():
            features = [row[f] for f in FEATURES]
            proba = self.model.predict(features)
            y_pred.append(1 if proba >= 0.5 else 0)
        return y_pred

    def evaluate(self, data_path):
        print("\n--- 📊 Évaluation du modèle ---")
        df = pd.read_csv(data_path)

        y_true = df['malignant'].tolist()
        y_pred = self._predict_all(df)

        # Métriques
        metrics = compute_all_metrics(y_true, y_pred)
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-score  : {metrics['f1']:.4f}")

        # Matrice de confusion
        print("\nMatrice de confusion :")
        print(compute_confusion_matrix(y_true, y_pred))

        # Rapport complet
        print("\nRapport de classification :")
        print(compute_classification_report(y_true, y_pred))

        return metrics
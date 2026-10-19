# pipeline/predictor.py
from core.model import FEATURES


class Predictor:
    def __init__(self, model, threshold: float = 0.5):
        """
        Args:
            model: modèle entraîné (doit avoir une méthode predict)
            threshold: seuil de décision (par défaut 0.5)
        """
        self.model = model
        self.threshold = threshold
        self.features = FEATURES

    def predict_proba(self, patient_data: dict) -> float:
        """Retourne la probabilité brute de malignité."""
        features = [patient_data[f] for f in self.features]
        return self.model.predict(features)

    def diagnose(self, patient_data: dict) -> str:
        """Retourne le diagnostic selon le seuil."""
        proba = self.predict_proba(patient_data)
        label = "⚠️ MALIN (Suspect de cancer)" if proba >= self.threshold \
                else "✅ BÉNIN (Pas de cancer)"
        return f"{label} — probabilité : {proba:.2%}"
# core/model.py
from sklearn.linear_model import LogisticRegression

# La liste des features est maintenant directement ici
FEATURES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
]

class CancerModel:
    def __init__(self):
        # Plus de Pipeline, plus de Scaler. Juste le modèle.
        self.model = LogisticRegression(max_iter=1000)
        self.features = FEATURES

    def fit(self, X, y):
        """Entraîne le modèle sur les données brutes."""
        self.model.fit(X, y)
        print(f"✅ Modèle entraîné sur {len(X)} patients.")

    def predict(self, features):
        """Reçoit une liste de features brutes et retourne la probabilité."""
        proba = self.model.predict_proba([features])[0][1]
        return proba
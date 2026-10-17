from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import FEATURES

class CancerModel:
    def __init__(self):
        # Pipeline : Standardisation + Régression Logistique
        # Très important en médical : normaliser les données !
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        self.features = FEATURES

    def fit(self, X, y):
        """Entraîne le modèle."""
        self.pipeline.fit(X, y)
        print(f"✅ Modèle entraîné sur {len(X)} patients.")
        acc = self.pipeline.score(X, y)
        print(f"   Précision (training) : {acc:.2%}")

    def predict(self, features):
        """Retourne la probabilité de malignité (0 à 1)."""
        proba = self.pipeline.predict_proba([features])[0][1]
        return proba
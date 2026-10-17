# pipeline/trainer.py
import pandas as pd
from core.model import CancerModel, FEATURES
from core.dataset import generate_dataset

class Trainer:
    def __init__(self):
        self.model = CancerModel()

    def train(self, data_path):
        print("--- 🏥 Début de l'entraînement (Cancer du Sein) ---")
        
        generate_dataset(data_path)
        df = pd.read_csv(data_path)
        
        # On passe les données brutes (pandas gère les colonnes)
        X = df[FEATURES]
        y = df['malignant']
        
        self.model.fit(X, y)
        print("--- ✅ Fin de l'entraînement ---")
        return self.model
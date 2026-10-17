import pandas as pd
from core.model import CancerModel
from core.dataset import generate_dataset
from utils.preprocessing import FEATURES

class Trainer:
    def __init__(self):
        self.model = CancerModel()

    def train(self, data_path):
        print("--- 🏥 Début de l'entraînement (Cancer du Sein) ---")
        
        
        
        # 2. Charger les données
        df = pd.read_csv(data_path)
        
        # 3. Séparer X et y
        X = df[FEATURES]
        y = df['malignant']
        
        # 4. Entraîner
        self.model.fit(X, y)
        
        print("--- ✅ Fin de l'entraînement ---")
        return self.model
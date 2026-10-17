# app/main.py
import os
import pandas as pd

from core.dataset import generate_dataset
from core.model import CancerModel, FEATURES  # FEATURES importé depuis le modèle
from pipeline.trainer import Trainer
from app.interface_clinique import ClinicalPredictor


class Main:
    DATASET_PATH = os.path.join("data", "breast_cancer.csv")

    def __init__(self):
        self.trained_model = None
        self.predictor = None

    def step_1_load_dataset(self):
        print("\n--- ÉTAPE 1 : Chargement du dataset ---")
        if not os.path.exists(self.DATASET_PATH):
            generate_dataset(self.DATASET_PATH)
        df = pd.read_csv(self.DATASET_PATH)
        print(f"✅ Dataset chargé : {len(df)} patients.")
        return df

    def step_2_train_model(self):
        print("\n--- ÉTAPE 2 : Entraînement ---")
        trainer = Trainer()
        self.trained_model = trainer.train(self.DATASET_PATH)
        return self.trained_model

    def step_3_init_predictor(self):
        print("\n--- ÉTAPE 3 : Initialisation du Prédicteur ---")
        self.predictor = ClinicalPredictor(self.trained_model)

    def step_4_test_predictions(self, df, indices=[0, 50, 100]):
        print("\n--- ÉTAPE 4 : Prédictions ---")
        for i in indices:
            patient_row = df.iloc[i]
            patient_data = {f: patient_row[f] for f in FEATURES}
            real_diag = "MALIN" if patient_row['malignant'] == 1 else "BÉNIN"
            
            print(f"\nPatient #{i} (Réel : {real_diag})")
            print(f"-> Prédiction : {self.predictor.diagnose(patient_data)}")

    def run(self):
        df = self.step_1_load_dataset()
        self.step_2_train_model()
        self.step_3_init_predictor()
        self.step_4_test_predictions(df)
        print("\n🎉 Processus terminé !")

if __name__ == "__main__":
    Main().run()
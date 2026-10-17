from utils.preprocessing import prepare_features, FEATURES
import pandas as pd

class ClinicalPredictor:
    def __init__(self, trained_model):
        self.model = trained_model

    def diagnose(self, patient_data):
        features = prepare_features(patient_data)
        proba = self.model.predict(features)
        print(f"Probabilité de malignité : {proba:.2%}")
        
        if proba >= 0.5:
            return "⚠️ MALIN (Suspect de cancer)"
        else:
            return "✅ BÉNIN (Pas de cancer)"

if __name__ == "__main__":
    from pipeline.trainer import Trainer

    print(">>> ÉTAPE 1 : Entraînement")
    trainer = Trainer()
    model = trainer.train("data/breast_cancer.csv")

    print("\n>>> ÉTAPE 2 : Prédictions sur patients réels")
    
    # On prend 3 vrais patients du dataset pour tester
    df = pd.read_csv("data/breast_cancer.csv")
    
    for i in [0, 50, 100]:  # On teste 3 patients différents
        patient = df.iloc[i]
        patient_data = {f: patient[f] for f in FEATURES}
        vrai_diag = "MALIN" if patient['malignant'] == 1 else "BÉNIN"
        
        print(f"\n📋 Patient #{i} (Diagnostic réel : {vrai_diag})")
        prediction = ClinicalPredictor(model).diagnose(patient_data)
        print(f"   -> Prédiction IA : {prediction}")
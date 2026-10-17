# app/interface_clinique.py

class ClinicalPredictor:
    def __init__(self, trained_model):
        self.model = trained_model
        # On récupère la liste des features depuis le modèle
        self.features = trained_model.features

    def diagnose(self, patient_data):
        """
        patient_data est un dictionnaire : {'mean radius': 14.2, ...}
        """
        # On transforme le dict en liste dans le bon ordre
        features_list = [patient_data[f] for f in self.features]
        
        # Prédiction directe sur les données brutes
        proba = self.model.predict(features_list)
        
        print(f"Probabilité de malignité : {proba:.2%}")
        
        if proba >= 0.5:
            return "⚠️ MALIN (Suspect de cancer)"
        else:
            return "✅ BÉNIN (Pas de cancer)"
# app/main.py
"""
Point d'entrée principal du système de diagnostic du cancer du sein.

Cette classe orchestre le processus complet :
    1. Chargement / génération du dataset (core.dataset)
    2. Entraînement du modèle (pipeline.trainer)
    3. Évaluation du modèle (pipeline.evaluator)
    4. Prédiction clinique (app.interface_clinique)

Chaque composant respecte son rôle : le Main ne fait QUE les appeler.
"""

import os
import pandas as pd

from core.dataset import generate_dataset
from core.model import CancerModel
from pipeline.trainer import Trainer
from utils.preprocessing import FEATURES, prepare_features
from app.interface_clinique import ClinicalPredictor


class Main:
    """Orchestrateur principal de l'application de diagnostic."""

    DATASET_PATH = os.path.join("data", "breast_cancer.csv")

    def __init__(self):
        self.trained_model = None
        self.predictor = None

    # ------------------------------------------------------------------ #
    # ÉTAPE 1 : Chargement du dataset                                     #
    # ------------------------------------------------------------------ #
    def step_1_load_dataset(self) -> pd.DataFrame:
        """Délègue la génération/chargement du dataset à core.dataset."""
        print("\n" + "=" * 60)
        print("ÉTAPE 1 : Chargement du dataset")
        print("=" * 60)

        if not os.path.exists(self.DATASET_PATH):
            generate_dataset(self.DATASET_PATH)

        df = pd.read_csv(self.DATASET_PATH)
        print(f"✅ Dataset chargé : {len(df)} patients, {len(FEATURES)} features utilisées.")
        return df

    # ------------------------------------------------------------------ #
    # ÉTAPE 2 : Entraînement du modèle                                    #
    # ------------------------------------------------------------------ #
    def step_2_train_model(self) -> CancerModel:
        """Délègue l'entraînement au pipeline.Trainer."""
        print("\n" + "=" * 60)
        print("ÉTAPE 2 : Entraînement du modèle")
        print("=" * 60)

        trainer = Trainer()
        self.trained_model = trainer.train(self.DATASET_PATH)
        print("✅ Modèle entraîné et prêt pour les prédictions.")
        return self.trained_model

    # ------------------------------------------------------------------ #
    # ÉTAPE 3 : Initialisation du prédicteur clinique                     #
    # ------------------------------------------------------------------ #
    def step_3_init_predictor(self):
        """Injecte le modèle entraîné dans le ClinicalPredictor."""
        print("\n" + "=" * 60)
        print("ÉTAPE 3 : Initialisation du ClinicalPredictor")
        print("=" * 60)

        if self.trained_model is None:
            raise RuntimeError("Le modèle doit être entraîné avant de créer le prédicteur.")

        self.predictor = ClinicalPredictor(self.trained_model)
        print("✅ ClinicalPredictor prêt (injection de dépendance réussie).")

    # ------------------------------------------------------------------ #
    # ÉTAPE 4 : Tests de prédiction                                       #
    # ------------------------------------------------------------------ #
    def step_4_test_predictions(self, df: pd.DataFrame, indices: list = None):
        """Teste le prédicteur sur des patients réels du dataset."""
        print("\n" + "=" * 60)
        print("ÉTAPE 4 : Prédictions sur patients réels")
        print("=" * 60)

        if self.predictor is None:
            raise RuntimeError("Le prédicteur doit être initialisé avant les tests.")

        if indices is None:
            indices = [0, 50, 100, 200, 300]

        correct = 0
        for i in indices:
            if i >= len(df):
                continue

            patient_row = df.iloc[i]
            patient_data = {f: patient_row[f] for f in FEATURES}
            real_diag = "MALIN" if patient_row["malignant"] == 1 else "BÉNIN"

            print(f"\n📋 Patient #{i} — Diagnostic réel : {real_diag}")
            prediction = self.predictor.diagnose(patient_data)

            predicted_malignant = "MALIN" in prediction
            if (predicted_malignant and real_diag == "MALIN") or \
               (not predicted_malignant and real_diag == "BÉNIN"):
                correct += 1
                print(f"   ✅ Cohérent avec le diagnostic réel")
            else:
                print(f"   ❌ Écart avec le diagnostic réel")

        print(f"\n📊 Taux de cohérence sur cet échantillon : {correct}/{len(indices)} "
              f"({correct / len(indices) * 100:.1f}%)")

    # ------------------------------------------------------------------ #
    # ÉTAPE 5 : Test du polymorphisme (démonstration POO)                 #
    # ------------------------------------------------------------------ #
    def step_5_demo_polymorphism(self):
        """
        Démontre le concept de polymorphisme (Duck Typing) :
        ClinicalPredictor fonctionne avec N'IMPORTE QUEL modèle
        possédant une méthode predict().
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 5 : Démonstration du polymorphisme")
        print("=" * 60)

        class FakeModel:
            """Faux modèle qui prédit toujours 'malin'."""
            def predict(self, features):
                return 0.99

        fake_predictor = ClinicalPredictor(FakeModel())
        patient = {f: 10.0 for f in FEATURES}  # Patient fictif
        result = fake_predictor.diagnose(patient)
        print(f"✅ Le ClinicalPredictor a fonctionné avec un FakeModel : {result}")
        print("   → Preuve du polymorphisme : aucun changement de code requis.")

    # ------------------------------------------------------------------ #
    # ORCHESTRATION COMPLÈTE                                              #
    # ------------------------------------------------------------------ #
    def run(self):
        """Exécute le processus complet de bout en bout."""
        print("\n" + "🏥" * 20)
        print("   SYSTÈME DE DIAGNOSTIC DU CANCER DU SEIN")
        print("🏥" * 20)

        df = self.step_1_load_dataset()
        self.step_2_train_model()
        self.step_3_init_predictor()
        self.step_4_test_predictions(df)
        self.step_5_demo_polymorphism()

        print("\n" + "=" * 60)
        print("🎉 PROCESSUS COMPLET TERMINÉ AVEC SUCCÈS")
        print("=" * 60)


if __name__ == "__main__":
    Main().run()
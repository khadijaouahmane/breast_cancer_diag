# main.py  (à la racine du projet)
import os
import pandas as pd

from core.dataset import generate_dataset
from core.model import FEATURES
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import Predictor


# ✅ Seuil de décision global ≥ 0.5
PREDICTION_THRESHOLD = 0.5

DATASET_PATH = os.path.join("data", "breast_cancer.csv")


def main():
    print("=" * 60)
    print("   🏥 SYSTÈME DE DIAGNOSTIC DU CANCER DU SEIN")
    print("=" * 60)

    # ---------- ÉTAPE 1 : Dataset ----------
    print("\n--- ÉTAPE 1 : Chargement du dataset ---")
    if not os.path.exists(DATASET_PATH):
        generate_dataset(DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ {len(df)} patients chargés.")

    # ---------- ÉTAPE 2 : Entraînement ----------
    print("\n--- ÉTAPE 2 : Entraînement (Trainer) ---")
    trainer = Trainer()
    model = trainer.train(DATASET_PATH)

    # ---------- ÉTAPE 3 : Évaluation ----------
    print("\n--- ÉTAPE 3 : Évaluation (Evaluator) ---")
    evaluator = Evaluator(model)
    metrics = evaluator.evaluate(DATASET_PATH)

    # ✅ Contrôle du seuil de performance (≥ 0.5)
    if metrics['accuracy'] < PREDICTION_THRESHOLD:
        print(f"\n⚠️  Accuracy ({metrics['accuracy']:.2f}) < seuil {PREDICTION_THRESHOLD}")
    else:
        print(f"\n✅ Accuracy ({metrics['accuracy']:.2f}) ≥ seuil {PREDICTION_THRESHOLD}")

    # ---------- ÉTAPE 4 : Prédictions individuelles ----------
    print("\n--- ÉTAPE 4 : Prédictions (Predictor) ---")
    predictor = Predictor(model, threshold=PREDICTION_THRESHOLD)

    for i in [0, 50, 100, 200]:
        patient_row = df.iloc[i]
        patient_data = {f: patient_row[f] for f in FEATURES}
        real_diag = "MALIN" if patient_row['malignant'] == 1 else "BÉNIN"

        print(f"\n📋 Patient #{i} — Diagnostic réel : {real_diag}")
        print(f"   → {predictor.diagnose(patient_data)}")

    print("\n" + "=" * 60)
    print("🎉 PROCESSUS COMPLET TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()
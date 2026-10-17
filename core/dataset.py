import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def generate_dataset(filepath):
    """
    Charge le vrai dataset Breast Cancer Wisconsin (569 patients, 30 features)
    et le sauvegarde en CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Chargement du dataset réel depuis scikit-learn
    data = load_breast_cancer()
    
    # Création du DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Ajout de la colonne cible (0 = maligne, 1 = bénigne dans sklearn)
    # On inverse pour avoir : 1 = maligne (cancéreuse), 0 = bénigne
    df['malignant'] = 1 - data.target  
    
    df.to_csv(filepath, index=False)
    
    print(f"✅ Dataset Breast Cancer généré : {filepath}")
    print(f"   - {len(df)} patients")
    print(f"   - {len(data.feature_names)} features")
    print(f"   - Malins : {(df['malignant'] == 1).sum()}")
    print(f"   - Bénins : {(df['malignant'] == 0).sum()}")
    
    return df
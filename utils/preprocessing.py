from sklearn.preprocessing import StandardScaler

# On utilise 5 features principales pour garder le modèle simple
FEATURES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

def prepare_features(patient_data):
    """Transforme un dict patient en liste de features."""
    return [patient_data[f] for f in FEATURES]
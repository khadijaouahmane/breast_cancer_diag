from sklearn.linear_model import LogisticRegression

FEATURES = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
]

class CancerModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.features = FEATURES

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict_proba([features])[0][1]
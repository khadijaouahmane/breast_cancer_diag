# utils/metrics.py
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def compute_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Bénin', 'Malin'])

def compute_all_metrics(y_true, y_pred):
    """Retourne toutes les métriques dans un dictionnaire."""
    return {
        'accuracy':  compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall':    compute_recall(y_true, y_pred),
        'f1':        compute_f1(y_true, y_pred),
    }
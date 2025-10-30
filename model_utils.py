# model_utils.py
import json
import os
import joblib
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, average_precision_score, confusion_matrix
)
from typing import Dict


def evaluate_classifier(model, X, y) -> Dict:
    """
    Compute key metrics for binary classification.
    Returns dict with roc_auc, pr_auc (avg precision), f1, accuracy, precision, recall, confusion_matrix.
    Expects model.predict_proba to be available; falls back to decision_function or predict.
    """
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        try:
            proba = model.decision_function(X)
        except Exception:
            proba = None

    y_pred = model.predict(X)
    res = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }
    if proba is not None:
        res["roc_auc"] = float(roc_auc_score(y, proba))
        res["pr_auc"] = float(average_precision_score(y, proba))
    else:
        res["roc_auc"] = None
        res["pr_auc"] = None
    return res


def save_model_artifact(model_obj, feature_names, out_dir="models", name_prefix="best_model"):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name_prefix}.joblib")
    meta_path = os.path.join(out_dir, f"{name_prefix}_meta.json")
    joblib.dump({
        "model": model_obj,
        "feature_names": feature_names
    }, model_path)
    # Write minimal metadata
    with open(meta_path, "w") as fh:
        json.dump({"model_path": model_path, "feature_names": feature_names}, fh, indent=2)
    return model_path, meta_path

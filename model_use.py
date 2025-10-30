# xgboost_model.py
"""
Generic model loader / inference wrapper.

Behavior:
- If `path` is provided to the constructor, it will load that artifact.
- If not provided, it will try, in order:
    1. "models/best_model.joblib"
    2. Any file matching "models/best_*.joblib" (chooses the most recently modified)
    3. Any joblib file under models/ (fallback)
- The saved artifact format expected (trainer.save_model_artifact) is:
    joblib.dump({"model": <model_obj>, "feature_names": <list_of_features>}, path)
  but the loader also supports older dumps that were just the model object.

Provides:
- predict(X) : array-like predictions
- predict_proba(X) : array-like probabilities if supported (otherwise raises)
- feature_names : list of feature names (if saved in artifact)
- save(path) : save current wrapped model to path (standardized "best_model.joblib" recommended)
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Any


class ModelLoader:
    def __init__(self, path: Optional[str] = None, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.feature_names = None
        self.artifact_path = None

        if path:
            self.load(path)
        else:
            found = self._find_best_model_artifact()
            if found:
                self.load(found)
            else:
                raise FileNotFoundError(f"No model artifact found in '{self.models_dir}'. Provide path to model.")

    def _find_best_model_artifact(self) -> Optional[str]:
        # 1) Exact canonical filename
        can_path = os.path.join(self.models_dir, "best_model.joblib")
        if os.path.exists(can_path):
            return can_path

        # 2) Look for trainer-style "best_<modelname>.joblib"
        pattern = os.path.join(self.models_dir, "best_*.joblib")
        files = glob.glob(pattern)
        if files:
            # pick most recently modified
            files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
            return files_sorted[0]

        # 3) fallback: pick any .joblib in models dir (most recent)
        any_pattern = os.path.join(self.models_dir, "*.joblib")
        any_files = glob.glob(any_pattern)
        if any_files:
            any_sorted = sorted(any_files, key=os.path.getmtime, reverse=True)
            return any_sorted[0]

        return None

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = joblib.load(path)
        # Support two formats:
        # 1) dict with keys "model" and "feature_names"
        # 2) raw estimator object
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.feature_names = obj.get("feature_names")
        else:
            self.model = obj
            self.feature_names = None
        self.artifact_path = path

    def save(self, path: str = None):
        if self.model is None:
            raise RuntimeError("No model loaded to save.")
        out_path = path or os.path.join(self.models_dir, "best_model.joblib")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, out_path)
        self.artifact_path = out_path
        return out_path

    def predict(self, X: Any):
        if self.model is None:
            raise RuntimeError("No model loaded.")
        # Accept DataFrame or numpy array; if feature_names are known, prefer DataFrame ordering
        if isinstance(X, pd.DataFrame) and self.feature_names:
            # reorder if necessary, error early if missing columns
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Input is missing features: {missing}")
            X_in = X[self.feature_names]
            return self.model.predict(X_in)
        else:
            # assume numpy-like
            arr = np.asarray(X)
            return self.model.predict(arr)

    def predict_proba(self, X: Any):
        if self.model is None:
            raise RuntimeError("No model loaded.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model does not support predict_proba.")
        if isinstance(X, pd.DataFrame) and self.feature_names:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Input is missing features: {missing}")
            X_in = X[self.feature_names]
            return self.model.predict_proba(X_in)
        else:
            arr = np.asarray(X)
            return self.model.predict_proba(arr)

    def get_feature_names(self):
        return self.feature_names

    def get_artifact_path(self):
        return self.artifact_path


# Example usage:
# loader = ModelLoader()                  # auto-find model in models/
# preds = loader.predict(X_df)            # X_df: pd.DataFrame with same feature names
# probs = loader.predict_proba(X_df)      # if supported
# loader.save()                           # will write models/best_model.joblib

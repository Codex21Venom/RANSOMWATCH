# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_and_preprocess_data(
    benign_path: str = "C:/Users/AgentxVenom/Documents/Soham Goswami/New_Ransomwatch/data/benign.csv",
    ransom_path: str = "C:/Users/AgentxVenom/Documents/Soham Goswami/New_Ransomwatch/data/ransom.csv",
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    time_column: Optional[str] = "Timestamp",
    use_time_split: bool = False,
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline, List[str]]:
    """
    Load and preprocess ransomware/benign datasets. Returns:
      X_train_df, X_test_df, y_train, y_test, full_pipeline, feature_names

    Notes:
      - Numeric features are imputed (median) + scaled (StandardScaler).
      - If use_time_split is True and time_column present, performs chronological split.
      - If group_column is provided and present, performs group-aware split (useful for host-based leakage avoidance).
    """
    try:
        benign = pd.read_csv(benign_path)
        ransom = pd.read_csv(ransom_path)

        benign = benign.copy()
        ransom = ransom.copy()
        benign['Label'] = 0
        ransom['Label'] = 1

        df = pd.concat([benign, ransom], ignore_index=True)

        if drop_cols is None:
            drop_cols = ['Flow ID', 'Source IP', 'Destination IP']
            # keep Timestamp for optional splitting/feature extraction

        cols_to_drop = [c for c in drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Cast timestamp if present
        if time_column and time_column in df.columns:
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except Exception:
                logging.warning("Timestamp column exists but could not be parsed; leaving as-is.")

        # Replace infs and keep index
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Separate target
        y = df['Label'].astype(int).copy()
        X = df.drop(columns=['Label']).copy()

        # Identify numeric & categorical
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        logging.info(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")
        logging.info(f"Total samples: {len(X)}")

        # Pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # If you have categorical columns and want to include them, extend pipeline here (OneHotEncoder etc.)
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_pipeline, numeric_cols),
            # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ], remainder='drop')

        full_pipeline = Pipeline([('preprocessor', preprocessor)])

        # Splitting
        if group_column and group_column in df.columns:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=df[group_column]))
            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)
        elif use_time_split and (time_column and time_column in df.columns):
            sorted_df = pd.concat([X, y, df[[time_column]]], axis=1).sort_values(time_column)
            cutoff = int((1 - test_size) * len(sorted_df))
            train_df = sorted_df.iloc[:cutoff]
            test_df = sorted_df.iloc[cutoff:]
            X_train_raw = train_df.drop(columns=[time_column])
            X_test_raw = test_df.drop(columns=[time_column])
            y_train = train_df['Label'].reset_index(drop=True)
            y_test = test_df['Label'].reset_index(drop=True)
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

        logging.info(f"X_train shape: {X_train_raw.shape}, X_test shape: {X_test_raw.shape}")

        # Fit pipeline on train and transform both
        full_pipeline.fit(X_train_raw)
        X_train_arr = full_pipeline.transform(X_train_raw)
        X_test_arr = full_pipeline.transform(X_test_raw)

        # Feature names (currently numeric only)
        feature_names = numeric_cols  # update if you add categorical encoders

        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index).reset_index(drop=True)
        X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index).reset_index(drop=True)

        logging.info("Preprocessing complete.")
        logging.info(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
        logging.info(f"y_test distribution:\n{y_test.value_counts(normalize=True)}")

        return X_train_df, X_test_df, y_train, y_test, full_pipeline, feature_names

    except Exception as e:
        raise RuntimeError(f"Data loading/preprocessing failed: {e}")

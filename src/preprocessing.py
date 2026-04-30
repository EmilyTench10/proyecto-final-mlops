"""
preprocessing.py
----------------
Construye el pipeline de preprocesamiento (escalado + one-hot)
y separa el dataset en train/test.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que aplica:
      - StandardScaler a las features numéricas.
      - OneHotEncoder a las features categóricas.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """
    Separa el DataFrame en X_train, X_test, y_train, y_test, manteniendo
    la proporción de clases del target (estratificación).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"--- Split: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test

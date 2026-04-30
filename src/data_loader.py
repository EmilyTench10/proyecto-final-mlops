"""
data_loader.py
--------------
Funciones de carga y limpieza del dataset Heart Disease.
El CSV está versionado dentro del repositorio en data/heart.csv.
"""

import os

import pandas as pd


def load_dataset(local_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde el CSV local.

    Parameters
    ----------
    local_path : str
        Ruta relativa o absoluta al archivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos crudos.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"No se encontró el dataset en '{local_path}'. "
            "Verifica que data/heart.csv esté en el repositorio."
        )

    print(f"--- Cargando dataset desde: {local_path}")
    df = pd.read_csv(local_path)
    print(f"--- Shape del dataset: {df.shape}")
    return df


def basic_clean(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Limpieza básica del dataset:
      1. Elimina filas duplicadas.
      2. Elimina filas con valores nulos.
      3. Asegura que el target sea binario (0 o 1). En la versión UCI
         original el target tiene valores 0-4; los mapeamos a 0/1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.
    target_column : str
        Nombre de la columna objetivo.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio.
    """
    n_initial = len(df)

    # 1. Eliminar duplicados
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Eliminar filas con nulos
    df = df.dropna().reset_index(drop=True)

    # 3. Convertir target a binario (algunos mirrors tienen 0-4)
    if df[target_column].nunique() > 2:
        df[target_column] = (df[target_column] > 0).astype(int)

    n_final = len(df)
    print(f"--- Limpieza: {n_initial} → {n_final} filas "
          f"({n_initial - n_final} eliminadas)")
    return df

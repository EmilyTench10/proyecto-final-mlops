"""
tests/test_data.py
------------------
Pruebas básicas de carga y limpieza del dataset.
"""

import pandas as pd
import pytest
import yaml

from src.data_loader import basic_clean, load_dataset


@pytest.fixture(scope="module")
def config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def raw_df(config):
    return load_dataset(local_path=config["data"]["local_path"])


def test_dataset_no_vacio(raw_df):
    """El dataset debe tener al menos una fila y una columna."""
    assert raw_df.shape[0] > 0, "El dataset está vacío (sin filas)"
    assert raw_df.shape[1] > 1, "El dataset solo tiene una columna"


def test_columna_target_existe(raw_df, config):
    """La columna target definida en config.yaml debe existir."""
    target = config["data"]["target_column"]
    assert target in raw_df.columns, f"Falta la columna target '{target}'"


def test_features_definidas_existen(raw_df, config):
    """Todas las features de config.yaml deben existir en el dataset."""
    expected = (
        config["preprocessing"]["numeric_features"]
        + config["preprocessing"]["categorical_features"]
    )
    missing = [c for c in expected if c not in raw_df.columns]
    assert not missing, f"Faltan columnas en el dataset: {missing}"


def test_basic_clean_elimina_duplicados_y_nulos(config):
    """basic_clean debe quitar duplicados y filas con nulos."""
    df = pd.DataFrame(
        {
            "age": [50, 50, 60, None],
            "target": [0, 0, 1, 1],
        }
    )
    cleaned = basic_clean(df, target_column="target")
    assert cleaned.isnull().sum().sum() == 0, "Quedaron valores nulos"
    assert cleaned.duplicated().sum() == 0, "Quedaron duplicados"


def test_basic_clean_target_binario(config):
    """basic_clean debe convertir target multiclase a binario (0/1)."""
    df = pd.DataFrame(
        {
            "age": [50, 60, 70, 45],
            "target": [0, 1, 2, 4],
        }
    )
    cleaned = basic_clean(df, target_column="target")
    assert set(cleaned["target"].unique()).issubset({0, 1}), \
        "El target no quedó binario"

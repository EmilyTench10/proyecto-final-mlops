"""
tests/test_model.py
-------------------
Pruebas básicas del modelo entrenado.
Asume que train.py ya se ejecutó previamente y existe model.pkl + metrics.json.
"""

import json
import os

import joblib
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def trained_model(config):
    model_path = config["output"]["model_pkl"]
    if not os.path.exists(model_path):
        pytest.skip(f"No existe {model_path}; ejecuta `make train` primero.")
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def metrics(config):
    metrics_path = config["output"]["metrics_json"]
    if not os.path.exists(metrics_path):
        pytest.skip(f"No existe {metrics_path}; ejecuta `make train` primero.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_modelo_es_pipeline(trained_model):
    """El modelo guardado debe ser un Pipeline de sklearn (preproc + clf)."""
    from sklearn.pipeline import Pipeline

    assert isinstance(trained_model, Pipeline), "El modelo no es un Pipeline"
    assert "preprocessor" in trained_model.named_steps
    assert "classifier" in trained_model.named_steps


def test_modelo_predice_correctamente(trained_model, config):
    """El modelo debe poder predecir sobre datos válidos sin errores."""
    sample = pd.DataFrame(
        [
            {
                "Age": 55,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 140,
                "Cholesterol": 250,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 150,
                "ExerciseAngina": "N",
                "Oldpeak": 1.0,
                "ST_Slope": "Up",
            }
        ]
    )
    pred = trained_model.predict(sample)
    assert len(pred) == 1, "El modelo debe devolver una predicción"
    assert pred[0] in (0, 1), "La predicción debe ser binaria (0 o 1)"


def test_metricas_superan_umbrales(metrics, config):
    """Accuracy y F1 deben superar los umbrales mínimos del config."""
    min_acc = config["validation"]["min_accuracy"]
    min_f1 = config["validation"]["min_f1"]

    assert metrics["accuracy"] >= min_acc, (
        f"Accuracy {metrics['accuracy']:.4f} < umbral {min_acc}"
    )
    assert metrics["f1_weighted"] >= min_f1, (
        f"F1 {metrics['f1_weighted']:.4f} < umbral {min_f1}"
    )


def test_metricas_son_validas(metrics):
    """Las métricas deben estar entre 0 y 1."""
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy fuera de rango"
    assert 0.0 <= metrics["f1_weighted"] <= 1.0, "F1 fuera de rango"

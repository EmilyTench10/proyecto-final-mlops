"""
src/train.py
------------
Pipeline principal de entrenamiento:
  1. Lee la configuración desde config.yaml.
  2. Carga y limpia el dataset Heart Disease.
  3. Construye el pipeline de preprocesamiento.
  4. Entrena un RandomForestClassifier.
  5. Evalúa con accuracy y f1_weighted.
  6. Registra todo en MLflow (params, métricas, firma, input_example, modelo).

Ejecución:
    python src/train.py
"""

import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# Hacer importable el paquete src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import basic_clean, load_dataset  # noqa: E402
from preprocessing import build_preprocessor, split_data  # noqa: E402


def load_config(path: str = "config.yaml") -> dict:
    """Carga el archivo de configuración YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(tracking_dir: str, experiment_name: str) -> str:
    """
    Configura MLflow para tracking local y devuelve el experiment_id.
    """
    workspace_dir = os.getcwd()
    mlruns_dir = os.path.join(workspace_dir, tracking_dir)
    os.makedirs(mlruns_dir, exist_ok=True)

    tracking_uri = "file://" + os.path.abspath(mlruns_dir)
    artifact_location = "file://" + os.path.abspath(mlruns_dir)
    mlflow.set_tracking_uri(tracking_uri)

    print(f"--- MLflow tracking URI: {tracking_uri}")

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id

    return mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Configuración
    # ------------------------------------------------------------------
    cfg = load_config("config.yaml")
    print("--- Configuración cargada desde config.yaml")

    # ------------------------------------------------------------------
    # 2. Cargar y limpiar datos
    # ------------------------------------------------------------------
    df = load_dataset(local_path=cfg["data"]["local_path"])
    df = basic_clean(df, target_column=cfg["data"]["target_column"])

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_column=cfg["data"]["target_column"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # ------------------------------------------------------------------
    # 4. Pipeline de preprocesamiento + modelo
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor(
        numeric_features=cfg["preprocessing"]["numeric_features"],
        categorical_features=cfg["preprocessing"]["categorical_features"],
    )

    classifier = RandomForestClassifier(**cfg["model"]["params"])

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # ------------------------------------------------------------------
    # 5. MLflow setup + entrenamiento
    # ------------------------------------------------------------------
    experiment_id = setup_mlflow(
        tracking_dir=cfg["mlflow"]["tracking_dir"],
        experiment_name=cfg["mlflow"]["experiment_name"],
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"--- MLflow run_id: {run_id}")

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Evaluar
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # ------------------------------------------------------------------
        # 6. Registro MLflow: params + metrics + firma + modelo
        # ------------------------------------------------------------------
        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_param("model_name", cfg["model"]["name"])
        mlflow.log_param("dataset", cfg["data"]["local_path"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=cfg["mlflow"]["artifact_subpath"],
            signature=signature,
            input_example=input_example,
        )

        # ------------------------------------------------------------------
        # 7. Guardar artefactos locales (para tests y CI)
        # ------------------------------------------------------------------
        joblib.dump(pipeline, cfg["output"]["model_pkl"])
        with open(cfg["output"]["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": accuracy, "f1_weighted": f1, "run_id": run_id},
                f,
                indent=2,
            )

        # ------------------------------------------------------------------
        # 8. Reporte final
        # ------------------------------------------------------------------
        print("\n=== Métricas en test ===")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"F1 weighted: {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"✅ Modelo registrado en MLflow (run_id={run_id})")
        print(f"✅ Modelo guardado en {cfg['output']['model_pkl']}")
        print(f"✅ Métricas guardadas en {cfg['output']['metrics_json']}")


if __name__ == "__main__":
    main()

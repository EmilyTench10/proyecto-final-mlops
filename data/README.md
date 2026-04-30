# data/

Esta carpeta contiene el dataset utilizado por el pipeline.

## `heart.csv`

- **Fuente:** [Heart Failure Prediction Dataset — Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Origen real:** combinación curada de 5 datasets de heart disease del UCI
  Machine Learning Repository:
  - Cleveland (303 instancias)
  - Hungarian (294 instancias)
  - Switzerland (123 instancias)
  - Long Beach VA (200 instancias)
  - Statlog (Heart) (270 instancias)
- **Tamaño:** 918 instancias × 12 columnas (sin duplicados, sin nulos).
- **Tarea:** clasificación binaria (presencia/ausencia de enfermedad cardiaca).
- **Target:** columna `HeartDisease` (0 = ausente, 1 = presente).

El CSV se versiona junto con el código para garantizar reproducibilidad
total del pipeline (no depende de descargas en tiempo de ejecución).

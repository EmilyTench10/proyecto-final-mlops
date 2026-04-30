# Proyecto Final — Pipeline CI/CD de ML con MLflow y GitHub Actions

**Estudiante:** Emily Delcarmen Tench Pérez
**Curso:** MLOps y Analítica en la Nube — 2026-1

Pipeline reproducible de Machine Learning que entrena un clasificador binario
para predecir la presencia de enfermedad cardiaca, completamente automatizado
con **GitHub Actions** y registrado con **MLflow**.

---

## 🎯 Objetivo

Diseñar un flujo de trabajo automatizado que integre:

- Carga, limpieza y preprocesamiento de datos.
- Entrenamiento y evaluación de un modelo.
- Registro de experimentos con MLflow (parámetros, métricas, firma, modelo).
- Pruebas automáticas con `pytest`.
- Verificación de calidad de código con `flake8`.
- Ejecución automatizada en **GitHub Actions** al hacer `push` a `main`.

---

## 📁 Estructura del proyecto

```
proyecto-final/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Carga y limpieza del dataset
│   ├── preprocessing.py    # Pipeline de preprocesamiento + split
│   └── train.py            # Script principal de entrenamiento
├── tests/
│   ├── conftest.py         # Configuración de pytest
│   ├── test_data.py        # Pruebas de carga y limpieza
│   └── test_model.py       # Pruebas del modelo entrenado
├── data/
│   ├── heart.csv           # Dataset (versionado en el repo)
│   └── README.md           # Información de la fuente
├── .github/
│   └── workflows/
│       └── ml.yml          # Pipeline de GitHub Actions
├── mlruns/                 # Tracking local de MLflow (auto-generado)
├── config.yaml             # Hiperparámetros, rutas y umbrales
├── Makefile                # Comandos de automatización
├── requirements.txt        # Dependencias Python
├── .gitignore
└── README.md               # Este archivo
```

---

## 🗂️ Dataset

**Heart Failure Prediction Dataset** (Kaggle - fedesoriano)

- **Fuente:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Origen real:** combinación de 5 datasets del UCI Machine Learning Repository
  (Cleveland, Hungarian, Switzerland, Long Beach VA y Statlog).
- **Instancias:** 918 (sin duplicados ni nulos).
- **Variables:** 11 features + 1 target binario.
- **Tarea:** Clasificación binaria (presencia/ausencia de enfermedad cardiaca).

### Variables del dataset

| Columna           | Tipo       | Descripción                                                       |
| ----------------- | ---------- | ----------------------------------------------------------------- |
| `Age`             | numérica   | Edad del paciente (años)                                          |
| `Sex`             | categórica | Género (`M`, `F`)                                                 |
| `ChestPainType`   | categórica | Tipo de dolor torácico (`TA`, `ATA`, `NAP`, `ASY`)                |
| `RestingBP`       | numérica   | Presión arterial en reposo (mm Hg)                                |
| `Cholesterol`     | numérica   | Colesterol sérico (mg/dl)                                         |
| `FastingBS`       | categórica | Glucemia en ayunas > 120 mg/dl (1 = sí, 0 = no)                   |
| `RestingECG`      | categórica | ECG en reposo (`Normal`, `ST`, `LVH`)                             |
| `MaxHR`           | numérica   | Frecuencia cardiaca máxima alcanzada                              |
| `ExerciseAngina`  | categórica | Angina inducida por ejercicio (`Y`, `N`)                          |
| `Oldpeak`         | numérica   | Depresión del ST inducida por ejercicio                           |
| `ST_Slope`        | categórica | Pendiente del segmento ST (`Up`, `Flat`, `Down`)                  |
| `HeartDisease`    | binaria    | **Variable objetivo** (1 = enfermedad presente, 0 = ausente)      |

### ¿Por qué este dataset?

- **Externo:** no pertenece a `sklearn.datasets` (requisito del proyecto).
- **Curado profesionalmente:** combina 5 datasets clínicos del UCI ML Repository,
  obteniendo un total de 918 instancias únicas — el dataset más grande de
  predicción de enfermedad cardiaca disponible públicamente.
- **Limpio:** sin duplicados ni valores nulos.
- **Realista:** datos médicos reales con variables numéricas y categóricas
  mixtas → ejercita el preprocesamiento (escalado + one-hot encoding).
- **Versionado:** el CSV se incluye en el repositorio para reproducibilidad total.

---

## 🧠 Modelo

**RandomForestClassifier** con un Pipeline de scikit-learn que combina:

1. **Preprocesamiento** (`ColumnTransformer`):
   - `StandardScaler` para 5 variables numéricas (`Age`, `RestingBP`,
     `Cholesterol`, `MaxHR`, `Oldpeak`).
   - `OneHotEncoder` para 6 variables categóricas (`Sex`, `ChestPainType`,
     `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope`).
2. **Clasificador**: Random Forest con 200 árboles, `max_depth=8`,
   `min_samples_split=4`, `random_state=42`.

Los hiperparámetros se leen desde `config.yaml`, evitando hardcodear valores.

---

## 📊 Resultados

Métricas obtenidas sobre el conjunto de test (20% del dataset, estratificado):

| Métrica            | Valor   | Umbral mínimo | Estado |
| ------------------ | ------- | ------------- | ------ |
| Accuracy (test)    | 0.9022  | 0.85          | ✅     |
| F1-weighted (test) | 0.9017  | 0.85          | ✅     |

```
              precision    recall  f1-score   support

           0       0.92      0.85      0.89        82
           1       0.89      0.94      0.91       102

    accuracy                           0.90       184
   macro avg       0.90      0.90      0.90       184
weighted avg       0.90      0.90      0.90       184
```

El modelo logra **alta sensibilidad (recall = 0.94)** en la clase positiva
(enfermedad presente), lo cual es importante en aplicaciones médicas:
un falso negativo (no detectar la enfermedad) es más costoso que un falso
positivo.

---

## 🛠️ Uso local

### 1. Instalación

```bash
pip install -r requirements.txt
# o
make install
```

### 2. Ejecutar el pipeline completo

```bash
make all   # install + lint + train + test
```

### 3. Ejecutar pasos individuales

```bash
make lint     # Verifica calidad del código con flake8
make train    # Entrena y registra el modelo en MLflow
make test     # Corre 9 pruebas unitarias con pytest
make clean    # Borra artefactos generados
```

### 4. Ver el tracking de MLflow

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Abrir http://127.0.0.1:5000
```

---

## 🔄 Pipeline CI/CD (GitHub Actions)

El workflow `.github/workflows/ml.yml` se dispara con `push` o
`pull_request` a `main`, y también puede ejecutarse manualmente.

### Pasos automatizados

1. ✅ Clonar repositorio
2. 🐍 Configurar Python 3.10
3. 📦 `make install` — instala dependencias
4. 🧹 `make lint` — verifica calidad del código con flake8
5. 🧪 `make train` — entrena el modelo y lo registra en MLflow
6. ✅ `make test` — ejecuta 9 pruebas unitarias con pytest
7. 📤 Sube artefactos: `model.pkl`, `metrics.json`, `mlruns/`

Si cualquier paso falla, el pipeline se detiene y los artefactos NO se publican.

---

## ⚙️ Configuración (`config.yaml`)

Toda la configuración del proyecto está centralizada en `config.yaml`:

- Ruta local del dataset.
- Lista de features numéricas y categóricas.
- Hiperparámetros del modelo.
- Nombre del experimento y carpeta de tracking de MLflow.
- Umbrales mínimos de Accuracy y F1 para que las pruebas pasen.

Esto permite cambiar el comportamiento del pipeline sin tocar el código,
facilitando la experimentación y la reproducibilidad.

---

## ✅ Pruebas (`tests/`)

### `test_data.py` — Pruebas de datos (5 tests)

- El dataset no está vacío.
- La columna target existe.
- Todas las features definidas en config existen en el dataset.
- `basic_clean` elimina duplicados y nulos.
- `basic_clean` convierte targets multiclase a binarios.

### `test_model.py` — Pruebas del modelo (4 tests)

- El modelo guardado es un `Pipeline` (preprocesamiento + clasificador).
- El modelo predice correctamente sobre datos de ejemplo.
- Las métricas superan los umbrales mínimos del config.
- Las métricas están en el rango válido `[0, 1]`.

**Total: 9 tests automatizados con `pytest`.**

---

## 📈 Evidencia de MLflow

Cada run de entrenamiento registra en MLflow:

- **Parámetros**: `n_estimators`, `max_depth`, `min_samples_split`,
  `random_state`, `n_jobs`, `model_name`, `dataset`, `n_features`,
  `n_train_samples`, `n_test_samples`.
- **Métricas**: `accuracy`, `f1_weighted`.
- **Artefactos**: modelo serializado con firma (`signature`) e
  `input_example` (para evitar errores de schema en producción).

Los artefactos completos del tracking se suben como artifact del workflow
de GitHub Actions, descargables desde la pestaña **Actions**.

---

## 📦 Entregables

- ✅ Código en `src/`, modular y reutilizable.
- ✅ `Makefile` funcional con `install`, `lint`, `train`, `test`, `clean`, `all`.
- ✅ Pipeline CI/CD activo en GitHub Actions (`ml.yml`).
- ✅ Pruebas automatizadas con `pytest` (9 tests).
- ✅ Tracking con MLflow (params + metrics + signature + input_example + model).
- ✅ Configuración externalizada en `config.yaml`.
- ✅ Documentación clara (este README).

---

## 👤 Autor

**Emily Delcarmen Tench Pérez**
Curso: MLOps y Analítica en la Nube — 2026-1

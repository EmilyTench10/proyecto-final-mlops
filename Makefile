# =====================================================================
# Makefile - Proyecto Final CI/CD ML con MLflow + GitHub Actions
# =====================================================================
# Comandos disponibles:
#   make install -> Instala las dependencias del proyecto.
#   make lint    -> Verifica calidad del código con flake8.
#   make train   -> Ejecuta el pipeline completo de entrenamiento.
#   make test    -> Corre pruebas básicas con pytest.
#   make clean   -> Borra artefactos generados (mlruns, model.pkl, etc.).
#   make all     -> install + lint + train + test (pipeline CI completo).
# =====================================================================

.PHONY: install lint train test clean all

install:
	@echo ">>> Actualizando pip e instalando dependencias..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	@echo ">>> Ejecutando lint con flake8..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

train:
	@echo ">>> Ejecutando pipeline de entrenamiento..."
	python src/train.py

test:
	@echo ">>> Ejecutando pruebas con pytest..."
	pytest tests/ -v

clean:
	@echo ">>> Borrando artefactos generados..."
	rm -rf mlruns mlartifacts __pycache__ .pytest_cache
	rm -f model.pkl metrics.json
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install lint train test

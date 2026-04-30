"""
conftest.py
-----------
Configuración compartida de pytest. Agrega el directorio raíz del proyecto
al sys.path para que los tests puedan importar el paquete `src`.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

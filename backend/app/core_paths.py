from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_CORE_DIR = PROJECT_ROOT / 'ml_core'
ML_RESULTS_DIR = ML_CORE_DIR / 'results'
ML_MODELS_DIR = ML_CORE_DIR / 'models'
ML_DATA_DIR = ML_CORE_DIR / 'data'

for path in (PROJECT_ROOT, BACKEND_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

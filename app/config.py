import os
from pathlib import Path

BASE_DIR = Path(os.getenv("MODELS_DIR", "../models"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

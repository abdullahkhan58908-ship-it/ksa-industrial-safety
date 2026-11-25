from pathlib import Path

# Base project directory (C:\Users\Admin\Desktop\PPE)
BASE_DIR = Path(__file__).resolve().parent

# Project directories
WEIGHTS_DIR = BASE_DIR / "weights"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
DATASETS_DIR = BASE_DIR / "datasets"
INCIDENTS_IMG_DIR = REPORTS_DIR / "incidents"

# Database path
DB_PATH = BASE_DIR / "database" / "safety_system.db"

# Default model paths
PERSON_MODEL_PATH = WEIGHTS_DIR / "yolo11n.pt"
PPE_MODEL_PATH = WEIGHTS_DIR / "ppe_best.pt"
FIRE_MODEL_PATH = WEIGHTS_DIR / "fire_smoke_best.pt"

# CPU-friendly video & processing settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 15
DEFAULT_CONFIDENCE = 0.45

# Alert cooldown in seconds (prevents repeated alerts for the same violation)
ALERT_COOLDOWN_SECONDS = 30
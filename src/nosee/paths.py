from pathlib import Path


APP_SUPPORT = Path.home() / "Library" / "Application Support" / "NoSee"
MODEL_DIR = APP_SUPPORT / "models"
GAZEMAP_DIR = MODEL_DIR / "gazemap"
GAZEMAP_CONFIG = MODEL_DIR / "config.json"
FACE_MODEL = MODEL_DIR / "6 points-based face model.mat"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
KEYCHAIN_SERVICE = "local.nosee.owner-face"
KEYCHAIN_ACCOUNT = "owner-embedding"


def ensure_app_support() -> None:
    APP_SUPPORT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

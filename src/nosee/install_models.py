"""One-time, user-initiated download of third-party on-device model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat

from .paths import (
    FACE_MODEL,
    GAZEMAP_CONFIG,
    GAZEMAP_DIR,
    LANDMARKER_MODEL,
    MODEL_DIR,
    SFACE_MODEL,
    ensure_app_support,
)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_LICENSE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/LICENSE"
)
GAZEMAP_SOURCE_URL = "https://github.com/fammad/GazeMap/archive/refs/heads/main.zip"
GAZEMAP_WEIGHTS_URL = "https://github.com/fammad/GazeMap/releases/latest/download/gazemap_models.zip"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
DEFAULT_GAZE_CONFIG = {"conv1": 20, "conv2": 50, "hidden": 500, "head_pose": True}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "NoSee/0.1 model setup"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_members(archive: zipfile.ZipFile):
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            continue
        yield member


def _extract_all(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in _safe_members(archive):
            archive.extract(member, destination)


def _find_file(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_file():
            return path
    return None


def _install_gazemap(temp: Path) -> None:
    existing_weights = list(GAZEMAP_DIR.glob("lopo_p*.pt")) if GAZEMAP_DIR.exists() else []
    if len(existing_weights) == 15 and FACE_MODEL.exists() and GAZEMAP_CONFIG.exists():
        print("GazeMap assets are already installed.")
        return
    source_archive = temp / "gazemap-source.zip"
    weights_archive = temp / "gazemap-weights.zip"
    source_tree = temp / "gazemap-source"
    weights_tree = temp / "gazemap-weights"
    print("Downloading GazeMap source assets...")
    _download(GAZEMAP_SOURCE_URL, source_archive)
    _extract_all(source_archive, source_tree)
    print("Downloading GazeMap pretrained weights (about 100 MB)...")
    _download(GAZEMAP_WEIGHTS_URL, weights_archive)
    _extract_all(weights_archive, weights_tree)

    GAZEMAP_DIR.mkdir(parents=True, exist_ok=True)
    weight_files = [p for p in weights_tree.rglob("lopo_p*.pt") if p.is_file()]
    if len(weight_files) != 15:
        raise RuntimeError(
            f"Expected 15 GazeMap LOPO weight files, found {len(weight_files)}."
        )
    for path in weight_files:
        shutil.copy2(path, GAZEMAP_DIR / path.name)

    model_file = _find_file(source_tree, "6 points-based face model.mat")
    if model_file is not None:
        shutil.copy2(model_file, FACE_MODEL)

    config_file = _find_file(weights_tree, "config.json") or _find_file(source_tree, "config.json")
    if config_file is not None:
        shutil.copy2(config_file, GAZEMAP_CONFIG)
    else:
        GAZEMAP_CONFIG.write_text(
            __import__("json").dumps(DEFAULT_GAZE_CONFIG, indent=2), "utf-8"
        )

    for filename in ("LICENSE", "NOTICE"):
        source = _find_file(source_tree, filename)
        if source is not None:
            shutil.copy2(source, MODEL_DIR / f"gazemap-{filename.lower()}")

    if not FACE_MODEL.exists():
        raise RuntimeError(
            "The GazeMap source archive did not include its 6-point face model. "
            "Place the MPIIGaze file '6 points-based face model.mat' at:\n"
            f"  {FACE_MODEL}\n"
            "The gaze model cannot estimate a 3D direction without this model."
        )
    manifest = {
        "gaze_project": "https://github.com/fammad/GazeMap",
        "gaze_source_archive": GAZEMAP_SOURCE_URL,
        "gaze_weights_archive": GAZEMAP_WEIGHTS_URL,
        "weights_sha256": {
            path.name: hashlib.sha256((GAZEMAP_DIR / path.name).read_bytes()).hexdigest()
            for path in weight_files
        },
        "weights_license": "CC BY-NC-SA 4.0",
        "source_license": "Apache-2.0",
    }
    (MODEL_DIR / "model-manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


def _install_sface() -> None:
    if SFACE_MODEL.exists() and hashlib.sha256(SFACE_MODEL.read_bytes()).hexdigest() == SFACE_SHA256:
        return
    print("Downloading the SFace identity model...")
    temporary = SFACE_MODEL.with_suffix(".download")
    _download(SFACE_URL, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != SFACE_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SFace checksum mismatch: received {digest}.")
    temporary.replace(SFACE_MODEL)
    try:
        _download(SFACE_LICENSE_URL, MODEL_DIR / "sface-LICENSE")
    except Exception as exc:
        print(f"Could not fetch the upstream SFace license file: {exc}")


def _install_face_model(source: Path | None) -> None:
    if source is not None:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Could not find the selected face model file: {source}")
        if source.name != "6 points-based face model.mat":
            raise ValueError("Select the MPIIGaze file named '6 points-based face model.mat'.")
        if source != FACE_MODEL.resolve():
            shutil.copy2(source, FACE_MODEL)

    if not FACE_MODEL.is_file():
        raise FileNotFoundError(
            "GazeMap requires MPIIGaze's '6 points-based face model.mat'. It is not included "
            "in GazeMap's source or weight archive. Download the MPIIGaze dataset from the "
            "official project page, extract this file, then run the installer again and select it.\n"
            "Official page: https://www.collaborative-ai.org/research/datasets/MPIIGaze/\n"
            f"Expected local path: {FACE_MODEL}"
        )

    model = loadmat(str(FACE_MODEL)).get("model")
    if model is None:
        raise ValueError("The selected MPIIGaze file does not contain a 'model' matrix.")
    values = np.asarray(model)
    if (
        values.shape != (3, 6)
        or not np.issubdtype(values.dtype, np.number)
        or not np.isfinite(values).all()
    ):
        raise ValueError("The MPIIGaze face model must be a numeric 3×6 matrix.")


def install_models(face_model_source: Path | None = None) -> None:
    ensure_app_support()
    _install_face_model(face_model_source)
    with tempfile.TemporaryDirectory(prefix="nosee-models-") as temporary:
        _install_gazemap(Path(temporary))
    _install_sface()
    if not LANDMARKER_MODEL.exists():
        print("Downloading MediaPipe Face Landmarker task file...")
        _download(LANDMARKER_URL, LANDMARKER_MODEL)
    print(f"Models are ready in: {MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install NoSee's offline model assets.")
    parser.add_argument(
        "--face-model",
        type=Path,
        help="path to MPIIGaze's '6 points-based face model.mat' file",
    )
    args = parser.parse_args()
    try:
        install_models(args.face_model)
    except Exception as exc:
        raise SystemExit(f"Model setup failed: {exc}") from exc


if __name__ == "__main__":
    main()

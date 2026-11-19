from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .paths import SFACE_MODEL

# Conservative fallback for profiles created before NoSee stored a
# per-enrollment threshold. New profiles derive a threshold from sample quality.
OWNER_COSINE_THRESHOLD = 0.42
IDENTITY_EMBEDDING_VERSION = 2


@dataclass(frozen=True)
class FaceSample:
    box: tuple[int, int, int, int]
    landmarks: tuple[tuple[float, float], ...]


class FaceIdentity:
    """SFace face descriptor extraction and owner comparison."""

    def __init__(self) -> None:
        if not SFACE_MODEL.is_file():
            raise FileNotFoundError("SFace is missing. Run: python -m nosee.install_models")
        factory = getattr(cv2, "FaceRecognizerSF", None)
        if factory is not None:
            self.recognizer = factory.create(str(SFACE_MODEL), "")
        elif hasattr(cv2, "FaceRecognizerSF_create"):
            self.recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")
        else:
            raise RuntimeError("OpenCV was installed without the SFace API. Install opencv-contrib-python.")

    @staticmethod
    def sample_from_landmarks(frame: np.ndarray, landmarks: list) -> FaceSample | None:
        height, width = frame.shape[:2]
        points = np.array([[p.x * width, p.y * height] for p in landmarks], dtype=np.float32)
        if len(points) < 363:
            return None
        left = max(0, int(points[:, 0].min()))
        top = max(0, int(points[:, 1].min()))
        right = min(width, int(points[:, 0].max()))
        bottom = min(height, int(points[:, 1].max()))
        if right - left < 40 or bottom - top < 40:
            return None
        # SFace expects YuNet's five-point layout: subject-right eye, subject-
        # left eye, nose tip, subject-right mouth corner, subject-left mouth
        # corner. MediaPipe 33/263 are eye outer corners, not eye centers, so
        # average each eye's two corners before alignment.
        right_eye = (points[33] + points[133]) * 0.5
        left_eye = (points[263] + points[362]) * 0.5
        five = tuple(
            (float(point[0]), float(point[1]))
            for point in (right_eye, left_eye, points[1], points[61], points[291])
        )
        return FaceSample((left, top, right - left, bottom - top), five)

    def embedding(self, frame: np.ndarray, sample: FaceSample) -> np.ndarray:
        x, y, width, height = sample.box
        box_and_landmarks = np.asarray(
            [[x, y, width, height, *[coordinate for point in sample.landmarks for coordinate in point]]],
            dtype=np.float32,
        )
        aligned = self.recognizer.alignCrop(frame, box_and_landmarks)
        feature = np.asarray(self.recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise RuntimeError("SFace returned an invalid face embedding.")
        return feature / norm

    @staticmethod
    def similarity(a: list[float] | np.ndarray, b: np.ndarray) -> float:
        first = np.asarray(a, dtype=np.float32).reshape(-1)
        second = np.asarray(b, dtype=np.float32).reshape(-1)
        if first.shape != second.shape:
            return -1.0
        return float(np.dot(first, second) / max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8))

    def is_owner(self, template: list[float] | None, embedding: np.ndarray) -> bool:
        return template is not None and self.similarity(template, embedding) >= OWNER_COSINE_THRESHOLD

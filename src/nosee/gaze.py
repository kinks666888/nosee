"""Multi-face wrapper around the offline GazeMap model and screen geometry."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import torch
from scipy.io import loadmat

from .gaze_model import GazeNet, pose_to_angles
from .identity import FaceIdentity, FaceSample
from .paths import FACE_MODEL, GAZEMAP_CONFIG, GAZEMAP_DIR, LANDMARKER_MODEL

FOCAL_NORM = 960.0
DISTANCE_NORM = 600.0
EYE_WIDTH = 60
EYE_HEIGHT = 36
MEDIAPIPE_IDS = (33, 133, 362, 263, 61, 291)
SCREEN_MARGIN = 0.10
MAX_FACES = 8


@dataclass
class FaceAnalysis:
    sample: FaceSample
    embedding: np.ndarray
    looking_at_screen: bool | None
    gaze_point: tuple[float, float] | None = None


@dataclass
class _PoseTrack:
    box: tuple[int, int, int, int]
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    rejects: int = 0
    last_seen: float = 0.0


class ScreenGeometry:
    """Approximate primary display plane, with the webcam at its top centre."""

    def __init__(self, screen_width_mm: float | None = None, camera_offset_mm: float = 0.0):
        self.width_px, self.height_px, detected_width_mm, detected_height_mm = self._display_size()
        self.width_mm = float(screen_width_mm or detected_width_mm or 340.0)
        self.height_mm = float(detected_height_mm or self.width_mm * self.height_px / self.width_px)
        self.camera_offset_mm = float(camera_offset_mm)

    @staticmethod
    def _display_size() -> tuple[int, int, float, float]:
        try:
            import Quartz

            display = Quartz.CGMainDisplayID()
            size = Quartz.CGDisplayScreenSize(display)
            return (
                int(Quartz.CGDisplayPixelsWide(display)),
                int(Quartz.CGDisplayPixelsHigh(display)),
                float(size.width),
                float(size.height),
            )
        except Exception:
            return 1440, 900, 340.0, 212.5

    def ray_hits_display(self, origin: np.ndarray, direction: np.ndarray) -> tuple[bool, tuple[float, float] | None]:
        if not np.isfinite(direction).all() or direction[2] >= -1e-6:
            return False, None
        t = -origin[2] / direction[2]
        if t <= 0:
            return False, None
        point = origin + t * direction
        px = (0.5 - point[0] / self.width_mm) * self.width_px
        py = (point[1] - self.camera_offset_mm) / self.height_mm * self.height_px
        x = px / self.width_px
        y = py / self.height_px
        margin = SCREEN_MARGIN
        inside = -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin
        return inside, (float(x), float(y))


class GazeEngine:
    def __init__(self, screen_width_mm: float | None = None, camera_offset_mm: float = 0.0):
        for path in (LANDMARKER_MODEL, FACE_MODEL, GAZEMAP_CONFIG):
            if not path.is_file():
                raise FileNotFoundError(f"Required gaze model asset is missing: {path}. Run python -m nosee.install_models")
        weight_paths = sorted(GAZEMAP_DIR.glob("lopo_p*.pt"))
        if len(weight_paths) != 15:
            raise FileNotFoundError(
                f"Expected all 15 GazeMap weights, found {len(weight_paths)}. "
                "Run python -m nosee.install_models."
            )

        config = json.loads(GAZEMAP_CONFIG.read_text("utf-8"))
        conv1 = int(config.get("conv1", 20))
        conv2 = int(config.get("conv2", 50))
        hidden = int(config.get("hidden", 500))
        self.models: list[GazeNet] = []
        checkpoint_head_pose: bool | None = None
        for path in weight_paths:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            output_weight = state.get("fc2.weight")
            if output_weight is None or output_weight.ndim != 2 or output_weight.shape[0] != 2:
                raise RuntimeError(f"GazeMap checkpoint {path.name} has no valid fc2.weight.")
            input_features = int(output_weight.shape[1])
            if input_features == hidden + 2:
                uses_head_pose = True
            elif input_features == hidden:
                uses_head_pose = False
            else:
                raise RuntimeError(
                    f"GazeMap checkpoint {path.name} expects {input_features} inputs in fc2; "
                    f"expected {hidden} or {hidden + 2}."
                )
            if checkpoint_head_pose is not None and uses_head_pose != checkpoint_head_pose:
                raise RuntimeError("GazeMap checkpoints use inconsistent head-pose settings.")
            checkpoint_head_pose = uses_head_pose
            # The checkpoint tensor shape is authoritative. Some GazeMap releases
            # ship a config.json whose head_pose flag disagrees with the weights.
            model = GazeNet(conv1, conv2, hidden, uses_head_pose)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

        raw_face_model = loadmat(str(FACE_MODEL)).get("model")
        if raw_face_model is None:
            raise RuntimeError("The MPIIGaze 6-point face model file has no 'model' matrix.")
        self.face_model = np.asarray(raw_face_model, dtype=np.float64)
        if self.face_model.shape != (3, 6):
            raise RuntimeError(f"Unexpected 6-point face model shape: {self.face_model.shape}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=MAX_FACES,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.identity = FaceIdentity()
        self.screen = ScreenGeometry(screen_width_mm, camera_offset_mm)
        self.last_timestamp_ms = -1
        self.camera_offset_mm = camera_offset_mm
        self._pose_tracks: list[_PoseTrack] = []
        self.last_face_count = 0
        self.last_skipped_faces = 0
        self.last_analysis_error: str | None = None

    def close(self) -> None:
        self.landmarker.close()

    @staticmethod
    def _angles_to_vector(pitch: float, yaw: float) -> np.ndarray:
        return np.array(
            [-math.cos(pitch) * math.sin(yaw), -math.sin(pitch), -math.cos(pitch) * math.cos(yaw)],
            dtype=np.float64,
        )

    def _predict_eyes(self, eye_images: list[np.ndarray], head_angles: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(eye_images).astype(np.float32)).div(255.0).unsqueeze(1)
        head = torch.from_numpy(np.stack(head_angles).astype(np.float32))
        with torch.inference_mode():
            predictions = [model(x, head).cpu().numpy() for model in self.models]
        return np.mean(np.stack(predictions, axis=0), axis=0)

    def _normalize_eye(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        head_rotation: np.ndarray,
        center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distance = float(np.linalg.norm(center))
        if distance <= 1e-6:
            raise ValueError("Invalid eye position.")
        forward = center / distance
        down = np.cross(forward, head_rotation[:, 0])
        down_norm = float(np.linalg.norm(down))
        if down_norm <= 1e-8:
            raise ValueError("Invalid face orientation.")
        down /= down_norm
        right = np.cross(down, forward)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        rotation = np.stack([right, down, forward])
        scale = np.diag([1.0, 1.0, DISTANCE_NORM / distance])
        normalized_camera = np.array(
            [[FOCAL_NORM, 0, EYE_WIDTH / 2], [0, FOCAL_NORM, EYE_HEIGHT / 2], [0, 0, 1.0]],
            dtype=np.float64,
        )
        warp = normalized_camera @ scale @ rotation @ np.linalg.inv(camera_matrix)
        warped = cv2.warpPerspective(frame, warp, (EYE_WIDTH, EYE_HEIGHT))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray), rotation

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def _pose_track(self, box: tuple[int, int, int, int], now: float, claimed: set[int]) -> _PoseTrack:
        self._pose_tracks = [track for track in self._pose_tracks if now - track.last_seen < 1.5]
        best_index, best_score = None, 0.0
        for index, track in enumerate(self._pose_tracks):
            if index in claimed:
                continue
            score = self._iou(box, track.box)
            if score > best_score:
                best_index, best_score = index, score
        if best_index is not None and best_score >= 0.12:
            track = self._pose_tracks[best_index]
            claimed.add(best_index)
        else:
            track = _PoseTrack(box=box)
            self._pose_tracks.append(track)
            claimed.add(len(self._pose_tracks) - 1)
        track.box = box
        track.last_seen = now
        return track

    @staticmethod
    def _rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
        cosine = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _orthonormalize(matrix: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(matrix)
        result = u @ vt
        if np.linalg.det(result) < 0:
            u[:, -1] *= -1
            result = u @ vt
        return result

    def _update_pose(self, points: np.ndarray, camera_matrix: np.ndarray, track: _PoseTrack):
        object_points = self.face_model.T.reshape(-1, 1, 3)
        image_points = points.reshape(-1, 1, 2).astype(np.float64)
        if track.rotation is None:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_EPNP
            )
            if not ok:
                return None
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                None,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return None
            track.rotation = cv2.Rodrigues(rvec)[0]
            track.translation = tvec.reshape(3)
            track.rvec = rvec.reshape(3, 1)
            track.tvec = tvec.reshape(3, 1)
            return track.rotation, track.translation

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None,
            track.rvec.copy(),
            track.tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return track.rotation, track.translation
        new_rotation = cv2.Rodrigues(rvec)[0]
        if self._rotation_distance(track.rotation, new_rotation) > 15.0 and track.rejects < 10:
            track.rejects += 1
            return track.rotation, track.translation
        track.rejects = 0
        alpha = 0.5
        track.rotation = self._orthonormalize(alpha * track.rotation + (1.0 - alpha) * new_rotation)
        track.translation = alpha * track.translation + (1.0 - alpha) * tvec.reshape(3)
        track.rvec = cv2.Rodrigues(track.rotation)[0]
        track.tvec = track.translation.reshape(3, 1).copy()
        return track.rotation, track.translation

    def _analyze_face(
        self,
        frame: np.ndarray,
        landmarks: list,
        camera_matrix: np.ndarray,
        sample: FaceSample,
        track: _PoseTrack,
    ) -> FaceAnalysis | None:
        embedding = self.identity.embedding(frame, sample)
        points = np.array(
            [[landmarks[index].x * frame.shape[1], landmarks[index].y * frame.shape[0]] for index in MEDIAPIPE_IDS],
            dtype=np.float64,
        )
        pose = self._update_pose(points, camera_matrix, track)
        if pose is None:
            return FaceAnalysis(sample, embedding, None)
        head_rotation, translation = pose
        face_points = head_rotation @ self.face_model + translation.reshape(3, 1)
        centers = [face_points[:, 0:2].mean(axis=1), face_points[:, 2:4].mean(axis=1)]
        origin = np.mean(centers, axis=0)

        try:
            eyes: list[np.ndarray] = []
            heads: list[np.ndarray] = []
            rotations: list[np.ndarray] = []
            for eye_index, center in enumerate(centers):
                image, eye_rotation = self._normalize_eye(frame, camera_matrix, head_rotation, center)
                normalized_head = cv2.Rodrigues(eye_rotation @ head_rotation)[0].reshape(1, 3)
                head = pose_to_angles(normalized_head)[0]
                if eye_index == 1:
                    image = image[:, ::-1]
                    head = head * np.array([1.0, -1.0])
                eyes.append(np.ascontiguousarray(image))
                heads.append(head)
                rotations.append(eye_rotation)

            predictions = self._predict_eyes(eyes, heads)
            gaze_vectors = []
            for eye_index, prediction in enumerate(predictions):
                pitch, yaw = float(prediction[0]), float(prediction[1])
                if eye_index == 1:
                    yaw = -yaw
                gaze = rotations[eye_index].T @ self._angles_to_vector(pitch, yaw)
                norm = float(np.linalg.norm(gaze))
                if norm > 1e-8 and np.isfinite(gaze).all():
                    gaze_vectors.append(gaze / norm)
        except (ValueError, cv2.error, FloatingPointError, np.linalg.LinAlgError):
            return FaceAnalysis(sample, embedding, None)

        if not gaze_vectors:
            return FaceAnalysis(sample, embedding, None)
        gaze = np.sum(gaze_vectors, axis=0)
        gaze /= max(float(np.linalg.norm(gaze)), 1e-8)
        looking, point = self.screen.ray_hits_display(origin, gaze)
        return FaceAnalysis(
            sample,
            embedding,
            looking,
            point,
        )

    def analyze(self, frame: np.ndarray) -> list[FaceAnalysis]:
        now_ms = int(time.monotonic() * 1000)
        self.last_timestamp_ms = max(now_ms, self.last_timestamp_ms + 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, self.last_timestamp_ms)
        faces = result.face_landmarks or []
        self.last_face_count = len(faces)
        self.last_skipped_faces = 0
        self.last_analysis_error = None
        if not faces:
            return []
        height, width = frame.shape[:2]
        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]],
            dtype=np.float64,
        )
        readings = []
        claimed_tracks: set[int] = set()
        now = time.monotonic()
        for landmarks in faces:
            try:
                sample = self.identity.sample_from_landmarks(frame, landmarks)
                if sample is None:
                    self.last_skipped_faces += 1
                    continue
                track = self._pose_track(sample.box, now, claimed_tracks)
                reading = self._analyze_face(frame, landmarks, camera_matrix, sample, track)
                if reading is not None:
                    readings.append(reading)
            except (ValueError, cv2.error, FloatingPointError, np.linalg.LinAlgError) as exc:
                self.last_skipped_faces += 1
                self.last_analysis_error = f"{type(exc).__name__}: {exc}"
                continue
        return readings


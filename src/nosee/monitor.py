from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np

from .camera import CameraError, CameraService
from .gaze import FaceAnalysis, GazeEngine
from .identity import IDENTITY_EMBEDDING_VERSION, FaceIdentity
from .paths import LANDMARKER_MODEL
from .storage import LocalStore

TRIGGER_SECONDS = 0.6
CLEAR_SECONDS = 1.0
ANALYSIS_INTERVAL = 0.20
ENROLLMENT_SAMPLES = 7
ENROLLMENT_SAMPLE_INTERVAL = 0.75


@dataclass(frozen=True)
class MonitorUpdate:
    status: str
    warning: bool
    other_faces: int = 0
    unknown_gaze: bool = False
    screen_facing_faces: int = 0
    uncertain_identity_faces: int = 0


@dataclass
class _IdentityTrack:
    box: tuple[int, int, int, int]
    scores: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    owner: bool | None = None
    last_seen: float = 0.0


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class AlertStateMachine:
    def __init__(self, trigger_seconds: float = TRIGGER_SECONDS, clear_seconds: float = CLEAR_SECONDS):
        self.trigger_seconds = trigger_seconds
        self.clear_seconds = clear_seconds
        self.warning = False
        self._trigger_since: float | None = None
        self._clear_since: float | None = None

    def update(self, readings: list[tuple[bool, bool | None]], now: float | None = None) -> bool:
        """Each reading is (is_other, screen_facing); None means gaze is unknown."""
        current = now if now is not None else time.monotonic()
        others = [gaze for is_other, gaze in readings if is_other]
        if any(gaze is True for gaze in others):
            self._trigger_since = self._trigger_since or current
            self._clear_since = None
            if current - self._trigger_since >= self.trigger_seconds:
                self.warning = True
            return self.warning

        self._trigger_since = None
        if self.warning:
            # Do not clear based on an unreadable face; wait until gaze is known
            # to be away, or until no other face remains visible.
            if others and any(gaze is None for gaze in others):
                self._clear_since = None
                return self.warning
            self._clear_since = self._clear_since or current
            if current - self._clear_since >= self.clear_seconds:
                self.warning = False
                self._clear_since = None
        else:
            self._clear_since = None
        return self.warning


class MonitorController:
    def __init__(self, callback: Callable[[MonitorUpdate], None], store: LocalStore | None = None):
        self.callback = callback
        self.store = store or LocalStore()
        self.camera = CameraService()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._running = False
        self._engine: GazeEngine | None = None
        self._latest_update = MonitorUpdate("监控已停止", False)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def busy(self) -> bool:
        return self._busy

    def _send(self, update: MonitorUpdate, force: bool = False) -> None:
        if not force and update == self._latest_update:
            return
        self._latest_update = update
        self.callback(update)

    def start_monitoring(self) -> None:
        with self._lock:
            if self._running or self._busy:
                return
            try:
                template, threshold, embedding_version = self.store.load_owner_profile()
            except Exception as exc:
                self._send(MonitorUpdate(f"无法读取本人资料：{exc}", False))
                return
            if template is None:
                self._send(MonitorUpdate("请先录入本人脸部资料", False))
                return
            if embedding_version != IDENTITY_EMBEDDING_VERSION:
                self._send(
                    MonitorUpdate("人脸对齐方式已更新，请重新录入本人后再开始监控", False)
                )
                return
            self._busy = True
            self._send(MonitorUpdate("正在启动本机模型和摄像头…", False))
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._monitor_loop,
                args=(template, threshold),
                name="nosee-monitor",
                daemon=True,
            )
            self._worker.start()

    def stop_monitoring(self) -> None:
        with self._lock:
            self._stop.set()
        self.camera.stop()
        worker = self._worker
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        if worker and worker.is_alive():
            self._running = False
            self._send(MonitorUpdate("正在停止监控…", False))
            return
        self._running = False
        self._busy = False
        self._send(MonitorUpdate("监控已暂停", False))

    def _monitor_loop(self, template: list[float], owner_threshold: float) -> None:
        state = AlertStateMachine()
        last_analysis = 0.0
        identity_tracks: list[_IdentityTrack] = []
        try:
            self._engine = GazeEngine()
            if self._stop.is_set():
                return
            self.camera.start()
            self._running = True
            self._busy = False
            self._send(MonitorUpdate("监控中：画面仅在本机处理", False))
            while not self._stop.is_set():
                if self.camera.error:
                    raise CameraError(self.camera.error)
                frame = self.camera.latest_frame()
                now = time.monotonic()
                if frame is None or now - last_analysis < ANALYSIS_INTERVAL:
                    time.sleep(0.025)
                    continue
                last_analysis = now
                readings: list[FaceAnalysis] = self._engine.analyze(frame)
                identity = self._engine.identity
                identity_tracks = [track for track in identity_tracks if now - track.last_seen < 1.5]
                claimed_identity_tracks: set[int] = set()
                classified: list[tuple[bool, bool | None]] = []
                other_gaze_points: list[tuple[float, float]] = []
                other_count = 0
                uncertain_identity_count = 0
                unknown_gaze = False
                owner_seen = False
                owner_scores: list[float] = []
                bystander_scores: list[float] = []
                for reading in readings:
                    best_index, best_iou = None, 0.0
                    for index, track in enumerate(identity_tracks):
                        if index in claimed_identity_tracks:
                            continue
                        overlap = _box_iou(reading.sample.box, track.box)
                        if overlap > best_iou:
                            best_index, best_iou = index, overlap
                    if best_index is not None and best_iou >= 0.15:
                        track = identity_tracks[best_index]
                        claimed_identity_tracks.add(best_index)
                    else:
                        track = _IdentityTrack(reading.sample.box)
                        identity_tracks.append(track)
                        claimed_identity_tracks.add(len(identity_tracks) - 1)
                    track.box = reading.sample.box
                    track.last_seen = now

                    score = identity.similarity(template, reading.embedding)
                    track.scores.append(score)
                    stable_score = float(np.median(track.scores))
                    release_threshold = max(0.35, owner_threshold - 0.08)
                    if track.owner is True and stable_score >= release_threshold:
                        is_owner: bool | None = True
                    elif stable_score >= owner_threshold and len(track.scores) >= 3:
                        is_owner = True
                    elif stable_score < release_threshold:
                        is_owner = False
                    else:
                        is_owner = None
                    track.owner = is_owner

                    if is_owner is True:
                        owner_seen = True
                        owner_scores.append(stable_score)
                    else:
                        if is_owner is False:
                            other_count += 1
                        else:
                            uncertain_identity_count += 1
                        bystander_scores.append(stable_score)
                        unknown_gaze = unknown_gaze or reading.looking_at_screen is None
                        classified.append((True, reading.looking_at_screen))
                        if reading.gaze_point is not None:
                            other_gaze_points.append(reading.gaze_point)
                warning = state.update(classified, now)
                screen_facing_count = sum(1 for _, gaze in classified if gaze is True)
                if other_count or uncertain_identity_count:
                    unknown_count = sum(1 for _, gaze in classified if gaze is None)
                    status = (
                        f"监控中：其他人脸 {other_count}；身份待确认 {uncertain_identity_count}；"
                        f"朝向屏幕 {screen_facing_count}"
                    )
                    if bystander_scores:
                        status += f"；最高相似度 {max(bystander_scores):.2f}/{owner_threshold:.2f}"
                    if unknown_count:
                        status += f"；视线未知 {unknown_count}"
                    if self._engine.last_skipped_faces:
                        status += f"；另有 {self._engine.last_skipped_faces} 张未分析"
                    if other_gaze_points:
                        point_x, point_y = other_gaze_points[0]
                        status += f"；视线落点 ({point_x:.2f}, {point_y:.2f})"
                elif owner_seen:
                    if self._engine.last_skipped_faces:
                        status = (
                            f"监控中：本人已识别；另有 "
                            f"{self._engine.last_skipped_faces} 张脸未分析"
                        )
                    else:
                        score_text = f"；相似度 {max(owner_scores):.2f}/{owner_threshold:.2f}" if owner_scores else ""
                        status = f"监控中：本人已识别，未见其他人{score_text}"
                elif self._engine.last_face_count:
                    reason = self._engine.last_analysis_error
                    if reason:
                        reason = f"（{reason.split(':', 1)[0]}）"
                    status = (
                        f"监控中：检测到 {self._engine.last_face_count} 张脸，"
                        f"未能提取身份特征{reason}"
                    )
                else:
                    status = "监控中：等待可识别的人脸"
                self._send(
                    MonitorUpdate(
                        status,
                        warning,
                        other_count,
                        unknown_gaze,
                        screen_facing_count,
                        uncertain_identity_count,
                    )
                )
        except Exception as exc:
            self._running = False
            self._busy = False
            if not self._stop.is_set():
                self._send(MonitorUpdate(f"监控中断：{exc}", False))
        finally:
            self.camera.stop()
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception:
                    pass
                self._engine = None
            self._running = False
            self._busy = False
            if self._stop.is_set():
                self._send(MonitorUpdate("监控已暂停", False), force=True)
            else:
                self._send(self._latest_update, force=True)

    def enroll_owner(self) -> None:
        with self._lock:
            if self._running or self._busy:
                self._send(MonitorUpdate("请先暂停监控，再录入本人", self._latest_update.warning))
                return
            self._busy = True
            self._stop.clear()
            self._send(
                MonitorUpdate(
                    "正在录入本人：请保持光线充足，先正对摄像头，再轻微向左、向右转头；画面中只出现你…",
                    False,
                )
            )
            self._worker = threading.Thread(target=self._enroll_loop, name="nosee-enrollment", daemon=True)
            self._worker.start()

    def _enroll_loop(self) -> None:
        landmarker = None
        try:
            if not LANDMARKER_MODEL.is_file():
                raise FileNotFoundError("尚未安装人脸模型。请先选择“安装离线模型”。")
            self.camera.start()
            identity = FaceIdentity()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=4,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            embeddings: list[np.ndarray] = []
            last_sample = 0.0
            last_timestamp = -1
            deadline = time.monotonic() + 30.0
            while (
                len(embeddings) < ENROLLMENT_SAMPLES
                and time.monotonic() < deadline
                and not self._stop.is_set()
            ):
                frame = self.camera.latest_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                now = time.monotonic()
                if now - last_sample < ENROLLMENT_SAMPLE_INTERVAL:
                    time.sleep(0.03)
                    continue
                last_sample = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                last_timestamp = max(int(now * 1000), last_timestamp + 1)
                result = landmarker.detect_for_video(image, last_timestamp)
                faces = result.face_landmarks or []
                if not faces:
                    self._send(MonitorUpdate(f"录入本人：尚未找到人脸（{len(embeddings)}/{ENROLLMENT_SAMPLES}）", False))
                    continue
                if len(faces) != 1:
                    self._send(MonitorUpdate("录入本人：请让画面中只出现你一个人", False))
                    continue
                samples = [identity.sample_from_landmarks(frame, face) for face in faces]
                samples = [sample for sample in samples if sample is not None]
                if len(samples) != 1:
                    self._send(MonitorUpdate("录入本人：无法清楚定位人脸，请调整距离和光线", False))
                    continue
                embeddings.append(identity.embedding(frame, samples[0]))
                self._send(MonitorUpdate(f"录入本人：已采集 {len(embeddings)}/{ENROLLMENT_SAMPLES} 张样本", False))
            if len(embeddings) < ENROLLMENT_SAMPLES:
                if self._stop.is_set():
                    raise RuntimeError("录入已取消。")
                raise RuntimeError("录入超时。请改善光线并让脸部完整出现在画面中后重试。")
            within_scores = [
                identity.similarity(embeddings[left], embeddings[right])
                for left in range(len(embeddings))
                for right in range(left + 1, len(embeddings))
            ]
            within_median = float(np.median(within_scores))
            if within_median < 0.50:
                raise RuntimeError(
                    "录入样本之间差异较大，无法建立稳定的人脸档案。"
                    "请正对摄像头、改善光线并重新录入。"
                )
            template = np.mean(np.stack(embeddings), axis=0)
            template /= max(float(np.linalg.norm(template)), 1e-8)
            template_scores = [identity.similarity(template, sample) for sample in embeddings]
            lower_quartile = float(np.quantile(template_scores, 0.25))
            owner_threshold = float(np.clip(lower_quartile - 0.08, 0.50, 0.65))
            self.store.save_owner_embedding(
                template.astype(float).tolist(),
                owner_threshold,
                IDENTITY_EMBEDDING_VERSION,
            )
            self._send(
                MonitorUpdate(
                    f"本人资料已保存到此 Mac 的钥匙串；匹配阈值 {owner_threshold:.2f}；只保存特征，不保存原始画面",
                    False,
                )
            )
        except Exception as exc:
            self._send(MonitorUpdate(f"录入失败：{exc}", False))
        finally:
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
            self.camera.stop()
            self._busy = False
            self._send(self._latest_update, force=True)

    def clear_owner(self) -> None:
        try:
            self.store.clear_owner_embedding()
            self._send(MonitorUpdate("本人资料已从钥匙串清除", False))
        except Exception as exc:
            self._send(MonitorUpdate(f"清除本人资料失败：{exc}", False))


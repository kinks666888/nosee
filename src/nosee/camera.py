from __future__ import annotations

import threading
import time

import cv2


class CameraError(RuntimeError):
    pass


class CameraService:
    """One camera reader with a bounded, in-memory latest-frame slot."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        try:
            self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._capture.set(cv2.CAP_PROP_FPS, 15)
        except Exception as exc:
            self._capture = None
            raise CameraError(str(exc)) from exc
        if self._capture is None or not self._capture.isOpened():
            self._release_capture()
            raise CameraError(
                "无法打开摄像头。请在“系统设置 > 隐私与安全性 > 相机”允许当前 NoSee/终端应用访问摄像头。"
            )
        self._thread = threading.Thread(target=self._read_loop, name="nosee-camera", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                self._error = "摄像头停止传输画面。请检查摄像头是否被其他应用占用。"
                break
            with self._lock:
                self._latest_frame = frame
            time.sleep(0.005)

    def latest_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def error(self) -> str | None:
        return self._error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._release_capture()
        with self._lock:
            self._latest_frame = None

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


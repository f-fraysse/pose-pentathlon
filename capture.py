import threading
import time


class CaptureThread:
    """Dedicated thread for webcam capture — main loop never blocks on cap.read()."""

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._running = False
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame is not None, self._frame

    def wait_for_first_frame(self, timeout=5.0):
        """Block until the capture thread has grabbed at least one frame."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.01)
        return False

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

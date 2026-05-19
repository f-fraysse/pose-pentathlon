import time
import cv2

import config as cfg
from capture import CaptureThread
from pose import PoseDetector
from circuit import build_demo_circuit
import ui


def main():
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg.CAMERA_INDEX}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {actual_w}x{actual_h}")

    cap_thread = CaptureThread(cap)
    if not cap_thread.wait_for_first_frame(timeout=5.0):
        cap_thread.stop()
        raise RuntimeError("Capture thread never produced a frame")

    detector = PoseDetector(cfg)
    circuit = build_demo_circuit()

    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)

    # FPS tracking — EMA
    fps_ema = 30.0
    last_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap_thread.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            results = detector.detect(frame, fps_ema)

            circuit.draw_skeleton(frame, results)
            circuit.update(results)
            circuit.draw(frame)
            ui.draw_fps(frame, fps_ema)

            cv2.imshow(cfg.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and circuit.on_key(key) == "quit":
                break

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * fps_inst
    finally:
        cap_thread.stop()
        cv2.destroyAllWindows()
        print(f"Final smoothed FPS: {fps_ema:.1f}")


if __name__ == "__main__":
    main()

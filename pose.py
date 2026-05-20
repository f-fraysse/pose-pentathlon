import numpy as np
from collections import deque, namedtuple
from rtmlib import Body, Wholebody

PoseResult = namedtuple('PoseResult', ['position', 'velocity', 'prev_position'])


class OneEuroFilter:
    """Speed-adaptive low-pass filter for 2D keypoint positions.

    Heavy smoothing at rest; light smoothing during fast gestures.
    Reference: https://inria.hal.science/hal-00670496/document
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x_raw, dt):
        if self._x_prev is None:
            self._x_prev = x_raw.copy()
            return x_raw.copy()
        dx_raw = (x_raw - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_prev
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a_x = self._alpha(cutoff, dt)
        x_hat = a_x * x_raw + (1.0 - a_x) * self._x_prev
        self._dx_prev = dx_hat
        self._x_prev = x_hat
        return x_hat


def calculate_velocity_direct(history):
    if len(history) < 2:
        return np.zeros(2, dtype=np.float32)
    return np.array(history[-1], dtype=np.float32) - np.array(history[-2], dtype=np.float32)


def calculate_velocity_ma(history, window_size=5):
    window_size = min(len(history), window_size, 10)
    if window_size < 2:
        return np.zeros(2, dtype=np.float32)
    recent = [np.array(pos, dtype=np.float32) for pos in list(history)[-window_size:]]
    velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    return sum(velocities) / len(velocities)


class PoseDetector:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.TRACKING_MODE == "body":
            self.model = Body(
                mode='lightweight',
                pose='rtmo',
                backend=cfg.POSE_BACKEND,
                device=cfg.POSE_DEVICE)
            self.model.pose_model.score_thr = cfg.RTMO_SCORE_THR
        else:
            self.model = Wholebody(
                mode='lightweight',
                backend=cfg.POSE_BACKEND,
                device=cfg.POSE_DEVICE)

        self.keypoint_selection = cfg.KEYPOINT_SELECTION
        self.keypoint_names = cfg.KEYPOINT_NAMES

        enabled_kpts = [cfg.KEYPOINT_NAMES[k]
                        for k, v in cfg.KEYPOINT_SELECTION.items() if v]
        print(f"[Tracking mode: {cfg.TRACKING_MODE}] Model initialized.")
        print(f"  Enabled keypoints ({len(enabled_kpts)}): {enabled_kpts}")

        self._history = {}
        self._smoothers = {}
        self._vel = {}

        if cfg.VELOCITY_CALCULATION_METHOD == "direct":
            self._calc_vel = lambda h: calculate_velocity_direct(h)
        else:
            self._calc_vel = lambda h: calculate_velocity_ma(h, cfg.VELOCITY_MA_WINDOW_SIZE)

    def detect(self, frame, fps):
        """Run pose detection and return results for enabled keypoints.

        Returns:
            dict of {kpt_idx: PoseResult(position, velocity, prev_position)}
        """
        keypoints, scores = self.model(frame)

        results = {}
        num_instances = min(1, keypoints.shape[0]) if keypoints.shape[0] > 0 else 0
        if num_instances == 0:
            return results

        dt_sec = 1.0 / fps if fps > 0 else 1.0 / 30.0

        for kpt_idx, enabled in self.keypoint_selection.items():
            if not enabled:
                continue

            score = float(scores[0, kpt_idx])
            if score < self.cfg.SCORE_THRESHOLD:
                continue

            kpt_pos = np.array([keypoints[0, kpt_idx, 0],
                                keypoints[0, kpt_idx, 1]], dtype=np.float32)

            if kpt_idx not in self._history:
                self._history[kpt_idx] = deque(maxlen=self.cfg.KEYPOINT_TRAIL_LENGTH)
                self._smoothers[kpt_idx] = OneEuroFilter(
                    self.cfg.ONE_EURO_MIN_CUTOFF,
                    self.cfg.ONE_EURO_BETA,
                    self.cfg.ONE_EURO_DCUTOFF,
                )
                self._vel[kpt_idx] = np.zeros(2, dtype=np.float32)

            if self.cfg.KEYPOINT_SMOOTHING_ENABLED:
                kpt_pos = self._smoothers[kpt_idx](kpt_pos, dt_sec)

            prev_pos = (np.array(self._history[kpt_idx][-1], dtype=np.float32)
                        if len(self._history[kpt_idx]) > 0 else None)

            self._history[kpt_idx].append((float(kpt_pos[0]), float(kpt_pos[1])))

            vel = self._calc_vel(self._history[kpt_idx])
            self._vel[kpt_idx] = vel

            results[kpt_idx] = PoseResult(
                position=kpt_pos,
                velocity=vel,
                prev_position=prev_pos,
            )

        return results

    def get_history(self, kpt_idx):
        """Return the position history deque for a keypoint, or None."""
        return self._history.get(kpt_idx, None)

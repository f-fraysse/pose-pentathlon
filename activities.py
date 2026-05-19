from collections import deque
from statistics import median

import config as cfg
import ui


class HighKneesActivity:
    name = "High Knees"
    instruction_text = "Run on the spot. Lift those knees as high as you can!"
    instruction_image = None
    duration_s = 20.0

    # Hip / knee / ankle on both sides
    HL_JOINTS = {11, 12, 13, 14, 15, 16}
    HL_BONES = {(11, 13), (13, 15), (12, 14), (14, 16)}

    UP_THRESHOLD = 0.20      # gap_norm < UP   -> leg counted as "up"
    DOWN_THRESHOLD = 0.40    # gap_norm > DOWN -> leg counted as "down"

    FREQ_WINDOW_S = 3.0      # rolling window for the power bar's frequency
    MAX_FREQ_HZ = 4.0        # 80 reps over 20s -> bar maxes at 4 reps/sec

    def __init__(self):
        self.reps = 0
        self._left = "down"
        self._right = "down"
        self._rep_times = deque()
        self.current_freq = 0.0
        self.peak_freq = 0.0

    def reset(self):
        self.reps = 0
        self._left = "down"
        self._right = "down"
        self._rep_times = deque()
        self.current_freq = 0.0
        self.peak_freq = 0.0

    def update(self, keypoints, t_elapsed):
        self._step(keypoints, t_elapsed, hip=11, knee=13, ankle=15, side='left')
        self._step(keypoints, t_elapsed, hip=12, knee=14, ankle=16, side='right')

        cutoff = t_elapsed - self.FREQ_WINDOW_S
        while self._rep_times and self._rep_times[0] < cutoff:
            self._rep_times.popleft()
        window = max(1.0, min(t_elapsed, self.FREQ_WINDOW_S))
        self.current_freq = len(self._rep_times) / window
        if self.current_freq > self.peak_freq:
            self.peak_freq = self.current_freq

    def _step(self, kpts, t_now, hip, knee, ankle, side):
        if hip not in kpts or knee not in kpts or ankle not in kpts:
            return
        hp = kpts[hip].position
        kp = kpts[knee].position
        ap = kpts[ankle].position
        leg_len = ((hp[0] - ap[0]) ** 2 + (hp[1] - ap[1]) ** 2) ** 0.5
        if leg_len < 1e-3:
            return
        gap_norm = (kp[1] - hp[1]) / leg_len  # screen y grows down

        attr = f"_{side}"
        cur = getattr(self, attr)
        if cur == "down" and gap_norm < self.UP_THRESHOLD:
            setattr(self, attr, "up")
            self.reps += 1
            self._rep_times.append(t_now)
        elif cur == "up" and gap_norm > self.DOWN_THRESHOLD:
            setattr(self, attr, "down")

    def draw(self, frame):
        cur_frac = self.current_freq / self.MAX_FREQ_HZ
        peak_frac = self.peak_freq / self.MAX_FREQ_HZ
        ui.draw_power_bar(frame, ui.left_power_bar_rect(frame),
                          cur_frac, peak_frac=peak_frac)

    def is_finished(self, t_elapsed):
        return t_elapsed >= self.duration_s

    def get_result(self):
        raw_min, raw_max = cfg.SCORE_MAP["high_knees"]
        f = (self.reps - raw_min) / max(1e-6, (raw_max - raw_min))
        f = max(0.0, min(1.0, f))
        pts = int(round(f * 1000))
        return {"points": pts, "raw": self.reps, "display_str": f"{self.reps}"}

    def highlight(self):
        return {"joints": self.HL_JOINTS, "bones": self.HL_BONES}


class VerticalJumpActivity:
    name = "Vertical Jump"
    instruction_text = "Crouch and jump as high as you can. Try a few times - your best counts!"
    instruction_image = None
    duration_s = 15.0

    # Hips + ankles + leg chains: measurement points and the legs that drive the jump
    HL_JOINTS = {11, 12, 15, 16}
    HL_BONES = {(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)}

    BASELINE_WINDOW_S = 1.0   # collect standing samples for the first second of ACTIVITY

    def __init__(self):
        self._baseline_hip_y_samples = []
        self._leg_len_samples = []
        self.baseline_hip_y = None
        self.leg_len = None
        self.peak_ratio = 0.0
        self.current_ratio = 0.0

    def reset(self):
        self._baseline_hip_y_samples = []
        self._leg_len_samples = []
        self.baseline_hip_y = None
        self.leg_len = None
        self.peak_ratio = 0.0
        self.current_ratio = 0.0

    def _hip_centre_and_leg_len(self, kpts):
        """Return (hip_y, leg_len) using whichever side(s) are visible.
        hip_y: mean of available hip Ys. leg_len: mean of |hip - ankle| for the
        side(s) where both endpoints are visible. Returns (None, None) if neither
        side is usable."""
        hips_y = []
        legs = []
        for hip, ankle in ((11, 15), (12, 16)):
            if hip in kpts:
                hips_y.append(float(kpts[hip].position[1]))
            if hip in kpts and ankle in kpts:
                hp = kpts[hip].position
                ap = kpts[ankle].position
                dx = float(hp[0] - ap[0])
                dy = float(hp[1] - ap[1])
                legs.append((dx * dx + dy * dy) ** 0.5)
        if not hips_y or not legs:
            return None, None
        return sum(hips_y) / len(hips_y), sum(legs) / len(legs)

    def update(self, keypoints, t_elapsed):
        hip_y, leg_len = self._hip_centre_and_leg_len(keypoints)
        if hip_y is None:
            return

        if t_elapsed < self.BASELINE_WINDOW_S:
            self._baseline_hip_y_samples.append(hip_y)
            self._leg_len_samples.append(leg_len)
            return

        # Finalise baseline on first post-window frame
        if self.baseline_hip_y is None:
            if not self._baseline_hip_y_samples:
                return  # never saw a usable frame during baseline — try this one
            self.baseline_hip_y = median(self._baseline_hip_y_samples)
            self.leg_len = median(self._leg_len_samples)

        if self.leg_len is None or self.leg_len < 1e-3:
            return

        rise = self.baseline_hip_y - hip_y          # screen y grows down -> rise positive when hip moves up
        jump_ratio = rise / self.leg_len
        self.current_ratio = jump_ratio
        if jump_ratio > self.peak_ratio:
            self.peak_ratio = jump_ratio

    def draw(self, frame):
        raw_min, raw_max = cfg.SCORE_MAP["vertical_jump"]
        span = max(1e-6, raw_max - raw_min)
        cur_frac = (self.current_ratio - raw_min) / span
        peak_frac = (self.peak_ratio - raw_min) / span
        ui.draw_power_bar(frame, ui.left_power_bar_rect(frame),
                          cur_frac, peak_frac=peak_frac)

    def is_finished(self, t_elapsed):
        return t_elapsed >= self.duration_s

    def get_result(self):
        raw_min, raw_max = cfg.SCORE_MAP["vertical_jump"]
        f = (self.peak_ratio - raw_min) / max(1e-6, (raw_max - raw_min))
        f = max(0.0, min(1.0, f))
        pts = int(round(f * 1000))
        return {"points": pts, "raw": self.peak_ratio, "display_str": f"{pts}"}

    def highlight(self):
        return {"joints": self.HL_JOINTS, "bones": self.HL_BONES}

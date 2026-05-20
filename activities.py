import math
import random
from collections import deque
from statistics import median

import cv2

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


class ReactionWallActivity:
    name = "Reaction Wall"
    instruction_text = "Touch the targets as fast as you can - use both hands!"
    instruction_image = None
    duration_s = 20.0

    # Shoulders, elbows, wrists
    HL_JOINTS = {5, 6, 7, 8, 9, 10}
    HL_BONES = {(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)}

    MAX_TARGETS = 2
    TARGET_RADIUS_FRAC = 0.06
    HIT_RADIUS_MULT = 1.3          # wrist within this * visual radius counts as a hit
    SPAWN_SIDES_FRAC = 0.20
    SPAWN_TOP_FRAC = 0.10
    SPAWN_BOTTOM_FRAC = 0.70
    MIN_SEPARATION_FRAC = 0.20
    SPAWN_TRIES = 20

    def __init__(self):
        self.hits = 0
        self._targets = []  # list of (x, y) pixel centres
        self._rng = random.Random()
        self._last_radius = 50
        self._pulse_t = 0.0

    def reset(self):
        self.hits = 0
        self._targets = []
        self._pulse_t = 0.0

    def update(self, keypoints, t_elapsed):
        if not self._targets:
            return
        wrists = []
        for wid in (9, 10):
            if wid in keypoints:
                wrists.append(keypoints[wid].position)
        if not wrists:
            return

        r = self._last_radius * self.HIT_RADIUS_MULT
        r2 = r * r
        survivors = []
        for (tx, ty) in self._targets:
            hit = False
            for w in wrists:
                dx = float(w[0]) - tx
                dy = float(w[1]) - ty
                if dx * dx + dy * dy <= r2:
                    hit = True
                    break
            if hit:
                self.hits += 1
            else:
                survivors.append((tx, ty))
        self._targets = survivors

    def _spawn_target(self, frame_w, frame_h, radius):
        x_min = int(self.SPAWN_SIDES_FRAC * frame_w + radius)
        x_max = int((1.0 - self.SPAWN_SIDES_FRAC) * frame_w - radius)
        y_min = int(self.SPAWN_TOP_FRAC * frame_h + radius)
        y_max = int(self.SPAWN_BOTTOM_FRAC * frame_h - radius)
        if x_max <= x_min or y_max <= y_min:
            return None
        min_sep = self.MIN_SEPARATION_FRAC * frame_h
        min_sep2 = min_sep * min_sep
        for _ in range(self.SPAWN_TRIES):
            x = self._rng.randint(x_min, x_max)
            y = self._rng.randint(y_min, y_max)
            ok = True
            for (tx, ty) in self._targets:
                if (x - tx) ** 2 + (y - ty) ** 2 < min_sep2:
                    ok = False
                    break
            if ok:
                return (x, y)
        return None

    def draw(self, frame):
        h, w = frame.shape[:2]
        radius = max(8, int(self.TARGET_RADIUS_FRAC * h))
        self._last_radius = radius

        while len(self._targets) < self.MAX_TARGETS:
            pos = self._spawn_target(w, h, radius)
            if pos is None:
                break
            self._targets.append(pos)

        self._pulse_t += 0.05
        pulse = 1.0 + 0.08 * math.sin(8.0 * self._pulse_t)
        r_draw = max(6, int(radius * pulse))
        for (tx, ty) in self._targets:
            cv2.circle(frame, (tx, ty), r_draw, cfg.COL_HUD_BAR,
                       thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), r_draw, (20, 20, 20),
                       thickness=2, lineType=cv2.LINE_AA)

        raw_min, raw_max = cfg.SCORE_MAP["reaction_wall"]
        span = max(1e-6, raw_max - raw_min)
        cur_frac = (self.hits - raw_min) / span
        ui.draw_power_bar(frame, ui.left_power_bar_rect(frame), cur_frac)

    def is_finished(self, t_elapsed):
        return t_elapsed >= self.duration_s

    def get_result(self):
        raw_min, raw_max = cfg.SCORE_MAP["reaction_wall"]
        f = (self.hits - raw_min) / max(1e-6, (raw_max - raw_min))
        f = max(0.0, min(1.0, f))
        pts = int(round(f * 1000))
        return {"points": pts, "raw": self.hits, "display_str": f"{self.hits}"}

    def highlight(self):
        return {"joints": self.HL_JOINTS, "bones": self.HL_BONES}


class PunchPowerActivity:
    name = "Punch Power"
    instruction_text = "Throw your hardest punches - either fist counts!"
    instruction_image = None
    duration_s = 10.0

    # Shoulders, elbows, wrists
    HL_JOINTS = {5, 6, 7, 8, 9, 10}
    HL_BONES = {(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)}

    # We deviate from design doc 8.4: it specifies windowed integrated
    # displacement, but that introduced a ~0.3s lag between the punch and
    # the bar filling. Instead use the per-frame horizontal wrist velocity
    # from PoseDetector (already MA-smoothed over 5 frames) normalised by
    # arm length. Horizontal-only so vertical chops don't count.
    # Units: arm-lengths per frame (so SCORE_MAP scale is FPS-dependent).

    # Deadband below which per-side speed is treated as zero — masks the
    # residual jitter on a still wrist after OneEuro smoothing. Tune up if
    # standing still still shows bar; tune down if slow punches don't
    # register.
    MIN_SPEED = 0.02

    # M7 polish ideas, deliberately deferred so M6a stays cheap:
    #   - Wrist motion trail (fading polyline of last ~0.3s of wrist positions).
    #   - Floating live-energy number near the leading wrist.
    #   - Brief screen flash when current energy crosses a "big punch" threshold.

    def _new_side(self, sh, el, wr):
        return {
            "sh": sh, "el": el, "wr": wr,
            "arm_len": 0.0,        # running mean of |sh-el| + |el-wr|
            "arm_n": 0,
        }

    def __init__(self):
        self._sides = {
            "L": self._new_side(5, 7, 9),
            "R": self._new_side(6, 8, 10),
        }
        self.current_energy = 0.0
        self.peak_energy = 0.0

    def reset(self):
        self._sides = {
            "L": self._new_side(5, 7, 9),
            "R": self._new_side(6, 8, 10),
        }
        self.current_energy = 0.0
        self.peak_energy = 0.0

    def update(self, keypoints, t_elapsed):
        speeds = []
        for s in self._sides.values():
            sh, el, wr = s["sh"], s["el"], s["wr"]
            # Update arm-length running mean when the full chain is visible
            if sh in keypoints and el in keypoints and wr in keypoints:
                shp = keypoints[sh].position
                elp = keypoints[el].position
                wrp = keypoints[wr].position
                upper = ((shp[0] - elp[0]) ** 2 + (shp[1] - elp[1]) ** 2) ** 0.5
                fore  = ((elp[0] - wrp[0]) ** 2 + (elp[1] - wrp[1]) ** 2) ** 0.5
                seg = float(upper + fore)
                s["arm_n"] += 1
                s["arm_len"] += (seg - s["arm_len"]) / s["arm_n"]

            # |horizontal velocity| / arm length, skip until arm_len bootstrapped
            if wr in keypoints and s["arm_len"] > 1e-3:
                vx = float(keypoints[wr].velocity[0])
                speed = abs(vx) / s["arm_len"]
                if speed >= self.MIN_SPEED:
                    speeds.append(speed)

        self.current_energy = max(speeds) if speeds else 0.0
        if self.current_energy > self.peak_energy:
            self.peak_energy = self.current_energy

    def draw(self, frame):
        raw_min, raw_max = cfg.SCORE_MAP["punch_power"]
        span = max(1e-6, raw_max - raw_min)
        cur_frac = (self.current_energy - raw_min) / span
        peak_frac = (self.peak_energy - raw_min) / span
        ui.draw_power_bar(frame, ui.left_power_bar_rect(frame),
                          cur_frac, peak_frac=peak_frac)

    def is_finished(self, t_elapsed):
        return t_elapsed >= self.duration_s

    def get_result(self):
        raw_min, raw_max = cfg.SCORE_MAP["punch_power"]
        f = (self.peak_energy - raw_min) / max(1e-6, (raw_max - raw_min))
        f = max(0.0, min(1.0, f))
        pts = int(round(f * 1000))
        return {"points": pts, "raw": self.peak_energy, "display_str": f"{pts}"}

    def highlight(self):
        return {"joints": self.HL_JOINTS, "bones": self.HL_BONES}

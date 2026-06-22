import math
import random
from collections import deque
from enum import Enum, auto
from statistics import median

import cv2

import config as cfg
import ui


class _StickPhase(Enum):
    STANCE = auto()
    HOP    = auto()
    LAND   = auto()
    DONE   = auto()


class HighKneesActivity:
    name = "High Knees"
    instruction_text = "Run on the spot. Lift those knees up!"
    instruction_image = None
    duration_s = 10.0

    # Hip / knee / ankle on both sides
    HL_JOINTS = {11, 12, 13, 14, 15, 16}
    HL_BONES = {(11, 13), (13, 15), (12, 14), (14, 16)}

    UP_THRESHOLD = 0.20      # gap_norm < UP   -> leg counted as "up"
    DOWN_THRESHOLD = 0.40    # gap_norm > DOWN -> leg counted as "down"

    FREQ_WINDOW_S = 2.0      # rolling window for the power bar's frequency
    MAX_FREQ_HZ = 5.0        # 80 reps over 20s -> bar maxes at 4 reps/sec

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
    instruction_text = "Crouch and jump as high as you can. Best try counts!"
    instruction_image = None
    duration_s = 10.0

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
    instruction_text = "Touch the targets as fast as you can!"
    instruction_image = None
    duration_s = 10.0

    # Shoulders, elbows, wrists
    HL_JOINTS = {5, 6, 7, 8, 9, 10}
    HL_BONES = {(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)}

    MAX_TARGETS = 2
    TARGET_RADIUS_FRAC = 0.06
    HIT_RADIUS_MULT = 1.4          # wrist within this * visual radius counts as a hit
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
    instruction_text = "Throw your hardest punch!"
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


class StickTheLandingActivity:
    """3-phase composite event: single-leg stance -> sideways hop -> stick the
    landing. No SCORE_MAP entry — the activity computes a composite 0..1
    quality (40% balance + 40% landing stability + 20% accuracy) internally
    and converts to points directly."""

    name = "Stick the Landing"
    instruction_text = ("Stand on one leg, hop to the target, "
                        "then stick the landing!")
    instruction_image = None
    duration_s = 12.0   # hard cap: 5 (stance budget) + 4 (hop) + 3 (land)

    # Whole body highlight — sway is now measured across hips+knees+elbows+wrists
    HL_JOINTS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    HL_BONES  = {(5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                 (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)}

    # Keypoints whose sway counts toward the balance/landing stillness score.
    # Equal weight per keypoint; flailing arms therefore tank stillness.
    TRACKED_KPTS = (7, 8, 9, 10, 11, 12, 13, 14)

    # Phase budgets and detection thresholds (all leg-length-normalised)
    STANCE_HOLD_S         = 2.0
    STANCE_TIMEOUT_S      = 5.0
    HOP_TIMEOUT_S         = 4.0
    LAND_HOLD_S           = 3.0
    LAND_SKIP_FRAMES      = 10     # ignore the first N LAND frames so the
                                  # impact spike doesn't tank the sway score
    STANCE_ANKLE_DIFF_THR = 0.25
    HOP_AIRBORNE_THR      = 0.08
    HOP_LAND_THR          = 0.04
    TARGET_DIST_FRAC      = 0.75   # × leg_len, distance from baseline_hip_x to target
    TARGET_RADIUS         = 0.25
    BALANCE_STD_BAD       = 0.08
    LAND_STD_BAD          = 0.21
    ACCURACY_HOT_DIST     = 0.25
    ACCURACY_COLD_DIST    = 0.75
    STEADY_WINDOW_S       = 0.5

    def __init__(self):
        self.reset()

    def reset(self):
        self._phase = _StickPhase.STANCE
        self._phase_started_t = 0.0
        # STANCE phase state
        self._stance_started_t = None
        # Per-keypoint sample dicts: {kpt_idx: [(x, y), ...]}
        self._stance_samples = {}
        self._land_samples = {}
        # Shared (computed at end of STANCE, used in HOP+LAND)
        self.standing_leg = None      # 'L' or 'R'
        self.ground_y = None
        self.baseline_hip_x = None
        self.leg_len = None
        self.target_x = None
        # HOP phase
        self._airborne_seen = False
        self.landing_x = None      # standing-ankle x at the landing instant
        # LAND phase
        self._land_frame_n = 0     # frames seen in LAND; first N are skipped
        # Live steadiness deque (for the power bar in STANCE/LAND)
        # Element shape: (t, {kpt_idx: (x, y)})
        self._steady_samples = deque()
        # Sub-scores (default 0 so graceful degrade is automatic)
        self.balance_quality = 0.0
        self.land_stability_quality = 0.0
        self.accuracy_quality = 0.0
        # Last valid lower-body obs — used as fallback on STANCE timeout
        self._last_lower = None
        # Decorative pulse accumulator for the hop target
        self._pulse_t = 0.0
        # Temporary: print sub-scores once per run for lab tuning
        self._printed = False

    # ── Phase machinery ──────────────────────────────────────────────────

    def _enter_phase(self, new_phase, t_elapsed):
        self._phase = new_phase
        self._phase_started_t = t_elapsed

    def _lower_body_obs(self, kpts):
        """Return (hip_centre, leg_len, ankle_l, ankle_r) or None if missing."""
        if 11 not in kpts or 12 not in kpts:
            return None
        if 15 not in kpts or 16 not in kpts:
            return None
        hl = kpts[11].position
        hr = kpts[12].position
        al = kpts[15].position
        ar = kpts[16].position
        hip_centre = ((float(hl[0]) + float(hr[0])) * 0.5,
                      (float(hl[1]) + float(hr[1])) * 0.5)
        legs = []
        for hip, ankle in ((hl, al), (hr, ar)):
            dx = float(hip[0] - ankle[0])
            dy = float(hip[1] - ankle[1])
            legs.append((dx * dx + dy * dy) ** 0.5)
        leg_len = sum(legs) / len(legs)
        return (hip_centre, leg_len,
                (float(al[0]), float(al[1])),
                (float(ar[0]), float(ar[1])))

    def _extract_tracked(self, kpts):
        """Return {kpt_idx: (x, y)} for whichever TRACKED_KPTS are present."""
        out = {}
        for k in self.TRACKED_KPTS:
            if k in kpts:
                p = kpts[k].position
                out[k] = (float(p[0]), float(p[1]))
        return out

    # ── update / draw entry points ───────────────────────────────────────

    def update(self, keypoints, t_elapsed):
        # Hard duration cap — if we run out the clock mid-phase, finalise
        # whatever sub-score is still pending and transition to DONE.
        if t_elapsed >= self.duration_s and self._phase is not _StickPhase.DONE:
            if self._phase is _StickPhase.LAND and self._land_samples:
                self._finalise_land()
            self._phase = _StickPhase.DONE
            return

        obs = self._lower_body_obs(keypoints)
        tracked = self._extract_tracked(keypoints)
        if obs is not None:
            self._last_lower = obs
        if tracked:
            self._steady_samples.append((t_elapsed, tracked))
            cutoff = t_elapsed - self.STEADY_WINDOW_S
            while self._steady_samples and self._steady_samples[0][0] < cutoff:
                self._steady_samples.popleft()

        if self._phase is _StickPhase.STANCE:
            self._update_stance(obs, tracked, t_elapsed)
        elif self._phase is _StickPhase.HOP:
            self._update_hop(obs, t_elapsed)
        elif self._phase is _StickPhase.LAND:
            self._update_land(tracked, t_elapsed)

    def _update_stance(self, obs, tracked, t_elapsed):
        if obs is None:
            self._stance_started_t = None
            self._stance_samples = {}
        else:
            leg_len, al, ar = obs[1], obs[2], obs[3]
            ankle_diff = abs(al[1] - ar[1])
            valid = ankle_diff > self.STANCE_ANKLE_DIFF_THR * leg_len
            if valid:
                if self._stance_started_t is None:
                    self._stance_started_t = t_elapsed
                    self._stance_samples = {}
                for k, p in tracked.items():
                    self._stance_samples.setdefault(k, []).append(p)
                if (t_elapsed - self._stance_started_t) >= self.STANCE_HOLD_S:
                    self._finalise_stance(obs)
                    self._enter_phase(_StickPhase.HOP, t_elapsed)
                    return
            else:
                self._stance_started_t = None
                self._stance_samples = {}

        if (t_elapsed - self._phase_started_t) >= self.STANCE_TIMEOUT_S:
            self.balance_quality = 0.0
            self._init_hop_defaults()
            self._enter_phase(_StickPhase.HOP, t_elapsed)

    def _finalise_stance(self, obs):
        leg_len, al, ar = obs[1], obs[2], obs[3]
        # Larger y = lower on screen = grounded foot.
        if al[1] >= ar[1]:
            self.standing_leg = 'L'
            self.ground_y = al[1]
        else:
            self.standing_leg = 'R'
            self.ground_y = ar[1]
        self.leg_len = leg_len
        # baseline_hip_x = mean of x across hip samples collected during stance
        hip_xs = []
        for k in (11, 12):
            if k in self._stance_samples:
                hip_xs.extend(p[0] for p in self._stance_samples[k])
        self.baseline_hip_x = (sum(hip_xs) / len(hip_xs)) if hip_xs else obs[0][0]
        # Multi-keypoint sway, leg-length-normalised
        sway_norm = self._multi_kp_sway(self._stance_samples, leg_len)
        self.balance_quality = max(0.0, min(1.0,
            1.0 - sway_norm / self.BALANCE_STD_BAD))
        # Hop towards the standing-leg side (target on the SAME side as the
        # grounded foot). kpt 15 = L ankle on image-left after mirror; for
        # the user, "standing on left leg" -> image-left -> direction -1.
        direction = -1.0 if self.standing_leg == 'L' else 1.0
        self.target_x = self.baseline_hip_x + direction * self.TARGET_DIST_FRAC * leg_len

    def _init_hop_defaults(self):
        """Seed HOP-phase state from last-known obs after a STANCE timeout."""
        if self._last_lower is None:
            self.standing_leg = 'L'
            self.leg_len = 100.0
            self.ground_y = 500.0
            self.baseline_hip_x = 400.0
        else:
            hip_centre, leg_len, al, ar = self._last_lower
            if al[1] >= ar[1]:
                self.standing_leg = 'L'
                self.ground_y = al[1]
            else:
                self.standing_leg = 'R'
                self.ground_y = ar[1]
            self.leg_len = leg_len
            self.baseline_hip_x = hip_centre[0]
        direction = -1.0 if self.standing_leg == 'L' else 1.0
        self.target_x = self.baseline_hip_x + direction * self.TARGET_DIST_FRAC * self.leg_len

    def _update_hop(self, obs, t_elapsed):
        if obs is not None and self.ground_y is not None and self.leg_len is not None:
            standing_ankle = obs[2] if self.standing_leg == 'L' else obs[3]
            if standing_ankle[1] < self.ground_y - self.HOP_AIRBORNE_THR * self.leg_len:
                self._airborne_seen = True
            elif (self._airborne_seen
                  and standing_ankle[1] >= self.ground_y - self.HOP_LAND_THR * self.leg_len):
                self.landing_x = standing_ankle[0]
                self._compute_accuracy()
                self._enter_phase(_StickPhase.LAND, t_elapsed)
                return

        if (t_elapsed - self._phase_started_t) >= self.HOP_TIMEOUT_S:
            self.accuracy_quality = 0.0
            self._enter_phase(_StickPhase.LAND, t_elapsed)

    def _compute_accuracy(self):
        if (self.landing_x is None
                or self.target_x is None
                or self.leg_len is None
                or self.leg_len < 1e-3):
            self.accuracy_quality = 0.0
            return
        dist_norm = abs(self.landing_x - self.target_x) / self.leg_len
        if dist_norm <= self.ACCURACY_HOT_DIST:
            self.accuracy_quality = 1.0
        elif dist_norm >= self.ACCURACY_COLD_DIST:
            self.accuracy_quality = 0.0
        else:
            span = self.ACCURACY_COLD_DIST - self.ACCURACY_HOT_DIST
            self.accuracy_quality = 1.0 - (dist_norm - self.ACCURACY_HOT_DIST) / span

    def _update_land(self, tracked, t_elapsed):
        self._land_frame_n += 1
        # Skip the first few frames so the post-landing impact spike doesn't
        # poison the sway average.
        if tracked and self._land_frame_n > self.LAND_SKIP_FRAMES:
            for k, p in tracked.items():
                self._land_samples.setdefault(k, []).append(p)
        if (t_elapsed - self._phase_started_t) >= self.LAND_HOLD_S:
            self._finalise_land()
            self._enter_phase(_StickPhase.DONE, t_elapsed)

    def _finalise_land(self):
        if not self._land_samples or self.leg_len is None or self.leg_len < 1e-3:
            self.land_stability_quality = 0.0
            return
        sway_norm = self._multi_kp_sway(self._land_samples, self.leg_len)
        if not math.isfinite(sway_norm):
            self.land_stability_quality = 0.0
            return
        self.land_stability_quality = max(0.0, min(1.0,
            1.0 - sway_norm / self.LAND_STD_BAD))

    def _multi_kp_sway(self, samples_dict, leg_len):
        """Mean per-keypoint (stdev_x + stdev_y) across keypoints, divided by
        leg_len. Keypoints with <2 samples are skipped. Returns +inf if no
        keypoint contributes (caller treats as "no data, score 0")."""
        if leg_len < 1e-3:
            return float("inf")
        per_kp = []
        for pts in samples_dict.values():
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            per_kp.append(self._stdev(xs) + self._stdev(ys))
        if not per_kp:
            return float("inf")
        return (sum(per_kp) / len(per_kp)) / leg_len

    @staticmethod
    def _stdev(values):
        n = len(values)
        if n < 2:
            return 0.0
        m = sum(values) / n
        var = sum((v - m) ** 2 for v in values) / n
        return var ** 0.5

    def _live_steadiness_frac(self):
        if len(self._steady_samples) < 2:
            return 0.0
        # Use the finalised leg_len after stance, else the latest observation
        # so the bar is live during the stance phase too.
        leg_len = self.leg_len
        if leg_len is None and self._last_lower is not None:
            leg_len = self._last_lower[1]
        if leg_len is None or leg_len < 1e-3:
            return 0.0
        # Re-shard the rolling deque into per-keypoint series
        per_kp = {}
        for entry in self._steady_samples:
            for k, p in entry[1].items():
                per_kp.setdefault(k, []).append(p)
        sway_norm = self._multi_kp_sway(per_kp, leg_len)
        if not math.isfinite(sway_norm):
            return 0.0
        thr = self.LAND_STD_BAD if self._phase is _StickPhase.LAND else self.BALANCE_STD_BAD
        return max(0.0, min(1.0, 1.0 - sway_norm / thr))

    # ── Drawing ──────────────────────────────────────────────────────────

    def draw(self, frame):
        h = frame.shape[0]

        # Steadiness power bar — only while we're measuring (STANCE + LAND)
        if self._phase in (_StickPhase.STANCE, _StickPhase.LAND):
            ui.draw_power_bar(frame, ui.left_power_bar_rect(frame),
                              self._live_steadiness_frac())

        scale_heading = ui._scale_for(h, cfg.UI_HEADING_FRAC)
        if self._phase is _StickPhase.STANCE:
            ui.text_centered(frame, "Stand on one leg!", y=int(h * 0.18),
                             scale=scale_heading, color=cfg.COL_PRIMARY,
                             panel_pad=14)
        elif self._phase is _StickPhase.HOP:
            self._draw_hop_overlay(frame)
        elif self._phase is _StickPhase.LAND:
            ui.text_centered(frame, "Stick it!", y=int(h * 0.18),
                             scale=scale_heading, color=cfg.COL_PRIMARY,
                             panel_pad=14)

    def _draw_hop_overlay(self, frame):
        h = frame.shape[0]
        if (self.target_x is not None
                and self.ground_y is not None
                and self.leg_len is not None):
            tx = int(self.target_x)
            ty = int(self.ground_y)
            r = max(8, int(self.TARGET_RADIUS * self.leg_len))
            self._pulse_t += 0.05
            pulse = 1.0 + 0.10 * math.sin(8.0 * self._pulse_t)
            r_draw = max(6, int(r * pulse))
            cv2.circle(frame, (tx, ty), r_draw, cfg.COL_ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), r_draw, (20, 20, 20), 2, cv2.LINE_AA)

        # Direction cue derived from where the target actually is — robust
        # to future direction changes.
        target_right = (self.target_x is not None
                        and self.baseline_hip_x is not None
                        and self.target_x >= self.baseline_hip_x)
        msg = "HOP >>" if target_right else "<< HOP"
        scale = ui._scale_for(h, cfg.UI_HEADING_FRAC * 1.2)
        ui.text_centered(frame, msg, y=int(h * 0.18),
                         scale=scale, color=cfg.COL_ACCENT, panel_pad=18)

    def is_finished(self, t_elapsed):
        return self._phase is _StickPhase.DONE or t_elapsed >= self.duration_s

    def get_result(self):
        # No defensive finalisation here — update() owns the hard-cap path.
        # get_result is called every frame for the HUD, so anything mutating
        # state here would end the activity early.
        final = (0.4 * self.balance_quality
                 + 0.4 * self.land_stability_quality
                 + 0.2 * self.accuracy_quality)
        final = max(0.0, min(1.0, final))
        pts = int(round(final * 1000))
        # Temporary tuning print (once per run, on first DONE call).
        # TODO: remove when sub-score thresholds are tuned.
        if self._phase is _StickPhase.DONE and not self._printed:
            print(f"[StickTheLanding] "
                  f"balance={self.balance_quality:.3f} (x0.4 -> {int(round(self.balance_quality * 400))}pt)  "
                  f"land_stab={self.land_stability_quality:.3f} (x0.4 -> {int(round(self.land_stability_quality * 400))}pt)  "
                  f"accuracy={self.accuracy_quality:.3f} (x0.2 -> {int(round(self.accuracy_quality * 200))}pt)  "
                  f"=> {pts}pt")
            self._printed = True
        # display_str drives the live HUD top-right cell. Show the current
        # phase so the spectator strip reads "STAND" / "HOP" / "STICK" while
        # the activity runs; once done, show the points (only seen briefly
        # if at all, since Circuit moves to TRANSITION immediately).
        if self._phase is _StickPhase.STANCE:
            disp = "STAND"
        elif self._phase is _StickPhase.HOP:
            disp = "HOP"
        elif self._phase is _StickPhase.LAND:
            disp = "STICK"
        else:
            disp = f"{pts}"
        return {"points": pts, "raw": final, "display_str": disp}

    def highlight(self):
        return {"joints": self.HL_JOINTS, "bones": self.HL_BONES}

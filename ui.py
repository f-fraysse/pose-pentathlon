import time
import cv2
import numpy as np

import config as cfg


# COCO-17 skeleton bone list (pairs of keypoint indices)
COCO17_BONES = [
    (5, 7), (7, 9),         # left arm
    (6, 8), (8, 10),        # right arm
    (5, 6),                 # shoulders
    (5, 11), (6, 12),       # torso sides
    (11, 12),               # hips
    (11, 13), (13, 15),     # left leg
    (12, 14), (14, 16),     # right leg
    (0, 1), (0, 2),         # nose <-> eyes
    (1, 3), (2, 4),         # eyes <-> ears
    (3, 5), (4, 6),         # ears <-> shoulders
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Probe to convert "fraction of frame height" into an OpenCV font scale.
# cv2.getTextSize at scale=1.0 puts upper-case digit height around 22px.
_PROBE_H_AT_SCALE_1 = cv2.getTextSize("8", _FONT, 1.0, 2)[0][1]


def _scale_for(frame_h, frac):
    """Convert a target text height (as a fraction of frame_h) into an OpenCV scale."""
    target_px = frame_h * frac
    return max(0.3, target_px / _PROBE_H_AT_SCALE_1)


def _thickness_for(scale):
    return max(1, int(round(scale * 1.8)))


# ── Primitives ────────────────────────────────────────────────────────────────

def panel(frame, rect, alpha=None, color=None):
    """Semi-transparent filled rectangle in-place via cv2.addWeighted.
    rect = (x, y, w, h). Clamps to frame bounds."""
    alpha = cfg.PANEL_ALPHA if alpha is None else alpha
    color = cfg.COL_PANEL_BG if color is None else color
    x, y, w, h = rect
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    overlay = np.full_like(sub, color, dtype=sub.dtype)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, dst=sub)


def progress_bar(frame, rect, frac, color=None, bg=None):
    """Background bar + foreground fill proportional to frac in [0, 1]."""
    color = cfg.COL_HUD_BAR if color is None else color
    bg = cfg.COL_HUD_BAR_BG if bg is None else bg
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    fw = int(round(w * frac))
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg.COL_OUTLINE, 1, cv2.LINE_AA)


def text_outlined(frame, text, org, scale, color, thickness=None, outline=None):
    """putText with a dark outline so it survives any background."""
    if thickness is None:
        thickness = _thickness_for(scale)
    if outline is None:
        outline = max(2, thickness * 2)
    cv2.putText(frame, text, org, _FONT, scale, cfg.COL_OUTLINE,
                thickness + outline, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def text_centered(frame, text, y, scale, color=None, thickness=None, panel_pad=None):
    """Draw text centered horizontally. y is the vertical center of the text.
    If panel_pad is non-None, draws a panel behind the text padded by that many pixels."""
    color = cfg.COL_TEXT if color is None else color
    h, w = frame.shape[:2]
    if thickness is None:
        thickness = _thickness_for(scale)
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x = (w - tw) // 2
    baseline_y = y + th // 2
    if panel_pad is not None:
        panel(frame,
              (x - panel_pad, baseline_y - th - panel_pad,
               tw + 2 * panel_pad, th + 2 * panel_pad))
    text_outlined(frame, text, (x, baseline_y), scale, color, thickness=thickness)


def big_digit(frame, label):
    """Huge centered digit/word (countdown 3/2/1/GO)."""
    fh, fw = frame.shape[:2]
    scale = _scale_for(fh, cfg.UI_BIG_DIGIT_FRAC)
    thickness = _thickness_for(scale)
    (tw, th), _ = cv2.getTextSize(label, _FONT, scale, thickness)
    x = (fw - tw) // 2
    y = (fh + th) // 2
    # darken the frame a touch behind the digit
    panel(frame, (x - 40, y - th - 30, tw + 80, th + 60), alpha=0.40)
    text_outlined(frame, label, (x, y), scale, cfg.COL_PRIMARY,
                  thickness=thickness, outline=thickness * 2)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame, results, score_thresh=None):
    """Draw COCO-17 bones + joints from a PoseDetector results dict."""
    pts = {k: (int(r.position[0]), int(r.position[1])) for k, r in results.items()}

    for a, b in COCO17_BONES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], cfg.COL_SKELETON,
                     cfg.SKELETON_THICKNESS, cv2.LINE_AA)

    for p in pts.values():
        cv2.circle(frame, p, cfg.JOINT_RADIUS, cfg.COL_JOINT, -1, cv2.LINE_AA)


def draw_fps(frame, fps):
    """Tiny FPS readout, top-left."""
    text_outlined(frame, f"{fps:5.1f} FPS", (20, 40), 0.9, cfg.COL_TEXT, thickness=2)


# ── Screens ───────────────────────────────────────────────────────────────────

def draw_attract(frame, leaderboard=None):
    """Pulsing 'STEP IN' prompt for the ATTRACT screen.
    leaderboard hook reserved for M7 — ignored for now."""
    fh = frame.shape[0]
    t = time.perf_counter()
    pulse = 0.85 + 0.15 * (0.5 + 0.5 * np.sin(t * 2.5))
    scale = _scale_for(fh, cfg.UI_TITLE_FRAC) * pulse
    text_centered(frame, cfg.ATTRACT_PROMPT, y=int(fh * 0.20),
                  scale=scale, color=cfg.COL_PRIMARY, panel_pad=24)


def draw_instructions(frame, event_name, instruction_text, image=None):
    """Title + instruction text + GET READY footer. Image arg reserved (M7)."""
    fh, fw = frame.shape[:2]
    # backdrop panel covering the middle band so text reads cleanly
    panel(frame, (int(fw * 0.05), int(fh * 0.12),
                  int(fw * 0.90), int(fh * 0.66)))
    text_centered(frame, event_name, y=int(fh * 0.22),
                  scale=_scale_for(fh, cfg.UI_TITLE_FRAC),
                  color=cfg.COL_PRIMARY)
    # wrap the instruction text into up to 2 lines roughly
    text_centered(frame, instruction_text, y=int(fh * 0.50),
                  scale=_scale_for(fh, cfg.UI_BODY_FRAC * 1.4),
                  color=cfg.COL_TEXT)
    text_centered(frame, "GET READY...", y=int(fh * 0.72),
                  scale=_scale_for(fh, cfg.UI_HEADING_FRAC),
                  color=cfg.COL_ACCENT)


def draw_countdown(frame, seconds_remaining):
    """seconds_remaining: 3, 2, 1 -> shown as digits; 0 -> 'GO'."""
    label = "GO" if seconds_remaining <= 0 else str(seconds_remaining)
    big_digit(frame, label)


def draw_activity_hud(frame, event_name, time_frac, score):
    """Top: event name + score on a thin strip. Bottom: time-left progress bar."""
    fh, fw = frame.shape[:2]
    strip_h = int(fh * 0.10)
    panel(frame, (0, 0, fw, strip_h))
    pad = int(fh * 0.02)
    heading_scale = _scale_for(fh, cfg.UI_HEADING_FRAC)
    text_outlined(frame, event_name, (pad, int(strip_h * 0.7)),
                  heading_scale, cfg.COL_PRIMARY)
    score_text = f"{score} pts"
    (tw, _), _ = cv2.getTextSize(score_text, _FONT, heading_scale,
                                 _thickness_for(heading_scale))
    text_outlined(frame, score_text, (fw - tw - pad, int(strip_h * 0.7)),
                  heading_scale, cfg.COL_TEXT)

    bar_h = int(fh * 0.025)
    bar_y = fh - bar_h - int(fh * 0.03)
    progress_bar(frame, (pad, bar_y, fw - 2 * pad, bar_h),
                 frac=1.0 - time_frac)  # bar drains as time runs out


def draw_transition(frame, just_finished, next_up, last_points):
    """Centered panel: '<event> complete — N points' / 'Next: <name>'."""
    fh, fw = frame.shape[:2]
    panel(frame, (int(fw * 0.10), int(fh * 0.30),
                  int(fw * 0.80), int(fh * 0.40)))
    text_centered(frame, f"{just_finished} complete",
                  y=int(fh * 0.40),
                  scale=_scale_for(fh, cfg.UI_HEADING_FRAC),
                  color=cfg.COL_PRIMARY)
    text_centered(frame, f"{last_points} points",
                  y=int(fh * 0.50),
                  scale=_scale_for(fh, cfg.UI_TITLE_FRAC),
                  color=cfg.COL_TEXT)
    text_centered(frame, f"Next: {next_up}",
                  y=int(fh * 0.63),
                  scale=_scale_for(fh, cfg.UI_HEADING_FRAC),
                  color=cfg.COL_ACCENT)


def draw_results(frame, breakdown, total, title):
    """breakdown: list of (event_name, points). Renders a tidy table on a panel."""
    fh, fw = frame.shape[:2]
    panel(frame, (int(fw * 0.05), int(fh * 0.05),
                  int(fw * 0.90), int(fh * 0.90)))

    text_centered(frame, "Your Pentathlon Result",
                  y=int(fh * 0.12),
                  scale=_scale_for(fh, cfg.UI_HEADING_FRAC),
                  color=cfg.COL_ACCENT)

    body_scale = _scale_for(fh, cfg.UI_BODY_FRAC * 1.4)
    row_h = int(fh * 0.07)
    rows_top = int(fh * 0.22)
    col_left = int(fw * 0.18)
    col_right = int(fw * 0.78)
    for i, (name, pts) in enumerate(breakdown):
        y = rows_top + i * row_h
        text_outlined(frame, name, (col_left, y), body_scale, cfg.COL_TEXT)
        pts_text = f"{pts} pts"
        (tw, _), _ = cv2.getTextSize(pts_text, _FONT, body_scale,
                                     _thickness_for(body_scale))
        text_outlined(frame, pts_text, (col_right - tw, y),
                      body_scale, cfg.COL_PRIMARY)

    text_centered(frame, f"Total: {total}",
                  y=int(fh * 0.72),
                  scale=_scale_for(fh, cfg.UI_TITLE_FRAC),
                  color=cfg.COL_TEXT)
    text_centered(frame, title,
                  y=int(fh * 0.84),
                  scale=_scale_for(fh, cfg.UI_TITLE_FRAC * 1.1),
                  color=cfg.COL_PRIMARY)
    text_centered(frame, "PRESS SPACE FOR NEXT STUDENT",
                  y=int(fh * 0.94),
                  scale=_scale_for(fh, cfg.UI_BODY_FRAC),
                  color=cfg.COL_ACCENT)

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


def left_power_bar_rect(frame):
    """Standard placement for an activity's left-side vertical power bar."""
    fh, fw = frame.shape[:2]
    return (int(fw * 0.04), int(fh * 0.14), int(fw * 0.06), int(fh * 0.72))


def draw_power_bar(frame, rect, frac, peak_frac=None):
    """Vertical fill bar with a red->yellow->green gradient.
    Fills from the bottom up to `frac` (0..1). Optional `peak_frac` draws a
    horizontal marker line at the persistent high-water mark."""
    x, y, w, h = rect
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    bar_h = y1 - y0
    bar_w = x1 - x0

    # Dark background
    frame[y0:y1, x0:x1] = (40, 40, 40)

    # Fill region with gradient (red at bottom, yellow at midpoint, green at top)
    frac = max(0.0, min(1.0, float(frac)))
    fill_h = int(round(bar_h * frac))
    if fill_h > 0:
        # v=0 at bottom row of fill, v=1 at top row of bar
        v = np.linspace(1.0 - fill_h / bar_h, 1.0, fill_h, dtype=np.float32)[::-1]
        g_ch = np.clip(2.0 * v * 255.0, 0, 255).astype(np.uint8)
        r_ch = np.clip(2.0 * (1.0 - v) * 255.0, 0, 255).astype(np.uint8)
        b_ch = np.zeros(fill_h, dtype=np.uint8)
        col = np.stack([b_ch, g_ch, r_ch], axis=-1)         # (fill_h, 3)
        fill_y0 = y1 - fill_h
        frame[fill_y0:y1, x0:x1] = col[:, None, :]          # broadcast over width

    # Peak marker
    if peak_frac is not None:
        pf = max(0.0, min(1.0, float(peak_frac)))
        if pf > 0.0:
            peak_y = y1 - int(round(bar_h * pf))
            peak_y = max(y0, min(y1 - 1, peak_y))
            cv2.line(frame, (x0 - 6, peak_y), (x1 + 5, peak_y),
                     cfg.COL_PRIMARY, 3, cv2.LINE_AA)

    # Outline
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), cfg.COL_OUTLINE, 2, cv2.LINE_AA)


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


def _wrap_text(text, scale, thickness, max_w):
    """Greedy word-wrap. Returns a list of lines that each fit max_w pixels."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, _FONT, scale, thickness)
        if tw > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def text_block_centered(frame, text, y, scale, color=None,
                        thickness=None, max_w_frac=0.85, line_spacing=1.3):
    """Word-wrapped, horizontally centered, vertically centered around y."""
    color = cfg.COL_TEXT if color is None else color
    if thickness is None:
        thickness = _thickness_for(scale)
    fh, fw = frame.shape[:2]
    max_w = int(fw * max_w_frac)
    lines = _wrap_text(text, scale, thickness, max_w)
    if not lines:
        return
    (_, th), _ = cv2.getTextSize("Ag", _FONT, scale, thickness)
    step = int(th * line_spacing)
    total_h = step * (len(lines) - 1) + th
    y0 = y - total_h // 2
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, _FONT, scale, thickness)
        x = (fw - tw) // 2
        text_outlined(frame, line, (x, y0 + i * step + th),
                      scale, color, thickness=thickness)


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

def draw_skeleton(frame, results, highlight_joints=None, highlight_bones=None,
                  score_thresh=None):
    """Draw COCO-17 bones + joints. Optional highlight sets render the given
    joints/bones in COL_ACCENT at increased size."""
    hj = set(highlight_joints) if highlight_joints else set()
    hb = set(highlight_bones) if highlight_bones else set()
    pts = {k: (int(r.position[0]), int(r.position[1])) for k, r in results.items()}

    for a, b in COCO17_BONES:
        if a in pts and b in pts:
            is_h = (a, b) in hb or (b, a) in hb
            color = cfg.COL_ACCENT if is_h else cfg.COL_SKELETON
            thick = cfg.SKELETON_THICKNESS + (3 if is_h else 0)
            cv2.line(frame, pts[a], pts[b], color, thick, cv2.LINE_AA)

    for k, p in pts.items():
        is_h = k in hj
        color = cfg.COL_ACCENT if is_h else cfg.COL_JOINT
        radius = cfg.JOINT_RADIUS + (4 if is_h else 0)
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)


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
    text_block_centered(frame, instruction_text, y=int(fh * 0.50),
                        scale=_scale_for(fh, cfg.UI_BODY_FRAC * 1.4),
                        color=cfg.COL_TEXT)
    text_centered(frame, "GET READY...", y=int(fh * 0.72),
                  scale=_scale_for(fh, cfg.UI_HEADING_FRAC),
                  color=cfg.COL_ACCENT)


def draw_countdown(frame, seconds_remaining):
    """seconds_remaining: 3, 2, 1 -> shown as digits; 0 -> 'GO'."""
    label = "GO" if seconds_remaining <= 0 else str(seconds_remaining)
    big_digit(frame, label)


def draw_activity_hud(frame, event_name, time_frac, display):
    """Top: event name + display string on a thin strip. Bottom: time-left progress bar."""
    fh, fw = frame.shape[:2]
    strip_h = int(fh * 0.10)
    panel(frame, (0, 0, fw, strip_h))
    pad = int(fh * 0.02)
    heading_scale = _scale_for(fh, cfg.UI_HEADING_FRAC)
    text_outlined(frame, event_name, (pad, int(strip_h * 0.7)),
                  heading_scale, cfg.COL_PRIMARY)
    (tw, _), _ = cv2.getTextSize(display, _FONT, heading_scale,
                                 _thickness_for(heading_scale))
    text_outlined(frame, display, (fw - tw - pad, int(strip_h * 0.7)),
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

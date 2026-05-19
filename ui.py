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


def text_outlined(frame, text, org, scale, color, thickness=2, outline=4, font=cv2.FONT_HERSHEY_SIMPLEX):
    """putText with a dark outline so it survives any background."""
    cv2.putText(frame, text, org, font, scale, cfg.COL_OUTLINE, thickness + outline, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def text_centered(frame, text, y, scale=2.0, color=None, thickness=3):
    color = color if color is not None else cfg.COL_TEXT
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (w - tw) // 2
    text_outlined(frame, text, (x, y + th // 2), scale, color, thickness=thickness)


def draw_skeleton(frame, results, score_thresh=None):
    """Draw COCO-17 bones + joints from a PoseDetector results dict."""
    pts = {k: (int(r.position[0]), int(r.position[1])) for k, r in results.items()}

    for a, b in COCO17_BONES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], cfg.COL_SKELETON, cfg.SKELETON_THICKNESS, cv2.LINE_AA)

    for p in pts.values():
        cv2.circle(frame, p, cfg.JOINT_RADIUS, cfg.COL_JOINT, -1, cv2.LINE_AA)


def draw_attract(frame):
    """Pulsing 'STEP IN' prompt for the ATTRACT screen."""
    t = time.perf_counter()
    pulse = 0.85 + 0.15 * (0.5 + 0.5 * np.sin(t * 2.5))  # 0.70 .. 1.00
    scale = 4.5 * pulse
    h = frame.shape[0]
    text_centered(frame, cfg.ATTRACT_PROMPT, y=int(h * 0.15), scale=scale,
                  color=cfg.COL_PRIMARY, thickness=8)


def draw_fps(frame, fps):
    """Tiny FPS readout, top-left."""
    text_outlined(frame, f"{fps:5.1f} FPS", (20, 40), 0.9, cfg.COL_TEXT, thickness=2)

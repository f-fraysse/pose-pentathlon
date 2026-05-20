# ── Tracking ──────────────────────────────────────────────────────────────────
TRACKING_MODE = "body"  # RTMO + COCO-17. No wholebody for the pentathlon.

# All 17 COCO joints enabled. Flag value is unused by the pentathlon
# (it's a holdover from rtmlib_vfx's emit/disturb role bitmask); non-zero just
# means "track this keypoint".
BODY_KEYPOINT_SELECTION = {i: 1 for i in range(17)}

BODY_KEYPOINT_NAMES = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}

KEYPOINT_SELECTION = BODY_KEYPOINT_SELECTION
KEYPOINT_NAMES = BODY_KEYPOINT_NAMES

# ── RTMPose ───────────────────────────────────────────────────────────────────
POSE_DEVICE = "cuda"
POSE_BACKEND = "onnxruntime"
SCORE_THRESHOLD = 0.3

# RTMO whole-person detection threshold. rtmlib default is 0.7, which
# drops the entire skeleton for 1-2 frames during fast motion. Lower =
# fewer dropouts but more false detections. Distinct from SCORE_THRESHOLD
# above (per-keypoint).
RTMO_SCORE_THR = 0.3

# ── Webcam ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0

# ── One Euro Filter ───────────────────────────────────────────────────────────
KEYPOINT_SMOOTHING_ENABLED = True
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_DCUTOFF = 1.0

# ── Velocity ──────────────────────────────────────────────────────────────────
VELOCITY_CALCULATION_METHOD = "moving_average"
VELOCITY_MA_WINDOW_SIZE = 5
KEYPOINT_TRAIL_LENGTH = 10

# ── UI / Display ──────────────────────────────────────────────────────────────
WINDOW_NAME = "Pose Pentathlon"
ATTRACT_PROMPT = "STEP IN"

# Colours (BGR for OpenCV)
COL_PRIMARY     = (60, 220, 255)   # warm yellow
COL_ACCENT      = (255, 120, 50)   # cyan-ish
COL_OUTLINE     = (0, 0, 0)
COL_TEXT        = (255, 255, 255)
COL_SKELETON    = (60, 220, 255)
COL_JOINT       = (255, 255, 255)
SKELETON_THICKNESS = 4
JOINT_RADIUS = 6

# ── State machine timings ────────────────────────────────────────────────────
COUNTDOWN_SEC         = 3       # 3 -> 2 -> 1 -> GO -> ACTIVITY (each ~1s)
TRANSITION_SEC        = 2.0     # "Event complete — Next: ..."
STUB_ACTIVITY_SEC     = 8.0     # short for M2 demoability; tune in M4
INSTRUCTIONS_AUTO_SEC = 4.0     # auto-advance to COUNTDOWN; SPACE skips

# ── UI sizing (fractions of frame height — capture-resolution independent) ──
UI_TITLE_FRAC     = 0.07        # ATTRACT "STEP IN", screen titles
UI_HEADING_FRAC   = 0.045       # screen subheadings, event names
UI_BODY_FRAC      = 0.026       # instruction text, hud labels
UI_BIG_DIGIT_FRAC = 0.42        # countdown 3/2/1/GO

# Semi-transparent panel behind text blocks
PANEL_ALPHA    = 0.55
COL_PANEL_BG   = (0, 0, 0)
COL_HUD_BAR    = (60, 220, 255)
COL_HUD_BAR_BG = (40, 40, 40)

# ── Pentathlon (filled in as we build each activity) ─────────────────────────
# Ordered list of activity keys for the circuit. Comment out entries to
# shorten the circuit for testing/tweaking a single event. Keys must match
# circuit.ACTIVITY_REGISTRY.
CIRCUIT_ACTIVITIES = [
    # "high_knees",
    # "vertical_jump",
    # "reaction_wall",
    # "punch_power",
    "stick_landing",
]

# Athlete titles by total points
ATHLETE_TITLES = [
    (1500, "Rookie"),
    (2500, "Athlete"),
    (3500, "Competitor"),
    (4500, "Elite"),
    (10_000, "Champion"),
]

# Per-activity scoring endpoints — tune by feel during M4+
# Each entry: (raw_min, raw_max) → linearly mapped to 0..1000 points
SCORE_MAP = {
    "high_knees":      (0, 30),       # reps
    "vertical_jump":   (0.0, 0.4),    # jump_ratio (rise / leg_length)
    "reaction_wall":   (0, 20),       # hits
    "punch_power":     (0.0, 0.9),    # |vx|/arm_len per frame (FPS-dep) — tune
    "javelin":         (0.0, 1.0),    # normalised burst — tune
}

# Pose Pentathlon — Implementation Plan

> Initial brief for Claude Code. Build target: an interactive open-day exhibit.
> Working name: **Pose Pentathlon** (rename freely).

---

## 1. Goal & context

An interactive exhibit for visiting high school students at a university
high-performance-sport biomechanics lab. A single USB webcam runs real-time
2D pose detection. Students see their own skeleton "wireframe" overlaid live,
then can run a **5-event athletic pentathlon** of quick physical mini-games,
ending with a composite score and an athlete title.

Two display modes:

- **Attract / mirror mode** — idle state. Live mirrored skeleton on screen,
  "STEP IN" prompt. This is the baseline experience everyone sees.
- **Pentathlon circuit** — an operator starts a run; the student is taken
  through 5 timed events with on-screen instructions, countdowns, live
  scoring, and a final results card.

Audience flow on the day: groups of ~25 students, ~30 min per group. Not every
student will complete the full circuit — the mirror is the always-on draw, the
pentathlon is the headline attraction a subset run while others watch the
leaderboard.

## 2. Hard constraints & non-goals

**Constraints**
- This is a one-off built in ~1.5 days. **Speed of delivery beats everything.**
- Single camera, no calibration, no heavy setup.
- Operator-controlled pacing (a lab helper presses keys).
- Must be visually clear to a 14–18 y/o seeing it for the first time.

**Explicit non-goals — do not spend time on these**
- Code quality, maintainability, architecture purity beyond what Section 5 asks.
- Scientific / measurement accuracy. All scoring mappings are arbitrary and
  tuned by feel.
- Calibration, 3D, multi-camera.
- Tests, packaging, persistence to disk, networking.
- Multi-person handling beyond "pick the most prominent person" (Section 11).

## 3. Foundation to reuse — `rtmlib_vfx` repo

The repo `f-fraysse/rtmlib_vfx` already has a working real-time pose pipeline.
**Reuse it; do not rebuild pose detection.** Relevant pieces:

- `particle_game/capture.py` — `CaptureThread`, threaded non-blocking webcam
  capture. Reuse as-is.
- `particle_game/pose.py` — `PoseDetector`, **`OneEuroFilter`**, velocity
  helpers. Reuse the detector and the One-Euro filter.
- `particle_game/config.py` — the "all constants in one file" pattern. Copy it.
- `particle_game/main.py` — reference for the main loop / orchestration shape.
- Conda env is `rtmlib`; run via `conda run -n rtmlib python main.py`.

**Use `TRACKING_MODE = "body"`** (RTMO, 17 COCO keypoints). The pentathlon only
needs body joints — no face/hands — and body mode is faster. Wholebody mode is
unnecessary here.

First task for Claude Code is to **inspect this repo** and confirm the exact
API for getting COCO-17 keypoints per frame before scaffolding anything.

## 4. Tech stack

- Python 3.10, conda env `rtmlib`.
- `rtmlib` for pose (RTMO body mode).
- **OpenCV (`cv2`) for ALL rendering** — `putText`, `circle`, `line`,
  `addWeighted` for panels, image overlay for instruction pictures. No GUI
  framework, no OpenGL needed for this project.
- One process, one main loop. NumPy for the keypoint math.

## 5. Architecture

A simple **state machine** plus a **uniform Activity interface**. This is the
one place to be disciplined — it is what makes the 2-day timeline safe, because
each activity can be tested standalone and the orchestrator never needs to know
what any activity does internally.

### State machine

```
ATTRACT  --SPACE-->  INSTRUCTIONS --> COUNTDOWN --> ACTIVITY --> TRANSITION
   ^                                                                |
   |                          (loop over 5 activities)              |
   |                                                                v
   +-------------------- RESULTS  <----------------------------------+
                         (SPACE returns to ATTRACT for next student)
```

### Uniform Activity interface

Every mini-game implements the same base class:

```python
class Activity:
    name: str               # e.g. "High Knees"
    instruction_text: str    # one-line "what to do"
    instruction_image: str   # path under instruction_images/ (may be missing)
    duration_s: float | None # fixed-time; None = attempt-based (e.g. jump)

    def reset(self): ...
    def update(self, keypoints, t_elapsed): ...   # update internal state
    def draw(self, frame): ...                     # draw activity overlay
    def is_finished(self, t_elapsed) -> bool: ...
    def get_result(self) -> dict:                  # {points, raw, display_str}
        ...
```

A `Circuit` object holds the ordered list of `Activity` instances, the current
index, the current state, and timing. The main loop:

```
grab frame -> run pose -> pick prominent person -> smooth keypoints
   -> dispatch to current state handler -> render screen -> cv2.imshow
```

Reset = re-instantiate / `reset()` all activities and return to ATTRACT.

## 6. The display / UI layer — the bulk of the work

The activity logic is small; **the polish is in the screens**. Build a
dedicated `ui.py` with reusable primitives, then compose screens from them.

### Reusable primitives

- `text_centered(frame, text, y, scale, color)` — with a dark outline/shadow
  drawn underneath so text reads against any background.
- `panel(frame, rect, alpha)` — semi-transparent dark rounded panel
  (`addWeighted`), used behind all text blocks.
- `progress_bar(frame, rect, frac, color)` — used for time-left and fill bars.
- `skeleton(frame, keypoints, conf_thresh)` — the wireframe (bones + joints).
- `big_digit(frame, n)` — huge centered countdown digit.

### Screens

| Screen        | Contents |
|---------------|----------|
| ATTRACT       | Live mirrored skeleton + pulsing "STEP IN — PRESS SPACE TO START" + day leaderboard (top 5) |
| INSTRUCTIONS  | Event title, instruction image, one-line text, "GET READY..." |
| COUNTDOWN     | Live feed + huge `3 · 2 · 1 · GO` |
| ACTIVITY      | Live feed + skeleton + the activity's own overlay + HUD (event name, time-left bar, current score) |
| TRANSITION    | "Event complete — Next: <name>" (~2 s) |
| RESULTS       | Per-event score breakdown + total + athlete title + "PRESS SPACE for next student" |

### Style rules
- Large, high-contrast text; outlined so it survives any background.
- Consistent colour palette (define in `config.py`).
- **Mirror the displayed frame** (horizontal flip) so it behaves like a mirror —
  `rtmlib_vfx` already does this; reuse the approach.
- Instruction images live in `instruction_images/` (one per event). The
  instruction screen **must tolerate a missing image** and fall back to drawn
  text — the operator may add athlete photos / diagrams later.

## 7. Operator controls (keyboard)

| Key   | Action |
|-------|--------|
| SPACE | Start circuit (from ATTRACT) / advance (from RESULTS) |
| N     | Skip current activity |
| R     | Abort, reset to ATTRACT |
| Q     | Quit |

Activities also auto-advance on their timer. Operator keys override.

## 8. The five activities

Build them **in this order** (see milestones). Each returns `points` on a
common 0–1000 scale via a clamped linear mapping from its raw metric — tuned by
feel, not physics.

### 8.1 High Knees  *(safe anchor — also warms students up)*
- Duration ~20 s. Count knee-raises across both legs.
- Per leg, `gap = knee_y - hip_y`, normalised by leg length
  (`|hip - ankle|`). Standing `gap_norm ≈ 0.5`; knee raised high `gap_norm`
  small.
- **Hysteresis**: leg is "up" when `gap_norm < ~0.2`, "down" when
  `gap_norm > ~0.4`. Count a rep on each down→up transition. Sum both legs.
- Score: reps → points.

### 8.2 Vertical Jump  *(safe anchor)*
- Attempt-based, best of 2.
- Use **hip-centre Y** (mean of L/R hip) — stabler than head.
- Capture standing baseline (median hip-centre Y over a ~1 s still window).
- `rise = baseline_y - min(hip_centre_y)` during the attempt window.
- Normalise: `jump_ratio = rise / leg_length` (leg length measured standing).
- Score: `jump_ratio` → points.

### 8.3 Reaction Wall
- Duration ~20 s. Targets (circles) spawn at random reachable screen positions,
  1–2 at a time.
- A hit = either wrist keypoint within the (generous) target radius → target
  disappears, +1, respawn elsewhere.
- Scored as **hit count**, not reaction time — so display latency is irrelevant.
- Score: hits → points.

### 8.4 Punch Power
- Single best punch (or best over a ~10 s window).
- Per frame, accumulate each wrist's displacement over a short sliding window
  (~0.3 s) → "punch energy". **Use integrated displacement, not peak velocity**
  — robust to low/jittery FPS.
- Normalise by arm length (`|shoulder - wrist|`).
- A fill bar shows the current best energy.
- Score: peak energy → points.

### 8.5 Stick the Landing
- 3-phase composite event:
  1. **Stance (3 s)** — hold a single-leg stand; detect a raised ankle
     `> 0.25 · leg_length` above the grounded one. Log sway across a set of
     tracked keypoints (elbows, wrists, hips, knees) — flailing tanks the
     score. → static balance sub-score.
  2. **Hop (≤4 s)** — a glowing target appears on the standing-leg side
     (~0.75 · leg_length away from the baseline hip). Cue "HOP". Detect the
     hop as a brief airborne window from the standing-ankle Y rising above
     ground and returning. Capture: did the hip-centre X end up near the
     target? → landing-accuracy sub-score.
  3. **Stabilise (3 s)** — after landing, log sway again for 3 s →
     landing-stability sub-score.
- Composite: 0.4 · balance + 0.4 · landing stability + 0.2 · accuracy →
  0..1000. All sub-metrics body-unit normalised (sway divided by leg length;
  hip-to-target distance divided by leg length).
- **Graceful degrade**: each phase has a timeout (stance 5 s, hop 4 s, land
  fixed 3 s). On timeout the corresponding sub-score is set to 0 and the
  activity continues — nobody gets stuck on stage.

> Originally event 5 was a **Javelin Throw** sharing the displacement-burst
> detector with Punch Power. Stick the Landing was chosen instead — it
> exercises a different physical quality (balance) and gives the codebase a
> multi-phase activity template (an internal `_StickPhase` enum with
> per-phase dispatch from `update()` and `draw()`). Javelin is deferred to
> M8 polish.

## 9. Scoring & pentathlon composite

- Each activity → 0–1000 points (clamped linear map from raw metric; tune the
  endpoints by testing on yourself).
- Total out of 5000. Show a per-event breakdown on the RESULTS screen.
- Map total to an athlete title (tune thresholds):

| Total      | Title |
|------------|-----------|
| < 1500     | Rookie |
| 1500–2500  | Athlete |
| 2500–3500  | Competitor |
| 3500–4500  | Elite |
| > 4500     | Champion |

- **Day leaderboard**: keep an in-memory list of completed totals; show top 5
  on the ATTRACT and RESULTS screens ("Today's best"). No disk persistence.

## 10. COCO-17 keypoint reference

```
0  nose            5  L shoulder    9  L wrist     13 L knee
1  L eye           6  R shoulder    10 R wrist     14 R knee
2  R eye           7  L elbow       11 L hip       15 L ankle
3  L ear           8  R elbow       12 R hip       16 R ankle
4  R ear
```

## 11. Technical notes & gotchas

- **Smooth keypoints** with the One-Euro filter from `rtmlib_vfx/pose.py`
  before any logic — critical for jump peak detection and the displacement
  bursts.
- **Counters use hysteresis** (separate up/down thresholds) or students rack up
  phantom reps hovering at the line.
- **Body-size invariance**: anywhere a metric is compared across students,
  normalise by a body segment length (leg length, arm length).
- **Low-confidence / missing keypoints**: treat below-threshold keypoints as
  missing — hold last good value, or skip that frame's update for counters.
- **Most-prominent person**: the room will have onlookers. Each frame, pick the
  detected person with the largest bounding box (closest to camera) and ignore
  the rest.
- **Latency-tolerant by design**: count-based events (reaction wall, high knees)
  and velocity / sway events (punch, stick the landing) all tolerate modest
  webcam FPS and display latency. Do not introduce any true reaction-time
  scoring.
- Run at a sensible display size (~1280×720). Threaded capture handles the rest.

## 12. Build order — milestones

Build a **vertical slice first**: a complete, runnable 2-event circuit before
touching events 3–5. If time runs out, a 2- or 3-event pentathlon is still a
finished, demoable exhibit.

1. **M1 — Mirror + ATTRACT.** Reuse `capture.py` + `pose.py`; draw the skeleton,
   mirrored; ATTRACT screen with prompt. Runnable end-to-end.
2. **M2 — Display layer.** `ui.py` primitives + screen renderers (countdown,
   instruction card, HUD, results — results/instructions can be stubs).
3. **M3 — Orchestrator.** `Activity` base + `Circuit` state machine wired with
   2 stub activities (just a number ticking up) → the full
   INSTRUCTIONS→COUNTDOWN→ACTIVITY→TRANSITION→RESULTS→reset flow runs.
4. **M4 — Real events 1 & 2.** Implement High Knees + Vertical Jump.
   ← **This is the MVP: a complete, polished 2-event pentathlon.**
5. **M5 — Reaction Wall.**
6. **M6 — Punch Power.** Per-frame `|horizontal wrist velocity| / arm_length`
   (arm_length = running mean of upper-arm + forearm, bend-invariant). 5-frame
   MA-smoothed velocity, deadband threshold, peak-hold power bar.
   *Originally paired with Javelin via a shared displacement-burst detector;
   Javelin deferred to M8.*
7. **M7 — Stick the Landing** (replaces the original Javelin event as #5).
   3-phase composite (single-leg stance → sideways hop → stabilise). Composite
   score from balance + landing stability + landing accuracy. First multi-phase
   activity; introduces a `_StickPhase` enum + per-phase dispatch pattern.
8. **M8 — Polish.** Instruction images, day leaderboard, optional sound;
   Javelin throw if time permits (reuse the punch detector).

## 13. First task for Claude Code

Before writing any project code:

1. Inspect the `rtmlib_vfx` repo — specifically `particle_game/capture.py`,
   `pose.py`, `config.py`, `main.py`. Confirm the exact API for running RTMO
   "body" mode and getting per-frame COCO-17 keypoints + confidences.
2. Propose the new project file layout, the `Activity` base class signature,
   and the `Circuit` orchestrator skeleton.
3. **Stop and confirm** that scaffold before implementing any activity logic.
4. Then build **M1** and verify it runs against the webcam.

Proceed milestone by milestone; pause for a quick check after M1 and after M4.

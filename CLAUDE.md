# CLAUDE.md — pose_pentathlon

Interactive open-day exhibit: webcam -> RTMO COCO-17 pose -> mirrored
"attract" view + 5-event athletic mini-pentathlon scored out of 5000.
Built in ~1.5 days for a uni biomechanics-lab open day.

Authoritative design: [docs/POSE_PENTATHLON_PLAN.md](docs/POSE_PENTATHLON_PLAN.md).

## Run

```
conda run -n pentathlon python main.py
```

Requires `onnxruntime-gpu==1.20.1` on a CUDA 12.x box. NEVER let plain
`onnxruntime` (CPU) co-exist in the env — it import-shadows the GPU build
silently. See Gotchas.

## Gotchas (from M1)

- **onnxruntime CPU shadowing.** If FPS is awful, run
  `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
  and confirm `CUDAExecutionProvider` is present. Fix:
  `pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.20.1`.
  `requirements.txt` already carries a comment guard.
- **Mirror BEFORE detect/draw.** Flipping the final composited frame reverses
  the on-screen text too. [main.py](main.py) flips immediately after
  `cap_thread.read()`; keep it that way.
- **Do not call `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`.** On Windows DSHOW
  this negotiates an uncompressed stream that caps the camera at ~5 FPS at
  1080p. `particle_game` ran at the webcam default and was always fast — we
  match that.

## Conventions

- **No emojis in any file** (code, docs, commit messages).
- **cv2 for ALL rendering.** No GUI framework, no OpenGL.
- **`capture.py` and `pose.py` are copied verbatim** from
  `d:\PythonProjects\rtmlib\particle_game`. Don't refactor; pull upstream
  fixes by re-copying if they appear there.
- **Speed of delivery beats code quality.** The design doc is explicit: no
  tests, packaging, disk persistence, networking, multi-camera, calibration.
- **Skeleton rendering is currently flat single-colour** via
  `cfg.COL_SKELETON / COL_JOINT / SKELETON_THICKNESS / JOINT_RADIUS`. See
  Forward-looking before generalising.
- **M2 scores are linear stubs.** `StubActivity.update` just sets
  `_score = t_elapsed * score_rate`. Real per-event metrics
  (rep-counting with hysteresis, jump ratio, hit counts, displacement
  integrals) arrive at M4 — don't read the stub formula as intent.

## Architecture (one paragraph)

State machine: `ATTRACT -> INSTRUCTIONS -> COUNTDOWN -> ACTIVITY -> TRANSITION`
(loop x N events) `-> RESULTS -> ATTRACT`. Implemented in [circuit.py](circuit.py):
`State` enum, `StubActivity` (placeholder with the design-doc §5 method
signatures), and `Circuit` (owns activity list, current index, current
state, stage clock; handles key events). [main.py](main.py) is a thin
dispatcher: per frame it captures, flips, runs pose, draws the skeleton,
then calls `circuit.update(keypoints)` / `circuit.draw(frame)` and routes
keys through `circuit.on_key(...)`. Full design detail:
[docs/POSE_PENTATHLON_PLAN.md](docs/POSE_PENTATHLON_PLAN.md) section 5.

## Forward-looking notes

- **Per-event skeleton styling.** Activities at M4+ will want to emphasise
  specific joints/bones (hip-knee-ankle for High Knees; wrists for Reaction
  Wall + Punch). Plan to extend `ui.draw_skeleton` with an optional per-joint
  / per-bone style override (dict keyed by COCO-17 index / bone tuple). Defer
  the API shape until the first concrete need lands at M4.

## Status

- M1 complete: mirror + ATTRACT verified.
- M2 complete: UI primitives, screen renderers (instructions / countdown /
  HUD / transition / results), `Circuit` state machine + 2 stub activities,
  full flow walkable end-to-end. Scores are linear stubs.
- Next: M3 (formal `Activity` ABC) — likely collapses into M4 if we
  jump straight to real High Knees + Vertical Jump scoring.

See [README.md](README.md) for the full milestone ladder.

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
- **Per-event skeleton emphasis.** `ui.draw_skeleton` takes optional
  `highlight_joints` / `highlight_bones` sets. Each activity exposes a
  `highlight()` dict; the `Circuit.draw_skeleton` method consults the
  current activity during ACTIVITY state only. Other states stay
  flat-colour.
- **Activities are duck-typed, no formal ABC.** All activity classes
  (`StubActivity` in `circuit.py`, real events in `activities.py`)
  implement the same implicit contract: class attrs `name`,
  `instruction_text`, `instruction_image`, `duration_s`; methods
  `reset()`, `update(keypoints, t_elapsed)`, `draw(frame)`,
  `is_finished(t_elapsed)`, `get_result()` returning
  `{"points": int, "raw": Any, "display_str": str}`, and `highlight()`
  returning `{"joints": set, "bones": set}` or `{}`. Design doc §5 is
  the authoritative contract.
- **Scoring**: every activity maps its raw metric through
  `cfg.SCORE_MAP[event_key]` to 0..1000 points via clamped linear
  interpolation. Tune endpoints in `config.py`, not in the activity.

## Architecture (one paragraph)

State machine: `ATTRACT -> INSTRUCTIONS -> COUNTDOWN -> ACTIVITY -> TRANSITION`
(loop x N events) `-> RESULTS -> ATTRACT`. Implemented in [circuit.py](circuit.py):
`State` enum, `StubActivity` (placeholder, still used for stub events),
and `Circuit` (owns activity list, current index, current state, stage
clock; handles key events; routes the per-state skeleton highlight via
`Circuit.draw_skeleton`). Real events live in [activities.py](activities.py).
[main.py](main.py) is a thin dispatcher: per frame it captures, flips,
runs pose, then calls `circuit.draw_skeleton(frame, results)` /
`circuit.update(results)` / `circuit.draw(frame)` and routes keys
through `circuit.on_key(...)`. Full design detail:
[docs/POSE_PENTATHLON_PLAN.md](docs/POSE_PENTATHLON_PLAN.md) section 5.

## Power bars

Vertical fill bar on the left side with a red->yellow->green gradient and
a persistent yellow peak marker. Built from `ui.draw_power_bar(frame,
rect, frac, peak_frac=None)`; layout is standard via
`ui.left_power_bar_rect(frame)`. Each activity decides what `frac` and
`peak_frac` mean — High Knees uses **rep frequency** (reps/sec over a
rolling 3s window, max at `MAX_FREQ_HZ = 4.0`); Vertical Jump uses
**jump ratio** (rise / leg-length, max at `cfg.SCORE_MAP["vertical_jump"][1]`).

## Lab-tuning notes

The two real events were developed against a desk webcam. Tomorrow on
the lab machine, watch for:

- **Vertical Jump score saturation.** Desk-webcam testing reached 1000pt
  too easily. Likely needs `cfg.SCORE_MAP["vertical_jump"]` upper bound
  raised from `0.4` to something higher (`0.5`-`0.6`) once you see real
  student jumps under the lab camera. Leg-length normalisation depends
  on camera framing — closer camera = bigger leg-pixels = smaller ratio
  for the same physical jump, so a tighter framing in the lab will
  effectively make the bar harder to fill.
- **High Knees frequency target.** `HighKneesActivity.MAX_FREQ_HZ = 4.0`
  (= 80 reps in 20s) was set against the user's self-test of 60 reps.
  Re-eyeball with real students.
- **High Knees hysteresis thresholds.** `UP_THRESHOLD = 0.20` /
  `DOWN_THRESHOLD = 0.40` at the top of `HighKneesActivity`. If reps
  feel too easy/hard, narrow or widen.
- **Vertical Jump baseline window.** `BASELINE_WINDOW_S = 1.0` —
  increase if the lab setup has students drifting before settling.

## Status

- **M1 complete**: mirror + ATTRACT.
- **M2 complete**: UI primitives, all five screen renderers, `Circuit`
  state machine. Skeleton highlight + bare-number HUD landed too.
- **M3 skipped**: formal `Activity` ABC was not added; duck typing
  covers it (see Conventions).
- **M4 complete** (the design-doc MVP): both real events implemented.
  - High Knees: hysteresis on hip-knee-ankle, rep counting, frequency
    power bar with persistent peak marker.
  - Vertical Jump: hip-centre Y tracked against a 1s standing baseline,
    normalised by leg length, peak ratio scored via `cfg.SCORE_MAP`,
    height power bar with persistent peak marker.
- **Next options**: M5 (Reaction Wall), M6 (Punch Power / Javelin —
  shared displacement-burst detector), or M7 polish (instruction images,
  day leaderboard, sound). All scoring values likely need lab-tuning
  first (see Lab-tuning notes above).

See [README.md](README.md) for the full milestone ladder.

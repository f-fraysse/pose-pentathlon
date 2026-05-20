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
  `requirements.txt` already carries a comment guard. **Recurs across
  machines** because `rtmlib==0.0.15` declares `onnxruntime` (CPU) as a
  hard dep, so any `pip install rtmlib` pulls the CPU build back in
  alongside the GPU one. Don't reinstall rtmlib after fixing.
- **RTMO whole-skeleton flicker.** rtmlib's `Body(..., pose='rtmo')`
  hardcodes the RTMO person-detection threshold to 0.7, which drops the
  entire skeleton for 1-2 frames during fast motion. We override it via
  `cfg.RTMO_SCORE_THR` applied as `self.model.pose_model.score_thr` in
  [pose.py](pose.py) right after `Body(...)` construction. Distinct from
  `cfg.SCORE_THRESHOLD` which is the **per-keypoint** filter inside
  `PoseDetector.detect`.
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
**jump ratio** (rise / leg-length, max at `cfg.SCORE_MAP["vertical_jump"][1]`);
Reaction Wall uses **hits / SCORE_MAP max** (no peak marker — count is
monotone).

## Circuit composition

The activity list is built by `circuit.build_demo_circuit()` from
`cfg.CIRCUIT_ACTIVITIES` (ordered list of keys) via the
`circuit.ACTIVITY_REGISTRY` `{key: Class}` dict. To test a single event,
comment out the others in `CIRCUIT_ACTIVITIES`. New events register by
adding one line to `ACTIVITY_REGISTRY` and listing the key.

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
- **Reaction Wall hit feel.** `TARGET_RADIUS_FRAC` controls the visual
  size of targets; `HIT_RADIUS_MULT` (default 1.3) is the multiplier
  that defines the actual hit-test radius (so hits register when the
  wrist is just outside the visible circle). Bump
  `cfg.SCORE_MAP["reaction_wall"]` upper bound from 20 if students
  routinely exceed that in 20 s.

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
- **M5 complete**: Reaction Wall. Two cyan circles spawn in a reachable
  zone (avoiding edges + feet); hit on wrist (kpts 9/10) within
  `HIT_RADIUS_MULT * visual_radius` counts +1 and respawns. Hit count
  scored via `cfg.SCORE_MAP["reaction_wall"]`.
- **Next options**: M6 (Punch Power / Javelin — shared displacement-burst
  detector) or M7 polish (instruction images, day leaderboard, sound).
  All scoring values likely need lab-tuning first (see Lab-tuning notes
  above).

See [README.md](README.md) for the full milestone ladder.

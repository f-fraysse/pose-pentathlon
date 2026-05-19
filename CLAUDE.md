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

## Architecture (one paragraph)

State machine: `ATTRACT -> INSTRUCTIONS -> COUNTDOWN -> ACTIVITY -> TRANSITION`
(loop x5) `-> RESULTS -> ATTRACT`. Every event implements a uniform `Activity`
interface (`name`, `instruction_text`, `instruction_image`, `duration_s`,
`reset/update/draw/is_finished/get_result`). A `Circuit` owns the activity
list, current index, state, timing. Main loop: capture -> flip -> pose ->
pick most-prominent person -> smooth -> dispatch to state handler -> draw ->
`imshow`. Full detail: [docs/POSE_PENTATHLON_PLAN.md](docs/POSE_PENTATHLON_PLAN.md)
section 5.

## Forward-looking notes

- **Per-event skeleton styling.** Activities at M4+ will want to emphasise
  specific joints/bones (hip-knee-ankle for High Knees; wrists for Reaction
  Wall + Punch). Plan to extend `ui.draw_skeleton` with an optional per-joint
  / per-bone style override (dict keyed by COCO-17 index / bone tuple). Defer
  the API shape until the first concrete need lands at M4.

## Status

M1 complete (mirror + ATTRACT verified). Next: M2 display-layer primitives +
screen renderers. See [README.md](README.md) for the milestone ladder.

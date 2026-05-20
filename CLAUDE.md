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
- **Punch Power score saturation.** `cfg.SCORE_MAP["punch_power"]` upper
  bound is `0.9` (units: arm-lengths-of-horizontal-wrist-velocity per
  *frame* — so the scale is FPS-dependent; re-tune on the lab machine if
  its FPS differs significantly from the desk rig). Lower it if the bar
  never moves; raise it if students max out in one punch.
- **Punch Power jitter deadband.** `PunchPowerActivity.MIN_SPEED = 0.02`
  is the per-side speed threshold below which motion is ignored, so a
  still wrist after OneEuro smoothing doesn't flicker the bar. Tune up
  if standing still still shows bar; tune down if slow punches don't
  register.
- **Stick the Landing — single-leg detection.**
  `StickTheLandingActivity.STANCE_ANKLE_DIFF_THR = 0.25` (fraction of
  leg length). Raise to 0.30 if students cheat with a slightly lifted
  foot; lower to 0.20 if a clean stance gets rejected as "not single-leg".
- **Stick the Landing — steadiness scoring.** `BALANCE_STD_BAD = 0.05`
  (stance) and `LAND_STD_BAD = 0.08` (landing) set how forgiving the
  steadiness scoring is. Both are leg-length-normalised thresholds on
  the *mean* per-keypoint (stdev_x + stdev_y) across `TRACKED_KPTS =
  (7, 8, 9, 10, 11, 12, 13, 14)` — elbows, wrists, hips, knees,
  equally weighted. So flailing arms tank the score along with hip
  sway. Raise if everyone scores 0; lower if everyone scores 1000. If
  you want arms to dominate / be ignored, edit `TRACKED_KPTS`.
- **Stick the Landing — hop detection.** `HOP_AIRBORNE_THR = 0.08`
  (standing-ankle must rise this much × leg_len above ground to count
  as airborne) and `HOP_LAND_THR = 0.04` (must return within this ×
  leg_len of ground to count as landed). Raise airborne threshold if
  shuffling triggers the hop detector; lower if students can't get
  high enough on the lab floor.
- **Stick the Landing — accuracy zone.** `ACCURACY_HOT_DIST = 0.25` /
  `ACCURACY_COLD_DIST = 0.75` (leg_len multiples) define the full-bonus
  and zero-bonus distances of the landing hip-x from the target. Widen
  the cold distance if even decent hops score 0 accuracy.
- **Stick the Landing — phase budgets.** `STANCE_HOLD_S = 3.0` /
  `STANCE_TIMEOUT_S = 5.0` / `HOP_TIMEOUT_S = 4.0` / `LAND_HOLD_S = 3.0`.
  Total hard cap = `duration_s = 12.0` (matches worst-case timeout
  chain 5+4+3). Note that STANCE_HOLD=3 inside a 5s budget leaves
  only 2s of wobble allowed — bump `STANCE_TIMEOUT_S` (and `duration_s`)
  if students struggle to settle. Match `duration_s` if you change any
  budget.
- **Stick the Landing — smoothing revisit.** Variances are computed
  from the OneEuro-smoothed positions (same stream the skeleton uses).
  OneEuro preserves slow sway and suppresses jitter, which is the right
  trade-off for balance scoring. If lab measurements come out too
  tightly clustered to discriminate students, consider exposing a
  `position_raw` field from `PoseDetector` and computing sway from
  that instead.

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
- **M6a complete**: Punch Power. Per-frame `|horizontal wrist velocity| /
  arm_length`, where arm_length is a running mean of `|sh-el| + |el-wr|`
  (bend-invariant). Velocity comes from `PoseDetector` (MA-smoothed over
  5 frames). Per-frame current energy is `max(left, right)`; peak across
  the 10 s duration is scored via `cfg.SCORE_MAP["punch_power"]`. Live
  power bar with persistent peak marker matches High Knees / Vertical
  Jump. Deviates from design doc 8.4 (which specifies windowed integrated
  displacement) because that introduced a visible ~0.3 s lag between the
  punch and the bar filling — velocity-based is instant. Horizontal-only
  so vertical motion doesn't count.
- **M7 complete**: Stick the Landing. First composite/multi-phase
  activity. Internal `_StickPhase` enum (STANCE -> HOP -> LAND -> DONE)
  with per-phase dispatch from `update()` and `draw()`. Composite score
  = 0.4 · balance + 0.4 · land_stability + 0.2 · accuracy, all
  computed in body-units (leg-length-normalised). Stillness measured
  as the mean per-keypoint (stdev_x + stdev_y) across `TRACKED_KPTS`
  (elbows + wrists + hips + knees) — flailing arms reduce the score.
  Graceful degrade: phase timeouts zero that sub-score and continue
  rather than aborting the activity. No `SCORE_MAP` entry — points are
  computed directly from the composite quality. The phase-enum +
  dispatch pattern is the template if future events grow phases.
- **Next options**: M6b (Javelin — reuse the displacement-burst pattern
  from `PunchPowerActivity`; the design doc says this is "cheap" after
  Punch Power) or M8 polish (instruction images, day leaderboard, sound,
  Punch Power wrist trail / energy-number / hit flash, raw-position
  source for Stick the Landing variance). All scoring values likely
  need lab-tuning first (see Lab-tuning notes above).

See [README.md](README.md) for the full milestone ladder.

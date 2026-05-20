# Pose Pentathlon

**Disclaimer**: extremely vibe coded. This is for a one-off event. The foundation (video stream capture + rtmlib pose detection) was taken from a previous project - done cleanly, everything on top is quick AI code.

Interactive open-day exhibit: a single webcam runs real-time 2D pose detection
(RTMO / COCO-17). Students see their own skeleton mirrored live, then an
operator runs them through a 5-event athletic mini-pentathlon for a composite
score and athlete title.

Design doc: [docs/POSE_PENTATHLON_PLAN.md](docs/POSE_PENTATHLON_PLAN.md).

## Quickstart

```
conda create -n pentathlon python=3.10 -y
conda run -n pentathlon pip install -r requirements.txt
conda run -n pentathlon python main.py
```

Requires a CUDA-capable GPU (onnxruntime-gpu, pinned to 1.20.1).

## Controls

| Key     | Action                                                |
|---------|-------------------------------------------------------|
| SPACE   | Start circuit (from ATTRACT) / advance / next student |
| N       | Skip current activity                                 |
| R       | Abort, reset to ATTRACT                               |
| Q       | Quit                                                  |

## Status

- **M1 — Mirror + ATTRACT** (done) — live mirrored skeleton + pulsing "STEP IN" prompt
- **M2 — Display layer + state-machine spine** (done) — UI primitives + screen renderers (instructions / countdown / HUD / transition / results); Circuit state machine; full ATTRACT -> ... -> RESULTS flow walkable
- **M3 — Formal `Activity` ABC** (skipped) — duck typing covers it; see CLAUDE.md
- **M4 — Real events: High Knees + Vertical Jump (MVP)** (done) — hysteresis-based rep counting, hip-centre jump tracking, per-event skeleton highlight, vertical power bars with persistent peak markers. Score endpoints in `config.py` likely need lab-tuning.
- **M5 — Reaction Wall** (done) — two simultaneous target circles in a reachable zone, wrist-keypoint hit detection with tunable `HIT_RADIUS_MULT`, hit count scored via `cfg.SCORE_MAP["reaction_wall"]`. Circuit composition is now driven by `cfg.CIRCUIT_ACTIVITIES`, so individual events can be commented out for testing.
- M6 — Punch Power + Javelin
- M7 — Polish: instruction images, day leaderboard, optional sound

## Project files

- `main.py` — capture/pose loop + circuit dispatch
- `circuit.py` — `State`, `StubActivity`, `Circuit`, `build_demo_circuit`
- `activities.py` — real event classes: `HighKneesActivity`, `VerticalJumpActivity`
- `ui.py` — primitives (panel, progress_bar, power_bar, big_digit, text helpers) + screen renderers
- `config.py` — single source for tracking / UI sizing / state-machine timings / `SCORE_MAP`

## Reused from `rtmlib_vfx`

- `capture.py` — `CaptureThread` (threaded non-blocking webcam capture)
- `pose.py` — `PoseDetector`, `OneEuroFilter`

Copied verbatim except for two lines stripped from `pose.py` (a Warp CUDA
sync that's redundant when ONNX is the only GPU consumer).

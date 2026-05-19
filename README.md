# Pose Pentathlon

Interactive open-day exhibit: a single webcam runs real-time 2D pose detection
(RTMO / COCO-17). Students see their own skeleton mirrored live, then an
operator runs them through a 5-event athletic mini-pentathlon for a composite
score and athlete title.

Design doc lives in the parent `rtmlib_vfx` repo at
[docs/POSE_PENTATHLON_PLAN.md](https://github.com/f-fraysse/rtmlib_vfx/blob/main/docs/POSE_PENTATHLON_PLAN.md).

## Quickstart

```
conda create -n pentathlon python=3.10 -y
conda run -n pentathlon pip install -r requirements.txt
conda run -n pentathlon python main.py
```

Requires a CUDA-capable GPU (onnxruntime-gpu, pinned to 1.20.1).

## Controls (M1)

| Key | Action |
|-----|--------|
| `q` | Quit |

(More keys land with M2/M3.)

## Status

- **M1 — Mirror + ATTRACT** ✅ live mirrored skeleton + pulsing "STEP IN" prompt
- M2 — UI primitives + screen renderers
- M3 — `Activity` base + `Circuit` state machine + 2 stub activities
- M4 — Real events: High Knees + Vertical Jump (MVP)
- M5 — Reaction Wall
- M6 — Punch Power + Javelin
- M7 — Polish: instruction images, day leaderboard, optional sound

## Reused from `rtmlib_vfx`

- `capture.py` — `CaptureThread` (threaded non-blocking webcam capture)
- `pose.py` — `PoseDetector`, `OneEuroFilter`

Copied verbatim except for two lines stripped from `pose.py` (a Warp CUDA
sync that's redundant when ONNX is the only GPU consumer).

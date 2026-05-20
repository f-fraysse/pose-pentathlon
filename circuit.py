import time
from enum import Enum, auto

import config as cfg
import ui
from activities import HighKneesActivity, VerticalJumpActivity, ReactionWallActivity


class State(Enum):
    ATTRACT      = auto()
    INSTRUCTIONS = auto()
    COUNTDOWN    = auto()
    ACTIVITY     = auto()
    TRANSITION   = auto()
    RESULTS      = auto()


class StubActivity:
    """Placeholder activity. Matches the design-doc §5 method signatures so
    M3 can promote this into an Activity ABC without changing call sites."""

    def __init__(self, name, instruction_text, duration_s, score_rate):
        self.name = name
        self.instruction_text = instruction_text
        self.instruction_image = None
        self.duration_s = duration_s
        self._score_rate = score_rate
        self._score = 0.0

    def reset(self):
        self._score = 0.0

    def update(self, keypoints, t_elapsed):
        self._score = t_elapsed * self._score_rate

    def draw(self, frame):
        pass

    def is_finished(self, t_elapsed):
        return t_elapsed >= self.duration_s

    def get_result(self):
        pts = int(round(self._score))
        return {"points": pts, "raw": self._score, "display_str": f"{pts}"}

    def highlight(self):
        return {}


def athlete_title(total_points):
    for ceiling, title in cfg.ATHLETE_TITLES:
        if total_points < ceiling:
            return title
    return cfg.ATHLETE_TITLES[-1][1]


class Circuit:
    """Owns the activity list, current index, current state, and stage clock."""

    def __init__(self, activities):
        self.activities = activities
        self.state = State.ATTRACT
        self.idx = 0
        self.stage_start = time.perf_counter()
        self.results = []  # list of (name, points)

    def _enter(self, new_state):
        self.state = new_state
        self.stage_start = time.perf_counter()

    def _elapsed(self):
        return time.perf_counter() - self.stage_start

    def _finish_current(self):
        act = self.activities[self.idx]
        self.results.append((act.name, act.get_result()["points"]))
        self.idx += 1
        if self.idx >= len(self.activities):
            self._enter(State.RESULTS)
        else:
            self._enter(State.TRANSITION)

    def update(self, keypoints):
        t = self._elapsed()
        s = self.state

        if s is State.INSTRUCTIONS and t >= cfg.INSTRUCTIONS_AUTO_SEC:
            self._enter(State.COUNTDOWN)
        elif s is State.COUNTDOWN and t >= cfg.COUNTDOWN_SEC + 1:
            self.activities[self.idx].reset()
            self._enter(State.ACTIVITY)
        elif s is State.ACTIVITY:
            act = self.activities[self.idx]
            act.update(keypoints, t)
            if act.is_finished(t):
                self._finish_current()
        elif s is State.TRANSITION and t >= cfg.TRANSITION_SEC:
            self._enter(State.INSTRUCTIONS)

    def draw_skeleton(self, frame, results):
        hl = {}
        if self.state is State.ACTIVITY:
            hl = self.activities[self.idx].highlight() or {}
        ui.draw_skeleton(frame, results,
                         highlight_joints=hl.get("joints"),
                         highlight_bones=hl.get("bones"))

    def draw(self, frame):
        s = self.state
        if s is State.ATTRACT:
            ui.draw_attract(frame)
        elif s is State.INSTRUCTIONS:
            act = self.activities[self.idx]
            ui.draw_instructions(frame, act.name, act.instruction_text,
                                 act.instruction_image)
        elif s is State.COUNTDOWN:
            remaining = cfg.COUNTDOWN_SEC - int(self._elapsed())
            ui.draw_countdown(frame, remaining)
        elif s is State.ACTIVITY:
            act = self.activities[self.idx]
            act.draw(frame)
            t = self._elapsed()
            ui.draw_activity_hud(frame, act.name,
                                 time_frac=min(1.0, t / act.duration_s),
                                 display=act.get_result()["display_str"])
        elif s is State.TRANSITION:
            prev_name, prev_pts = self.results[-1]
            next_name = self.activities[self.idx].name
            ui.draw_transition(frame, prev_name, next_name, prev_pts)
        elif s is State.RESULTS:
            total = sum(p for _, p in self.results)
            ui.draw_results(frame, self.results, total, athlete_title(total))

    def on_key(self, key):
        """Returns 'quit' if the app should exit; otherwise None."""
        if key == ord('q'):
            return "quit"
        if key == ord('r'):
            self._reset_all()
            return None
        if key == ord('n') and self.state is State.ACTIVITY:
            self._finish_current()
            return None
        if key == ord(' '):
            if self.state is State.ATTRACT:
                self.idx = 0
                self.results = []
                for a in self.activities:
                    a.reset()
                self._enter(State.INSTRUCTIONS)
            elif self.state is State.INSTRUCTIONS:
                self._enter(State.COUNTDOWN)
            elif self.state is State.RESULTS:
                self._reset_all()
        return None

    def _reset_all(self):
        self.idx = 0
        self.results = []
        for a in self.activities:
            a.reset()
        self._enter(State.ATTRACT)


ACTIVITY_REGISTRY = {
    "high_knees":    HighKneesActivity,
    "vertical_jump": VerticalJumpActivity,
    "reaction_wall": ReactionWallActivity,
}


def build_demo_circuit():
    """Build the circuit from cfg.CIRCUIT_ACTIVITIES (ordered list of keys)."""
    activities = []
    for key in cfg.CIRCUIT_ACTIVITIES:
        if key not in ACTIVITY_REGISTRY:
            raise ValueError(
                f"Unknown activity '{key}' in cfg.CIRCUIT_ACTIVITIES. "
                f"Known keys: {sorted(ACTIVITY_REGISTRY)}"
            )
        activities.append(ACTIVITY_REGISTRY[key]())
    if not activities:
        raise ValueError("cfg.CIRCUIT_ACTIVITIES is empty")
    return Circuit(activities)

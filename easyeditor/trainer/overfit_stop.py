"""Train-loss overfit detector for MEND meta-training.

The 2026-09-23 LLaVA-OV VQA outer-ASAM 10k run kept improving train
accuracy after ~4k steps while the 10k final_eval collapsed (edit CE
7.4 vs 0.9 on the 30k non-ASAM control).  This stopper watches the
logged train window, ignores the noisy pre-basin region, and cuts the
run on a sustained rebound — with a hard cap at 10 000 steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence
import argparse
import os
import re
import signal
import time


DEFAULT_KEY = "loss/total_train"
DEFAULT_MIN_STEP = 4000
DEFAULT_HARD_MAX = 10000
DEFAULT_PATIENCE_LOGS = 3
DEFAULT_REL_REBOUND = 0.25
DEFAULT_SMOOTH_WINDOW = 3

_STEP_RE = re.compile(r"Step\s+(\d+):")
_METRIC_RE = re.compile(r"([A-Za-z0-9_./+-]+):\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


@dataclass
class OverfitStopDecision:
    stop: bool
    is_best: bool
    reason: str
    step: int
    best_step: int
    best_value: float
    current_value: Optional[float]
    smoothed_value: Optional[float]
    worse_logs: int


class OverfitStopper:
    """Smoothed-loss early stop with a delayed start and a hard step cap.

    * Updates are the trainer's log_interval windows (not raw steps).
    * ``min_step`` blocks stopping before the first expected basin.
    * A log counts against patience only if the smoothed value is at
      least ``rel_rebound`` worse than the best (so 15% noise does not
      kill a still-descending run).
    * ``hard_max_step`` always stops, even if loss is still falling.
    """

    def __init__(
        self,
        key: str = DEFAULT_KEY,
        min_step: int = DEFAULT_MIN_STEP,
        hard_max_step: int = DEFAULT_HARD_MAX,
        patience_logs: int = DEFAULT_PATIENCE_LOGS,
        rel_rebound: float = DEFAULT_REL_REBOUND,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        higher_is_better: Optional[bool] = None,
    ):
        if min_step < 0:
            raise ValueError("min_step must be >= 0")
        if hard_max_step <= 0:
            raise ValueError("hard_max_step must be > 0")
        if patience_logs < 1:
            raise ValueError("patience_logs must be >= 1")
        if smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        if rel_rebound < 0:
            raise ValueError("rel_rebound must be >= 0")
        self.key = key
        self.min_step = int(min_step)
        self.hard_max_step = int(hard_max_step)
        self.patience_logs = int(patience_logs)
        self.rel_rebound = float(rel_rebound)
        self.smooth_window = int(smooth_window)
        if higher_is_better is None:
            higher_is_better = "acc" in key
        self.higher_is_better = bool(higher_is_better)
        self._window: Deque[float] = deque(maxlen=self.smooth_window)
        self.best_value = float("-inf") if self.higher_is_better else float("inf")
        self.best_step = 0
        self.worse_logs = 0
        self._stop = False
        self._reason = ""

    def _smoothed(self) -> float:
        return sum(self._window) / len(self._window)

    def _is_improvement(self, value: float) -> bool:
        if self.higher_is_better:
            return value > self.best_value
        return value < self.best_value

    def _is_rebound(self, value: float) -> bool:
        if self.best_step == 0:
            return False
        best = self.best_value
        if best == 0:
            delta = abs(value - best)
            return delta > self.rel_rebound
        if self.higher_is_better:
            return value < best * (1.0 - self.rel_rebound)
        return value > best * (1.0 + self.rel_rebound)

    def update(self, step: int, stats: Dict[str, float]) -> OverfitStopDecision:
        step = int(step)
        current: Optional[float] = None
        smoothed: Optional[float] = None
        is_best = False
        if self.key in stats and stats[self.key] is not None:
            current = float(stats[self.key])
            self._window.append(current)
            smoothed = self._smoothed()
            if self._is_improvement(smoothed):
                self.best_value = smoothed
                self.best_step = step
                self.worse_logs = 0
                is_best = True
            elif step >= self.min_step and self._is_rebound(smoothed):
                self.worse_logs += 1
            else:
                self.worse_logs = 0

        reason = ""
        stop = self._stop
        if step >= self.hard_max_step:
            stop = True
            reason = (
                f"hard_max_step={self.hard_max_step} reached at {step}"
            )
        elif (
            step >= self.min_step
            and self.best_step >= self.min_step
            and self.worse_logs >= self.patience_logs
        ):
            stop = True
            reason = (
                f"smoothed {self.key} rebounded {self.worse_logs} logs "
                f"(rel>={self.rel_rebound}) after best {self.best_value:.5f} "
                f"at step {self.best_step}"
            )
        if stop:
            self._stop = True
            self._reason = reason or self._reason

        return OverfitStopDecision(
            stop=self._stop,
            is_best=is_best,
            reason=self._reason,
            step=step,
            best_step=self.best_step,
            best_value=self.best_value,
            current_value=current,
            smoothed_value=smoothed,
            worse_logs=self.worse_logs,
        )

    def should_stop(self) -> bool:
        return self._stop

    def state_dict(self) -> Dict:
        return {
            "key": self.key,
            "min_step": self.min_step,
            "hard_max_step": self.hard_max_step,
            "patience_logs": self.patience_logs,
            "rel_rebound": self.rel_rebound,
            "smooth_window": self.smooth_window,
            "higher_is_better": self.higher_is_better,
            "window": list(self._window),
            "best_value": self.best_value,
            "best_step": self.best_step,
            "worse_logs": self.worse_logs,
            "_stop": self._stop,
            "_reason": self._reason,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._window = deque(state.get("window") or [], maxlen=self.smooth_window)
        self.best_value = float(state["best_value"])
        self.best_step = int(state["best_step"])
        self.worse_logs = int(state.get("worse_logs", 0))
        self._stop = bool(state.get("_stop", False))
        self._reason = str(state.get("_reason", ""))


def overfit_stopper_from_config(config) -> Optional[OverfitStopper]:
    if not bool(getattr(config, "overfit_stop", False)):
        return None
    key = str(getattr(config, "overfit_stop_key", DEFAULT_KEY) or DEFAULT_KEY)
    return OverfitStopper(
        key=key,
        min_step=int(getattr(config, "overfit_stop_min_step", DEFAULT_MIN_STEP)),
        hard_max_step=int(getattr(config, "overfit_stop_hard_max", DEFAULT_HARD_MAX)),
        patience_logs=int(
            getattr(config, "overfit_stop_patience_logs", DEFAULT_PATIENCE_LOGS)
        ),
        rel_rebound=float(
            getattr(config, "overfit_stop_rel_rebound", DEFAULT_REL_REBOUND)
        ),
        smooth_window=int(
            getattr(config, "overfit_stop_smooth_window", DEFAULT_SMOOTH_WINDOW)
        ),
    )


def cap_max_iters(max_iters, hard_max: int = DEFAULT_HARD_MAX) -> int:
    hard_max = int(hard_max)
    if max_iters is None:
        return hard_max
    return min(int(max_iters), hard_max)


def parse_train_log_records(text: str) -> List[Dict[str, float]]:
    """Parse BaseTrainer echo lines into ``{step, <metrics>}`` records."""
    lines = text.splitlines()
    records: List[Dict[str, float]] = []
    i = 0
    while i < len(lines):
        match = _STEP_RE.search(lines[i])
        if not match:
            i += 1
            continue
        step = int(match.group(1))
        blob = lines[i]
        j = i + 1
        while j < len(lines) and not _STEP_RE.search(lines[j]):
            blob += " " + lines[j]
            if "loss/total" in lines[j] or "edit/acc" in lines[j]:
                break
            j += 1
        metrics = {k: float(v) for k, v in _METRIC_RE.findall(blob)}
        if any(k.startswith("loss/") and k.endswith("_train") for k in metrics):
            rec: Dict[str, float] = {"step": step, **metrics}
            records.append(rec)
        i += 1
    return records


def replay(
    records: Sequence[Dict[str, float]],
    stopper: Optional[OverfitStopper] = None,
) -> Optional[OverfitStopDecision]:
    stopper = stopper or OverfitStopper()
    last: Optional[OverfitStopDecision] = None
    for record in records:
        last = stopper.update(int(record["step"]), record)
        if last.stop:
            return last
    return last


def _pid_from_file(path: str) -> int:
    return int(open(path).read().strip().split()[0])


def _watch_and_stop(log_path: str, pid: int, poll_s: float = 15.0) -> int:
    stopper = OverfitStopper()
    seen = 0
    while True:
        if pid and os.path.exists(f"/proc/{pid}"):
            pass
        elif pid:
            print(f"PID {pid} is gone; exiting watcher")
            return 0
        text = open(log_path, errors="replace").read()
        records = parse_train_log_records(text)
        if len(records) > seen:
            stopper = OverfitStopper()
            decision = replay(records, stopper)
            seen = len(records)
            if decision is not None:
                print(
                    f"step={decision.step} smoothed={decision.smoothed_value} "
                    f"best={decision.best_value}@{decision.best_step} "
                    f"worse={decision.worse_logs} stop={decision.stop} "
                    f"{decision.reason}"
                )
            if decision is not None and decision.stop:
                if pid:
                    print(f"Sending SIGTERM to {pid}: {decision.reason}")
                    os.kill(pid, signal.SIGTERM)
                return 0
        time.sleep(poll_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide when to stop a MEND VQA ASAM run from train logs."
    )
    parser.add_argument("--log", required=True, help="Trainer log to parse")
    parser.add_argument("--pid", type=int, default=0, help="Process to SIGTERM")
    parser.add_argument("--pid-file", default="", help="File containing a PID")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll the log until stop (or the process exits)",
    )
    parser.add_argument("--min-step", type=int, default=DEFAULT_MIN_STEP)
    parser.add_argument("--hard-max", type=int, default=DEFAULT_HARD_MAX)
    parser.add_argument("--patience-logs", type=int, default=DEFAULT_PATIENCE_LOGS)
    parser.add_argument("--rel-rebound", type=float, default=DEFAULT_REL_REBOUND)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--key", default=DEFAULT_KEY)
    args = parser.parse_args(argv)

    pid = args.pid
    if args.pid_file:
        pid = _pid_from_file(args.pid_file)

    if args.watch:
        return _watch_and_stop(args.log, pid)

    records = parse_train_log_records(open(args.log, errors="replace").read())
    stopper = OverfitStopper(
        key=args.key,
        min_step=args.min_step,
        hard_max_step=args.hard_max,
        patience_logs=args.patience_logs,
        rel_rebound=args.rel_rebound,
        smooth_window=args.smooth_window,
    )
    decision = replay(records, stopper)
    if decision is None:
        print("NO_RECORDS")
        return 2
    print(
        f"STOP={int(decision.stop)} step={decision.step} "
        f"best_step={decision.best_step} best={decision.best_value:.5f} "
        f"smoothed={decision.smoothed_value} worse_logs={decision.worse_logs}"
    )
    if decision.reason:
        print(decision.reason)
    if decision.stop and not (args.min_step <= decision.step <= args.hard_max):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

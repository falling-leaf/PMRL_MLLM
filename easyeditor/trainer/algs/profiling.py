"""Opt-in section profiler for the MEND meta-training step.

Enable with ``MEND_PROFILE=1`` (optionally ``MEND_PROFILE_EVERY=<n>`` to change
how often the table is logged, default every 20 steps).  When disabled every
:func:`section` call is a no-op context manager, so the training path only pays
a single ``if`` branch.

A section boundary is only meaningful after ``torch.cuda.synchronize()``
because CUDA work is asynchronous; that synchronisation is the reason the
profiler is opt-in rather than always on.
"""

import os
import time
from collections import OrderedDict
from contextlib import contextmanager

PROFILE = os.environ.get("MEND_PROFILE", "0").lower() not in ("", "0", "false", "no")
PROFILE_EVERY = int(os.environ.get("MEND_PROFILE_EVERY", "20") or 20)

_durations = OrderedDict()
_counts = OrderedDict()
_dump_calls = 0


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - never break training for profiling
        pass


@contextmanager
def section(name):
    """Accumulate wall-clock time (per device sync) spent in ``name``."""
    if not PROFILE:
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _durations[name] = _durations.get(name, 0.0) + (time.perf_counter() - start)
        _counts[name] = _counts.get(name, 0) + 1


def dump(logger, steps=None, reset=True):
    """Log the accumulated table every ``MEND_PROFILE_EVERY`` calls."""
    global _dump_calls
    if not PROFILE or not _durations:
        return
    _dump_calls += 1
    if PROFILE_EVERY > 1 and _dump_calls % PROFILE_EVERY:
        return
    total = sum(_durations.values())
    lines = [f"MEND_PROFILE step={steps} steps_covered={_dump_calls} measured_total_s={total:.2f}"]
    for name, secs in sorted(_durations.items(), key=lambda kv: -kv[1]):
        n = _counts[name]
        lines.append(
            f"MEND_PROFILE   {name:<26s} sum={secs:8.3f}s n={n:<7d} "
            f"mean={secs / n * 1000:8.2f}ms {secs / total * 100:5.1f}%"
        )
    logger.info("\n".join(lines))
    if reset:
        _durations.clear()
        _counts.clear()
        _dump_calls = 0

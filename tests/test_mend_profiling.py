"""Functional tests for the opt-in section profiler (`MEND_PROFILE=1`).

The profiler is the instrument used to attribute step time to sections, so its
contract has to be pinned down too: a no-op when disabled, wall-clock
accumulation per section (inclusive of nested sections), periodic dumping and
reset, and no state leaking between dumps.

Run: ``python -m pytest tests/test_mend_profiling.py -q``
"""

import time

from easyeditor.trainer.algs import profiling


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _reset():
    profiling._durations.clear()
    profiling._counts.clear()
    profiling._dump_calls = 0


def test_section_is_a_noop_when_profiling_is_disabled():
    saved, profiling.PROFILE = profiling.PROFILE, False
    try:
        _reset()
        with profiling.section("disabled"):
            pass
        assert not profiling._durations
        assert not profiling._counts
        logger = _Logger()
        profiling.dump(logger, steps=1)
        assert logger.messages == []
    finally:
        profiling.PROFILE = saved
        _reset()


def test_sections_accumulate_inclusive_times_and_dump_resets():
    saved = (profiling.PROFILE, profiling.PROFILE_EVERY)
    profiling.PROFILE, profiling.PROFILE_EVERY = True, 1
    try:
        _reset()
        with profiling.section("a"):
            time.sleep(0.01)
        with profiling.section("a"):
            pass
        with profiling.section("b"):
            pass
        assert profiling._counts == {"a": 2, "b": 1}
        assert profiling._durations["a"] >= 0.01

        # Nested sections: the outer section includes the inner one.
        with profiling.section("outer"):
            with profiling.section("inner"):
                time.sleep(0.005)
        assert profiling._durations["outer"] >= profiling._durations["inner"] >= 0.005

        logger = _Logger()
        profiling.dump(logger, steps=7)
        assert len(logger.messages) == 1
        report = logger.messages[0]
        assert "step=7" in report and "measured_total_s" in report
        assert "outer" in report and " inner" in report and "a" in report
        # sorted by cost, so the sleep-heavy sections come first
        assert report.index("outer") < report.index("b")
        # a dumped report resets the accumulators, so sections do not bleed
        # across dump windows
        assert not profiling._durations and not profiling._counts
        assert profiling._dump_calls == 0
    finally:
        profiling.PROFILE, profiling.PROFILE_EVERY = saved
        _reset()


def test_dump_respects_profile_every_gate():
    saved = (profiling.PROFILE, profiling.PROFILE_EVERY)
    profiling.PROFILE, profiling.PROFILE_EVERY = True, 3
    try:
        _reset()
        logger = _Logger()
        for call in range(1, 7):
            with profiling.section("x"):
                pass
            profiling.dump(logger, steps=call)
        # printed on call 3 and 6 only
        assert len(logger.messages) == 2
        assert "step=3" in logger.messages[0]
        assert "step=6" in logger.messages[1]
        assert profiling._dump_calls == 0
    finally:
        profiling.PROFILE, profiling.PROFILE_EVERY = saved
        _reset()

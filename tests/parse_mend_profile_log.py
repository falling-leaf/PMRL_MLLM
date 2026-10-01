"""Parse a MEND_PROFILE log into per-step section timing + wall-clock stats.

Sections are nested (edit.loss wraps edit.lap_probe/lap_view/edit.pmrl), so a
naive sum of ``measured_total_s`` double counts.  This script reports both the
raw sum and a de-duplicated sum (children of any section removed from the
parent's total).

Usage: python tests/parse_mend_profile_log.py run_logs/<log> [--window N]
"""

import re
import sys
from datetime import datetime

TS = re.compile(r"^(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - .*? - (.*)$")
SEC = re.compile(
    r"^MEND_PROFILE\s+(\S+)\s+sum=\s*([\d.]+)s n=(\d+)\s+mean=\s*([\d.]+)ms"
)
HDR = re.compile(r"^MEND_PROFILE step=(\d+) steps_covered=(\d+) measured_total_s=([\d.]+)")
STEP = re.compile(r"^Step (\d+):$")

# child -> parent (a child's time is already inside the parent's sum)
PARENT = {
    "edit.lap_probe": "edit.loss",
    "edit.lap_view": "edit.loss",
    "edit.pmrl": "edit.loss",
}


def parse(path):
    windows = []
    cur = None
    echoes = []  # (wall_clock, step)
    with open(path, errors="replace") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            m = TS.match(raw)
            ts = datetime.strptime(m.group(1), "%m/%d/%Y %H:%M:%S") if m else None
            body = (m.group(2) if m else raw).strip()
            h = HDR.match(body)
            if h:
                cur = {
                    "step": int(h.group(1)),
                    "covered": int(h.group(2)),
                    "raw_total": float(h.group(3)),
                    "secs": {},
                    "n": {},
                    "ts": ts,
                }
                windows.append(cur)
                continue
            s = SEC.match(body)
            if s and cur is not None:
                cur["secs"][s.group(1)] = float(s.group(2))
                cur["n"][s.group(1)] = int(s.group(3))
                continue
            st = STEP.match(body)
            if st:
                echoes.append((ts, int(st.group(1))))
    return windows, echoes


def main():
    path = sys.argv[1]
    windows, echoes = parse(path)
    if not windows:
        print("no MEND_PROFILE tables found")
        return
    print(f"{'window(step)':>14} {'cov':>4} {'raw_s':>8} {'dedup_s':>8} "
          f"{'dedup/step':>10} {'wall_s':>8} {'wall/step':>9} {'uncovered':>10}")
    prev_ts = None
    all_dedup = []
    all_wall = []
    for w in windows:
        dedup = sum(v for k, v in w["secs"].items() if k not in PARENT)
        per_step = dedup / w["covered"]
        wall = (w["ts"] - prev_ts).total_seconds() if prev_ts else float("nan")
        wall_step = wall / w["covered"] if prev_ts else float("nan")
        unc = (wall - dedup) / w["covered"] if prev_ts else float("nan")
        prev_ts = w["ts"]
        all_dedup.append(per_step)
        if wall == wall:
            all_wall.append(wall_step)
        print(f"{w['step']:>14} {w['covered']:>4} {w['raw_total']:>8.2f} "
              f"{dedup:>8.2f} {per_step:>10.2f} {wall:>8.1f} {wall_step:>9.2f} "
              f"{unc:>10.2f}")
    if len(all_wall) > 1:
        # drop the first (warm-up) window
        steady = all_wall[1:]
        print()
        print(f"steady windows: n={len(steady)} "
              f"wall/step mean={sum(steady) / len(steady):.3f}s "
              f"min={min(steady):.3f} max={max(steady):.3f}")
        sd = sum(all_dedup[1:]) / len(all_dedup[1:])
        print(f"section-sum/step (dedup, steady) mean={sd:.3f}s")
    last = windows[-1]
    print("\nlast window sections (dedup, per step):")
    for k, v in sorted(last["secs"].items(), key=lambda kv: -kv[1]):
        tag = "  (in parent)" if k in PARENT else ""
        print(f"  {k:<32} {v / last['covered'] * 1000:8.1f} ms/step{tag}")
    if echoes:
        print()
        print("step echoes:")
        for i, (ts, st) in enumerate(echoes):
            d = (ts - echoes[i - 1][0]).total_seconds() if i else float("nan")
            per = d / 10 if i else float("nan")
            print(f"  step {st:>5} at {ts} delta={d:8.1f}s per_step={per:5.2f}s")


if __name__ == "__main__":
    main()

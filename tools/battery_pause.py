#!/usr/bin/env python3
"""Rest between long CPU/GPU jobs so a laptop on a small adapter can recharge.

Reads the battery controller through `ioreg` (no root needed). A rest is a
fixed sleep followed by a gate: while the machine runs on battery, or the
charge is below `low`, it waits until it is on external power with at least
`high` percent. Without a readable battery the gate is a no-op.

A recharge break (`gain`) waits until the charge has risen by `gain` points,
capped at `cap` percent; it ends early when the charger stops charging while
connected (full, or held by optimised charging).

As a module: `rest(work_seconds, ratio, low, high, gain)` between work items.
As a command: `battery_pause.py --rest 300 --gain 15` between test blocks.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

POLL_SECONDS = 60


def _signed(value):
    value = int(value)
    return value - (1 << 64) if value >= 1 << 63 else value


def read():
    """Battery status, or None when no battery controller is readable."""
    try:
        text = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True,
                              timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    def field(name):
        match = re.search(r'"%s" = (\w+)' % name, text)
        return match.group(1) if match else None

    percent = field("CurrentCapacity")
    if percent is None:
        return None
    telemetry = re.search(r'"PowerTelemetryData" = \{([^}]*)\}', text)
    values = dict(re.findall(r'"(\w+)"=(\d+)', telemetry.group(1))) if telemetry else {}
    return {"percent": int(percent), "external": field("ExternalConnected") == "Yes",
            "charging": field("IsCharging") == "Yes",
            "battery_w": _signed(values["BatteryPower"]) / 1000.0 if "BatteryPower" in values else None,
            "load_w": int(values["SystemLoad"]) / 1000.0 if "SystemLoad" in values else None}


def rest(work_seconds=0.0, ratio=0.0, low=None, high=None, gain=None, cap=90, log=sys.stderr):
    """Sleep ratio*work_seconds, then wait for the battery gate and the recharge break."""
    pause = max(0.0, ratio * work_seconds)
    before = read()
    if pause > 0:
        time.sleep(pause)
    status = read()
    waited = 0
    target = None
    if status and low is not None and (not status["external"] or status["percent"] < low):
        target = max(low, high if high is not None else low)
    if status and gain and status["percent"] < cap:
        target = max(target or 0, min(cap, status["percent"] + gain))
    while status and target is not None and (not status["external"] or status["percent"] < target):
        if status["external"] and not status["charging"]:
            break  # connected but not charging: full or held by optimised charging
        print(json.dumps({"battery_wait": status, "target_percent": target}), file=log, flush=True)
        time.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        status = read()
    record = {"rest_seconds": round(pause, 1), "battery_wait_seconds": waited, "target_percent": target,
              "battery_before": before, "battery_after": status}
    print(json.dumps(record), file=log, flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest", type=float, default=0.0, help="fixed sleep in seconds before the gate")
    parser.add_argument("--low", type=int, help="wait when the charge is below this percent (or on battery)")
    parser.add_argument("--high", type=int, help="charge to reach before resuming (default: --low)")
    parser.add_argument("--gain", type=int, help="recharge break: wait until the charge rose by this many points")
    parser.add_argument("--cap", type=int, default=90, help="upper bound of the recharge break target")
    parser.add_argument("--status", action="store_true", help="print the battery status and exit")
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(read()))
        return 0
    rest(args.rest, 1.0, args.low, args.high, args.gain, args.cap, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

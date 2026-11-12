#!/usr/bin/env python3
"""Unit tests for tools/battery_pause.py (no battery or sleeping needed)."""
import io
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import battery_pause  # noqa: E402


def status(percent, external=True, charging=True):
    return {"percent": percent, "external": external, "charging": charging, "battery_w": 0.0, "load_w": 0.0}


class Rest(unittest.TestCase):
    def run_rest(self, readings, **kwargs):
        sleeps = []
        with mock.patch.object(battery_pause, "read", side_effect=list(readings)), \
                mock.patch.object(battery_pause.time, "sleep", side_effect=sleeps.append):
            record = battery_pause.rest(log=io.StringIO(), **kwargs)
        return record, sleeps

    def test_fixed_rest_without_gate(self):
        record, sleeps = self.run_rest([status(40), status(40)], work_seconds=80, ratio=0.25)
        self.assertEqual(sleeps, [20.0])
        self.assertIsNone(record["target_percent"])

    def test_low_gate_waits_for_high(self):
        record, sleeps = self.run_rest([status(49), status(49), status(60), status(70)], low=50, high=70)
        self.assertEqual(record["target_percent"], 70)
        self.assertEqual(record["battery_wait_seconds"], 2 * battery_pause.POLL_SECONDS)
        self.assertEqual(record["battery_after"]["percent"], 70)

    def test_unplugged_waits_even_above_low(self):
        readings = [status(80, external=False), status(80, external=False), status(80)]
        record, _ = self.run_rest(readings, low=50, high=70)
        self.assertEqual(record["battery_wait_seconds"], battery_pause.POLL_SECONDS)

    def test_recharge_break_gains_points_up_to_the_cap(self):
        record, _ = self.run_rest([status(53), status(53), status(65), status(73)], gain=20)
        self.assertEqual(record["target_percent"], 73)
        record, _ = self.run_rest([status(85), status(85), status(90)], gain=20, cap=90)
        self.assertEqual(record["target_percent"], 90)
        record, sleeps = self.run_rest([status(95), status(95)], gain=20, cap=90)
        self.assertIsNone(record["target_percent"])
        self.assertEqual(sleeps, [])

    def test_break_ends_when_the_charger_holds(self):
        record, _ = self.run_rest([status(78), status(78), status(80, charging=False)], gain=20)
        self.assertEqual(record["target_percent"], 90)
        self.assertEqual(record["battery_wait_seconds"], battery_pause.POLL_SECONDS)

    def test_no_battery_is_a_no_op(self):
        record, sleeps = self.run_rest([None, None], low=50, high=70, gain=20)
        self.assertIsNone(record["target_percent"])
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()

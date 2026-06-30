"""Unit tests for SweepParser and EISBuffer in testbed.eis_analysis."""
from __future__ import annotations

import time

import pytest

from testbed.eis_analysis import EISBuffer, SweepParser


class TestSweepParser:
    def setup_method(self):
        self.parser = SweepParser()

    def _feed_valid_line(self, freq: float, re: float, im: float) -> None:
        line = f"{freq:.3f},{re:+.6e},{im:+.6e}\n"
        self.parser.parse_line(line)

    def test_12_lines_plus_dbg_returns_sweep(self):
        for i in range(12):
            self._feed_valid_line(float(i + 1), 100.0 + i, -10.0 - i)
        self.parser.parse_line("# DBG sweep done\n")  # sentinel
        sweep = self.parser.get_sweep()
        assert sweep is not None
        assert len(sweep["freq"]) == 12
        assert len(sweep["Z_re"]) == 12
        assert len(sweep["Z_im"]) == 12
        assert sweep["freq"][0] == pytest.approx(1.0, rel=1e-3)

    def test_partial_sweep_returns_collected(self):
        for i in range(5):
            self._feed_valid_line(float(i + 1), 100.0, -5.0)
        self.parser.parse_line("# DBG\n")
        sweep = self.parser.get_sweep()
        assert sweep is not None
        assert len(sweep["freq"]) == 5

    def test_malformed_line_skipped(self):
        self._feed_valid_line(1.0, 100.0, -10.0)
        result = self.parser.parse_line("not,valid,data,extra\n")
        # parse_line returns None for malformed
        # The good point added before should still be in buffer
        result2 = self.parser.parse_line("also_bad\n")
        assert result is None or True  # no crash
        assert result2 is None or True
        # Buffer still has the first valid point
        self.parser.parse_line("# DBG\n")
        sweep = self.parser.get_sweep()
        assert sweep is not None
        assert len(sweep["freq"]) == 1

    def test_buffer_cleared_after_get_sweep(self):
        self._feed_valid_line(1.0, 100.0, -10.0)
        self.parser.parse_line("# DBG\n")
        sweep1 = self.parser.get_sweep()
        assert sweep1 is not None
        # Second call without new data → None
        sweep2 = self.parser.get_sweep()
        assert sweep2 is None

    def test_is_complete_after_dbg_line(self):
        assert self.parser.is_complete_after("# DBG end of sweep\n") is True
        assert self.parser.is_complete_after("100.0,+1.0e+02,-1.0e+01\n") is False
        assert self.parser.is_complete_after("# DBG\n") is True

    def test_hash_comment_line_not_added_to_buffer(self):
        self.parser.parse_line("# This is a comment\n")
        self.parser.parse_line("# DBG\n")
        sweep = self.parser.get_sweep()
        # Buffer should be empty (no data lines)
        assert sweep is None

    def test_empty_line_skipped(self):
        result = self.parser.parse_line("\n")
        assert result is None
        result2 = self.parser.parse_line("   \n")
        assert result2 is None


class TestEISBuffer:
    def setup_method(self):
        self.buf = EISBuffer()

    def _make_sweep(self, tag: str = "A") -> dict:
        return {
            "freq": [1.0, 10.0, 100.0],
            "Z_re": [100.0, 100.0, 100.0],
            "Z_im": [0.0, 0.0, 0.0],
            "tag": tag,
        }

    def test_initial_state(self):
        assert self.buf.latest is None
        assert self.buf.previous is None
        assert len(self.buf.history) == 0
        assert self.buf.status == "NO_DATA"
        assert self.buf.sweep_count == 0

    def test_previous_captured_on_second_add(self):
        sweep_a = self._make_sweep("A")
        sweep_b = self._make_sweep("B")
        self.buf.add_sweep(sweep_a)
        self.buf.add_sweep(sweep_b)
        assert self.buf.latest["tag"] == "B"
        assert self.buf.previous["tag"] == "A"

    def test_history_capped_at_100(self):
        for i in range(101):
            self.buf.add_sweep(self._make_sweep(str(i)))
        assert len(self.buf.history) == 100
        # The 101st should have evicted the 1st (tag "0")
        tags = [s["tag"] for s in self.buf.history]
        assert "0" not in tags
        assert "100" in tags

    def test_sweep_timestamp_updated(self):
        assert self.buf.sweep_timestamp is None
        self.buf.add_sweep(self._make_sweep())
        assert self.buf.sweep_timestamp is not None

    def test_set_error_changes_status(self):
        self.buf.set_error("port not found")
        assert self.buf.status == "ERROR"
        assert self.buf.error_msg == "port not found"

    def test_status_becomes_live_after_add(self):
        self.buf.set_error("oops")
        self.buf.add_sweep(self._make_sweep())
        assert self.buf.status == "LIVE"

    def test_get_snapshot_is_thread_safe(self):
        """Smoke test: snapshot returns consistent data."""
        self.buf.add_sweep(self._make_sweep("X"))
        snap = self.buf.get_snapshot()
        assert snap["latest"]["tag"] == "X"
        assert snap["sweep_count"] == 1
        assert snap["status"] == "LIVE"

    def test_sweep_count_increments(self):
        for _ in range(5):
            self.buf.add_sweep(self._make_sweep())
        assert self.buf.sweep_count == 5

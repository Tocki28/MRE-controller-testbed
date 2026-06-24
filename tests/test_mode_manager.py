"""Unit tests for ModeManager state machine."""

from __future__ import annotations

from testbed.mode_manager import ModeManager


def test_initial_state_is_idle() -> None:
    mm = ModeManager()
    assert mm.mode == "IDLE"


def test_idle_to_heating_on_start() -> None:
    mm = ModeManager()
    mm.safe_trigger("start")
    assert mm.mode == "HEATING"


def test_heating_to_run_nominal_on_target_reached() -> None:
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    assert mm.mode == "RUN_NOMINAL"


def test_fault_trip_from_run_nominal() -> None:
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("fault_detected")
    assert mm.mode == "FAULT_RECOVERY"


def test_recovery_complete_returns_to_run_nominal() -> None:
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("fault_detected")
    assert mm.mode == "FAULT_RECOVERY"
    mm.safe_trigger("recovery_complete")
    assert mm.mode == "RUN_NOMINAL"

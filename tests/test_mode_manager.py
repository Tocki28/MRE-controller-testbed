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


def test_run_nominal_to_electrode_degrading():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    assert mm.mode == "RUN_NOMINAL"
    mm.safe_trigger("electrode_degrading")
    assert mm.mode == "ELECTRODE_DEGRADING"


def test_electrode_degrading_to_electrode_swap():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("electrode_degrading")
    mm.safe_trigger("hot_swap_command")
    assert mm.mode == "ELECTRODE_SWAP"


def test_electrode_swap_to_run_nominal_on_complete():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("electrode_degrading")
    mm.safe_trigger("hot_swap_command")
    mm.safe_trigger("swap_complete")
    assert mm.mode == "RUN_NOMINAL"


def test_heating_abort_to_safe_shutdown():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("abort")
    assert mm.mode == "SAFE_SHUTDOWN"


def test_bath_depleted_fault_to_fault_recovery():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("bath_depleted")
    mm.safe_trigger("fault_detected")
    assert mm.mode == "FAULT_RECOVERY"


def test_run_nominal_to_bath_depleted():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("bath_depleted")
    assert mm.mode == "BATH_DEPLETED"


def test_bath_depleted_to_draining():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("bath_depleted")
    mm.safe_trigger("drain_command")
    assert mm.mode == "DRAINING"


def test_draining_to_cleanout():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("drain_command")
    mm.safe_trigger("drain_complete")
    assert mm.mode == "CLEANOUT"


def test_cleanout_to_idle():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("drain_command")
    mm.safe_trigger("drain_complete")
    mm.safe_trigger("cleanout_complete")
    assert mm.mode == "IDLE"


def test_fault_recovery_to_safe_shutdown_on_unrecoverable():
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    mm.safe_trigger("fault_detected")
    assert mm.mode == "FAULT_RECOVERY"
    mm.safe_trigger("unrecoverable")
    assert mm.mode == "SAFE_SHUTDOWN"

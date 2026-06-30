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


# ---------------------------------------------------------------------------
# M5.2 — helpers
# ---------------------------------------------------------------------------

def _mode_in_run_nominal() -> ModeManager:
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    return mm


# ---------------------------------------------------------------------------
# M5.2 — non-recoverable fault triggers (→ SAFE_SHUTDOWN)
# ---------------------------------------------------------------------------

def test_anode_burnout_from_run_nominal_to_safe_shutdown():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("anode_burnout_detected", fault_name="anode_burnout")
    assert mm.mode == "SAFE_SHUTDOWN"


def test_power_loss_from_run_nominal_to_safe_shutdown():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("power_loss_detected", fault_name="power_loss")
    assert mm.mode == "SAFE_SHUTDOWN"


def test_power_loss_from_heating_to_safe_shutdown():
    """power_loss can occur during HEATING before RUN_NOMINAL."""
    mm = ModeManager()
    mm.safe_trigger("start")
    assert mm.mode == "HEATING"
    mm.safe_trigger("power_loss_detected", fault_name="power_loss")
    assert mm.mode == "SAFE_SHUTDOWN"


# ---------------------------------------------------------------------------
# M5.2 — recoverable fault triggers (→ FAULT_RECOVERY)
# ---------------------------------------------------------------------------

def test_melt_freeze_from_run_nominal_to_fault_recovery():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("melt_freeze_detected", fault_name="melt_freeze")
    assert mm.mode == "FAULT_RECOVERY"


def test_electrode_short_from_run_nominal_to_fault_recovery():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("electrode_short_detected", fault_name="electrode_short")
    assert mm.mode == "FAULT_RECOVERY"


def test_bath_depletion_detected_from_run_nominal_to_bath_depleted():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("bath_depletion_detected", fault_name="bath_depletion")
    assert mm.mode == "BATH_DEPLETED"


def test_sensor_dropout_from_run_nominal_to_fault_recovery():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("sensor_dropout_detected", fault_name="sensor_dropout")
    assert mm.mode == "FAULT_RECOVERY"


def test_cathode_flooding_from_run_nominal_to_fault_recovery():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("cathode_flooding_detected", fault_name="cathode_flooding")
    assert mm.mode == "FAULT_RECOVERY"


def test_offgas_blockage_from_run_nominal_to_fault_recovery():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("offgas_blockage_detected", fault_name="offgas_blockage")
    assert mm.mode == "FAULT_RECOVERY"


# ---------------------------------------------------------------------------
# M5.2 — active_fault tracking
# ---------------------------------------------------------------------------

def test_active_fault_is_set_on_named_trigger():
    mm = _mode_in_run_nominal()
    assert mm.active_fault is None
    mm.safe_trigger("melt_freeze_detected", fault_name="melt_freeze")
    assert mm.active_fault == "melt_freeze"


def test_active_fault_is_none_initially():
    mm = ModeManager()
    assert mm.active_fault is None


def test_clear_active_fault():
    mm = _mode_in_run_nominal()
    mm.safe_trigger("sensor_dropout_detected", fault_name="sensor_dropout")
    assert mm.active_fault == "sensor_dropout"
    mm.clear_active_fault()
    assert mm.active_fault is None


# ---------------------------------------------------------------------------
# M5.2 — invalid trigger from IDLE is silently ignored
# ---------------------------------------------------------------------------

def test_invalid_fault_trigger_from_idle_is_ignored():
    """Fault triggers must not crash when machine is in IDLE (wrong source)."""
    mm = ModeManager()
    assert mm.mode == "IDLE"
    mm.safe_trigger("melt_freeze_detected")   # invalid source — should be silently ignored
    assert mm.mode == "IDLE"                  # state unchanged

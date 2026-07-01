"""M5.5 — Recovery sequence validation for all fault types."""

from __future__ import annotations

import pytest

from testbed.mode_manager import ModeManager


def _run_nominal() -> ModeManager:
    mm = ModeManager()
    mm.safe_trigger("start")
    mm.safe_trigger("target_reached")
    return mm


class TestNonRecoverableFaults:
    @pytest.mark.parametrize("trigger", [
        "anode_burnout_detected",
        "power_loss_detected",
    ])
    def test_goes_to_safe_shutdown(self, trigger: str) -> None:
        mm = _run_nominal()
        mm.safe_trigger(trigger)
        assert mm.mode == "SAFE_SHUTDOWN"


class TestRecoverableFaults:
    @pytest.mark.parametrize("trigger", [
        "melt_freeze_detected",
        "electrode_short_detected",
        "sensor_dropout_detected",
        "cathode_flooding_detected",
        "offgas_blockage_detected",
    ])
    def test_fault_recovery_then_run_nominal(self, trigger: str) -> None:
        mm = _run_nominal()
        mm.safe_trigger(trigger)
        assert mm.mode == "FAULT_RECOVERY"
        mm.safe_trigger("recovery_complete")
        assert mm.mode == "RUN_NOMINAL"


class TestBathDepletionPath:
    def test_bath_depletion_goes_to_bath_depleted(self) -> None:
        mm = _run_nominal()
        mm.safe_trigger("bath_depletion_detected")
        assert mm.mode == "BATH_DEPLETED"

    def test_bath_depleted_drains_on_command(self) -> None:
        mm = _run_nominal()
        mm.safe_trigger("bath_depletion_detected")
        assert mm.mode == "BATH_DEPLETED"
        mm.safe_trigger("drain_command")
        assert mm.mode == "DRAINING"


class TestRecoveryToELECTRODE_DEGRADING:
    def test_recovery_complete_degraded_returns_to_electrode_degrading(self) -> None:
        mm = _run_nominal()
        mm.safe_trigger("electrode_degrading")
        assert mm.mode == "ELECTRODE_DEGRADING"
        mm.safe_trigger("melt_freeze_detected")
        assert mm.mode == "FAULT_RECOVERY"
        mm.safe_trigger("recovery_complete_degraded")
        assert mm.mode == "ELECTRODE_DEGRADING"


class TestUnrecoverableAfterRepeatedFailures:
    def test_three_faults_in_fault_recovery_trigger_unrecoverable(self) -> None:
        mm = _run_nominal()
        mm.safe_trigger("fault_detected")
        assert mm.mode == "FAULT_RECOVERY"
        mm.safe_trigger("unrecoverable")
        assert mm.mode == "SAFE_SHUTDOWN"

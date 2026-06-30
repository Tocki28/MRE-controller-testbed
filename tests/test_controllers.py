"""Tests for HeaterPID, AdaptiveCurrent, and CompositeController.

Tests cover:
- HeaterPID: saturation at max error, clamping at positive error,
  anti-windup integral bounds, and derivative damping sign.
- AdaptiveCurrent: health scaling at 1.0, 0.0, 0.5, and output clamping.
- CompositeController.compute_setpoints: all 9 named mode branches plus
  the SAFE_SHUTDOWN / unknown fallback.
"""

from __future__ import annotations

import pytest

from testbed.controllers import (
    AdaptiveCurrent,
    CompositeController,
    HeaterPID,
    BASE_CURRENT,
    T_SETPOINT,
    _PHASE_CURRENT,
)
from testbed.plant import PlantState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(T_bulk: float = 1580.0, health: float = 1.0) -> PlantState:
    """Return a PlantState with the given bulk temperature and electrode health."""
    s = PlantState()
    s.T_bulk = T_bulk
    s.electrode_health = health
    return s


def make_controller() -> CompositeController:
    """Return a fresh CompositeController (fresh PID state)."""
    return CompositeController()


# ---------------------------------------------------------------------------
# HeaterPID
# ---------------------------------------------------------------------------

class TestHeaterPID:
    def test_far_below_setpoint_output_is_max(self):
        """Large negative error → output saturates at OUT_MAX (10 000 W)."""
        pid = HeaterPID()
        out = pid.compute(T_bulk=1000.0, dt=1.0)
        assert out == pytest.approx(HeaterPID.OUT_MAX)

    def test_above_setpoint_output_is_zero(self):
        """T_bulk above setpoint → output clamped to 0 (no negative power)."""
        pid = HeaterPID()
        out = pid.compute(T_bulk=T_SETPOINT + 50.0, dt=1.0)
        assert out == pytest.approx(0.0)

    def test_anti_windup_integral_bounded(self):
        """After 1000 steps at max positive error, integral stays within [-OUT_MAX, OUT_MAX]."""
        pid = HeaterPID()
        for _ in range(1000):
            pid.compute(T_bulk=0.0, dt=1.0)   # large positive error every step
        assert -HeaterPID.OUT_MAX <= pid._integral <= HeaterPID.OUT_MAX

    def test_derivative_damps_on_decreasing_error(self):
        """Two calls with decreasing error → derivative > 0 (braking contribution)."""
        pid = HeaterPID()
        # First call: error = T_SETPOINT - 1400 = 180
        pid.compute(T_bulk=1400.0, dt=1.0)
        # Second call: error = T_SETPOINT - 1500 = 80  → error decreased
        # derivative = (80 - 180) / 1.0 = -100
        # KD * derivative = 10 * -100 = -1000  (positive braking on output reduction)
        # We verify by checking that KD * (e2 - e1) / dt is negative when error shrinks,
        # but the _damping_ is in the right direction: it resists the change.
        # A simpler observable: the derivative contribution sign.
        # error went down → derivative term = KD*(e_new - e_old)/dt < 0, reducing over-shoot.
        # We'll measure the actual output change relative to a PID with KD=0.
        pid_no_d = HeaterPID()
        pid_no_d.KD = 0.0  # type: ignore[attr-defined] — patch for comparison
        pid_no_d.compute(T_bulk=1400.0, dt=1.0)

        pid_with_d = HeaterPID()
        pid_with_d.compute(T_bulk=1400.0, dt=1.0)

        out_no_d = pid_no_d.compute(T_bulk=1500.0, dt=1.0)
        out_with_d = pid_with_d.compute(T_bulk=1500.0, dt=1.0)

        # With derivative, error is falling → derivative damps output downward.
        # Both may saturate at 10 000; let's use T_bulk close to setpoint to avoid saturation.
        pid_nd2 = HeaterPID()
        pid_nd2.KD = 0.0  # type: ignore[attr-defined]
        pid_nd2.compute(T_bulk=1575.0, dt=1.0)

        pid_wd2 = HeaterPID()
        pid_wd2.compute(T_bulk=1575.0, dt=1.0)

        out_nd2 = pid_nd2.compute(T_bulk=1579.0, dt=1.0)  # error now = 1.0 (fell from 5.0)
        out_wd2 = pid_wd2.compute(T_bulk=1579.0, dt=1.0)

        # Decreasing error → derivative contribution is negative → with-D output < without-D
        assert out_wd2 < out_nd2, (
            f"Expected derivative to reduce output when error is falling, "
            f"got with_D={out_wd2:.2f}, no_D={out_nd2:.2f}"
        )


# ---------------------------------------------------------------------------
# AdaptiveCurrent
# ---------------------------------------------------------------------------

class TestAdaptiveCurrent:
    """AdaptiveCurrent uses bath phase 'Fe' (base=80 A) by default in compute()."""

    # The default bath_phase is "Fe" → base = 80.0 A
    _FE_BASE = _PHASE_CURRENT["Fe"]   # 80.0

    def test_health_1_gives_base_current(self):
        """health=1.0 → I_sp = BASE*Fe * (0.8*1 + 0.2) = 80*1.0 = 80 A (Fe phase)."""
        ac = AdaptiveCurrent()
        out = ac.compute(electrode_health_est=1.0, bath_phase="Fe")
        assert out == pytest.approx(self._FE_BASE * 1.0)

    def test_health_0_gives_20_percent(self):
        """health=0.0 → I_sp = 80 * 0.2 = 16 A (above I_MIN=10)."""
        ac = AdaptiveCurrent()
        out = ac.compute(electrode_health_est=0.0, bath_phase="Fe")
        assert out == pytest.approx(self._FE_BASE * 0.2)

    def test_health_half(self):
        """health=0.5 → I_sp = 80 * (0.8*0.5 + 0.2) = 80 * 0.6 = 48 A."""
        ac = AdaptiveCurrent()
        out = ac.compute(electrode_health_est=0.5, bath_phase="Fe")
        assert out == pytest.approx(self._FE_BASE * 0.6)

    def test_nominal_health_with_base_current_phase(self):
        """Using BASE_CURRENT (no phase override) → health=1.0 gives BASE_CURRENT."""
        ac = AdaptiveCurrent()
        # Pass an unknown phase so it falls back to BASE_CURRENT inside compute
        # Actually, looking at the code: unknown phase → base = BASE_CURRENT = 150
        out = ac.compute(electrode_health_est=1.0, bath_phase="unknown_phase")
        assert out == pytest.approx(BASE_CURRENT * 1.0)

    def test_health_0_with_base_current_phase(self):
        """Unknown phase → base=150; health=0.0 → 150*0.2=30 A (above I_MIN=10)."""
        ac = AdaptiveCurrent()
        out = ac.compute(electrode_health_est=0.0, bath_phase="unknown_phase")
        assert out == pytest.approx(BASE_CURRENT * 0.2)

    def test_health_half_with_base_current_phase(self):
        """Unknown phase → base=150; health=0.5 → 150*0.6=90 A."""
        ac = AdaptiveCurrent()
        out = ac.compute(electrode_health_est=0.5, bath_phase="unknown_phase")
        assert out == pytest.approx(BASE_CURRENT * 0.6)

    def test_output_never_below_i_min(self):
        """Pathological health value still clamps to I_MIN=10."""
        ac = AdaptiveCurrent()
        # Using a very small base phase current and health=0 → might dip below 10
        # Force it: use Fe phase (80 A base) at health=0 → 16 A, but test the clamp
        # directly by using a mock-low computation.  Instead test with "complete" phase
        # (base=40): 40 * 0.2 = 8 A → clamped to 10.
        out = ac.compute(electrode_health_est=0.0, bath_phase="complete")
        assert out == pytest.approx(AdaptiveCurrent.I_MIN)

    def test_output_never_above_i_max(self):
        """High-voltage phase at health=1.0 must stay at or below I_MAX=200."""
        ac = AdaptiveCurrent()
        # Al_Ti base=160: 160*1.0=160, well under 200
        out = ac.compute(electrode_health_est=1.0, bath_phase="Al_Ti")
        assert out <= AdaptiveCurrent.I_MAX


# ---------------------------------------------------------------------------
# CompositeController — mode branches
# ---------------------------------------------------------------------------

class TestCompositeControllerModes:
    """Test all 9 named mode branches plus SAFE_SHUTDOWN fallback."""

    # ------------------------------------------------------------------
    # 5. IDLE
    # ------------------------------------------------------------------
    def test_idle_all_off(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1580.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "IDLE")
        assert result["heater_power"] == pytest.approx(0.0)
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 6. HEATING — T_bulk well below setpoint
    # ------------------------------------------------------------------
    def test_heating_heater_on_no_current(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1000.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "HEATING")
        assert result["heater_power"] > 0.0
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 7. RUN_NOMINAL — at setpoint, full health
    # ------------------------------------------------------------------
    def test_run_nominal_at_setpoint_full_health(self):
        ctrl = make_controller()
        state = make_state(T_bulk=T_SETPOINT, health=1.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "RUN_NOMINAL")
        # PID error = 0 on first call (integral=0, derivative from 0 error) → heater near 0
        # Allow small positive value due to first-call derivative (prev_error=0 → deriv=0 too)
        assert result["heater_power"] == pytest.approx(0.0, abs=1.0)
        # Adaptive current: Fe phase (default), health=1.0 → 80*1.0=80 A
        expected_i = _PHASE_CURRENT["Fe"] * 1.0   # 80.0
        assert result["I_cell_setpoint"] == pytest.approx(expected_i)

    # ------------------------------------------------------------------
    # 8. FAULT_RECOVERY — heater at half power, no current
    # ------------------------------------------------------------------
    def test_fault_recovery_half_heater_no_current(self):
        ctrl = make_controller()
        # Use T_bulk well below setpoint so PID output is non-zero (not just 0*0.5=0)
        state = make_state(T_bulk=1000.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "FAULT_RECOVERY")
        # PID would return OUT_MAX (10000); half = 5000
        assert result["heater_power"] == pytest.approx(HeaterPID.OUT_MAX * 0.5)
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 9. ELECTRODE_DEGRADING — half of adaptive current
    # ------------------------------------------------------------------
    def test_electrode_degrading_half_current(self):
        ctrl = make_controller()
        health = 0.5
        state = make_state(T_bulk=T_SETPOINT, health=health)
        result = ctrl.compute_setpoints(
            state, {"electrode_health_est": health}, "ELECTRODE_DEGRADING"
        )
        # Expected: adaptive_current(0.5, "Fe") * 0.5 = 48 * 0.5 = 24 A
        normal_i = AdaptiveCurrent().compute(health, "Fe")
        expected_i = float(max(10.0, min(200.0, normal_i * 0.5)))
        assert result["I_cell_setpoint"] == pytest.approx(expected_i)
        # Heater should still be running (T at setpoint → near 0 but that's fine)
        # The key is the mode doesn't zero out the heater
        assert "heater_power" in result

    # ------------------------------------------------------------------
    # 10. ELECTRODE_SWAP — heater on, no current
    # ------------------------------------------------------------------
    def test_electrode_swap_heater_on_no_current(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1000.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "ELECTRODE_SWAP")
        assert result["heater_power"] > 0.0
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 11. BATH_DEPLETED — heater on, no current
    # ------------------------------------------------------------------
    def test_bath_depleted_heater_on_no_current(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1000.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "BATH_DEPLETED")
        assert result["heater_power"] > 0.0
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 12. DRAINING — heater on, no current
    # ------------------------------------------------------------------
    def test_draining_heater_on_no_current(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1000.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "DRAINING")
        assert result["heater_power"] > 0.0
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 13. CLEANOUT — everything off
    # ------------------------------------------------------------------
    def test_cleanout_all_off(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1580.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "CLEANOUT")
        assert result["heater_power"] == pytest.approx(0.0)
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    # ------------------------------------------------------------------
    # 14. SAFE_SHUTDOWN (and unknown fallback) — everything off
    # ------------------------------------------------------------------
    def test_safe_shutdown_all_off(self):
        ctrl = make_controller()
        state = make_state(T_bulk=1580.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "SAFE_SHUTDOWN")
        assert result["heater_power"] == pytest.approx(0.0)
        assert result["I_cell_setpoint"] == pytest.approx(10.0)

    def test_unknown_mode_all_off(self):
        """An unrecognised mode falls back to SAFE_SHUTDOWN behaviour."""
        ctrl = make_controller()
        state = make_state(T_bulk=1580.0)
        result = ctrl.compute_setpoints(state, {"electrode_health_est": 1.0}, "TOTALLY_UNKNOWN")
        assert result["heater_power"] == pytest.approx(0.0)
        assert result["I_cell_setpoint"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TemperaturePID — step response and anti-windup
# ---------------------------------------------------------------------------

def test_pid_step_response():
    from testbed.plant import PlantSimulator
    from testbed.controllers import TemperaturePID
    T_SP = 1700.0
    pid = TemperaturePID(T_setpoint=T_SP)
    plant = PlantSimulator()
    dt = 1.0
    T_history = []
    state = plant.get_state()
    for _ in range(300):
        hp = pid.compute(state.T_bulk, dt)
        state = plant.step(dt=dt, setpoints={"heater_power": hp, "I_cell_setpoint": 10.0})
        T_history.append(state.T_bulk)
    # Must reach setpoint ± 5°C within 200 s
    reached = any(abs(T - T_SP) <= 5.0 for T in T_history[:200])
    assert reached, f"PID did not reach {T_SP} ± 5°C within 200 s"
    # Steady-state error at 300 s
    ss_error = abs(T_history[-1] - T_SP)
    assert ss_error <= 5.0, f"Steady-state error {ss_error:.1f} °C > 5 °C"


def test_pid_no_windup_on_large_step():
    from testbed.plant import PlantSimulator
    from testbed.controllers import TemperaturePID
    T_SP = 2000.0
    pid = TemperaturePID(T_setpoint=T_SP)
    plant = PlantSimulator()
    dt = 1.0
    state = plant.get_state()
    peak_T = state.T_bulk
    for _ in range(500):
        hp = pid.compute(state.T_bulk, dt)
        assert hp <= 10_000.0
        state = plant.step(dt=dt, setpoints={"heater_power": hp, "I_cell_setpoint": 10.0})
        peak_T = max(peak_T, state.T_bulk)
    assert peak_T <= T_SP + 100.0, f"Overshoot too large: {peak_T:.1f}"

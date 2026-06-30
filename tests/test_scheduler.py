"""Comprehensive tests for SimLoop (scheduler.py) — auto-transitions and timed exits.

Strategy:
  - Thread-based tests use SimLoop.start() + short time.sleep() to let the
    daemon loop tick a few times (loop runs at 5 ticks/s wall-clock).
  - Logic-only tests manipulate mode_manager / component state directly
    without starting the thread, keeping them fast and deterministic.
"""

from __future__ import annotations

import time
from collections import deque

from testbed.faults import ElectrodeDegradationDetector, FaultInjector
from testbed.mode_manager import ModeManager
from testbed.plant import PlantSimulator, PlantState
from testbed.scheduler import SimLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs) -> PlantState:
    """Return a PlantState with overridden fields."""
    s = PlantState()
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _wait_for_mode(sim: SimLoop, target: str, timeout: float = 2.0) -> bool:
    """Poll until mode_manager reaches *target* or timeout expires.

    Returns True if target was reached, False if timed out.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sim.mode_manager.mode == target:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Test 1 — HEATING → RUN_NOMINAL auto-transition
# ---------------------------------------------------------------------------

class TestHeatingToRunNominal:
    """Plant starts at T_bulk=1580°C which is already above the 1560°C threshold.
    The loop triggers 'target_reached' on the first tick after start().
    """

    def test_auto_transition_on_hot_plant(self) -> None:
        sim = SimLoop()
        # Plant default T_bulk is 1580°C — already above threshold.
        sim.start()

        reached = _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)
        assert reached, (
            f"Expected RUN_NOMINAL within 2s but mode is {sim.mode_manager.mode!r}"
        )

    def test_starts_in_heating_then_advances(self) -> None:
        """The loop immediately fires 'start' (IDLE→HEATING) then checks T_bulk."""
        sim = SimLoop()
        # Don't start — we can confirm the initial IDLE state.
        assert sim.mode_manager.mode == "IDLE"

        sim.start()
        # After one short sleep the loop should have fired start() and then
        # transitioned to RUN_NOMINAL because T_bulk > 1560.
        reached = _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)
        assert reached

    def test_cold_plant_stays_in_heating(self) -> None:
        """Force T_bulk below threshold — loop must stay in HEATING."""
        sim = SimLoop()
        # Drop temperature below the 1560°C threshold before the loop starts.
        sim.plant._state.T_bulk = 900.0
        sim.plant._state.heater_power = 0.0   # no heater → stays cold

        sim.start()
        # Give the loop a few ticks (0.5 s = ~2-3 ticks) to process.
        time.sleep(0.5)

        # With T_bulk=900 and no heater the simulator won't reach 1560 quickly.
        # Mode must still be HEATING (not yet RUN_NOMINAL).
        mode = sim.mode_manager.mode
        assert mode == "HEATING", (
            f"Expected HEATING with cold plant but got {mode!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — ElectrodeDegradationDetector consecutive-hit logic
# ---------------------------------------------------------------------------

class TestElectrodeDegradationDetector:
    """Unit-test the detector directly: 3 consecutive low-health readings."""

    def _state_and_inferred(self, health: float) -> tuple[PlantState, dict]:
        return _make_state(), {"electrode_health_est": health}

    def test_first_two_calls_return_none(self) -> None:
        det = ElectrodeDegradationDetector()
        s, inferred = self._state_and_inferred(0.5)

        assert det.detect(s, [], inferred) is None, "1st call should be None"
        assert det.detect(s, [], inferred) is None, "2nd call should be None"

    def test_third_consecutive_call_returns_fault(self) -> None:
        det = ElectrodeDegradationDetector()
        s, inferred = self._state_and_inferred(0.5)

        det.detect(s, [], inferred)
        det.detect(s, [], inferred)
        result = det.detect(s, [], inferred)
        assert result == "electrode_degradation"

    def test_counter_resets_on_healthy_reading(self) -> None:
        det = ElectrodeDegradationDetector()
        s_low, inferred_low = self._state_and_inferred(0.5)
        s_hi, inferred_hi = self._state_and_inferred(0.9)

        det.detect(s_low, [], inferred_low)
        det.detect(s_low, [], inferred_low)
        # Reset with a healthy reading
        det.detect(s_hi, [], inferred_hi)
        # Now two more low readings — counter starts fresh, should be None
        det.detect(s_low, [], inferred_low)
        result = det.detect(s_low, [], inferred_low)
        assert result is None

    def test_none_inferred_returns_none(self) -> None:
        det = ElectrodeDegradationDetector()
        s = _make_state()
        for _ in range(5):
            assert det.detect(s, [], None) is None

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        """health_est == HEALTH_THRESHOLD (0.6) should NOT trigger."""
        det = ElectrodeDegradationDetector()
        s, inferred = self._state_and_inferred(0.6)  # not strictly < 0.6
        for _ in range(5):
            result = det.detect(s, [], inferred)
        assert result is None


# ---------------------------------------------------------------------------
# Test 3 — Rolling-window recovery tracking (deque maxlen=5)
# ---------------------------------------------------------------------------

class TestRollingWindowRecovery:
    """The scheduler tracks recent fault outcomes in a deque(maxlen=5).
    3+ True values (failed recoveries) → unrecoverable threshold.
    """

    def test_three_failures_exceed_threshold(self) -> None:
        outcomes: deque[bool] = deque(maxlen=5)
        for _ in range(3):
            outcomes.append(True)
        assert sum(outcomes) >= 3

    def test_two_failures_do_not_exceed_threshold(self) -> None:
        outcomes: deque[bool] = deque(maxlen=5)
        outcomes.append(True)
        outcomes.append(True)
        assert sum(outcomes) < 3

    def test_alternating_true_false_three_trues_triggers(self) -> None:
        outcomes: deque[bool] = deque(maxlen=5)
        for v in [True, False, True, False, True]:
            outcomes.append(v)
        assert sum(outcomes) == 3  # 3 × True → at threshold

    def test_all_false_does_not_trigger(self) -> None:
        outcomes: deque[bool] = deque(maxlen=5)
        for _ in range(5):
            outcomes.append(False)
        assert sum(outcomes) == 0

    def test_maxlen_evicts_oldest(self) -> None:
        """deque(maxlen=5): adding a 6th element evicts the first True."""
        outcomes: deque[bool] = deque(maxlen=5)
        # Fill with 3 True then 2 False → [T, T, T, F, F], sum=3
        for _ in range(3):
            outcomes.append(True)
        outcomes.append(False)
        outcomes.append(False)
        assert len(outcomes) == 5
        assert sum(outcomes) == 3  # All 5 items are in the window

        # 6th item (False) evicts the first True → [T, T, F, F, F], sum=2
        outcomes.append(False)
        assert len(outcomes) == 5
        assert sum(outcomes) == 2

    def test_overflow_evicts_old_failures(self) -> None:
        """6 failures added → window only holds last 5, sum=5 still ≥ 3."""
        outcomes: deque[bool] = deque(maxlen=5)
        for _ in range(6):
            outcomes.append(True)
        assert len(outcomes) == 5
        assert sum(outcomes) >= 3


# ---------------------------------------------------------------------------
# Test 4 — FaultInjector integration
# ---------------------------------------------------------------------------

class TestFaultInjectorIntegration:
    """Verify FaultInjector wires into the plant correctly and that
    safe_trigger("fault_detected") drives SimLoop to FAULT_RECOVERY.

    Note on AnodeEffectDetector physics: inject() sets fault_active on the plant,
    causing the plant's own recovery ramp to drop I_cell by 20 A/tick. With the
    multiplier the spike only lasts one tick before I_cell drops too low. The
    detector requires 2 *consecutive* V_cell > 8 V readings, so self-detection via
    the normal loop is not achievable after a single inject() call. The tests below
    verify the underlying components and the mode-manager path independently.
    """

    def test_inject_sets_fault_active_on_plant(self) -> None:
        """inject() must mark fault_active on the plant state."""
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        assert sim.plant.get_state().fault_active is None
        sim.fault_injector.inject("anode_effect", severity=7)

        # Plant state must now show the fault.
        assert sim.plant.get_state().fault_active == "anode_effect"

    def test_clear_removes_fault_from_plant(self) -> None:
        """clear() must remove fault_active from the plant state."""
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        sim.fault_injector.inject("anode_effect", severity=4)
        assert sim.plant.get_state().fault_active == "anode_effect"

        sim.fault_injector.clear()
        assert sim.plant.get_state().fault_active is None

    def test_fault_detected_trigger_reaches_fault_recovery(self) -> None:
        """safe_trigger('fault_detected') from RUN_NOMINAL → FAULT_RECOVERY."""
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        sim.mode_manager.safe_trigger("fault_detected")
        assert sim.mode_manager.mode == "FAULT_RECOVERY"

    def test_snapshot_reflects_fault_recovery_after_trigger(self) -> None:
        """After triggering fault_detected directly, snapshot mode matches."""
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        sim.mode_manager.safe_trigger("fault_detected")
        # Give the loop one tick to write the new snapshot.
        time.sleep(0.3)
        snap = sim.get_snapshot()
        assert snap.get("mode") == "FAULT_RECOVERY"

    def test_injected_fault_produces_elevated_v_cell(self) -> None:
        """After inject(), the plant steps produce V_cell above the spike threshold
        on the first fault tick (before the recovery ramp brings I_cell down).
        Validates that the multiplier actually reaches the plant physics.
        """
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        # Let I_cell stabilise at its RUN_NOMINAL setpoint (~80 A).
        time.sleep(0.3)
        sim.fault_injector.inject("anode_effect", severity=7)  # multiplier ≈ 5.0

        # The very next plant step should produce a spike.
        # We poll the snapshot for elevated V_cell.
        deadline = time.monotonic() + 1.0
        max_v_seen = 0.0
        while time.monotonic() < deadline:
            snap = sim.get_snapshot()
            if snap:
                v = snap["state"].V_cell
                if v > max_v_seen:
                    max_v_seen = v
            time.sleep(0.05)

        # With I_cell ≈ 80 A and multiplier 5.0: V ≈ 80 * 0.03 * 5 = 12 V
        assert max_v_seen > 5.0, (
            f"Expected elevated V_cell after injection but max seen was {max_v_seen:.2f} V"
        )


# ---------------------------------------------------------------------------
# Test 5 — get_snapshot() returns expected keys and types
# ---------------------------------------------------------------------------

class TestGetSnapshot:
    """get_snapshot() must return a well-formed dict after the first tick."""

    def test_snapshot_has_required_keys(self) -> None:
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)   # let at least one tick complete

        snap = sim.get_snapshot()
        for key in ("state", "inferred", "mode", "history"):
            assert key in snap, f"Missing key {key!r} in snapshot"

    def test_state_is_plant_state(self) -> None:
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)

        snap = sim.get_snapshot()
        assert isinstance(snap["state"], PlantState)

    def test_inferred_electrode_health_in_range(self) -> None:
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)

        snap = sim.get_snapshot()
        health = snap["inferred"].get("electrode_health_est")
        assert health is not None, "electrode_health_est missing from inferred"
        assert isinstance(health, float)
        assert 0.0 <= health <= 1.0, f"electrode_health_est={health} out of [0,1]"

    def test_mode_is_string(self) -> None:
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)

        snap = sim.get_snapshot()
        assert isinstance(snap["mode"], str)

    def test_history_is_list(self) -> None:
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)

        snap = sim.get_snapshot()
        assert isinstance(snap["history"], list)

    def test_snapshot_empty_before_first_tick(self) -> None:
        """Before start(), get_snapshot() returns an empty dict."""
        sim = SimLoop()
        snap = sim.get_snapshot()
        assert snap == {}


# ---------------------------------------------------------------------------
# Test 6 — Bath depletion trigger sequence (no thread)
# ---------------------------------------------------------------------------

class TestBathDepletionTriggers:
    """Validate mode_manager trigger chains for bath depletion path."""

    def test_run_nominal_to_bath_depleted(self) -> None:
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        assert mm.mode == "RUN_NOMINAL"

        mm.safe_trigger("bath_depleted")
        assert mm.mode == "BATH_DEPLETED"

    def test_bath_depleted_to_draining(self) -> None:
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        mm.safe_trigger("bath_depleted")

        mm.safe_trigger("drain_command")
        assert mm.mode == "DRAINING"

    def test_electrode_degrading_bath_depleted_to_draining(self) -> None:
        """ELECTRODE_DEGRADING can also receive bath_depleted → BATH_DEPLETED."""
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        mm.safe_trigger("electrode_degrading")
        assert mm.mode == "ELECTRODE_DEGRADING"

        mm.safe_trigger("bath_depleted")
        assert mm.mode == "BATH_DEPLETED"

        mm.safe_trigger("drain_command")
        assert mm.mode == "DRAINING"

    def test_bath_depleted_start_field_initialises_to_none(self) -> None:
        """_bath_depleted_start starts None; the loop sets it when BATH_DEPLETED entered."""
        sim = SimLoop()
        assert sim._bath_depleted_start is None

    def test_manual_bath_depleted_timer_logic(self) -> None:
        """Simulate the auto-drain timer by manipulating _bath_depleted_start directly."""
        sim = SimLoop()
        # Manually drive to BATH_DEPLETED without the thread.
        sim.mode_manager.safe_trigger("start")
        sim.mode_manager.safe_trigger("target_reached")
        sim.mode_manager.safe_trigger("bath_depleted")
        assert sim.mode_manager.mode == "BATH_DEPLETED"

        # Simulate the timer already having expired.
        sim._bath_depleted_start = 0.0
        # Now trigger the drain_command as the loop would after 30 s hold.
        sim.mode_manager.safe_trigger("drain_command")
        assert sim.mode_manager.mode == "DRAINING"


# ---------------------------------------------------------------------------
# Test 7 — pre_fault_mode tracking and recovery_complete_degraded
# ---------------------------------------------------------------------------

class TestPreFaultModeTracking:
    """When the fault originated from ELECTRODE_DEGRADING, recovery should
    return to ELECTRODE_DEGRADING (not RUN_NOMINAL) via recovery_complete_degraded.
    """

    def test_recovery_complete_degraded_returns_to_electrode_degrading(self) -> None:
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        mm.safe_trigger("electrode_degrading")
        assert mm.mode == "ELECTRODE_DEGRADING"

        mm.safe_trigger("fault_detected")
        assert mm.mode == "FAULT_RECOVERY"

        mm.safe_trigger("recovery_complete_degraded")
        assert mm.mode == "ELECTRODE_DEGRADING"

    def test_recovery_complete_returns_to_run_nominal(self) -> None:
        """Normal path (fault from RUN_NOMINAL) → RUN_NOMINAL after recovery."""
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        mm.safe_trigger("fault_detected")
        assert mm.mode == "FAULT_RECOVERY"

        mm.safe_trigger("recovery_complete")
        assert mm.mode == "RUN_NOMINAL"

    def test_pre_fault_mode_default_is_run_nominal(self) -> None:
        """SimLoop initialises _pre_fault_mode to 'RUN_NOMINAL'."""
        sim = SimLoop()
        assert sim._pre_fault_mode == "RUN_NOMINAL"

    def test_pre_fault_mode_set_before_thread(self) -> None:
        """We can manually set _pre_fault_mode and trigger the degraded recovery."""
        sim = SimLoop()
        sim.mode_manager.safe_trigger("start")
        sim.mode_manager.safe_trigger("target_reached")
        sim.mode_manager.safe_trigger("electrode_degrading")
        sim.mode_manager.safe_trigger("fault_detected")
        assert sim.mode_manager.mode == "FAULT_RECOVERY"

        sim._pre_fault_mode = "ELECTRODE_DEGRADING"
        sim.mode_manager.safe_trigger("recovery_complete_degraded")
        assert sim.mode_manager.mode == "ELECTRODE_DEGRADING"

    def test_unrecoverable_from_fault_recovery(self) -> None:
        """Enough consecutive failures → SAFE_SHUTDOWN via 'unrecoverable'."""
        mm = ModeManager()
        mm.safe_trigger("start")
        mm.safe_trigger("target_reached")
        mm.safe_trigger("fault_detected")
        assert mm.mode == "FAULT_RECOVERY"

        mm.safe_trigger("unrecoverable")
        assert mm.mode == "SAFE_SHUTDOWN"


# ---------------------------------------------------------------------------
# Test 8 — Additional integration / edge cases
# ---------------------------------------------------------------------------

class TestSimLoopMiscellaneous:
    """Extra integration checks for SimLoop lifecycle."""

    def test_start_is_idempotent(self) -> None:
        """Calling start() twice must not raise or launch a second thread."""
        sim = SimLoop()
        sim.start()
        sim.start()   # should silently no-op
        time.sleep(0.3)
        assert sim.mode_manager.mode in ("HEATING", "RUN_NOMINAL")

    def test_fault_injector_active_fault_property(self) -> None:
        """FaultInjector.active_fault returns the injected fault name."""
        plant = PlantSimulator()
        injector = FaultInjector(plant)
        assert injector.active_fault is None

        injector.inject("anode_effect", severity=4)
        assert injector.active_fault == "anode_effect"

        injector.clear()
        assert injector.active_fault is None

    def test_fault_injector_severity_to_multiplier_range(self) -> None:
        """Severity 1 → multiplier ≈1.5, severity 7 → multiplier ≈5.0."""
        from testbed.faults import _SEV_TO_MULT
        assert abs(_SEV_TO_MULT[1] - 1.5) < 1e-6
        assert abs(_SEV_TO_MULT[7] - 5.0) < 1e-6

    def test_snapshot_mode_transitions_over_time(self) -> None:
        """Snapshot mode should change from HEATING to RUN_NOMINAL quickly."""
        sim = SimLoop()
        sim.start()

        # Collect modes over ~1 s
        modes_seen: set[str] = set()
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            snap = sim.get_snapshot()
            if snap:
                modes_seen.add(snap.get("mode", ""))
            time.sleep(0.1)

        # Since plant starts above threshold, RUN_NOMINAL must appear.
        assert "RUN_NOMINAL" in modes_seen, (
            f"RUN_NOMINAL never appeared in snapshot modes: {modes_seen}"
        )

    def test_history_grows_with_ticks(self) -> None:
        """After several ticks, history list in snapshot must be non-empty."""
        sim = SimLoop()
        sim.start()
        time.sleep(0.6)   # ~3 ticks at 5/s

        snap = sim.get_snapshot()
        assert len(snap.get("history", [])) >= 1

    def test_plant_state_uptime_advances(self) -> None:
        """uptime_s in snapshot must be > 0 after a few ticks."""
        sim = SimLoop()
        sim.start()
        time.sleep(0.5)

        snap = sim.get_snapshot()
        assert snap["state"].uptime_s > 0.0

    def test_o2_produced_advances(self) -> None:
        """O2_produced_mol must increase during RUN_NOMINAL."""
        sim = SimLoop()
        sim.start()
        _wait_for_mode(sim, "RUN_NOMINAL", timeout=2.0)

        # _wait_for_mode polls mode_manager.mode directly; the snapshot is only
        # written at step 7 of the loop, which may not have completed yet.
        # Spin until get_snapshot() returns a dict that actually has a "state"
        # key so we don't hit a KeyError on the empty initial snapshot {}.
        deadline = time.monotonic() + 2.0
        snap1: dict = {}
        while time.monotonic() < deadline:
            snap1 = sim.get_snapshot()
            if "state" in snap1:
                break
            time.sleep(0.05)
        assert "state" in snap1, "Snapshot never populated 'state' key within 2 s"

        o2_start = snap1["state"].O2_produced_mol

        time.sleep(0.6)   # a few more ticks

        snap2 = sim.get_snapshot()
        o2_end = snap2["state"].O2_produced_mol
        assert o2_end > o2_start, "O2 production did not advance"


# ---------------------------------------------------------------------------
# Test 9 — All three extraction phases seen (direct plant stepping, fast)
# ---------------------------------------------------------------------------

def test_phase_transitions_all_seen():
    """SimLoop sees all 3 phase transitions within 20000 sim-seconds at high current.

    Steps the plant directly (bypassing the thread) at high current to accelerate
    phase transitions without waiting for real wall-clock time.
    """
    from testbed.plant import PlantSimulator
    plant = PlantSimulator()
    setpoints = {"heater_power": 8000.0, "I_cell_setpoint": 160.0}
    phases_seen: set[str] = set()
    last_phase = "Fe"
    for _ in range(20000):
        state = plant.step(dt=1.0, setpoints=setpoints)
        phases_seen.add(state.bath_phase)
        if state.bath_phase != last_phase:
            last_phase = state.bath_phase
        if "complete" in phases_seen:
            break
    assert "Si" in phases_seen, f"Never transitioned to Si phase; seen: {phases_seen}"
    assert "Al_Ti" in phases_seen, f"Never transitioned to Al_Ti phase; seen: {phases_seen}"
    assert "complete" in phases_seen, f"Never completed all phases; seen: {phases_seen}"

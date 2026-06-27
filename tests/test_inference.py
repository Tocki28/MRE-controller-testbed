"""Comprehensive tests for MOEInference and related functions.

Covers:
- fit_randles_scipy: parameter recovery on clean and noisy synthetic data
- SyntheticEISSource.acquire_spectrum: output shape, dtype, sign
- MOEInference.update: health proxy bounds, fresh vs degraded electrode
- EMA smoothing: damping of step change
- reset_for_new_batch: health resets to 1.0
- EIS interval gating: sweep only fires every EIS_INTERVAL_S sim-seconds
- TTF prediction: positive finite estimate when health is declining
- Custom EISDataSource injection: mock source is called on sweep tick
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from testbed.interfaces import EISDataSource
from testbed.inference import (
    ALPHA,
    EIS_INTERVAL_S,
    MOEInference,
    SyntheticEISSource,
    fit_randles_scipy,
)
from testbed.plant import PlantState
from testbed.randles_model import (
    MOE_CDL_BASELINE,
    MOE_RCT_END,
    MOE_RCT_NEW,
    MOE_RS_BASELINE,
    MOE_SIGMA_BASELINE,
    moe_frequencies,
    randles_impedance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(electrode_health: float = 1.0, uptime_s: float = 0.0) -> PlantState:
    """Return a PlantState with the given health and uptime."""
    s = PlantState()
    s.electrode_health = electrode_health
    s.uptime_s = uptime_s
    return s


# ---------------------------------------------------------------------------
# 1. fit_randles_scipy
# ---------------------------------------------------------------------------

class TestFitRandlesScipy:
    """Tests for the scipy TRF Randles fitter."""

    def _clean_spectrum(
        self,
        Rs: float = MOE_RS_BASELINE,
        Rct: float = MOE_RCT_NEW,
        Cdl: float = MOE_CDL_BASELINE,
        sigma: float = MOE_SIGMA_BASELINE,
    ) -> tuple[np.ndarray, np.ndarray]:
        freq = moe_frequencies()
        omega = 2.0 * np.pi * freq
        Z = randles_impedance(omega, Rs, Rct, Cdl, sigma)
        return omega, Z

    def test_rct_recovery_within_10pct_on_clean_data(self) -> None:
        """Fitter recovers true Rct within 10% on noise-free synthetic data."""
        Rct_true = MOE_RCT_NEW   # 8.0 Ω — fresh electrode
        omega, Z = self._clean_spectrum(Rct=Rct_true)
        _, Rct_fit, _, _ = fit_randles_scipy(omega, Z)
        assert abs(Rct_fit - Rct_true) / Rct_true < 0.10, (
            f"Rct_fit={Rct_fit:.3f} is more than 10% off true {Rct_true}"
        )

    def test_rct_recovery_degraded_electrode_within_10pct(self) -> None:
        """Fitter recovers Rct for a degraded (high Rct) electrode within 10%."""
        # Halfway to end-of-life: health=0.5 → Rct = 8 + 0.5*(50-8) = 29 Ω
        Rct_true = MOE_RCT_NEW + 0.5 * (MOE_RCT_END - MOE_RCT_NEW)
        omega, Z = self._clean_spectrum(Rct=Rct_true)
        _, Rct_fit, _, _ = fit_randles_scipy(omega, Z)
        assert abs(Rct_fit - Rct_true) / Rct_true < 0.10, (
            f"Rct_fit={Rct_fit:.3f} is more than 10% off true {Rct_true}"
        )

    def test_rct_recovery_with_gaussian_noise_within_20pct(self) -> None:
        """With Gaussian noise std=0.1, Rct stays within 20% of true value."""
        rng = np.random.default_rng(seed=99)
        Rct_true = MOE_RCT_NEW
        omega, Z_clean = self._clean_spectrum(Rct=Rct_true)
        noise = (
            rng.normal(0, 0.1, size=len(omega))
            + 1j * rng.normal(0, 0.1, size=len(omega))
        )
        Z_noisy = Z_clean + noise
        _, Rct_fit, _, _ = fit_randles_scipy(omega, Z_noisy)
        assert abs(Rct_fit - Rct_true) / Rct_true < 0.20, (
            f"Rct_fit={Rct_fit:.3f} is more than 20% off true {Rct_true} under noise"
        )

    def test_return_has_four_elements(self) -> None:
        """fit_randles_scipy returns exactly (Rs, Rct, Cdl, sigma)."""
        omega, Z = self._clean_spectrum()
        result = fit_randles_scipy(omega, Z)
        assert len(result) == 4

    def test_all_params_positive(self) -> None:
        """All four fitted parameters must be positive (physical constraint)."""
        omega, Z = self._clean_spectrum()
        Rs, Rct, Cdl, sigma = fit_randles_scipy(omega, Z)
        assert Rs > 0 and Rct > 0 and Cdl > 0 and sigma > 0


# ---------------------------------------------------------------------------
# 2. SyntheticEISSource.acquire_spectrum
# ---------------------------------------------------------------------------

class TestSyntheticEISSource:
    """Tests for SyntheticEISSource.acquire_spectrum output contract."""

    def setup_method(self) -> None:
        self.source = SyntheticEISSource()
        self.state = _make_state(electrode_health=1.0)

    def test_returns_two_arrays(self) -> None:
        result = self.source.acquire_spectrum(self.state)
        assert len(result) == 2

    def test_equal_length_arrays(self) -> None:
        omega, Z = self.source.acquire_spectrum(self.state)
        assert len(omega) == len(Z)

    def test_omega_all_positive(self) -> None:
        omega, _ = self.source.acquire_spectrum(self.state)
        assert np.all(omega > 0), "All omega values must be positive (rad/s)"

    def test_Z_is_complex(self) -> None:
        _, Z = self.source.acquire_spectrum(self.state)
        assert np.issubdtype(Z.dtype, np.complexfloating), (
            f"Z should be complex, got dtype={Z.dtype}"
        )

    def test_degraded_electrode_higher_Z_real_at_low_freq(self) -> None:
        """A degraded electrode (high Rct) should show larger real impedance."""
        state_fresh = _make_state(electrode_health=1.0)
        state_degraded = _make_state(electrode_health=0.1)

        # Use a fixed-seed source to compare without noise variance
        src = SyntheticEISSource()
        _, Z_fresh = src.acquire_spectrum(state_fresh)
        src2 = SyntheticEISSource()
        _, Z_degraded = src2.acquire_spectrum(state_degraded)

        # At lower frequencies the Rct contribution dominates; pick the max real part
        assert np.max(Z_degraded.real) > np.max(Z_fresh.real), (
            "Degraded electrode should have higher real impedance"
        )


# ---------------------------------------------------------------------------
# 3. MOEInference.update — health proxy
# ---------------------------------------------------------------------------

class TestMOEInferenceHealthProxy:
    """Tests for electrode_health_est produced by MOEInference.update."""

    def test_fresh_electrode_health_near_1(self) -> None:
        """After one sweep on a health=1.0 electrode, health_est should be >= 0.7."""
        inf = MOEInference()
        state = _make_state(electrode_health=1.0, uptime_s=0.0)
        result = inf.update(state)
        assert result["electrode_health_est"] >= 0.7, (
            f"Fresh electrode health_est={result['electrode_health_est']:.3f} too low"
        )

    def test_degraded_electrode_health_drops(self) -> None:
        """After several sweeps on health=0.1 electrode, health_est drops below 0.5."""
        inf = MOEInference()
        # Run enough sweeps to let EMA settle: each sweep is EIS_INTERVAL_S apart
        t = 0.0
        result = {}
        for _ in range(5):
            state = _make_state(electrode_health=0.1, uptime_s=t)
            result = inf.update(state)
            t += EIS_INTERVAL_S + 1.0  # advance past the gate each time

        assert result["electrode_health_est"] < 0.5, (
            f"Degraded electrode health_est={result['electrode_health_est']:.3f} should be < 0.5"
        )

    def test_health_est_always_in_0_1(self) -> None:
        """electrode_health_est must always be in [0, 1]."""
        inf = MOEInference()
        t = 0.0
        for health in [1.0, 0.5, 0.1, 0.01]:
            state = _make_state(electrode_health=health, uptime_s=t)
            result = inf.update(state)
            h = result["electrode_health_est"]
            assert 0.0 <= h <= 1.0, f"health_est={h} out of [0,1] for electrode_health={health}"
            t += EIS_INTERVAL_S + 1.0

    def test_health_est_starts_at_1_before_any_sweep(self) -> None:
        """Before any sweep fires, the default health_est is 1.0."""
        inf = MOEInference()
        # Call update without advancing time past the interval — no sweep fires
        # since _last_eis_time starts at -999, uptime_s=0 triggers immediately.
        # Instead, verify initial internal state directly.
        assert inf._health_est == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. EMA smoothing
# ---------------------------------------------------------------------------

class TestEMASmoothing:
    """EMA (α=0.3) damps abrupt changes in health_est."""

    def test_ema_damps_step_change(self) -> None:
        """After one sweep, degraded health_est is between raw and 1.0 (previous)."""
        # Start fresh, fire one sweep
        inf_healthy = MOEInference()
        state_healthy = _make_state(electrode_health=1.0, uptime_s=0.0)
        result_h = inf_healthy.update(state_healthy)
        # After first sweep on fresh electrode, health_est is between 0.7 and 1.0

        # Now start a second inference instance at degraded state (first sweep)
        inf_deg = MOEInference()
        state_deg = _make_state(electrode_health=0.0, uptime_s=0.0)
        result_d = inf_deg.update(state_deg)

        # The EMA starts from 1.0 and moves toward raw by α=0.3
        # So health_est_deg should be: 0.3*raw + 0.7*1.0 > raw
        h_deg = result_d["electrode_health_est"]
        # Raw for health=0.0 → Rct = MOE_RCT_END → raw health = 0.0
        # EMA: 0.3*0.0 + 0.7*1.0 = 0.7  (but noise might push raw slightly above 0)
        # Key assertion: health_est is strictly above the minimum possible raw (0.0)
        # and below 1.0 (the prior), proving EMA damped it.
        assert h_deg > 0.0, "EMA should keep health_est above raw=0 on first sweep"
        assert h_deg < 1.0, "EMA should reduce health_est below the initial 1.0"

    def test_ema_alpha_matches_module_constant(self) -> None:
        """Verify the module-level ALPHA constant is 0.3 (used in EMA)."""
        assert ALPHA == pytest.approx(0.3)

    def test_two_sweeps_damp_more_than_one(self) -> None:
        """Two degraded sweeps should push health_est further down than one sweep."""
        inf = MOEInference()
        state0 = _make_state(electrode_health=0.0, uptime_s=0.0)
        result1 = inf.update(state0)
        h_after_1 = result1["electrode_health_est"]

        state1 = _make_state(electrode_health=0.0, uptime_s=EIS_INTERVAL_S + 1.0)
        result2 = inf.update(state1)
        h_after_2 = result2["electrode_health_est"]

        assert h_after_2 < h_after_1, (
            "Second degraded sweep should lower health_est further due to EMA"
        )


# ---------------------------------------------------------------------------
# 5. reset_for_new_batch
# ---------------------------------------------------------------------------

class TestResetForNewBatch:
    """reset_for_new_batch must restore health_est to 1.0."""

    def test_reset_restores_health_to_1(self) -> None:
        """After degrading, reset_for_new_batch brings health_est back to 1.0."""
        inf = MOEInference()
        t = 0.0
        for _ in range(3):
            state = _make_state(electrode_health=0.1, uptime_s=t)
            inf.update(state)
            t += EIS_INTERVAL_S + 1.0

        # Confirm it's degraded
        assert inf._health_est < 0.9

        inf.reset_for_new_batch()
        assert inf._health_est == pytest.approx(1.0), (
            "reset_for_new_batch must set health_est to 1.0"
        )

    def test_reset_clears_health_history(self) -> None:
        """reset_for_new_batch clears the internal health history list."""
        inf = MOEInference()
        t = 0.0
        for _ in range(5):
            state = _make_state(electrode_health=0.5, uptime_s=t)
            inf.update(state)
            t += EIS_INTERVAL_S + 1.0

        inf.reset_for_new_batch()
        assert inf._health_history == [], "Health history should be empty after reset"

    def test_reset_allows_immediate_new_sweep(self) -> None:
        """After reset, a new sweep at t=0 fires immediately (last_eis_time=-999)."""
        inf = MOEInference()
        # Advance time so a sweep fires
        state0 = _make_state(electrode_health=0.1, uptime_s=0.0)
        inf.update(state0)

        inf.reset_for_new_batch()

        # Now call at uptime_s=5 — should trigger a sweep because last_eis_time=-999
        state1 = _make_state(electrode_health=1.0, uptime_s=5.0)
        result = inf.update(state1)
        # If sweep fired, health_est should have moved from 1.0 toward fresh value
        # (i.e. EIS data was acquired). We can confirm eis_freq is populated.
        assert len(result["eis_freq"]) > 0, (
            "After reset, update should run a sweep and populate eis_freq"
        )


# ---------------------------------------------------------------------------
# 6. EIS interval gating
# ---------------------------------------------------------------------------

class TestEISIntervalGating:
    """Sweep fires only when uptime_s advances >= EIS_INTERVAL_S since last sweep."""

    def test_second_call_within_interval_no_sweep(self) -> None:
        """Calling update at uptime=30 (< 60s after first sweep at t=0) skips sweep."""
        inf = MOEInference()

        # First call at t=0 — triggers sweep
        state0 = _make_state(electrode_health=1.0, uptime_s=0.0)
        result0 = inf.update(state0)
        h_after_first = inf._health_est

        # Second call at t=30 — within interval, no sweep
        state1 = _make_state(electrode_health=0.0, uptime_s=30.0)
        result1 = inf.update(state1)
        h_after_second = inf._health_est

        assert h_after_second == pytest.approx(h_after_first), (
            f"health_est should not change when no sweep fires: "
            f"before={h_after_first:.4f}, after={h_after_second:.4f}"
        )

    def test_call_past_interval_triggers_sweep(self) -> None:
        """Calling update at uptime=70 (>= 60s after first sweep) triggers new sweep."""
        inf = MOEInference()

        # First sweep at t=0 (fresh electrode)
        state0 = _make_state(electrode_health=1.0, uptime_s=0.0)
        inf.update(state0)
        h_after_first = inf._health_est

        # Second call within interval — no sweep
        state1 = _make_state(electrode_health=0.0, uptime_s=30.0)
        inf.update(state1)

        # Third call past interval — sweep fires with degraded electrode
        state2 = _make_state(electrode_health=0.0, uptime_s=70.0)
        inf.update(state2)
        h_after_third = inf._health_est

        assert h_after_third < h_after_first, (
            "Sweep at t=70 with degraded electrode should lower health_est "
            f"(was {h_after_first:.4f}, now {h_after_third:.4f})"
        )

    def test_interval_boundary_exact(self) -> None:
        """At exactly EIS_INTERVAL_S from the last sweep, a new sweep fires."""
        inf = MOEInference()

        state0 = _make_state(electrode_health=1.0, uptime_s=0.0)
        inf.update(state0)
        h_before = inf._health_est

        # Exact boundary: uptime_s = 0 + 60 = 60.0 → needs_sweep is True
        state1 = _make_state(electrode_health=0.0, uptime_s=EIS_INTERVAL_S)
        inf.update(state1)
        h_after = inf._health_est

        assert h_after < h_before, (
            "At exact EIS_INTERVAL_S, a sweep should fire and lower health_est"
        )


# ---------------------------------------------------------------------------
# 7. TTF prediction
# ---------------------------------------------------------------------------

class TestTTFPrediction:
    """predicted_ttf_hrs should be positive finite after enough history."""

    def test_ttf_positive_finite_after_decline(self) -> None:
        """After 10+ declining measurements, predicted_ttf_hrs is positive and finite."""
        inf = MOEInference()
        t = 0.0
        result = {}
        # Run 12 sweeps with steadily degrading health
        for i in range(12):
            health = max(0.01, 1.0 - i * 0.08)  # declining: 1.0, 0.92, ..., 0.12
            state = _make_state(electrode_health=health, uptime_s=t)
            result = inf.update(state)
            t += EIS_INTERVAL_S + 1.0

        ttf = result["predicted_ttf_hrs"]
        assert math.isfinite(ttf), f"predicted_ttf_hrs={ttf} must be finite"
        assert ttf > 0, f"predicted_ttf_hrs={ttf} must be positive"

    def test_ttf_default_before_enough_history(self) -> None:
        """Before 10 measurements, predicted_ttf_hrs stays at the default 999.0."""
        inf = MOEInference()
        t = 0.0
        result = {}
        # Only 3 sweeps — not enough history for TTF
        for i in range(3):
            state = _make_state(electrode_health=max(0.1, 1.0 - i * 0.3), uptime_s=t)
            result = inf.update(state)
            t += EIS_INTERVAL_S + 1.0

        # TTF prediction requires >= 10 history entries; should remain at 999.0
        assert result["predicted_ttf_hrs"] == pytest.approx(999.0), (
            "predicted_ttf_hrs should remain 999.0 before 10 history entries"
        )

    def test_ttf_stays_within_clip_bound(self) -> None:
        """predicted_ttf_hrs must never exceed 999.0 (upper clip)."""
        inf = MOEInference()
        t = 0.0
        result = {}
        # Run with near-constant health so degradation rate is near zero → large TTF
        for i in range(15):
            state = _make_state(electrode_health=0.99, uptime_s=t)
            result = inf.update(state)
            t += EIS_INTERVAL_S + 1.0

        assert result["predicted_ttf_hrs"] <= 999.0, (
            f"predicted_ttf_hrs={result['predicted_ttf_hrs']} exceeds clip bound 999.0"
        )


# ---------------------------------------------------------------------------
# 8. Custom EISDataSource injection
# ---------------------------------------------------------------------------

class FixedSpectrumEISSource(EISDataSource):
    """Concrete mock EISDataSource that returns a fixed synthetic spectrum.

    Counts how many times acquire_spectrum has been called so tests can verify
    it is actually invoked on each sweep tick.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        # Build a fixed spectrum once (fresh electrode)
        freq = moe_frequencies()
        self._omega = 2.0 * np.pi * freq
        self._Z = randles_impedance(
            self._omega, MOE_RS_BASELINE, MOE_RCT_NEW, MOE_CDL_BASELINE, MOE_SIGMA_BASELINE
        )

    def acquire_spectrum(self, state: PlantState) -> tuple[np.ndarray, np.ndarray]:
        self.call_count += 1
        return self._omega.copy(), self._Z.copy()


class TestCustomEISSourceInjection:
    """MOEInference accepts a custom EISDataSource via the constructor."""

    def test_mock_source_is_called_on_sweep_tick(self) -> None:
        """After one sweep tick, the injected mock source's call_count == 1."""
        mock = FixedSpectrumEISSource()
        inf = MOEInference(eis_source=mock)
        assert mock.call_count == 0, "No calls yet before update()"

        state = _make_state(electrode_health=1.0, uptime_s=0.0)
        inf.update(state)
        assert mock.call_count == 1, (
            f"Mock source should have been called once; call_count={mock.call_count}"
        )

    def test_mock_source_not_called_within_interval(self) -> None:
        """Within the gating interval, the mock source is NOT called again."""
        mock = FixedSpectrumEISSource()
        inf = MOEInference(eis_source=mock)

        # First call fires a sweep
        state0 = _make_state(electrode_health=1.0, uptime_s=0.0)
        inf.update(state0)
        assert mock.call_count == 1

        # Second call within interval — no sweep
        state1 = _make_state(electrode_health=1.0, uptime_s=30.0)
        inf.update(state1)
        assert mock.call_count == 1, (
            f"Mock source should not be called within interval; call_count={mock.call_count}"
        )

    def test_mock_source_called_again_after_interval(self) -> None:
        """Past the gating interval, the mock source is called a second time."""
        mock = FixedSpectrumEISSource()
        inf = MOEInference(eis_source=mock)

        state0 = _make_state(electrode_health=1.0, uptime_s=0.0)
        inf.update(state0)

        state1 = _make_state(electrode_health=1.0, uptime_s=EIS_INTERVAL_S + 5.0)
        inf.update(state1)
        assert mock.call_count == 2, (
            f"Mock source should be called twice after one full interval; "
            f"call_count={mock.call_count}"
        )

    def test_mock_source_stored_on_instance(self) -> None:
        """The injected source is accessible as inf._eis_source."""
        mock = FixedSpectrumEISSource()
        inf = MOEInference(eis_source=mock)
        assert inf._eis_source is mock

    def test_fixed_spectrum_yields_near_full_health(self) -> None:
        """A fixed fresh-electrode spectrum should yield health_est close to 1.0."""
        mock = FixedSpectrumEISSource()
        inf = MOEInference(eis_source=mock)
        state = _make_state(electrode_health=0.0, uptime_s=0.0)  # plant health irrelevant
        result = inf.update(state)
        # The mock always returns Rct=MOE_RCT_NEW → raw health = 1.0
        # EMA: 0.3*1.0 + 0.7*1.0 = 1.0
        assert result["electrode_health_est"] > 0.8, (
            f"Fixed fresh spectrum should give health_est > 0.8; "
            f"got {result['electrode_health_est']:.3f}"
        )


# ---------------------------------------------------------------------------
# 9. update() output contract
# ---------------------------------------------------------------------------

class TestUpdateOutputContract:
    """update() must always return all required keys with correct types."""

    REQUIRED_KEYS = {
        "electrode_health_est",
        "composition_est",
        "predicted_ttf_hrs",
        "eis_freq",
        "eis_Z_re",
        "eis_Z_im",
    }

    def test_all_keys_present(self) -> None:
        inf = MOEInference()
        state = _make_state()
        result = inf.update(state)
        missing = self.REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing keys in update() output: {missing}"

    def test_composition_est_sums_to_one(self) -> None:
        """Normalised composition estimate must sum to ~1.0."""
        inf = MOEInference()
        state = _make_state()
        result = inf.update(state)
        total = sum(result["composition_est"].values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"composition_est sums to {total}, expected 1.0"
        )

    def test_eis_arrays_populated_after_sweep(self) -> None:
        """After a sweep, eis_freq, eis_Z_re, eis_Z_im must be non-empty."""
        inf = MOEInference()
        state = _make_state(uptime_s=0.0)
        result = inf.update(state)
        assert len(result["eis_freq"]) > 0
        assert len(result["eis_Z_re"]) > 0
        assert len(result["eis_Z_im"]) > 0

    def test_eis_arrays_equal_length_after_sweep(self) -> None:
        """eis_freq, eis_Z_re, and eis_Z_im must have the same length."""
        inf = MOEInference()
        state = _make_state(uptime_s=0.0)
        result = inf.update(state)
        assert (
            len(result["eis_freq"])
            == len(result["eis_Z_re"])
            == len(result["eis_Z_im"])
        )

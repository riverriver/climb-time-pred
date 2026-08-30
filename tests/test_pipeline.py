"""合成データによるエンドツーエンドの健全性チェック。

既知の CdA/Crr で合成した登坂データを解析し、逆算値が元の値に
十分近いこと、予測タイムが妥当な範囲に入ることを確認する。
"""

import numpy as np
import pytest

from climbpred.altitude import ThresholdLinear
from climbpred.constants import DEFAULT_ETA
from climbpred.course import load_course
from climbpred.fit_ingest import synthetic_ride
from climbpred.physics import air_density, speed_at_power
from climbpred.pipeline import analyze_rides, run_prediction

TRUE_CDA = 0.30
TRUE_CRR = 0.0050
MASS = 74.0


def _rides():
    specs = [
        (1, 6.0, 0.055, 235), (2, 4.0, 0.085, 255), (3, 9.0, 0.045, 220),
        (4, 3.0, 0.10, 270), (5, 12.0, 0.04, 210), (6, 5.0, 0.07, 245),
        (7, 7.5, 0.06, 230), (8, 2.5, 0.11, 280), (9, 10.0, 0.05, 218),
        (10, 5.5, 0.065, 240),
    ]
    return {
        f"r{s}": synthetic_ride(
            seed=s, length_km=L, avg_grade=g, base_power=p,
            base_alt=120 + 30 * s, cda=TRUE_CDA, crr=TRUE_CRR, mass=MASS, eta=DEFAULT_ETA,
        )
        for s, L, g, p in specs
    }


def test_air_density_decreases_with_altitude():
    assert air_density(0) == pytest.approx(1.225, abs=1e-3)
    assert air_density(2000) < air_density(0)
    assert air_density(2305) == pytest.approx(0.977, abs=0.03)


def test_speed_power_roundtrip():
    from climbpred.physics import power_at_speed

    v = speed_at_power(250, 0.06, MASS, TRUE_CRR, TRUE_CDA, 1.1, DEFAULT_ETA)
    p = power_at_speed(v, 0.06, MASS, TRUE_CRR, TRUE_CDA, 1.1, DEFAULT_ETA)
    assert p == pytest.approx(250, rel=1e-3)


def test_calibration_recovers_parameters():
    analysis = analyze_rides(_rides(), mass_total=MASS, eta=DEFAULT_ETA)
    cal = analysis.calibration
    assert cal.cda == pytest.approx(TRUE_CDA, abs=0.03)
    assert cal.crr == pytest.approx(TRUE_CRR, abs=0.0015)
    assert cal.n_points > 500
    assert not cal.crr_at_bound  # 真値が範囲内なので張り付かない


def test_calibration_clamps_low_crr_to_bound():
    """真の Crr が物理下限より低いと、Crr は下限に固定され CdA が残差を吸収する。"""
    from climbpred.fit_ingest import synthetic_ride

    specs = [
        (1, 6.0, 0.055, 235), (2, 4.0, 0.085, 255), (3, 9.0, 0.045, 220),
        (4, 3.0, 0.10, 270), (5, 12.0, 0.04, 210), (6, 5.0, 0.07, 245),
        (7, 7.5, 0.06, 230), (8, 2.5, 0.11, 280),
    ]
    rides = {
        f"r{s}": synthetic_ride(seed=s, length_km=L, avg_grade=g, base_power=p,
                                base_alt=120 + 30 * s, cda=0.32, crr=0.0002,
                                mass=MASS, eta=DEFAULT_ETA)
        for s, L, g, p in specs
    }
    cal = analyze_rides(rides, mass_total=MASS, eta=DEFAULT_ETA).calibration
    assert cal.crr_at_bound
    assert cal.crr == pytest.approx(0.0015, abs=1e-6)
    assert cal.crr_unconstrained < 0.0015
    assert not cal.plausible and any("Crr" in w for w in cal.warnings)


def test_prediction_is_plausible():
    analysis = analyze_rides(_rides(), mass_total=MASS, eta=DEFAULT_ETA)
    course = load_course()
    pred = run_prediction(analysis, course, MASS, ThresholdLinear())
    # おおむね 60〜120 分に収まること
    assert 3000 < pred.time_s < 9000
    # 標高補正は必ずタイムを悪化させる
    assert pred.time_s > pred.time_no_altitude_s
    assert pred.altitude_factor < 1.0
    assert pred.time_lo_s < pred.time_s < pred.time_hi_s

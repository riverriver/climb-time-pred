"""FTP ベースの段階的予測フロー(predict_fuji_from_inputs / power_model)。"""

import pytest

from climbpred.constants import DEFAULT_ETA
from climbpred.course import load_course
from climbpred.fit_ingest import synthetic_ride
from climbpred.pipeline import predict_fuji_from_inputs
from climbpred.power_model import build_power_model, ftp_duration_factor

COURSE = load_course()


def test_duration_factor_monotone():
    assert ftp_duration_factor(3600) == pytest.approx(1.0, abs=1e-6)
    assert ftp_duration_factor(5400) < 1.0
    assert ftp_duration_factor(7200) < ftp_duration_factor(5400)
    assert ftp_duration_factor(1800) >= 1.0
    assert ftp_duration_factor(99999) >= 0.7  # 下限で頭打ち


def test_power_model_tiers():
    # FTP のみ
    m0 = build_power_model(ftp=250, mmp={}, est_duration_s=4800)
    assert m0.mode == "ftp" and m0.n_data_points == 0
    assert 200 < m0.power(4800) < 250

    # 実測 mean-max が複数点 → CP モデル
    m2 = build_power_model(ftp=250, mmp={300: 290, 1200: 240, 1800: 232})
    assert m2.mode == "cp" and m2.n_data_points == 3
    assert m2.power(1800) == pytest.approx(232, abs=12)


def test_tier0_prediction_plausible_and_monotone_in_ftp():
    times = []
    for ftp in (220, 260, 300):
        ip = predict_fuji_from_inputs(COURSE, mass_total=75.0, ftp=ftp)
        assert ip.tier == "簡易"
        assert 3000 < ip.result.time_s < 9000
        assert ip.result.time_lo_s < ip.result.time_s < ip.result.time_hi_s
        assert ip.params.cda == pytest.approx(0.32, abs=1e-6)  # 標準値
        times.append(ip.result.time_s)
    assert times[0] > times[1] > times[2]  # FTP が高いほど速い


def test_adding_climbs_shifts_params_from_prior():
    rides = {
        f"r{s}": synthetic_ride(seed=s, length_km=L, avg_grade=g, base_power=p,
                                base_alt=100 + 20 * s, cda=0.28, crr=0.0060,
                                mass=75.0, eta=DEFAULT_ETA)
        for s, L, g, p in [(1, 6, 0.05, 235), (2, 4, 0.08, 255), (3, 9, 0.045, 220),
                           (4, 3, 0.10, 270), (5, 7, 0.06, 240), (6, 5, 0.07, 245)]
    }
    ip = predict_fuji_from_inputs(COURSE, mass_total=75.0, ftp=250, rides=rides)
    assert ip.tier == "登坂補正"
    assert ip.params.data_weight > 0.5
    # 登坂では Crr は識別できるので真値 0.006 方向へ標準値 0.005 から動く
    assert ip.params.crr > 0.0050
    # CdA は登坂だと分離が弱いので標準値近傍に留まる(正則化が効いている)
    assert abs(ip.params.cda - 0.32) < 0.05
    assert 3000 < ip.result.time_s < 9000

    # strength=0(正則化なし)にすると素の推定に近づく
    ip2 = predict_fuji_from_inputs(COURSE, mass_total=75.0, ftp=250, rides=rides,
                                   prior_strength=0.0)
    assert ip2.params.data_weight == pytest.approx(1.0)

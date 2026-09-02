"""富士ヒルクライム予測エンジン(仕様書 11 節・反復計算)。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .altitude import AltitudeModel, ThresholdLinear
from .calibration import CalibrationResult
from .course import CourseProfile
from .pdcurve import PDCurve
from .physics import air_density, speed_at_power


@dataclass
class PredictionResult:
    time_s: float
    time_lo_s: float
    time_hi_s: float
    p_sea_level: float          # T に対応する海面出力 [W]
    p_altitude: float           # 標高補正後の出力 [W]
    altitude_factor: float      # f(h_avg)
    time_no_altitude_s: float   # 標高補正を切ったときの推定
    iterations: int
    converged: bool
    segment_table: pd.DataFrame = field(repr=False, default=None)

    @staticmethod
    def fmt(seconds: float) -> str:
        if not np.isfinite(seconds):
            return "--:--:--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}"


def predict_fuji(
    course: CourseProfile,
    pd_curve: PDCurve,
    cal: CalibrationResult,
    mass: float,
    altitude_model: AltitudeModel | None = None,
    race_temp_c: float | None = None,
    tol_s: float = 5.0,
    max_iter: int = 50,
    power_sigma_w: float | None = None,
    report_bin_m: float = 500.0,
) -> PredictionResult:
    """反復計算で完走タイムを予測する。

    物理計算はコースプロファイルの分解能(既定 100 m)で行い、
    区間別テーブルは ``report_bin_m``(既定 500 m)ごとに集約して表示する。
    """
    altitude_model = altitude_model or ThresholdLinear()
    h_avg = course.elev_avg
    f_h = altitude_model.factor(h_avg)

    def course_time(power: float):
        total = 0.0
        fine = []
        for seg in course.segments.itertuples():
            rho = float(air_density(seg.elev_mid, race_temp_c))
            v = speed_at_power(power, seg.grade, mass, cal.crr, cal.cda, rho, cal.eta)
            dt = seg.length / v if v > 0 else float("inf")
            total += dt
            fine.append((seg.d0, seg.d1, seg.length, seg.grade, seg.elev_mid, dt))
        return total, fine

    def report_table(fine) -> pd.DataFrame:
        rows: dict[int, dict] = {}
        for d0, d1, length, grade, elev_mid, dt in fine:
            b = int(d0 // report_bin_m)
            r = rows.setdefault(b, {"d0": d0, "d1": d1, "length": 0.0, "time": 0.0,
                                    "g_num": 0.0, "e_num": 0.0})
            r["d1"] = d1
            r["length"] += length
            r["time"] += dt
            r["g_num"] += grade * length
            r["e_num"] += elev_mid * length
        ordered = [r for _, r in sorted(rows.items())]
        # 端数の最終区間(半分未満)は 1 つ前に統合
        if len(ordered) >= 2 and ordered[-1]["length"] < 0.5 * report_bin_m:
            last = ordered.pop()
            for k in ("length", "time", "g_num", "e_num"):
                ordered[-1][k] += last[k]
            ordered[-1]["d1"] = last["d1"]
        out = []
        for r in ordered:
            length, t = r["length"], r["time"]
            out.append({
                "区間": f"{r['d0']/1000:.1f}-{r['d1']/1000:.1f} km",
                "距離[m]": round(length),
                "勾配[%]": round(100 * r["g_num"] / length, 1) if length else 0.0,
                "標高[m]": round(r["e_num"] / length) if length else 0,
                "速度[km/h]": round(length / t * 3.6, 1) if t else 0.0,
                "時間": PredictionResult.fmt(t)[2:] if t < 3600 else PredictionResult.fmt(t),
            })
        return pd.DataFrame(out)

    # 1. 初期タイム推定(平均勾配ベースの粗い計算)
    t_guess, _ = course_time(pd_curve.cp)
    t0 = t_guess

    converged = False
    it = 0
    fine = []
    for it in range(1, max_iter + 1):
        p_sl = pd_curve.power(t0)
        p_alt = p_sl * f_h
        t1, fine = course_time(p_alt)
        if abs(t1 - t0) < tol_s:
            t0 = t1
            converged = True
            break
        t0 = 0.5 * (t0 + t1)  # 減衰付き更新で振動を防ぐ

    p_sl = pd_curve.power(t0)
    p_alt = p_sl * f_h
    _, fine = course_time(p_alt)

    # 標高補正なしの参考推定
    t_no_alt = t0
    for _ in range(max_iter):
        p = pd_curve.power(t_no_alt)
        tt, _ = course_time(p)
        if abs(tt - t_no_alt) < tol_s:
            t_no_alt = tt
            break
        t_no_alt = 0.5 * (t_no_alt + tt)

    # 信頼区間: パワーの不確かさ (W) を伝播。
    # power_sigma_w が与えられればそれと残差の大きい方、無ければ残差のみ。
    resid_sigma = cal.residual_std_w if np.isfinite(cal.residual_std_w) else 0.0
    sigma_p = max(resid_sigma, power_sigma_w or 0.0)
    if sigma_p <= 0:
        sigma_p = 10.0
    t_lo, _ = course_time(p_alt + sigma_p)
    t_hi, _ = course_time(p_alt - sigma_p)

    return PredictionResult(
        time_s=t0,
        time_lo_s=t_lo,
        time_hi_s=t_hi,
        p_sea_level=p_sl,
        p_altitude=p_alt,
        altitude_factor=f_h,
        time_no_altitude_s=t_no_alt,
        iterations=it,
        converged=converged,
        segment_table=report_table(fine),
    )

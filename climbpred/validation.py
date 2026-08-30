"""現時点の予測モデルの精度確認(キャリブレーションではなく検証)。

`data/rides/` の練習ライドから CdA/Crr/PD カーブを求め、富士ヒルのタイムを
複数の標高補正設定で予測し、実走の計測区間タイム(あれば)と並べる。
"""

from __future__ import annotations

from dataclasses import dataclass

import dataclasses

import pandas as pd

from .altitude import ThresholdLinear, make_model
from .course import CourseProfile
from .pipeline import AnalysisResult, analyze_rides, run_prediction
from .predict import PredictionResult, predict_fuji


DEFAULT_ALT_MODELS = [
    ("標高補正なし (k=0)", ThresholdLinear(k=0.0)),
    ("推奨 k=0.5 %/1000m", ThresholdLinear(k=0.0005)),
    ("文献 k=1.0 %/1000m", ThresholdLinear(k=0.0010)),
    ("Bassett 多項式", make_model("bassett_poly")),
]


@dataclass
class AccuracyReport:
    analysis: AnalysisResult
    table: pd.DataFrame           # 標高モデル別の予測タイム
    actual_timed_s: float | None  # 実走の計測区間タイム
    n_rides: int
    sensitivity: pd.DataFrame | None = None  # CdA/Crr を振ったときの予測(k=0)


# 参考: ロードバイク・ブラケットポジションの文献値
_REFERENCE_PARAMS = [
    ("推定値(回帰)", None, None),
    ("CdA 0.33 / Crr 0.004", 0.33, 0.004),
    ("CdA 0.30 / Crr 0.005", 0.30, 0.005),
    ("CdA 0.36 / Crr 0.004", 0.36, 0.004),
]


def _sensitivity_table(analysis, course, mass, actual_timed_s):
    cal = analysis.calibration
    rows = []
    for label, cda, crr in _REFERENCE_PARAMS:
        cc = cal if cda is None else dataclasses.replace(cal, cda=cda, crr=crr)
        pred = predict_fuji(course, analysis.pd_curve, cc, mass, ThresholdLinear(k=0.0))
        row = {"CdA/Crr": label, "予測タイム(k=0)": PredictionResult.fmt(pred.time_s)}
        if actual_timed_s:
            row["誤差[%]"] = round(100 * (pred.time_s - actual_timed_s) / actual_timed_s, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_check(
    practice_rides: dict[str, pd.DataFrame],
    course: CourseProfile,
    mass_total: float,
    actual_timed_s: float | None = None,
    alt_models=DEFAULT_ALT_MODELS,
    eta: float | None = None,
) -> AccuracyReport:
    kw = {} if eta is None else {"eta": eta}
    analysis = analyze_rides(practice_rides, mass_total=mass_total, **kw)

    rows = []
    for label, model in alt_models:
        pred: PredictionResult = run_prediction(analysis, course, mass_total, model)
        row = {
            "標高補正モデル": label,
            "予測タイム": PredictionResult.fmt(pred.time_s),
            "予測秒": round(pred.time_s),
            "f(h_avg)": round(pred.altitude_factor, 3),
            "海面出力[W]": round(pred.p_sea_level),
            "標高補正後[W]": round(pred.p_altitude),
        }
        if actual_timed_s:
            row["実測との差"] = PredictionResult.fmt(abs(pred.time_s - actual_timed_s))
            row["誤差[%]"] = round(100 * (pred.time_s - actual_timed_s) / actual_timed_s, 1)
        rows.append(row)

    return AccuracyReport(
        analysis=analysis,
        table=pd.DataFrame(rows),
        actual_timed_s=actual_timed_s,
        n_rides=len(practice_rides),
        sensitivity=_sensitivity_table(analysis, course, mass_total, actual_timed_s),
    )

"""取り込みから予測までを束ねるパイプライン。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .altitude import AltitudeModel
from .calibration import CalibrationResult, calibrate
from .climb_detect import ClimbSegment, detect_climbs
from .constants import CLIMB_MERGE_GAP, CLIMB_MIN_GRADE, CLIMB_MIN_LENGTH, DEFAULT_ETA
from .course import CourseProfile
from .pdcurve import PDCurve, extract_mmp, fit_cp_wprime
from .predict import PredictionResult, predict_fuji


@dataclass
class AnalysisResult:
    climbs: list[ClimbSegment]
    mmp: dict
    pd_curve: PDCurve
    calibration: CalibrationResult
    climb_overview: pd.DataFrame = field(repr=False, default=None)


def analyze_rides(
    rides: dict[str, pd.DataFrame],
    mass_total: float,
    eta: float = DEFAULT_ETA,
    min_grade: float = CLIMB_MIN_GRADE,
    min_length: float = CLIMB_MIN_LENGTH,
    merge_gap: float = CLIMB_MERGE_GAP,
) -> AnalysisResult:
    """複数ライドを解析し、PD カーブとキャリブレーション結果を返す。"""
    all_climbs: list[ClimbSegment] = []
    for ride_id, df in rides.items():
        all_climbs.extend(
            detect_climbs(df, ride_id, min_grade=min_grade, min_length=min_length, merge_gap=merge_gap)
        )

    if not all_climbs:
        raise ValueError(
            "登坂区間を検出できませんでした。勾配・距離の閾値を下げるか、"
            "登坂を含むライドを追加してください。"
        )

    mmp = extract_mmp(all_climbs)
    pd_curve = fit_cp_wprime(mmp)
    cal = calibrate(all_climbs, mass=mass_total, eta=eta)

    overview = pd.DataFrame(
        [
            {
                "登坂": c.label,
                "距離[km]": round(c.distance_m / 1000, 2),
                "獲得標高[m]": round(c.ascent_m),
                "平均勾配[%]": round(c.avg_grade * 100, 1),
                "所要[min]": round(c.duration_s / 60, 1),
                "平均パワー[W]": round(c.avg_power),
                "NP[W]": round(c.np_power),
            }
            for c in all_climbs
        ]
    )

    return AnalysisResult(
        climbs=all_climbs,
        mmp=mmp,
        pd_curve=pd_curve,
        calibration=cal,
        climb_overview=overview,
    )


def run_prediction(
    analysis: AnalysisResult,
    course: CourseProfile,
    mass_total: float,
    altitude_model: AltitudeModel,
    race_temp_c: float | None = None,
) -> PredictionResult:
    return predict_fuji(
        course=course,
        pd_curve=analysis.pd_curve,
        cal=analysis.calibration,
        mass=mass_total,
        altitude_model=altitude_model,
        race_temp_c=race_temp_c,
    )

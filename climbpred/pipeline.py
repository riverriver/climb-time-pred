"""取り込みから予測までを束ねるパイプライン。

- ``predict_fuji_from_inputs``: メインの予測フロー。FTP + 体重があれば走行データ
  ゼロでも予測でき、FIT/GPX を足すほど精度が上がる。
- ``analyze_rides`` / ``run_prediction``: 多数の走行データからフル解析する
  診断用パス(検出登坂一覧・PD カーブ・精度チェックで使用)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .altitude import AltitudeModel, ThresholdLinear
from .calibration import CalibrationResult, ParamPrior, calibrate
from .climb_detect import ClimbSegment, detect_climbs
from .constants import CLIMB_MERGE_GAP, CLIMB_MIN_GRADE, CLIMB_MIN_LENGTH, DEFAULT_ETA
from .course import CourseProfile
from .pdcurve import PDCurve, extract_mmp, fit_cp_wprime
from .power_model import PowerModel, build_power_model, combined_mmp
from .predict import PredictionResult, predict_fuji


# ---------------------------------------------------------------------------
# メインフロー
# ---------------------------------------------------------------------------
@dataclass
class InputPrediction:
    result: PredictionResult
    power_model: PowerModel
    params: CalibrationResult
    tier: str                       # "簡易" | "パワー補正" | "登坂補正"
    tier_detail: str
    notes: list[str] = field(default_factory=list)
    climbs: list[ClimbSegment] = field(default_factory=list)
    mmp: dict = field(default_factory=dict)
    climb_overview: pd.DataFrame | None = None


def predict_fuji_from_inputs(
    course: CourseProfile,
    mass_total: float,
    *,
    ftp: float | None = None,
    target_intensity: float = 1.0,
    cda_prior: float | None = None,
    crr_prior: float | None = None,
    prior_strength: float | None = None,
    eta: float = DEFAULT_ETA,
    altitude_model: AltitudeModel | None = None,
    race_temp_c: float | None = None,
    rides: dict[str, pd.DataFrame] | None = None,
    min_grade: float = CLIMB_MIN_GRADE,
    min_length: float = CLIMB_MIN_LENGTH,
    merge_gap: float = CLIMB_MERGE_GAP,
) -> InputPrediction:
    """FTP + 体重(+任意の走行データ)から富士ヒルの完走タイムを予測する。"""
    rides = rides or {}
    altitude_model = altitude_model or ThresholdLinear()
    prior = ParamPrior(
        cda=cda_prior if cda_prior is not None else ParamPrior().cda,
        crr=crr_prior if crr_prior is not None else ParamPrior().crr,
        strength=prior_strength if prior_strength is not None else ParamPrior().strength,
    )
    notes: list[str] = []

    # 1. 走行データから登坂を検出
    all_climbs: list[ClimbSegment] = []
    for rid, df in rides.items():
        all_climbs.extend(
            detect_climbs(df, rid, min_grade=min_grade, min_length=min_length, merge_gap=merge_gap)
        )

    # 2. 持続可能パワーモデル(全走行データの mean-max + 登坂の mean-max)
    mmp = combined_mmp(rides)
    if all_climbs:
        for d, p in extract_mmp(all_climbs).items():
            if d not in mmp or p > mmp[d]:
                mmp[d] = p
    # 初期の所要時間推定(FTP 補正係数の当て込み用)。粗くてよい。
    rough_t = _rough_time(course, ftp or (max(mmp.values()) if mmp else 200.0))
    power_model = build_power_model(
        ftp=ftp, mmp=mmp, intensity=target_intensity, est_duration_s=rough_t
    )
    notes.append(f"パワー: {power_model.source}")

    # 3. CdA / Crr(標準値 + 実走登坂で正則化補正)
    params = calibrate(all_climbs, mass=mass_total, eta=eta, prior=prior)
    if all_climbs:
        notes.append(
            f"CdA/Crr: 実走 {params.n_climbs} 本の登坂で補正"
            f"(データ寄与 {params.data_weight*100:.0f}%)"
        )
    else:
        notes.append("CdA/Crr: 標準値(登坂を含む走行データ未入力)")

    # 4. 予測
    result = predict_fuji(
        course=course,
        pd_curve=power_model,
        cal=params,
        mass=mass_total,
        altitude_model=altitude_model,
        race_temp_c=race_temp_c,
        power_sigma_w=power_model.sigma_w(rough_t),
    )

    # 5. 確度ラベル
    if all_climbs:
        tier, detail = "登坂補正", f"走行データ {len(rides)} 本(登坂 {len(all_climbs)} 区間)"
    elif rides:
        tier, detail = "パワー補正", f"走行データ {len(rides)} 本(登坂なし・パワーのみ反映)"
    else:
        tier, detail = "簡易", "入力値のみ(FTP + 体重)"

    overview = _climb_overview(all_climbs) if all_climbs else None
    return InputPrediction(
        result=result, power_model=power_model, params=params,
        tier=tier, tier_detail=detail, notes=notes,
        climbs=all_climbs, mmp=mmp, climb_overview=overview,
    )


def _rough_time(course: CourseProfile, power: float) -> float:
    """平均勾配・標準空気密度での粗いタイム推定 [s]。"""
    from .physics import air_density, speed_at_power

    total = 0.0
    rho = float(air_density(course.elev_avg))
    p = ParamPrior()
    for seg in course.segments.itertuples():
        v = speed_at_power(power, seg.grade, 80.0, p.crr, p.cda, rho, DEFAULT_ETA)
        total += seg.length / max(v, 0.5)
    return total


def _climb_overview(climbs: list[ClimbSegment]) -> pd.DataFrame:
    return pd.DataFrame(
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
            for c in climbs
        ]
    )


# ---------------------------------------------------------------------------
# 診断用フルパス(多数の走行データが前提)
# ---------------------------------------------------------------------------
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

    return AnalysisResult(
        climbs=all_climbs,
        mmp=mmp,
        pd_curve=pd_curve,
        calibration=cal,
        climb_overview=_climb_overview(all_climbs),
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

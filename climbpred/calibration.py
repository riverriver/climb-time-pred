"""パーソナルキャリブレーション(線形最小二乗で Crr / CdA を逆算)。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from .constants import (
    CDA_BOUNDS,
    CRR_BOUNDS,
    DEFAULT_ETA,
    G,
    PRIOR_CDA,
    PRIOR_CRR,
    PRIOR_STRENGTH,
)
from .physics import air_density, grade_to_angle, power_at_speed


@dataclass
class ParamPrior:
    """実走データが少ないときに CdA / Crr を引き寄せる標準値(事前分布)。"""

    cda: float = PRIOR_CDA
    crr: float = PRIOR_CRR
    strength: float = PRIOR_STRENGTH   # 相当サンプル数

    @staticmethod
    def default() -> "ParamPrior":
        return ParamPrior()


@dataclass
class CalibrationResult:
    crr: float
    cda: float
    eta: float
    n_points: int
    rmse_w: float           # 出力残差の RMSE [W]
    residual_std_w: float   # 出力残差の標準偏差 [W]
    crr_se: float           # Crr の標準誤差
    cda_se: float           # CdA の標準誤差
    per_climb: pd.DataFrame  # 登坂ごとの実測 vs モデル逆算タイム
    n_climbs: int = 0
    warnings: tuple[str, ...] = ()
    crr_unconstrained: float = float("nan")  # 有界化前の生の Crr
    cda_unconstrained: float = float("nan")
    crr_at_bound: bool = False               # Crr が物理下限/上限に張り付いたか
    cda_at_bound: bool = False
    data_weight: float = 0.0                 # 推定に占める実走データの割合(0=標準値のみ)
    prior: ParamPrior | None = None

    @property
    def summary(self) -> str:
        w = f", データ寄与 {self.data_weight*100:.0f}%" if self.prior is not None else ""
        return (
            f"CdA = {self.cda:.4f} m^2, Crr = {self.crr:.5f}, "
            f"eta = {self.eta:.3f}  (実走点 n={self.n_points}{w})"
        )

    @property
    def plausible(self) -> bool:
        return not self.warnings


def _check_plausibility(n_climbs, data_weight, crr_at_bound, cda_at_bound) -> tuple[str, ...]:
    w = []
    if crr_at_bound or cda_at_bound:
        w.append("CdA/Crr が物理レンジの端に達しました。登坂データの速度域が狭く、"
                 "空力項と転がり項を分離できていません。標準値寄りに補正しています。")
    if n_climbs == 0:
        w.append("実走の登坂データが無いため、CdA/Crr は標準値です。"
                 "登坂を含む走行データを 1 本足すと簡易補正が効きます。")
    elif data_weight < 0.5:
        w.append(f"登坂データが少なく(実走寄与 {data_weight*100:.0f}%)、CdA/Crr は"
                 "まだ標準値寄りです。勾配・速度域の異なる登坂を増やすと精度が上がります。")
    return tuple(w)


def _sample_frame(segments, eta, accel_limit, min_speed, min_power):
    """全登坂の全サンプルを 1 つの回帰用テーブルにまとめる。"""
    frames = []
    for seg in segments:
        s = seg.samples.copy()
        t = s["t"].to_numpy(dtype=float)
        dist = s["distance"].to_numpy(dtype=float)
        alt = s["altitude"].to_numpy(dtype=float)
        v = s["speed"].to_numpy(dtype=float)
        power = s["power"].to_numpy(dtype=float)

        # 局所勾配(平滑化した高度の距離微分)
        alt_s = pd.Series(alt).rolling(5, min_periods=1, center=True).mean().to_numpy()
        d_dist = np.gradient(dist)
        d_alt = np.gradient(alt_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            grade = np.where(d_dist > 0.1, d_alt / d_dist, np.nan)

        accel = np.gradient(v, t, edge_order=1)
        temp = s["temperature"].to_numpy(dtype=float) if "temperature" in s else np.full_like(v, np.nan)

        frames.append(
            pd.DataFrame(
                {
                    "seg": seg.label,
                    "v": v,
                    "power": power,
                    "grade": grade,
                    "alt": alt,
                    "accel": accel,
                    "temp": temp,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["v", "power", "grade", "alt"])
    mask = (
        (df["v"] >= min_speed)
        & (df["power"] >= min_power)
        & (df["grade"] > 0.0)
        & (df["accel"].abs() <= accel_limit)
    )
    return df[mask].reset_index(drop=True)


def params_from_prior(prior: ParamPrior, eta: float = DEFAULT_ETA) -> CalibrationResult:
    """実走データが無いときに標準値をそのまま返す。"""
    return CalibrationResult(
        crr=prior.crr,
        cda=prior.cda,
        eta=eta,
        n_points=0,
        rmse_w=float("nan"),
        residual_std_w=18.0,   # モデル前提の不確かさ(W)。信頼区間の下限として使う
        crr_se=float("nan"),
        cda_se=float("nan"),
        per_climb=pd.DataFrame(),
        n_climbs=0,
        warnings=_check_plausibility(0, 0.0, False, False),
        data_weight=0.0,
        prior=prior,
    )


def calibrate(
    segments,
    mass: float,
    eta: float = DEFAULT_ETA,
    accel_limit: float = 0.15,
    min_speed: float = 1.5,
    min_power: float = 80.0,
    crr_bounds: tuple[float, float] = CRR_BOUNDS,
    cda_bounds: tuple[float, float] = CDA_BOUNDS,
    prior: ParamPrior | None = None,
) -> CalibrationResult:
    """Crr / CdA を有界リッジ最小二乗で推定する(仕様書 8 節 + 標準値への正則化)。

        y  = eta*P_rider - M g sinθ v
        x1 = M g cosθ v
        x2 = 0.5 ρ v^3
        y  = Crr*x1 + CdA*x2

    ``prior`` を与えると、実走点数がその強度を大きく超えるまでは推定を
    標準値(CdA0/Crr0)寄りに保つ(擬似観測による Tikhonov 正則化)。
    """
    if not segments:
        if prior is not None:
            return params_from_prior(prior, eta)
        raise ValueError("キャリブレーションに使う登坂区間がありません。")

    data = _sample_frame(segments, eta, accel_limit, min_speed, min_power)
    if len(data) < 50:
        if prior is not None:
            r = params_from_prior(prior, eta)
            r.n_climbs = len(segments)
            return r
        raise ValueError(
            f"有効なサンプル点が不足しています (n={len(data)})。"
            "定常走行の登坂データを増やしてください。"
        )

    theta = grade_to_angle(data["grade"].to_numpy())
    v = data["v"].to_numpy()
    temp = data["temp"].to_numpy()
    temp = np.where(np.isnan(temp), None, temp)
    rho = np.array(
        [
            float(air_density(a, tc if tc is not None else None))
            for a, tc in zip(data["alt"].to_numpy(), temp)
        ]
    )

    y = eta * data["power"].to_numpy() - mass * G * np.sin(theta) * v
    x1 = mass * G * np.cos(theta) * v
    x2 = 0.5 * rho * v ** 3
    A = np.column_stack([x1, x2])
    n = len(y)

    # 制約なしの生の解(診断用)
    raw, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    crr_raw, cda_raw = float(raw[0]), float(raw[1])

    # 標準値への正則化(擬似観測)。data_weight = n / (n + strength)。
    data_weight = 1.0
    A_fit, y_fit = A, y
    if prior is not None and prior.strength > 0:
        data_weight = n / (n + prior.strength)
        rows, rhs = [], []
        for i, x0 in ((0, prior.crr), (1, prior.cda)):
            lam = np.sqrt(float(A[:, i] @ A[:, i]) * prior.strength / n)
            row = np.zeros(2)
            row[i] = lam
            rows.append(row)
            rhs.append(lam * x0)
        A_fit = np.vstack([A, rows])
        y_fit = np.append(y, rhs)

    lo = [crr_bounds[0], cda_bounds[0]]
    hi = [crr_bounds[1], cda_bounds[1]]
    sol = lsq_linear(A_fit, y_fit, bounds=(lo, hi), method="bvls")
    crr, cda = float(sol.x[0]), float(sol.x[1])
    tol = 1e-6
    crr_at_bound = crr <= crr_bounds[0] + tol or crr >= crr_bounds[1] - tol
    cda_at_bound = cda <= cda_bounds[0] + tol or cda >= cda_bounds[1] - tol

    resid = y - A @ sol.x
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        crr_se, cda_se = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        crr_se = cda_se = float("nan")

    # 出力ベースの残差 RMSE(W 換算)
    rmse_w = float(np.sqrt(np.mean(resid ** 2)) / eta)
    residual_std_w = float(np.std(resid) / eta)

    per_climb = _back_predict(segments, mass, crr, cda, eta)
    warns = _check_plausibility(len(segments), data_weight, crr_at_bound, cda_at_bound)

    return CalibrationResult(
        crr=crr,
        cda=cda,
        eta=eta,
        n_points=n,
        rmse_w=rmse_w,
        residual_std_w=residual_std_w,
        crr_se=crr_se,
        cda_se=cda_se,
        per_climb=per_climb,
        n_climbs=len(segments),
        warnings=warns,
        data_weight=data_weight,
        prior=prior,
        crr_unconstrained=crr_raw,
        cda_unconstrained=cda_raw,
        crr_at_bound=crr_at_bound,
        cda_at_bound=cda_at_bound,
    )


def _back_predict(segments, mass, crr, cda, eta) -> pd.DataFrame:
    """各登坂を平均パワーでモデル逆算し、実測タイムと比較する(仕様書 12 節)。"""
    from .physics import speed_at_power

    rows = []
    for seg in segments:
        alt_mid = float(seg.samples["altitude"].mean())
        temp = seg.samples["temperature"].mean() if "temperature" in seg.samples else np.nan
        rho = float(air_density(alt_mid, None if np.isnan(temp) else float(temp)))
        v = speed_at_power(seg.avg_power, seg.avg_grade, mass, crr, cda, rho, eta)
        model_time = seg.distance_m / v if v > 0 else float("inf")
        err = model_time - seg.duration_s
        rows.append(
            {
                "登坂": seg.label,
                "実測タイム[s]": round(seg.duration_s, 0),
                "モデル逆算[s]": round(model_time, 0),
                "誤差[s]": round(err, 0),
                "誤差[%]": round(100 * err / seg.duration_s, 1) if seg.duration_s else np.nan,
                "平均パワー[W]": round(seg.avg_power, 0),
            }
        )
    return pd.DataFrame(rows)

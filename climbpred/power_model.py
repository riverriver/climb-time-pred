"""持続可能パワーモデル。

入力の少なさに応じて段階的に精度が上がる:

1. **FTP のみ**  … FTP(≒1時間パワー)にレース所要時間の補正係数を掛ける。
2. **実走データ 1〜数本**  … 走行データの mean-max power を 1〜数点使い、
   FTP を実測値で補強する。
3. **実走データ多数**  … 複数の継続時間で CP + W'/t をフィットする。

`predict_fuji` は `power(t)` と `cp`(初期推定用)だけを使うので、
`PDCurve` と同じインターフェースを持たせてある。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import (
    FTP_FACTOR_CEIL,
    FTP_FACTOR_FLOOR,
    FTP_FADE_PER_HOUR,
    FTP_HOUR_S,
    PD_DURATIONS,
)
from .pdcurve import _best_rolling_power


def ftp_duration_factor(t_s: float) -> float:
    """FTP に対する、継続時間 t 秒での持続可能パワー比。

    t = 3600 s(FTP の定義)で 1.0。それより長いと 1 時間あたり
    ``FTP_FADE_PER_HOUR`` ずつ低下、短いと少しだけ上振れ。
    """
    t = max(float(t_s), 1.0)
    if t <= FTP_HOUR_S:
        f = 1.0 + 0.04 * (FTP_HOUR_S - t) / FTP_HOUR_S
        return float(min(f, FTP_FACTOR_CEIL))
    f = 1.0 - FTP_FADE_PER_HOUR * (t - FTP_HOUR_S) / FTP_HOUR_S
    return float(max(f, FTP_FACTOR_FLOOR))


@dataclass
class PowerModel:
    """海面上での持続可能パワー [W]。"""

    mode: str                 # "ftp" | "cp"
    cp: float                 # CP [W](ftp モードでは FTP×補正の代表値)
    w_prime: float            # W' [J]
    ftp: float | None
    intensity: float          # ユーザー指定の強度倍率(1.0 = 自動)
    n_data_points: int        # フィットに使った実測 mean-max 点数
    sigma_frac: float         # 相対的な不確かさ(信頼区間用)
    source: str               # 表示用の説明
    points: dict = field(default_factory=dict)

    def power(self, t: float) -> float:
        if self.mode == "ftp":
            return self.ftp * ftp_duration_factor(t) * self.intensity
        return (self.cp + self.w_prime / max(t, 1.0)) * self.intensity

    def sigma_w(self, t: float) -> float:
        return self.sigma_frac * self.power(t)


def mean_max_power(df: pd.DataFrame, durations=PD_DURATIONS) -> dict:
    """1 本の走行データ全体から各継続時間のベストパワーを抽出。"""
    if "power" not in df or "t" not in df:
        return {}
    power = df["power"].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    out: dict[float, float] = {}
    for d in durations:
        val = _best_rolling_power(power, t, d)
        if val is not None and val > 0:
            out[float(d)] = val
    return out


def combined_mmp(rides: dict[str, pd.DataFrame], durations=PD_DURATIONS) -> dict:
    """複数の走行データから継続時間ごとのベストパワーを取る。"""
    best: dict[float, float] = {}
    for df in rides.values():
        for d, p in mean_max_power(df, durations).items():
            if d not in best or p > best[d]:
                best[d] = p
    return best


def _fit_cp(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    t = np.array([d for d, _ in points], dtype=float)
    p = np.array([w for _, w in points], dtype=float)
    A = np.column_stack([np.ones_like(t), 1.0 / t])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    cp, w_prime = float(coef[0]), float(max(coef[1], 0.0))
    pred = A @ coef
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 - float(np.sum((p - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return cp, w_prime, r2


def build_power_model(
    ftp: float | None = None,
    mmp: dict | None = None,
    intensity: float = 1.0,
    est_duration_s: float = 4800.0,
) -> PowerModel:
    """FTP と(あれば)実測 mean-max power から持続可能パワーモデルを作る。"""
    mmp = {float(k): float(v) for k, v in (mmp or {}).items() if v and v > 0}
    n = len(mmp)

    # 実測点が 2 点以上 → CP + W'/t をフィット。
    # 20 分以上の実測点が無い場合のみ FTP を 3600s の点として補う。
    if n >= 2:
        points = sorted(mmp.items())
        has_long = any(d >= 1200 for d, _ in points)
        used_ftp = bool(ftp) and not has_long and not any(abs(d - FTP_HOUR_S) < 1 for d, _ in points)
        if used_ftp:
            points = sorted(points + [(FTP_HOUR_S, ftp)])
        cp, w_prime, r2 = _fit_cp(points)
        sigma = 0.05 if n < 3 else (0.04 if n < 5 else 0.03)
        src = f"実測パワーカーブ({n}点" + ("・FTP補完" if used_ftp else "") + f")、CP≈{cp:.0f}W"
        return PowerModel("cp", cp, w_prime, ftp, intensity, n, sigma, src, dict(points))

    # 実測点 1 点 → FTP と合わせて 2 点フィット、無ければ 1 点で FTP モデル補強
    if n == 1:
        d, w = next(iter(mmp.items()))
        if ftp:
            cp, w_prime, _ = _fit_cp(sorted([(d, w), (FTP_HOUR_S, ftp)]))
            src = f"FTP + 実測 {int(d)}s パワー({w:.0f}W)"
            return PowerModel("cp", cp, w_prime, ftp, intensity, 1, 0.05, src, {d: w, FTP_HOUR_S: ftp})
        # FTP 無し・1 点のみ → その点を FTP 相当とみなす
        eff_ftp = w * (FTP_HOUR_S / d) ** 0.0  # そのまま
        src = f"実測 {int(d)}s パワー({w:.0f}W)を FTP 相当として使用"
        return PowerModel("ftp", w, 0.0, w, intensity, 1, 0.07, src, {d: w})

    # 実測なし → FTP のみ
    if ftp:
        f = ftp_duration_factor(est_duration_s)
        src = f"FTP {ftp:.0f}W × 所要時間補正 {f*100:.0f}%(実測データなし)"
        return PowerModel("ftp", ftp * f, 0.0, ftp, intensity, 0, 0.08, src, {})

    raise ValueError("FTP または実走データのいずれかが必要です。")

"""パワーデュレーションモデル(CP / W' の2パラメータ)。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import PD_DURATIONS


@dataclass
class PDCurve:
    cp: float          # Critical Power [W]
    w_prime: float     # W' [J]
    points: dict       # {duration_s: best_power_W}
    r2: float

    def power(self, t: float) -> float:
        """継続時間 t [s] における持続可能パワー [W]。"""
        return self.cp + self.w_prime / max(t, 1.0)

    def time_for_power(self, p: float) -> float:
        """パワー p を出せる継続時間 [s](P > CP のとき有限)。"""
        if p <= self.cp:
            return float("inf")
        return self.w_prime / (p - self.cp)


def _best_rolling_power(power: np.ndarray, t: np.ndarray, window_s: float) -> float | None:
    if t.size < 2:
        return None
    duration = t[-1] - t[0]
    if duration < window_s:
        return None
    dt = np.median(np.diff(t)) or 1.0
    win = max(int(round(window_s / dt)), 1)
    if win > power.size:
        return None
    roll = pd.Series(power).rolling(win, min_periods=win).mean().to_numpy()
    if np.all(np.isnan(roll)):
        return None
    return float(np.nanmax(roll))


def extract_mmp(segments, durations=PD_DURATIONS) -> dict:
    """全登坂区間から各継続時間のベストパワー(mean-max power)を抽出。"""
    best: dict[float, float] = {}
    for seg in segments:
        power = seg.samples["power"].to_numpy(dtype=float)
        t = seg.samples["t"].to_numpy(dtype=float)
        for d in durations:
            val = _best_rolling_power(power, t, d)
            if val is None:
                continue
            if d not in best or val > best[d]:
                best[d] = val
    return best


def fit_cp_wprime(mmp: dict) -> PDCurve:
    """P = CP + W'/t を線形最小二乗でフィット。"""
    items = [(d, p) for d, p in sorted(mmp.items()) if p and p > 0]
    if len(items) < 2:
        raise ValueError(
            "PD カーブのフィットには少なくとも 2 点の継続時間データが必要です。"
            "より長い登坂、または継続時間の幅がある登坂を追加してください。"
        )
    t = np.array([d for d, _ in items], dtype=float)
    p = np.array([pw for _, pw in items], dtype=float)

    A = np.column_stack([np.ones_like(t), 1.0 / t])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    cp, w_prime = float(coef[0]), float(coef[1])

    pred = A @ coef
    ss_res = float(np.sum((p - pred) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return PDCurve(cp=cp, w_prime=w_prime, points=dict(items), r2=r2)


def pdcurve_from_intervals(power_curve: dict) -> PDCurve:
    """Intervals.icu が算出済みのパワーカーブ({秒: W})からフィット(Phase 2)。"""
    mmp = {float(k): float(v) for k, v in power_curve.items() if float(k) >= 60}
    return fit_cp_wprime(mmp)

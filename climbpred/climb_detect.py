"""クライム自動抽出アルゴリズム(仕様書 5 節)。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import CLIMB_MERGE_GAP, CLIMB_MIN_GRADE, CLIMB_MIN_LENGTH


@dataclass
class ClimbSegment:
    ride_id: str
    start_idx: int
    end_idx: int
    distance_m: float
    ascent_m: float
    avg_grade: float
    duration_s: float
    avg_power: float
    np_power: float
    samples: pd.DataFrame = field(repr=False, default=None)

    @property
    def label(self) -> str:
        return (
            f"{self.ride_id}: {self.distance_m/1000:.1f}km / "
            f"+{self.ascent_m:.0f}m / {self.avg_grade*100:.1f}% / "
            f"{self.duration_s/60:.1f}min / {self.avg_power:.0f}W"
        )


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or x.size < window:
        return x
    return (
        pd.Series(x)
        .rolling(window, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )


def _normalized_power(power: np.ndarray, t: np.ndarray) -> float:
    if power.size < 5:
        return float(np.mean(power)) if power.size else 0.0
    dt = np.median(np.diff(t)) or 1.0
    win = max(int(round(30.0 / dt)), 1)
    roll = pd.Series(power).rolling(win, min_periods=1).mean().to_numpy()
    return float((np.mean(roll ** 4)) ** 0.25)


def detect_climbs(
    df: pd.DataFrame,
    ride_id: str,
    min_grade: float = CLIMB_MIN_GRADE,
    min_length: float = CLIMB_MIN_LENGTH,
    merge_gap: float = CLIMB_MERGE_GAP,
    smooth_window_m: float = 100.0,
) -> list[ClimbSegment]:
    """1 ライドから登坂区間を検出する。"""
    if len(df) < 10:
        return []

    dist = df["distance"].to_numpy()
    alt = df["altitude"].to_numpy()
    t = df["t"].to_numpy()

    step = np.median(np.diff(dist))
    step = step if step and step > 0 else 5.0
    win = max(int(round(smooth_window_m / step)), 3)
    alt_s = _smooth(alt, win)

    # サンプル間勾配
    d_dist = np.gradient(dist)
    d_alt = np.gradient(alt_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(d_dist > 0.1, d_alt / d_dist, 0.0)
    grade = np.clip(grade, -0.30, 0.30)

    climbing = grade >= min_grade

    # 連続する登坂ブロックを抽出
    blocks: list[tuple[int, int]] = []
    i = 0
    n = len(df)
    while i < n:
        if climbing[i]:
            j = i
            while j + 1 < n and climbing[j + 1]:
                j += 1
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1

    # ギャップ結合
    merged: list[tuple[int, int]] = []
    for b in blocks:
        if merged and dist[b[0]] - dist[merged[-1][1]] <= merge_gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(list(b))
    merged = [tuple(b) for b in merged]

    segments: list[ClimbSegment] = []
    for s, e in merged:
        seg_dist = dist[e] - dist[s]
        seg_ascent = alt_s[e] - alt_s[s]
        if seg_dist < min_length:
            continue
        avg_grade = seg_ascent / seg_dist if seg_dist > 0 else 0.0
        if avg_grade < min_grade:
            continue
        sub = df.iloc[s : e + 1].copy()
        power = sub["power"].to_numpy(dtype=float)
        segments.append(
            ClimbSegment(
                ride_id=ride_id,
                start_idx=int(s),
                end_idx=int(e),
                distance_m=float(seg_dist),
                ascent_m=float(seg_ascent),
                avg_grade=float(avg_grade),
                duration_s=float(t[e] - t[s]),
                avg_power=float(np.mean(power)),
                np_power=_normalized_power(power, sub["t"].to_numpy()),
                samples=sub,
            )
        )
    return segments

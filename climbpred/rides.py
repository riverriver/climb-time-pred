"""リファレンス実走データの保管・要約(power / heart_rate / cadence / time)。

`data/rides/` に置いた FIT / GPX をライブラリとして扱う。ここに置いたファイルは
キャリブレーション用の登坂抽出だけでなく、レース分析(区間タイム・パワー配分・
心拍推移)をあとから参照するための記録として保持する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .course import CourseProfile
from .fit_ingest import load_ride

RIDES_DIR = Path(__file__).resolve().parent.parent / "data" / "rides"


def list_rides(rides_dir: Path = RIDES_DIR) -> list[Path]:
    if not rides_dir.exists():
        return []
    return sorted(
        p for p in rides_dir.rglob("*")
        if p.suffix.lower() in (".fit", ".gpx")
    )


def load_ride_file(path: str | Path) -> pd.DataFrame:
    return load_ride(Path(path))


def _np_power(power: pd.Series, t: pd.Series) -> float:
    dt = float(np.median(np.diff(t))) or 1.0
    win = max(int(round(30.0 / dt)), 1)
    roll = power.clip(lower=0).rolling(win, min_periods=1).mean()
    return float((roll ** 4).mean() ** 0.25)


def _agg(series: pd.Series) -> dict:
    s = series.dropna()
    s = s[s > 0]
    if s.empty:
        return {"avg": None, "max": None}
    return {"avg": float(s.mean()), "max": float(s.max())}


@dataclass
class RideSummary:
    name: str
    started_at: pd.Timestamp
    duration_s: float
    distance_m: float
    ascent_m: float
    avg_speed_kmh: float
    power: dict
    np_power: float
    heart_rate: dict
    cadence: dict
    temperature: dict
    timed_section_s: float | None = None  # 富士ヒル計測区間の実測タイム

    def as_row(self) -> dict:
        def fmt(sec):
            if sec is None or not np.isfinite(sec):
                return "-"
            sec = int(round(sec))
            return f"{sec//3600}:{sec%3600//60:02d}:{sec%60:02d}"

        return {
            "記録": self.name,
            "日付": self.started_at.strftime("%Y-%m-%d"),
            "全体時間": fmt(self.duration_s),
            "計測区間タイム": fmt(self.timed_section_s),
            "距離[km]": round(self.distance_m / 1000, 1),
            "獲得[m]": round(self.ascent_m),
            "平均P[W]": round(self.power["avg"]) if self.power["avg"] else None,
            "NP[W]": round(self.np_power),
            "最大P[W]": round(self.power["max"]) if self.power["max"] else None,
            "平均HR": round(self.heart_rate["avg"]) if self.heart_rate["avg"] else None,
            "平均Cad": round(self.cadence["avg"]) if self.cadence["avg"] else None,
            "気温[℃]": round(self.temperature["avg"], 1) if self.temperature["avg"] else None,
        }


def _deg2m(dlat, dlon, lat0):
    return 111_320.0 * np.hypot(dlat, dlon * np.cos(np.radians(lat0)))


def _nearest(df: pd.DataFrame, lat: float, lon: float) -> tuple[float, float]:
    """(経過時間 [s], 最近点までの距離 [m])。"""
    dm = _deg2m(df["lat"] - lat, df["lon"] - lon, lat)
    i = int(np.nanargmin(dm.values))
    return float(df["t"].iloc[i]), float(dm.iloc[i])


def timed_section_seconds(
    df: pd.DataFrame, course: CourseProfile, tol_m: float = 400.0
) -> float | None:
    """ライドがコースのスタート/ゴール座標付近を通る場合のみ、その経過時間差を返す。

    練習ライドなど富士ヒルのコースを走っていない記録では None。
    """
    if course.start_latlon is None or course.finish_latlon is None:
        return None
    if df[["lat", "lon"]].isna().all().any():
        return None
    t0, d0 = _nearest(df, *course.start_latlon)
    t1, d1 = _nearest(df, *course.finish_latlon)
    if d0 > tol_m or d1 > tol_m or t1 <= t0:
        return None
    return t1 - t0


def summarize_ride(
    df: pd.DataFrame, name: str, course: CourseProfile | None = None
) -> RideSummary:
    ascent = float(df["altitude"].diff().clip(lower=0).sum())
    dur = float(df["t"].iloc[-1])
    dist = float(df["distance"].iloc[-1])
    return RideSummary(
        name=name,
        started_at=df["timestamp"].iloc[0],
        duration_s=dur,
        distance_m=dist,
        ascent_m=ascent,
        avg_speed_kmh=(dist / dur * 3.6) if dur else 0.0,
        power=_agg(df["power"]),
        np_power=_np_power(df["power"], df["t"]),
        heart_rate=_agg(df["heart_rate"]),
        cadence=_agg(df["cadence"]),
        temperature=_agg(df["temperature"]),
        timed_section_s=timed_section_seconds(df, course) if course is not None else None,
    )

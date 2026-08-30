"""データ取り込み層:FIT / GPX を time-series の DataFrame に変換する。

共通スキーマ(列): t, timestamp, lat, lon, altitude[m], speed[m/s],
power[W], heart_rate[bpm], cadence[rpm], temperature[degC], distance[m]
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)
_GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}

REQUIRED = ["timestamp", "power", "altitude", "speed"]
CHANNELS = ("altitude", "speed", "power", "heart_rate", "cadence", "temperature", "distance")

Source = Union[str, Path, bytes, io.IOBase]


def _first(record, names):
    for n in names:
        v = record.get_value(n)
        if v is not None:
            return v
    return None


def parse_fit(source: Union[str, bytes, io.IOBase]) -> pd.DataFrame:
    """FIT ファイルをパースして record メッセージを DataFrame 化する。

    列: timestamp, lat, lon, altitude[m], speed[m/s], power[W],
        temperature[degC], distance[m]
    """
    from fitparse import FitFile

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    fitfile = FitFile(source)

    rows = []
    for rec in fitfile.get_messages("record"):
        lat = rec.get_value("position_lat")
        lon = rec.get_value("position_long")
        rows.append(
            {
                "timestamp": rec.get_value("timestamp"),
                "lat": lat * SEMICIRCLE_TO_DEG if lat is not None else np.nan,
                "lon": lon * SEMICIRCLE_TO_DEG if lon is not None else np.nan,
                "altitude": _first(rec, ["enhanced_altitude", "altitude"]),
                "speed": _first(rec, ["enhanced_speed", "speed"]),
                "power": rec.get_value("power"),
                "heart_rate": rec.get_value("heart_rate"),
                "cadence": rec.get_value("cadence"),
                "temperature": rec.get_value("temperature"),
                "distance": rec.get_value("distance"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("FIT ファイルに record データが見つかりません。")
    return _finalize_ride(df)


def parse_gpx_ride(source: Source) -> pd.DataFrame:
    """GPX ライドをパースする(power/hr/cad/atemp 拡張に対応)。"""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    root = ET.parse(source).getroot()

    rows = []
    for tp in root.iterfind(".//g:trkpt", _GPX_NS):
        ele = tp.find("g:ele", _GPX_NS)
        tm = tp.find("g:time", _GPX_NS)
        rows.append(
            {
                "timestamp": tm.text if tm is not None else None,
                "lat": float(tp.get("lat")),
                "lon": float(tp.get("lon")),
                "altitude": float(ele.text) if ele is not None else np.nan,
                "speed": np.nan,
                "power": _gpx_num(tp, "power"),
                "heart_rate": _gpx_num(tp, "hr"),
                "cadence": _gpx_num(tp, "cad"),
                "temperature": _gpx_num(tp, "atemp"),
                "distance": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("GPX に trkpt が見つかりません。")
    return _finalize_ride(df)


def load_ride(source: Source, kind: str | None = None) -> pd.DataFrame:
    """拡張子(または kind)で FIT / GPX を振り分けてライドを読む。"""
    if kind is None:
        name = str(source).lower() if not isinstance(source, (bytes, io.IOBase)) else ""
        kind = "gpx" if name.endswith(".gpx") else "fit"
    return parse_gpx_ride(source) if kind == "gpx" else parse_fit(source)


def _gpx_num(trkpt, tag):
    el = trkpt.find(f".//{{*}}{tag}")
    if el is None or el.text is None:
        return np.nan
    try:
        return float(el.text)
    except ValueError:
        return np.nan


def _finalize_ride(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    for col in CHANNELS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # 速度が無ければ位置・距離から生成
    if df["speed"].isna().all():
        if df["distance"].notna().any():
            d = df["distance"].interpolate()
        else:
            d = _haversine_cum(df["lat"].to_numpy(), df["lon"].to_numpy())
            df["distance"] = d
        dt = df["t"].diff()
        df["speed"] = (pd.Series(d).diff() / dt).clip(lower=0).fillna(0.0)

    # 距離が無ければ速度を積分
    if df["distance"].isna().all():
        dt = df["t"].diff().fillna(0.0).clip(lower=0)
        df["distance"] = (df["speed"].fillna(0.0) * dt).cumsum()

    missing = [c for c in ("altitude",) if df[c].isna().all()]
    if missing:
        raise ValueError(f"必須フィールドが欠落しています: {missing}")

    df["power"] = df["power"].fillna(0.0)
    for col in ("altitude", "speed", "distance"):
        df[col] = df[col].interpolate().bfill().ffill()
    df["speed"] = df["speed"].rolling(5, min_periods=1, center=True).mean()
    return df


def _haversine_cum(lat, lon):
    r = 6371000.0
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(a)))])


def synthetic_ride(
    seed: int,
    length_km: float,
    avg_grade: float,
    base_power: float,
    base_alt: float = 200.0,
    cda: float = 0.32,
    crr: float = 0.005,
    mass: float = 75.0,
    eta: float = 0.976,
    dt: float = 1.0,
) -> pd.DataFrame:
    """デモ用の合成登坂ライドを生成する(既知の CdA/Crr から順計算)。"""
    from .physics import air_density, speed_at_power

    rng = np.random.default_rng(seed)
    dist_target = length_km * 1000.0
    t = 0.0
    dist = 0.0
    alt = base_alt
    rows = []
    while dist < dist_target:
        # 勾配は平均まわりに緩やかに変動(滑らかな地形を模擬)
        grade = avg_grade + 0.02 * np.sin(dist / 1200.0)
        grade = max(grade, 0.005)
        power = base_power + rng.normal(0, 10)
        rho = float(air_density(alt))
        v = speed_at_power(power, grade, mass, crr, cda, rho, eta)
        v = max(v, 0.5)
        dist += v * dt
        alt += v * dt * grade
        t += dt
        rows.append(
            {
                "t": t,
                "timestamp": pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(seconds=t),
                "lat": np.nan,
                "lon": np.nan,
                "altitude": alt,
                "speed": v,
                "power": max(power, 0.0),
                "temperature": np.nan,
                "distance": dist,
            }
        )
    return pd.DataFrame(rows)

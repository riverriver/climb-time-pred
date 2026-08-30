"""トラック(公式コース FIT / 実走 GPX)から計測区間プロファイルを生成する。

**優先**: Garmin/公式の *course* FIT(`file_id.type == "course"`)があればそれを使う。
計測スタート/ゴールがトラックの端そのものなので最も正確。

GPX トラックはセグメント境界を持たないため、計測スタート/ゴールを次のどれかで決める:

- ``anchor="track_ends"``(course FIT の既定): トラックの先頭と末尾をそのまま使う。
- ``anchor="finish_distance"``(GPX の既定): 最高標高地点をゴールとし、
  その ``timed_length_m`` 手前を計測スタートとする。
- ``anchor="elevation"``: 指定標高でスタート/ゴールを切り出す。
- ``anchor="manual"``: スタート/ゴールの (lat, lon) を直接指定する。
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .course import CourseProfile

_GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}
TIMED_LENGTH_M = 24000.0
_SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)

Source = Union[str, Path, bytes, io.IOBase]


@dataclass
class Track:
    lat: np.ndarray
    lon: np.ndarray
    ele: np.ndarray
    dist: np.ndarray          # 累積距離 [m]
    temp: np.ndarray          # 気温 [degC](無ければ NaN)
    is_course: bool = False   # 公式コース定義由来か


def _haversine_cumulative(lat, lon):
    r = 6371000.0
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return np.concatenate([[0.0], np.cumsum(2 * r * np.arcsin(np.sqrt(a)))])


def _finalize(lat, lon, ele, temp, dist=None, is_course=False) -> Track:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ele = pd.Series(ele, dtype=float).interpolate().bfill().ffill().to_numpy()
    if dist is None or np.all(~np.isfinite(dist)):
        dist = _haversine_cumulative(lat, lon)
    return Track(lat=lat, lon=lon, ele=ele, dist=np.asarray(dist, dtype=float),
                 temp=np.asarray(temp, dtype=float), is_course=is_course)


def parse_gpx_track(source: Source) -> Track:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    root = ET.parse(source).getroot()

    lat, lon, ele, temp = [], [], [], []
    for tp in root.iterfind(".//g:trkpt", _GPX_NS):
        lat.append(float(tp.get("lat")))
        lon.append(float(tp.get("lon")))
        e = tp.find("g:ele", _GPX_NS)
        ele.append(float(e.text) if e is not None else np.nan)
        atemp = tp.find(".//{*}atemp")
        temp.append(float(atemp.text) if atemp is not None else np.nan)

    if len(lat) < 10:
        raise ValueError("GPX トラックポイントが不足しています。")
    return _finalize(lat, lon, ele, temp)


def parse_fit_track(source: Source) -> Track:
    """FIT の record メッセージからトラックを取得(course FIT / ride FIT 両対応)。"""
    from fitparse import FitFile

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    elif isinstance(source, Path):
        source = str(source)
    fitfile = FitFile(source)

    is_course = False
    try:
        for m in fitfile.get_messages("file_id"):
            if m.get_value("type") == "course":
                is_course = True
    except Exception:  # noqa: BLE001
        pass

    lat, lon, ele, temp, dist = [], [], [], [], []
    for rec in fitfile.get_messages("record"):
        la = rec.get_value("position_lat")
        lo = rec.get_value("position_long")
        if la is None or lo is None:
            continue
        lat.append(la * _SEMICIRCLE_TO_DEG)
        lon.append(lo * _SEMICIRCLE_TO_DEG)
        ele.append(rec.get_value("enhanced_altitude") or rec.get_value("altitude"))
        temp.append(rec.get_value("temperature"))
        dist.append(rec.get_value("distance"))

    if len(lat) < 10:
        raise ValueError("FIT に位置つき record が不足しています。")
    d = np.array([np.nan if v is None else v for v in dist], dtype=float)
    return _finalize(lat, lon, ele, temp, dist=d, is_course=is_course)


def load_track(source: Source, kind: str | None = None) -> Track:
    if kind is None:
        name = str(source).lower() if not isinstance(source, (bytes, io.IOBase)) else ""
        kind = "fit" if name.endswith(".fit") else "gpx"
    return parse_fit_track(source) if kind == "fit" else parse_gpx_track(source)


def _nearest_index(track: Track, lat: float, lon: float) -> int:
    return int(np.argmin((track.lat - lat) ** 2 + (track.lon - lon) ** 2))


def _resolve_bounds(track, anchor, timed_length_m, start_elev, finish_elev, start_ll, finish_ll):
    d, e = track.dist, track.ele

    if anchor == "track_ends":
        return 0, len(d) - 1

    if anchor == "manual":
        si = _nearest_index(track, *start_ll)
        fi = _nearest_index(track, *finish_ll)
        return (si, fi) if si < fi else (fi, si)

    tail = d > d[-1] - 2000.0
    fi = int(np.where(tail)[0][np.argmax(e[tail])])

    if anchor == "elevation":
        si = int(np.argmin(np.abs(e[: fi + 1] - start_elev)))
        if finish_elev is not None:
            fi = int(np.argmin(np.abs(e[: fi + 1] - finish_elev)))
        return si, fi

    # finish_distance
    si = int(np.searchsorted(d, d[fi] - timed_length_m))
    return si, fi


def build_course_profile(
    source: Source,
    kind: str | None = None,
    name: str = "Mt.富士ヒルクライム",
    anchor: str | None = None,
    timed_length_m: float = TIMED_LENGTH_M,
    start_elev: float = 1035.0,
    finish_elev: float | None = None,
    start_latlon: tuple[float, float] | None = None,
    finish_latlon: tuple[float, float] | None = None,
    resample_m: float = 100.0,
    smooth_m: float = 100.0,
) -> tuple[CourseProfile, pd.DataFrame]:
    """トラックから CourseProfile と (distance_m, elevation_m) の DataFrame を返す。"""
    track = load_track(source, kind)
    if anchor is None:
        anchor = "track_ends" if track.is_course else "finish_distance"

    si, fi = _resolve_bounds(
        track, anchor, timed_length_m, start_elev, finish_elev, start_latlon, finish_latlon
    )

    seg_d = track.dist[si : fi + 1] - track.dist[si]
    seg_e = track.ele[si : fi + 1]

    fine_x = np.arange(0.0, seg_d[-1] + 1e-6, 25.0)
    fine_e = np.interp(fine_x, seg_d, seg_e)
    win = max(int(round(smooth_m / 25.0)), 1)
    if win > 1:
        kernel = np.ones(win) / win
        padded = np.pad(fine_e, (win, win), mode="edge")
        fine_e = np.convolve(padded, kernel, mode="same")[win:-win]

    out_x = np.arange(0.0, seg_d[-1], resample_m)
    out_e = np.interp(out_x, fine_x, fine_e)
    out_x = np.append(out_x, seg_d[-1])
    out_e = np.append(out_e, fine_e[-1])

    df = pd.DataFrame({"distance_m": np.round(out_x).astype(int), "elevation_m": np.round(out_e, 1)})

    d = df["distance_m"].to_numpy(float)
    e = df["elevation_m"].to_numpy(float)
    seg = pd.DataFrame(
        {
            "d0": d[:-1],
            "d1": d[1:],
            "length": d[1:] - d[:-1],
            "grade": (e[1:] - e[:-1]) / (d[1:] - d[:-1]),
            "elev_mid": (e[1:] + e[:-1]) / 2,
        }
    )
    seg = seg[seg["length"] > 0].reset_index(drop=True)
    profile = CourseProfile(
        name=name,
        segments=seg,
        start_latlon=(float(track.lat[si]), float(track.lon[si])),
        finish_latlon=(float(track.lat[fi]), float(track.lon[fi])),
    )
    return profile, df


# 後方互換
def build_course_profile_from_gpx(source, **kw):
    kw.pop("kind", None)
    return build_course_profile(source, kind="gpx", **kw)

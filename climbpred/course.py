"""富士ヒルクライム コースデータ(仕様書 10 節)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PROFILE = DATA_DIR / "fuji_course_profile.csv"


@dataclass
class CourseProfile:
    name: str
    segments: pd.DataFrame   # 列: d0, d1, length, grade, elev_mid
    start_latlon: tuple[float, float] | None = None
    finish_latlon: tuple[float, float] | None = None

    @property
    def distance_m(self) -> float:
        return float(self.segments["d1"].iloc[-1])

    @property
    def ascent_m(self) -> float:
        return float(self.segments["length"] @ self.segments["grade"].clip(lower=0))

    @property
    def elev_start(self) -> float:
        return float(self.segments["elev_mid"].iloc[0] - self.segments["grade"].iloc[0] * self.segments["length"].iloc[0] / 2)

    @property
    def elev_finish(self) -> float:
        last = self.segments.iloc[-1]
        return float(last["elev_mid"] + last["grade"] * last["length"] / 2)

    @property
    def elev_avg(self) -> float:
        w = self.segments["length"].to_numpy()
        return float(np.average(self.segments["elev_mid"].to_numpy(), weights=w))

    @property
    def avg_grade(self) -> float:
        return (self.elev_finish - self.elev_start) / self.distance_m


def load_course(path: str | Path = DEFAULT_PROFILE, name: str = "Mt.富士ヒルクライム") -> CourseProfile:
    if str(path).lower().endswith((".gpx", ".fit")):
        from .course_build import build_course_profile

        return build_course_profile(path, name=name)[0]

    start_ll, finish_ll = _read_latlon_header(path)
    df = pd.read_csv(path, comment="#")
    df = df.sort_values("distance_m").reset_index(drop=True)
    d = df["distance_m"].to_numpy(dtype=float)
    e = df["elevation_m"].to_numpy(dtype=float)

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
    return CourseProfile(
        name=name, segments=seg, start_latlon=start_ll, finish_latlon=finish_ll
    )


def _read_latlon_header(path: str | Path):
    start = finish = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                low = line.lower()
                for key in ("start_latlon", "finish_latlon"):
                    if key in low:
                        try:
                            lat, lon = (float(x) for x in low.split(":", 1)[1].split(","))
                            if key == "start_latlon":
                                start = (lat, lon)
                            else:
                                finish = (lat, lon)
                        except ValueError:
                            pass
    except OSError:
        pass
    return start, finish

"""コーストラック(公式コース FIT / 実走 GPX)から
data/fuji_course_profile.csv を再生成する。

使い方:
    python scripts/build_course.py data/fuji_hillclimb_course.fit
    python scripts/build_course.py <実走の富士ヒル>.gpx --anchor finish_distance
    python scripts/build_course.py track.gpx --anchor manual \
        --start-latlon 35.4512 138.7583 --finish-latlon 35.3941 138.7320
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from climbpred.course_build import build_course_profile  # noqa: E402

OUT = ROOT / "data" / "fuji_course_profile.csv"

HEADER = [
    "# Mt.富士ヒルクライム 計測区間 距離-標高プロファイル",
    "# 出典: {src}(scripts/build_course.py, anchor={anchor})",
    "# 距離 {km:.2f} km / 標高差 {gain:.0f} m / 平均勾配 {grade:.2f} %",
    "# start_latlon: {slat:.6f},{slon:.6f}",
    "# finish_latlon: {flat:.6f},{flon:.6f}",
    "# distance_m: 計測スタートからの累積距離 [m] / elevation_m: 標高 [m]",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", type=Path, help=".fit(公式コース)または .gpx")
    ap.add_argument("--anchor", default=None,
                    choices=["track_ends", "finish_distance", "elevation", "manual"],
                    help="既定: course FIT は track_ends、GPX は finish_distance")
    ap.add_argument("--timed-length", type=float, default=24000.0)
    ap.add_argument("--start-elev", type=float, default=1035.0)
    ap.add_argument("--finish-elev", type=float, default=None)
    ap.add_argument("--start-latlon", type=float, nargs=2, default=None)
    ap.add_argument("--finish-latlon", type=float, nargs=2, default=None)
    ap.add_argument("--resample", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    course, df = build_course_profile(
        args.track,
        anchor=args.anchor,
        timed_length_m=args.timed_length,
        start_elev=args.start_elev,
        finish_elev=args.finish_elev,
        start_latlon=tuple(args.start_latlon) if args.start_latlon else None,
        finish_latlon=tuple(args.finish_latlon) if args.finish_latlon else None,
        resample_m=args.resample,
    )

    gain = course.elev_finish - course.elev_start
    slat, slon = course.start_latlon or (0.0, 0.0)
    flat, flon = course.finish_latlon or (0.0, 0.0)
    lines = [
        h.format(src=args.track.name, anchor=args.anchor or "auto",
                 km=course.distance_m / 1000, gain=gain, grade=course.avg_grade * 100,
                 slat=slat, slon=slon, flat=flat, flon=flon)
        for h in HEADER
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
        df.to_csv(f, index=False)

    print(f"書き出し: {args.out}")
    print(f"距離 {course.distance_m/1000:.2f} km / 標高差 {gain:.0f} m / "
          f"平均勾配 {course.avg_grade*100:.2f} % / 区間数 {len(course.segments)}")


if __name__ == "__main__":
    main()

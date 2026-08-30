"""コーストラック(公式コース FIT / 実走 GPX)からのプロファイル生成。"""

from pathlib import Path

import pytest

from climbpred.course import load_course
from climbpred.course_build import build_course_profile, load_track

DATA = Path(__file__).resolve().parent.parent / "data"
COURSE_FIT = DATA / "fuji_hillclimb_course.fit"
RIDE_GPX = DATA / "rides" / "fuji_hillclimb.gpx"

fit_only = pytest.mark.skipif(not COURSE_FIT.exists(), reason="コース FIT が無い")
gpx_only = pytest.mark.skipif(not RIDE_GPX.exists(), reason="実走 GPX が無い")


@fit_only
def test_official_course_fit():
    course, df = build_course_profile(COURSE_FIT)
    assert course.distance_m == pytest.approx(24000, abs=100)
    assert course.avg_grade == pytest.approx(0.052, abs=0.004)
    assert 1200 < course.elev_finish - course.elev_start < 1320
    assert course.start_latlon is not None and course.finish_latlon is not None
    assert (df["distance_m"].diff().dropna() > 0).all()


@fit_only
def test_load_track_detects_course_fit():
    tr = load_track(COURSE_FIT)
    assert tr.is_course is True


@gpx_only
def test_ride_gpx_finish_distance_anchor():
    course, _ = build_course_profile(RIDE_GPX, anchor="finish_distance")
    assert course.distance_m == pytest.approx(24000, abs=150)


@fit_only
def test_csv_carries_latlon_header():
    course = load_course()  # data/fuji_course_profile.csv(FIT 由来)
    assert course.start_latlon is not None
    assert 35.3 < course.start_latlon[0] < 35.5

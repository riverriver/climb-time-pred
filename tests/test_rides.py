"""リファレンス実走データの読み込みと要約。"""

from pathlib import Path

import pytest

from climbpred.course import load_course
from climbpred.rides import list_rides, load_ride_file, summarize_ride

RIDE = Path(__file__).resolve().parent.parent / "data" / "rides" / "fuji_hillclimb.gpx"
pytestmark = pytest.mark.skipif(not RIDE.exists(), reason="実走 GPX が無い")


def test_ride_channels_present():
    df = load_ride_file(RIDE)
    for col in ("power", "heart_rate", "cadence", "t", "altitude", "distance"):
        assert col in df.columns
    assert (df["power"] > 0).mean() > 0.8
    assert (df["heart_rate"] > 0).mean() > 0.8
    assert (df["cadence"] > 0).mean() > 0.8


def test_summary_and_timed_section():
    df = load_ride_file(RIDE)
    s = summarize_ride(df, "fuji", load_course())
    assert 150 < s.power["avg"] < 260
    assert 120 < s.heart_rate["avg"] < 185
    # 計測区間タイムは全体時間より短い(パレード分)
    assert s.timed_section_s is not None
    assert s.timed_section_s < s.duration_s
    assert 3600 < s.timed_section_s < 7200

"""精度確認(検証)パイプラインのスモークテスト。"""

from pathlib import Path

import pytest

from climbpred.course import load_course
from climbpred.rides import list_rides, load_ride_file, timed_section_seconds
from climbpred.validation import accuracy_check

RIDES = Path(__file__).resolve().parent.parent / "data" / "rides"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        len(list(RIDES.glob("i*.fit"))) < 3, reason="練習 FIT が無い"
    ),
]


def _split():
    course = load_course()
    practice, actual = {}, None
    for p in list_rides():
        df = load_ride_file(p)
        ts = timed_section_seconds(df, course)
        if ts is not None:
            actual = ts
        else:
            practice[p.name] = df
    return course, practice, actual


def test_timed_section_only_matches_race_ride():
    _, practice, actual = _split()
    # 練習ライドは富士ヒルのコースを通らないので計測区間タイムは付かない
    assert all(name.startswith("i") for name in practice)
    # 実走 GPX があれば計測区間タイムが取れる(60〜100 分)
    if actual is not None:
        assert 3600 < actual < 6000


def test_accuracy_report_shape():
    course, practice, actual = _split()
    rep = accuracy_check(practice, course, mass_total=75.0, actual_timed_s=actual)
    assert len(rep.table) == 4
    assert rep.table["予測秒"].between(3000, 7200).all()
    # 標高補正を強めるほど遅くなる
    secs = rep.table["予測秒"].tolist()
    assert secs[0] <= secs[2]

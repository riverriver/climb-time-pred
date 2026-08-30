"""data/rides/ の練習ライドで現行モデルの富士ヒル予測精度を確認する。

    python scripts/accuracy_check.py --mass 75.0

実走の富士ヒル(コース座標付近を通る記録)が data/rides/ にあれば、
その計測区間タイムを実測値として自動で並べる。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from climbpred.course import load_course  # noqa: E402
from climbpred.rides import list_rides, load_ride_file, timed_section_seconds  # noqa: E402
from climbpred.validation import accuracy_check  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, required=True, help="総質量(ライダー+バイク+装備)[kg]")
    args = ap.parse_args()

    course = load_course()
    practice, actual = {}, None
    for p in list_rides():
        df = load_ride_file(p)
        ts = timed_section_seconds(df, course)
        if ts is not None:
            actual = ts
            print(f"実走(富士ヒル): {p.name}  計測区間タイム {ts/60:.1f} 分")
        else:
            practice[p.name] = df

    if not practice:
        raise SystemExit("data/rides/ に練習ライドがありません。")
    print(f"練習ライド {len(practice)} 本で解析中...")

    rep = accuracy_check(practice, course, args.mass, actual_timed_s=actual)
    cal = rep.analysis.calibration
    pdc = rep.analysis.pd_curve

    print("\n=== キャリブレーション(検証目的・校正には非採用)===")
    print(cal.summary)
    if not cal.plausible:
        for w in cal.warnings:
            print("  ! " + w)
    print(f"PD カーブ: CP {pdc.cp:.0f} W / W' {pdc.w_prime/1000:.1f} kJ / R2 {pdc.r2:.2f}")
    print(f"MMP: {{{', '.join(f'{int(k)}s:{v:.0f}W' for k, v in sorted(rep.analysis.mmp.items()))}}}")

    print("\n=== 富士ヒル予測(標高補正モデル別)===")
    print(rep.table.to_string(index=False))

    if rep.sensitivity is not None:
        print("\n=== CdA/Crr 感度(標高補正なし)===")
        print(rep.sensitivity.to_string(index=False))

    if actual:
        print(f"\n実測(参考): {actual/60:.1f} 分")


if __name__ == "__main__":
    main()

"""富士ヒルクライム タイム予測 Web アプリ(Streamlit)。

実行:  streamlit run app.py
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from climbpred.altitude import available_models, make_model
from climbpred.constants import (
    CLIMB_MERGE_GAP,
    CLIMB_MIN_GRADE,
    CLIMB_MIN_LENGTH,
    DEFAULT_ETA,
    DEFAULT_H_THRESHOLD,
    DEFAULT_K_DECAY,
    RECOMMENDED_K_RANGE,
)
from climbpred.course import DEFAULT_PROFILE, load_course
from climbpred.fit_ingest import parse_fit, synthetic_ride
from climbpred.pipeline import analyze_rides, run_prediction
from climbpred.predict import PredictionResult
from climbpred.rides import list_rides, load_ride_file, summarize_ride

st.set_page_config(page_title="富士ヒルクライム タイム予測", page_icon="🚴", layout="wide")

st.title("🚴 富士ヒルクライム タイム予測")
st.caption(
    "過去の登坂データ(FIT)から CdA・Crr を逆算し、標高補正込みで完走タイムを予測します。"
    "物理パラメータの手入力は不要です。"
)

# --------------------------------------------------------------------------------------
# サイドバー:入力
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1. 体重・装備")
    rider_kg = st.number_input("ライダー体重 [kg]", 40.0, 120.0, 68.0, 0.1)
    bike_kg = st.number_input("バイク + 装備 [kg]", 5.0, 20.0, 8.0, 0.1)
    mass_total = rider_kg + bike_kg
    st.metric("総質量", f"{mass_total:.1f} kg")

    st.header("2. 走行データ")
    data_source = st.radio(
        "入力方法",
        ["FIT ファイルをアップロード", "デモデータで試す"],
        help="Intervals.icu 同期(API トークン認証)は Phase 2 で対応予定。",
    )
    uploaded = None
    if data_source == "FIT ファイルをアップロード":
        uploaded = st.file_uploader(
            "登坂を含む FIT ファイル(約10本推奨)",
            type=["fit", "FIT"],
            accept_multiple_files=True,
        )

    with st.expander("クライム自動抽出の閾値"):
        min_grade = st.slider("最小平均勾配 [%]", 1.0, 8.0, CLIMB_MIN_GRADE * 100, 0.5) / 100
        min_length = st.slider("最小継続距離 [m]", 300, 3000, int(CLIMB_MIN_LENGTH), 100)
        merge_gap = st.slider("結合許容ギャップ [m]", 0, 1000, int(CLIMB_MERGE_GAP), 50)

    st.header("3. 標高補正モデル")
    model_name = st.selectbox("f(h) モデル", available_models(), index=0)
    if model_name == "threshold_linear":
        h_threshold = st.number_input("閾値標高 [m]", 500, 2500, int(DEFAULT_H_THRESHOLD), 50,
                                      help="この標高までは出力低下なし(f=1.0)")
        lo, hi = RECOMMENDED_K_RANGE
        k_pm = st.number_input(
            "減衰係数 k(%/1000m、手入力)",
            0.0, 20.0, DEFAULT_K_DECAY * 1000, 0.1,
            help="閾値標高を超えた分に対する、1000m あたりの持続可能パワー低下率。",
        )
        k_decay = k_pm / 1000.0
        st.caption(
            f"推奨 k ≈ {lo*1000:.1f}〜{hi*1000:.1f} %/1000m。"
            "文献でよく使う k=1.0(10%/1000m)は富士ヒルの標高帯では過大になりやすい。"
            "物理モデルで扱いたい場合は Bassett を選択。"
        )
        alt_kwargs = {"h_threshold": float(h_threshold), "k": float(k_decay)}
    else:
        h_threshold = st.number_input("閾値標高 [m]", 500, 2500, 1500, 50)
        st.caption("Bassett 多項式:1500m 未満は無視、以降は加速度的に低下(係数入力なし)。")
        alt_kwargs = {"h_threshold": float(h_threshold)}

    st.header("4. レース当日(任意)")
    use_temp = st.checkbox("気温を指定して空気密度を補正", value=False)
    race_temp = st.number_input("五合目付近の平均気温 [°C]", -10.0, 30.0, 8.0, 0.5) if use_temp else None

    eta = st.number_input("駆動効率 η(固定)", 0.90, 1.0, DEFAULT_ETA, 0.001, disabled=True)
    run = st.button("予測を実行", type="primary", use_container_width=True)


# --------------------------------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _demo_rides() -> dict[str, pd.DataFrame]:
    specs = [
        (1, 6.0, 0.055, 235), (2, 4.0, 0.08, 255), (3, 9.0, 0.045, 220),
        (4, 3.0, 0.10, 270), (5, 12.0, 0.04, 210), (6, 5.0, 0.07, 245),
        (7, 7.5, 0.06, 230), (8, 2.5, 0.11, 280), (9, 10.0, 0.05, 218),
        (10, 5.5, 0.065, 240),
    ]
    return {
        f"demo{seed:02d}": synthetic_ride(
            seed=seed, length_km=L, avg_grade=g, base_power=p,
            base_alt=150 + 40 * seed, cda=0.31, crr=0.0048, mass=75.0, eta=DEFAULT_ETA,
        )
        for seed, L, g, p in specs
    }


def _load_rides() -> dict[str, pd.DataFrame]:
    if data_source == "デモデータで試す":
        return _demo_rides()
    rides: dict[str, pd.DataFrame] = {}
    for f in uploaded or []:
        rides[f.name] = parse_fit(f.getvalue())
    return rides


@st.cache_data(show_spinner=False)
def _course_from_csv():
    return load_course(DEFAULT_PROFILE)


@st.cache_data(show_spinner=False)
def _course_from_track(raw: bytes, ext: str, anchor: str, timed_km: float, start_elev: float):
    from climbpred.course_build import build_course_profile

    return build_course_profile(
        raw, kind=ext, anchor=anchor if anchor != "auto" else None,
        timed_length_m=timed_km * 1000, start_elev=start_elev,
    )[0]


st.subheader("コースプロファイル")
st.caption("既定: data/fuji_course_profile.csv(公式コース FIT 由来・24.0km)")
track_file = st.file_uploader(
    "コースを差し替える(公式コース .fit / 実走 .gpx)", type=["gpx", "GPX", "fit", "FIT"],
)
if track_file is not None:
    ext = "fit" if track_file.name.lower().endswith(".fit") else "gpx"
    gc1, gc2, gc3 = st.columns(3)
    anchor = gc1.selectbox(
        "計測区間の切り出し", ["auto", "track_ends", "finish_distance", "elevation"],
        help="auto: コース FIT は端をそのまま / GPX は最高標高地点の N km 手前",
    )
    timed_km = gc2.number_input("計測距離 [km]", 10.0, 30.0, 24.0, 0.1)
    start_elev = gc3.number_input("スタート標高 [m](elevation 時)", 800.0, 1500.0, 1035.0, 5.0)
    course = _course_from_track(track_file.getvalue(), ext, anchor, timed_km, start_elev)
else:
    course = _course_from_csv()

c1, c2, c3, c4 = st.columns(4)
c1.metric("コース距離", f"{course.distance_m/1000:.1f} km")
c2.metric("獲得標高", f"{course.ascent_m:.0f} m")
c3.metric("平均勾配", f"{course.avg_grade*100:.1f} %")
c4.metric("フィニッシュ標高", f"{course.elev_finish:.0f} m")

with st.expander("標高プロファイルを表示"):
    seg = course.segments
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg["d1"] / 1000, y=seg["elev_mid"], fill="tozeroy", name="標高"))
    fig.update_layout(xaxis_title="距離 [km]", yaxis_title="標高 [m]", height=300, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------
# リファレンス実走データ(data/rides/)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="実走データを読み込み中...")
def _all_rides() -> dict:
    return {p.name: load_ride_file(p) for p in list_rides()}


@st.cache_data(show_spinner=False)
def _ride_summaries():
    course_ref = _course_from_csv()
    return [
        (name, df, summarize_ride(df, Path(name).stem, course_ref))
        for name, df in _all_rides().items()
    ]


_have_rides = bool(list_rides())
show_refs = _have_rides and st.checkbox(
    f"自分の実走データ / 精度チェックを開く({len(list_rides())} 本 — 初回読み込みに時間がかかります)",
    value=False,
)
_refs = _ride_summaries() if show_refs else []
if _refs:
    with st.expander(f"自分の実走データ({len(_refs)} 本) — パワー / 心拍 / ケイデンス / タイム", expanded=False):
        st.dataframe(
            pd.DataFrame([s.as_row() for _, _, s in _refs]),
            use_container_width=True, hide_index=True,
        )
        pick = st.selectbox("記録を選択", [name for name, _, _ in _refs])
        name, rdf, rs = next(r for r in _refs if r[0] == pick)

        m = st.columns(4)
        m[0].metric("計測区間タイム",
                    PredictionResult.fmt(rs.timed_section_s) if rs.timed_section_s else "-")
        m[1].metric("平均パワー / NP",
                    f"{rs.power['avg']:.0f} / {rs.np_power:.0f} W" if rs.power['avg'] else "-")
        m[2].metric("平均心拍", f"{rs.heart_rate['avg']:.0f} bpm" if rs.heart_rate['avg'] else "-")
        m[3].metric("平均ケイデンス", f"{rs.cadence['avg']:.0f} rpm" if rs.cadence['avg'] else "-")

        ch = st.multiselect("表示チャンネル", ["power", "heart_rate", "cadence", "speed", "altitude"],
                            default=["power", "heart_rate"])
        if ch:
            x_km = rdf["distance"] / 1000
            fig = go.Figure()
            for c in ch:
                y = rdf[c] * (3.6 if c == "speed" else 1.0)
                fig.add_trace(go.Scatter(x=x_km, y=y.rolling(15, min_periods=1).mean(), name=c))
            fig.update_layout(xaxis_title="距離 [km]", height=340, margin=dict(t=20),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"生データ: data/rides/{name} (raw を保持。列: {', '.join(rdf.columns)})")


# --------------------------------------------------------------------------------------
# 予測の精度チェック(検証:data/rides/ の練習ライドを使う。校正ではない)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="精度チェックを計算中...")
def _accuracy(mass: float):
    from climbpred.rides import timed_section_seconds
    from climbpred.validation import accuracy_check

    course_ref = _course_from_csv()
    practice, actual = {}, None
    for name, df in _all_rides().items():
        ts = timed_section_seconds(df, course_ref)
        if ts is not None:
            actual = ts
        else:
            practice[name] = df
    if len(practice) < 3:
        return None
    return accuracy_check(practice, course_ref, mass, actual_timed_s=actual), actual, len(practice)


_acc = _accuracy(mass_total) if show_refs else None
if _acc:
    rep, actual, n_practice = _acc
    with st.expander(
        f"予測の精度チェック(練習ライド {n_practice} 本・検証用/校正には非採用)", expanded=False
    ):
        if actual:
            st.metric("実走の計測区間タイム(実測)", PredictionResult.fmt(actual))
        st.markdown("**標高補正モデル別の予測**")
        st.dataframe(rep.table.drop(columns=["予測秒"]), use_container_width=True, hide_index=True)
        if rep.sensitivity is not None:
            st.markdown("**CdA/Crr 感度(標高補正なし)** — 予測が回帰値の不確かさに鈍感か確認")
            st.dataframe(rep.sensitivity, use_container_width=True, hide_index=True)
        cal = rep.analysis.calibration
        st.caption(
            f"検証キャリブレーション: {cal.summary}"
            + ("" if cal.plausible else "  ⚠ " + " / ".join(cal.warnings))
        )
        st.caption(
            f"PD カーブ: CP {rep.analysis.pd_curve.cp:.0f} W / "
            f"W′ {rep.analysis.pd_curve.w_prime/1000:.1f} kJ / R² {rep.analysis.pd_curve.r2:.2f}"
            "  — 校正方針は追って調整"
        )


# --------------------------------------------------------------------------------------
# 実行
# --------------------------------------------------------------------------------------
if run:
    try:
        rides = _load_rides()
        if not rides:
            st.warning("FIT ファイルをアップロードするか、デモデータを選択してください。")
            st.stop()

        with st.spinner("登坂抽出・PD カーブ・キャリブレーションを計算中..."):
            analysis = analyze_rides(
                rides, mass_total=mass_total, eta=eta,
                min_grade=min_grade, min_length=float(min_length), merge_gap=float(merge_gap),
            )
            alt_model = make_model(model_name, **alt_kwargs)
            pred: PredictionResult = run_prediction(
                analysis, course, mass_total, alt_model, race_temp_c=race_temp,
            )

        cal = analysis.calibration
        pdc = analysis.pd_curve

        st.header("予測結果")
        m1, m2, m3 = st.columns(3)
        m1.metric("予測タイム", PredictionResult.fmt(pred.time_s),
                  help="単独走行・一定出力・無風の前提")
        m2.metric("信頼区間", f"{PredictionResult.fmt(pred.time_lo_s)} 〜 {PredictionResult.fmt(pred.time_hi_s)}",
                  help="キャリブレーション残差 ±1σ を出力に伝播")
        m3.metric("標高補正の影響",
                  f"+{PredictionResult.fmt(pred.time_s - pred.time_no_altitude_s)}",
                  help="補正なし推定との差")

        if not pred.converged:
            st.warning("反復計算が収束しませんでした。結果は参考値です。")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("海面相当の目標出力", f"{pred.p_sea_level:.0f} W")
        b2.metric("標高補正後の出力", f"{pred.p_altitude:.0f} W")
        b3.metric("標高補正係数 f(h_avg)", f"{pred.altitude_factor:.3f}")
        b4.metric("補正なし推定", PredictionResult.fmt(pred.time_no_altitude_s))

        st.subheader("区間別の予測")
        st.dataframe(pred.segment_table, use_container_width=True, hide_index=True)

        st.divider()
        st.header("キャリブレーション")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CdA", f"{cal.cda:.4f} m²",
                  help=f"制約なしの生値 {cal.cda_unconstrained:.4f} / 標準誤差 ±{cal.cda_se:.4f}")
        k2.metric("Crr", f"{cal.crr:.5f}" + ("  ⛓" if cal.crr_at_bound else ""),
                  help=f"制約なしの生値 {cal.crr_unconstrained:.5f}")
        k3.metric("回帰点数", f"{cal.n_points:,}")
        k4.metric("出力残差 RMSE", f"{cal.rmse_w:.1f} W")

        if cal.crr_at_bound:
            st.info(
                f"Crr は物理下限 {cal.crr:.5f} に固定しました(制約なしの最小二乗では "
                f"{cal.crr_unconstrained:.5f} と負)。平地〜短い登坂中心で重力項と転がり項が"
                "分離できず、CdA が残差を吸収しています。**登坂での予測タイムはこの不確かさに"
                "鈍感**(重力項が支配的)ですが、CdA/Crr の個別値は参考程度に扱ってください。"
            )
        if not cal.plausible:
            st.warning("\n".join(f"- {w}" for w in cal.warnings))

        st.subheader("登坂ごとの寄与度(実測 vs モデル逆算)")
        st.dataframe(cal.per_climb, use_container_width=True, hide_index=True)

        st.subheader("検出された登坂")
        st.dataframe(analysis.climb_overview, use_container_width=True, hide_index=True)

        st.subheader("パワーデュレーションカーブ")
        pcol1, pcol2 = st.columns([2, 1])
        with pcol1:
            durs = sorted(analysis.mmp)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=durs, y=[analysis.mmp[d] for d in durs],
                                     mode="markers", name="実測ベストパワー"))
            tt = list(range(30, 3900, 30))
            fig.add_trace(go.Scatter(x=tt, y=[pdc.power(t) for t in tt],
                                     mode="lines", name="CP + W'/t フィット"))
            fig.update_layout(xaxis_title="継続時間 [s]", yaxis_title="パワー [W]",
                              height=350, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        with pcol2:
            st.metric("CP", f"{pdc.cp:.0f} W")
            st.metric("W′", f"{pdc.w_prime/1000:.1f} kJ")
            st.metric("フィット R²", f"{pdc.r2:.3f}")

        st.caption(f"標高補正モデル: {alt_model.describe()}  /  η = {eta:.3f}(固定)")

    except Exception as e:  # noqa: BLE001
        st.error(f"エラー: {e}")
        with st.expander("トレースバック"):
            st.code(traceback.format_exc())
else:
    st.info("左のサイドバーで入力を設定し、「予測を実行」を押してください。")

"""富士ヒルクライム タイム予測 Web アプリ(Streamlit)。

実行:  streamlit run app.py

構成:
  1. FTP + 体重 + 機材重量 → その場で予測(走行データ不要)
  2. FIT / GPX を足すと精度が上がる
     - 1 本でも:パワーを実測値に置き換え(簡易補正)
     - 登坂を含む: CdA / Crr も標準値から補正
     - 多いほど: 補正が強くなる
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from climbpred.altitude import available_models, make_model
from climbpred.constants import (
    ADDED_WEIGHT_DEFAULT,
    CDA_PRESETS,
    DEFAULT_ETA,
    DEFAULT_H_THRESHOLD,
    DEFAULT_K_DECAY,
    PRIOR_CRR,
    RECOMMENDED_K_RANGE,
)
from climbpred.course import DEFAULT_PROFILE, load_course
from climbpred.fit_ingest import load_ride, synthetic_ride
from climbpred.pipeline import predict_fuji_from_inputs
from climbpred.predict import PredictionResult
from climbpred.rides import list_rides, load_ride_file, summarize_ride

st.set_page_config(page_title="富士ヒルクライム タイム予測", page_icon="🚴", layout="wide")
fmt = PredictionResult.fmt

st.title("🚴 富士ヒルクライム タイム予測")
st.caption(
    "FTP と体重を入れるだけで予測できます。走行データ(FIT / GPX)を足すほど"
    "パワー・空力・転がり抵抗が実測で補正され、精度が上がります。"
)

# --------------------------------------------------------------------------------------
# サイドバー:基本情報
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("基本情報")
    ftp = st.number_input("FTP(1時間パワー)[W]", 80.0, 500.0, 250.0, 1.0,
                          help="最近の FTP。1時間持続できるおよそのパワー。")
    rider_kg = st.number_input("ライダー体重 [kg]", 35.0, 130.0, 62.0, 0.1)
    bike_kg = st.number_input("バイク + 装備 [kg]", 4.0, 20.0, 7.5, 0.1)
    added_kg = st.number_input("追加重量(ボトル・補給・工具)[kg]", 0.0, 6.0,
                               float(ADDED_WEIGHT_DEFAULT), 0.1)
    mass_total = rider_kg + bike_kg + added_kg
    st.metric("総質量", f"{mass_total:.1f} kg")
    st.metric("FTP / kg", f"{ftp / rider_kg:.2f} W/kg")

    with st.expander("詳細設定(任意)"):
        pos = st.selectbox("ポジション(CdA の目安)", list(CDA_PRESETS) + ["手入力"], index=1)
        cda_prior = (
            st.number_input("CdA [m²]", 0.20, 0.55, 0.32, 0.005)
            if pos == "手入力" else CDA_PRESETS[pos]
        )
        crr_prior = st.number_input("Crr(転がり抵抗係数)", 0.0020, 0.0120, float(PRIOR_CRR), 0.0005,
                                    format="%.4f", help="舗装路のクリンチャーで概ね 0.004〜0.006。")
        pace_pct = st.slider("ペーシング調整 [%](+ で攻める / − で余裕)", -12, 12, 0, 1,
                             help="FTP からの所要時間補正に対する上乗せ。まずは 0 で。")

        st.markdown("**標高によるパワー低下**")
        model_name = st.selectbox("f(h) モデル", available_models(), index=0)
        if model_name == "threshold_linear":
            h_threshold = st.number_input("閾値標高 [m]", 500, 2500, int(DEFAULT_H_THRESHOLD), 50,
                                          help="この標高までは低下なし(f=1.0)")
            lo, hi = RECOMMENDED_K_RANGE
            k_pm = st.number_input("減衰係数 k(%/1000m)", 0.0, 20.0,
                                   DEFAULT_K_DECAY * 1000, 0.1)
            st.caption(
                f"推奨 k ≈ {lo*1000:.1f}〜{hi*1000:.1f} %/1000m。"
                "文献値 1.0(10%/1000m)は富士ヒルの標高帯では過大になりやすい。"
            )
            alt_kwargs = {"h_threshold": float(h_threshold), "k": float(k_pm) / 1000.0}
        else:
            h_threshold = st.number_input("閾値標高 [m]", 500, 2500, 1500, 50)
            st.caption("Bassett 多項式:閾値未満は無視、以降は加速度的に低下。")
            alt_kwargs = {"h_threshold": float(h_threshold)}

        use_temp = st.checkbox("レース当日の気温で空気密度を補正", value=False)
        race_temp = (st.number_input("五合目付近の平均気温 [°C]", -10.0, 30.0, 8.0, 0.5)
                     if use_temp else None)

    run = st.button("予測する", type="primary", use_container_width=True)


# --------------------------------------------------------------------------------------
# コースプロファイル
# --------------------------------------------------------------------------------------
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


st.subheader("コース")
with st.expander("コースを差し替える(既定は公式コース FIT 由来の 24.0 km)"):
    track_file = st.file_uploader("公式コース .fit / 実走 .gpx", type=["gpx", "GPX", "fit", "FIT"])
    if track_file is not None:
        ext = "fit" if track_file.name.lower().endswith(".fit") else "gpx"
        gc1, gc2 = st.columns(2)
        anchor = gc1.selectbox("計測区間の切り出し",
                               ["auto", "track_ends", "finish_distance", "elevation"])
        timed_km = gc2.number_input("計測距離 [km]", 10.0, 30.0, 24.0, 0.1)
    else:
        ext = anchor = None
        timed_km = 24.0

course = (
    _course_from_track(track_file.getvalue(), ext, anchor, timed_km, 1035.0)
    if track_file is not None else _course_from_csv()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("距離", f"{course.distance_m/1000:.1f} km")
c2.metric("獲得標高", f"{course.ascent_m:.0f} m")
c3.metric("平均勾配", f"{course.avg_grade*100:.1f} %")
c4.metric("フィニッシュ標高", f"{course.elev_finish:.0f} m")


# --------------------------------------------------------------------------------------
# 走行データ(任意)
# --------------------------------------------------------------------------------------
st.subheader("走行データで補正(任意)")
st.caption(
    "1 本からで OK。パワーデータを実測に置き換え、登坂を含む走行なら CdA・Crr も"
    "標準値から補正します。本数が増えるほど補正が強くなります。"
)
col_u, col_d = st.columns([3, 1])
ride_files = col_u.file_uploader(
    "FIT / GPX(複数可)", type=["fit", "FIT", "gpx", "GPX"], accept_multiple_files=True,
)
use_demo = col_d.checkbox("デモデータ", value=False, help="合成した登坂データ 10 本で試す")


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


def _collect_rides() -> dict[str, pd.DataFrame]:
    if use_demo:
        return _demo_rides()
    out: dict[str, pd.DataFrame] = {}
    for f in ride_files or []:
        kind = "gpx" if f.name.lower().endswith(".gpx") else "fit"
        try:
            out[f.name] = load_ride(f.getvalue(), kind=kind)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{f.name} を読めませんでした: {e}")
    return out


# --------------------------------------------------------------------------------------
# 予測
# --------------------------------------------------------------------------------------
if run:
    try:
        rides = _collect_rides()
        with st.spinner("予測を計算中..."):
            alt_model = make_model(model_name, **alt_kwargs)
            ip = predict_fuji_from_inputs(
                course, mass_total=mass_total, ftp=ftp,
                target_intensity=1.0 + pace_pct / 100.0,
                cda_prior=cda_prior, crr_prior=crr_prior,
                altitude_model=alt_model, race_temp_c=race_temp,
                rides=rides or None,
            )
        pred, pm, params = ip.result, ip.power_model, ip.params

        badge = {"簡易": "🟡 簡易", "パワー補正": "🟢 パワー補正", "登坂補正": "🟢 登坂補正"}[ip.tier]
        st.header("予測結果")
        st.caption(f"{badge} — {ip.tier_detail}")

        m1, m2, m3 = st.columns(3)
        m1.metric("予測タイム", fmt(pred.time_s), help="単独走行・一定ペース・無風の前提")
        m2.metric("目安レンジ", f"{fmt(pred.time_lo_s)} 〜 {fmt(pred.time_hi_s)}")
        m3.metric("標高補正の影響", f"+{fmt(pred.time_s - pred.time_no_altitude_s)}",
                  help=f"補正なしなら {fmt(pred.time_no_altitude_s)}")
        if not pred.converged:
            st.warning("反復計算が収束しませんでした。結果は参考値です。")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("富士ヒル想定パワー", f"{pred.p_sea_level:.0f} W",
                  help=f"{pred.p_sea_level / rider_kg:.2f} W/kg")
        b2.metric("標高補正後の出力", f"{pred.p_altitude:.0f} W")
        b3.metric("CdA", f"{params.cda:.3f} m²")
        b4.metric("Crr", f"{params.crr:.4f}")

        for n in ip.notes:
            st.caption("・ " + n)
        if not params.plausible:
            st.info("\n".join(f"- {w}" for w in params.warnings))

        st.subheader("区間別の予測")
        st.dataframe(pred.segment_table, use_container_width=True, hide_index=True)

        # ---- 走行データがあるときの詳細 ----
        if ip.climbs:
            with st.expander(f"走行データの内訳(検出された登坂 {len(ip.climbs)} 区間)"):
                st.dataframe(ip.climb_overview, use_container_width=True, hide_index=True)
                st.markdown("**登坂ごとの実測 vs モデル逆算タイム**")
                st.dataframe(params.per_climb, use_container_width=True, hide_index=True)
                st.markdown("**パワーカーブ**")
                durs = sorted(ip.mmp)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[d / 60 for d in durs],
                                         y=[ip.mmp[d] for d in durs],
                                         mode="markers", name="実測 mean-max"))
                tt = list(range(30, 5400, 30))
                fig.add_trace(go.Scatter(x=[t / 60 for t in tt],
                                         y=[pm.power(t) for t in tt],
                                         mode="lines", name="モデル"))
                fig.update_layout(xaxis_title="継続時間 [min]", yaxis_title="パワー [W]",
                                  height=320, margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"CdA/Crr は標準値(CdA {cda_prior:.3f} / Crr {crr_prior:.4f})を"
                    f"実走データで {params.data_weight*100:.0f}% 補正。"
                )

        st.caption(f"標高補正: {alt_model.describe()}  /  η = {DEFAULT_ETA:.3f}(固定)")

    except Exception as e:  # noqa: BLE001
        st.error(f"エラー: {e}")
        with st.expander("トレースバック"):
            st.code(traceback.format_exc())
else:
    st.info("左のサイドバーで FTP と体重を入れて「予測する」を押してください。走行データは任意です。")


# --------------------------------------------------------------------------------------
# ローカルのリファレンス実走データ(data/rides/。デプロイ環境では非表示)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="実走データを読み込み中...")
def _all_rides() -> dict:
    return {p.name: load_ride_file(p) for p in list_rides()}


@st.cache_data(show_spinner=False)
def _ride_summaries():
    ref = _course_from_csv()
    return [(name, df, summarize_ride(df, Path(name).stem, ref))
            for name, df in _all_rides().items()]


if list_rides():
    st.divider()
    if st.checkbox(f"ローカルの実走データを見る({len(list_rides())} 本)", value=False):
        rows = pd.DataFrame([rs.as_row() for _, _, rs in _ride_summaries()])
        st.dataframe(rows, use_container_width=True, hide_index=True)

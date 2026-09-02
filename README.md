# 富士ヒルクライム タイム予測アプリ

**FTP と体重を入れるだけ**で Mt.富士ヒルクライムの完走タイムを予測する Web アプリ。
走行データ(FIT / GPX)を足すほど、パワー・空力(CdA)・転がり抵抗(Crr)が実測で
補正され精度が上がる。物理パラメータの手入力は必須ではない
(仕様書 `fuji-hillclimb-predictor-spec.md`)。

## 使い方(段階的に精度が上がる)

| 入力 | 何が起きるか | 確度 |
|---|---|---|
| **FTP + 体重 + 機材重量** | FTP に所要時間の補正を掛けて目標パワーを推定。CdA/Crr は標準値。 | 🟡 簡易 |
| **+ FIT/GPX 1〜数本(平地でも可)** | 走行データの mean-max power で目標パワーを実測補強。 | 🟢 パワー補正 |
| **+ 登坂を含む走行データ** | CdA/Crr を標準値から実走で正則化補正(本数が増えるほど強く)。 | 🟢 登坂補正 |

スマホからでも FTP+体重だけで予測でき、FIT は 1 本ずつ足せばよい。

## 仕様書との主な差分

| 項目 | 仕様書 | 本実装 |
|---|---|---|
| 入力 | 登坂 FIT を約10本 | FTP+体重で予測、FIT は任意(1本から)。`climbpred/pipeline.py: predict_fuji_from_inputs` |
| 目標パワー | PD カーブ(要データ) | FTP×所要時間補正 → 実測 mean-max があれば CP+W'/t に移行(`climbpred/power_model.py`) |
| Crr / CdA 推定 | 通常の最小二乗 | 標準値を事前分布とした**有界リッジ最小二乗**。実走点数が事前強度を超えるほどデータ主導に(`data_weight`)。 |
| 標高補正 k | 既定 0.0010 適用 | **GUI で手入力**(推奨レンジ表示)。既定 0.0002。実走検証で 0.0010 は過大と判明。 |
| 永続化 | MySQL | なし(セッション内)。Streamlit Community Cloud にそのままデプロイ可 |
| Intervals.icu 同期 | Phase 2 | 未実装。`climbpred/pdcurve.py: pdcurve_from_intervals()` に接続点のみ |
| コースデータ | 公式マップ / GPX | 公式コース定義 FIT から CSV 生成。GPX / FIT 差し替えもアプリで可 |
| 走行データ形式 | FIT のみ | FIT + GPX(power/hr/cad 拡張)対応(`climbpred/fit_ingest.py: load_ride`) |
| 3次方程式 | scipy | `numpy.roots` + Newton フォールバック(`climbpred/physics.py`) |
| 区間別表示 | — | 物理計算はコース分解能(100 m)、区間別テーブルは 500 m ごとに集約(`predict_fuji(..., report_bin_m=500)`)。ラベルは `0.0-0.5 km`、時間は mm:ss。 |

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows / PowerShell
pip install -r requirements.txt
streamlit run app.py
```

FTP と体重だけで予測できる。「走行データで補正」の「デモデータ」で合成登坂 10 本を使った
補正も試せる。

## デプロイ(Streamlit Community Cloud)

1. このディレクトリを GitHub リポジトリにする
2. Streamlit Community Cloud で `app.py` を指定
3. `requirements.txt` が自動で解決される。追加の secrets は不要

## モジュール構成(仕様書 5 層アーキテクチャに対応)

| ファイル | 層 |
|---|---|
| `climbpred/fit_ingest.py` | [1] データ取り込み(FIT / GPX パース、デモ合成) |
| `climbpred/climb_detect.py` | [2] クライム自動抽出 |
| `climbpred/pdcurve.py` | [3] PD カーブ(登坂からの mean-max / CP・W′ フィット) |
| `climbpred/power_model.py` | [3] 持続可能パワー(FTP → 実測 mean-max → CP+W'/t の段階モデル) |
| `climbpred/physics.py` + `calibration.py` | [4] 物理モデル & 有界リッジ最小二乗(標準値を事前分布に) |
| `climbpred/altitude.py` | [9] 標高別パワー減衰(プラグイン) |
| `climbpred/course.py` + `course_build.py` | [10] 富士ヒルコースデータ(CSV / 公式コース FIT / 実走 GPX) |
| `climbpred/rides.py` | 実走データの要約・計測区間タイム抽出 |
| `climbpred/predict.py` | [5] 予測エンジン(反復計算) |
| `climbpred/pipeline.py` | `predict_fuji_from_inputs`(メイン)/ `analyze_rides`(診断) |
| `climbpred/validation.py` | 多数の走行データでの精度確認(検証。校正ではない) |
| `app.py` | Streamlit UI |

## コースデータ

`data/fuji_course_profile.csv` は **公式コース定義 FIT**(`data/fuji_hillclimb_course.fit`、
ネット配布の Garmin Connect コース)から生成した 24.0 km の計測区間プロファイル
(距離 24.03 km / 標高差 1,236 m / 平均 5.14 %)。CSV ヘッダに計測スタート/ゴールの
緯度経度を保持する。

再生成:

```bash
python scripts/build_course.py data/fuji_hillclimb_course.fit
# セグメント情報の無い実走 GPX から作る場合(精度は落ちる):
python scripts/build_course.py <実走の富士ヒル>.gpx --anchor finish_distance
```

アプリ上でもコース(.fit / .gpx)を差し替え可能。詳細と前提は `docs/course_profile.md`。

## ローカルの実走データ(`data/rides/` — リポジトリには含めない)

`data/rides/` に FIT / GPX を置くと、`climbpred/rides.py` がパワー・心拍・
ケイデンス・気温の要約と、公式コース座標でクリップした**計測区間タイム**を算出し、
アプリ下部の「ローカルの実走データを見る」に表示する(ファイルが無ければ非表示)。

**このフォルダは走行位置・日時・心拍・パワーを含む個人データなので `.gitignore`
済み**(公開リポジトリに入れない)。各自ローカルに配置して使う。

## 予測モデルの精度確認(検証)

```bash
python scripts/accuracy_check.py --mass <総質量kg>
```

`data/rides/` に置いた練習ライドから CdA/Crr/PD カーブを求め、富士ヒルのタイムを
標高補正モデル別に予測する。富士ヒルのコース座標付近を通る記録が置いてあれば、
その計測区間タイムを実測値として並べる。

**これは校正ではない。** 練習データの時期がレースと前後している場合、現行モデルの
当たり具合を見るだけに留める。

## 設計上の判断(検証で得られた知見)

### 標高減衰 k は GUI 手入力(推奨レンジのみ表示)

標高別の有酸素パワー低下は文献値のばらつきと個人差が大きく、低地の練習データからは
自己較正できない。実走データでの検証でも、文献でよく使われる k=0.0010(10%/1000m)は
富士ヒルの標高帯(〜2300 m)では完走タイムを大幅に過大予測し、k≈0 や Bassett 多項式
モデルの方が実測に近かった。そのため k は決め打ちにせず GUI で手入力とし、既定は
控えめな 0.0002、推奨レンジ(0〜0.3 %/1000m)だけを表示する。物理的に妥当な曲線が
欲しい場合は `bassett_poly` を選ぶ。

### Crr / CdA は「標準値 + 実走で正則化」

走行データが無ければ標準値(CdA 0.32 / Crr 0.005)をそのまま使う。実走の登坂が
あれば、標準値を事前分布とした**有界リッジ最小二乗**で推定する:擬似観測で
標準値へ引き戻しつつ、実走点数 N が事前強度 S(既定 4000)を超えるほど推定は
データ主導になる(`CalibrationResult.data_weight = N/(N+S)`)。物理レンジ
Crr∈[0.0015, 0.02]・CdA∈[0.15, 0.6] は安全網として残す。

登坂中心のデータでは重力項と転がり項の速度依存が似通うため(仕様書 8 節の多重共線性)、
CdA と Crr を綺麗に分離できないことが多い。事前分布への正則化はこの不定性を
標準値側で吸収する。なお登坂予測は重力項が支配的なため、CdA/Crr の分離の
不確かさに対して完走タイムはもともと鈍感。

### 目標パワー(FTP → レース所要時間)

FTP は概ね 1 時間パワーなので、80〜95 分かかる富士ヒルではそれより低い値で走る。
`ftp_duration_factor(t)` が t=3600s で 1.0、それ以降 1 時間ごとに約 11% 低下
(下限 0.78)。実測 mean-max power が 2 点以上あれば FTP 前提を捨てて
CP + W'/t をフィット、20 分以上の実測点が無いときだけ FTP を 3600s の点として補う。
GUI の「ペーシング調整 [%]」で ±12% の微調整ができる。

### 未対応(校正方針で詰める)

- **クライム自動抽出が停止・待機を含む区間を 1 本の登坂として結合**することがある
  (同じ坂でも所要時間が大きくばらつく)。低出力ギャップでの分割・除外ロジックが要る。
- FTP → 所要時間補正の係数(`FTP_FADE_PER_HOUR`)は文献ベースの暫定値。

## テスト

```bash
pip install pytest
pytest
```

## 前提・非対象(仕様書 2 節)

- 単独走行・一定出力を前提(ドラフティング、ペーシング戦略は非対象)
- 初期実装は無風・標準大気(気温はオプション入力で補正可)

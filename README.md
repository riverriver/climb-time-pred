# 富士ヒルクライム タイム予測アプリ

過去の登坂データ(FIT)を数本入力するだけで、Mt.富士ヒルクライムの完走タイムを
標高補正込みで予測する Web アプリ。**CdA・Crr・駆動効率をユーザーに手入力させず、
実走データから逆算する** のが設計方針(仕様書 `fuji-hillclimb-predictor-spec.md`)。

## 仕様書との差分(Web アプリ化に伴う変更)

| 項目 | 仕様書 | 本実装 |
|---|---|---|
| 永続化 | MySQL | なし(セッション内で完結)。Streamlit Community Cloud にそのままデプロイ可能。将来 DB を足す場合は `climbpred/pipeline.py` の入出力を差し替える。 |
| Intervals.icu 同期 | Phase 2 | 未実装(UI にプレースホルダのみ)。`climbpred/pdcurve.py: pdcurve_from_intervals()` に接続点を用意済み。 |
| 3次方程式の数値解 | scipy | `numpy.roots` + Newton 法フォールバック(`climbpred/physics.py`) |
| コースデータ | 公式マップ / GPX を `docs/course_profile.md` に固定 | 公式コース定義 FIT から `data/fuji_course_profile.csv` を生成。GPX / FIT 差し替えもアプリで可能。 |
| 走行データ入力 | FIT のみ | FIT に加え GPX(power/hr/cad 拡張)にも対応(`climbpred/fit_ingest.py: load_ride`) |
| Crr / CdA 推定 | 通常の最小二乗 | **有界最小二乗**(`scipy.optimize.lsq_linear`)。Crr が負に出る問題への対処として Crr∈[0.0015, 0.02]・CdA∈[0.15, 0.6] に制約。下限に張り付いたら警告し、生値も併記。 |
| 標高補正 k | 既定値 0.0010 を適用 | **GUI で手入力**(推奨レンジのみ表示)。既定は控えめな 0.0002。実走検証で 0.0010 は過大と判明。 |

Phase 1(MVP)の範囲:FIT アップロード → クライム自動抽出 → 2 パラメータ線形
キャリブレーション → 自前 PD カーブ → 静的コースデータ → 標高補正込み予測。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows / PowerShell
pip install -r requirements.txt
streamlit run app.py
```

デモデータモードで FIT なしに動作確認できる。

## デプロイ(Streamlit Community Cloud)

1. このディレクトリを GitHub リポジトリにする
2. Streamlit Community Cloud で `app.py` を指定
3. `requirements.txt` が自動で解決される。追加の secrets は不要

## モジュール構成(仕様書 5 層アーキテクチャに対応)

| ファイル | 層 |
|---|---|
| `climbpred/fit_ingest.py` | [1] データ取り込み(FIT / GPX パース、デモ合成) |
| `climbpred/climb_detect.py` | [2] クライム自動抽出 |
| `climbpred/pdcurve.py` | [3] PD カーブ(CP / W′) |
| `climbpred/physics.py` + `calibration.py` | [4] 物理モデル & キャリブレーション |
| `climbpred/altitude.py` | [9] 標高別パワー減衰(プラグイン) |
| `climbpred/course.py` + `course_build.py` | [10] 富士ヒルコースデータ(CSV / 公式コース FIT / 実走 GPX) |
| `climbpred/rides.py` | リファレンス実走データの保管・要約・計測区間タイム抽出 |
| `climbpred/predict.py` | [5] 予測エンジン(反復計算) |
| `climbpred/pipeline.py` | 上記の統合 |
| `climbpred/validation.py` | 現行モデルの精度確認(検証。校正ではない) |
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

## リファレンス実走データ(`data/rides/` — リポジトリには含めない)

`data/rides/` に FIT / GPX を置くと、キャリブレーション用の登坂抽出とは別に
**レース分析用の記録**として扱われる。`climbpred/rides.py` がパワー・心拍・
ケイデンス・気温の要約と、公式コース座標でクリップした**計測区間タイム**を算出し、
アプリの「自分の実走データ」セクションに表示する。ファイルが無ければこの
セクションと「予測の精度チェック」は非表示になるだけ(コア予測機能には影響なし)。

**このフォルダは走行位置・日時・心拍・パワーを含む個人データなので `.gitignore`
済み**(公開リポジトリに入れない)。各自ローカルに配置して使う。

## 予測モデルの精度確認(検証)

```bash
python scripts/accuracy_check.py --mass <総質量kg>
```

`data/rides/` に置いた練習ライドから CdA/Crr/PD カーブを求め、富士ヒルのタイムを
標高補正モデル別に予測する。富士ヒルのコース座標付近を通る記録が置いてあれば、
その計測区間タイムを実測値として並べる。アプリでも「予測の精度チェック」で
同じ表を確認できる。

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

### Crr / CdA は有界最小二乗(`scipy.optimize.lsq_linear`)

登坂中心のデータでは重力項と転がり項の速度依存が似通うため(仕様書 8 節の多重共線性)、
素の最小二乗だと Crr が負に出ることがある。物理レンジ Crr∈[0.0015, 0.02]・
CdA∈[0.15, 0.6] で有界化し、下限に張り付いた場合は `crr_at_bound` フラグと警告を出す
(生値は `crr_unconstrained` に保持)。Crr を後から 0 付近へ置換するだけの手当てより、
CdA を再最適化する有界解の方が物理的に妥当で、残差もほぼ変わらない。なお登坂予測は
重力項が支配的なため、CdA/Crr の分離の不確かさに対して完走タイムは鈍感。

### 未対応(校正方針で詰める)

- **クライム自動抽出が停止・待機を含む区間を 1 本の登坂として結合**することがある
  (同じ坂でも所要時間が大きくばらつく)。低出力ギャップでの分割・除外ロジックが要る。
- PD カーブの長時間側(60 分〜)のデータが集まると CP 推定が安定する。

## テスト

```bash
pip install pytest
pytest
```

## 前提・非対象(仕様書 2 節)

- 単独走行・一定出力を前提(ドラフティング、ペーシング戦略は非対象)
- 初期実装は無風・標準大気(気温はオプション入力で補正可)

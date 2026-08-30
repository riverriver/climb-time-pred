# 富士ヒルクライム コースプロファイル

予測エンジンが読み込むのは `data/fuji_course_profile.csv`。
本ドキュメントは出典・生成方法・前提を記録する。

## 現在の値(公式コース FIT 由来)

| 項目 | 値 | 公式値(参考) |
|---|---|---|
| 計測距離 | 24.03 km | 24.0 km |
| スタート標高 | 約 1,055 m | 約 1,035 m |
| フィニッシュ標高 | 約 2,291 m | 約 2,305 m |
| 標高差(net) | 約 1,236 m | 約 1,270 m |
| 平均勾配 | 約 5.14 % | 約 5.2 % |
| スタート座標 | 35.451214, 138.758256 | 富士スバルライン料金所付近 |
| ゴール座標 | 35.394138, 138.731952 | 富士山五合目 |

区間別勾配はおおむね 4〜7 %。21〜23 km(標高 2,240 m 前後)に勾配 0〜1 % の
平坦区間があり、そこからゴールまで再び 5 % 前後で登る。

## 出典

`data/fuji_hillclimb_course.fit` — ネット配布の **Garmin Connect コース定義 FIT**
(`file_id.type == "course"`、628 record、`lap.total_distance = 24,026 m`)。
計測スタート/ゴールがトラックの端そのものなので、切り出し不要で最も正確。

`scripts/build_course.py` が `anchor="track_ends"`(course FIT の既定)で
そのまま 100 m 間隔・軽い平滑化を掛けて CSV 化する:

```
python scripts/build_course.py data/fuji_hillclimb_course.fit
```

CSV ヘッダ(`#` コメント)に `start_latlon` / `finish_latlon` を書き込み、
`load_course()` がそれを読み取って `CourseProfile.start_latlon/finish_latlon` に
反映する。実走データの「計測区間タイム」算出(`climbpred/rides.py`)に使う。

## GPX からの生成(代替)

セグメント情報の無い実走 GPX からも生成できる(精度は FIT に劣る):

- `anchor="finish_distance"`(GPX 既定): 最高標高地点をゴールとし、
  その 24.0 km 手前をスタートとみなす。
- `anchor="elevation"`: 指定標高で切り出す。
- `anchor="manual"`: スタート/ゴールの緯度経度を直接指定。

参考: 開発時に実走 GPX を `finish_distance` で切った結果は 24.00 km / 平均 5.18 %
で、公式コース FIT(24.03 km / 5.14 %)とほぼ一致した。

## さらに精度を上げるなら

- 公式コースマップの km ポスト別標高表(あれば FIT も不要で置換)
- 大会リザルトの区間ラップ(スタート〜1合目〜…〜五合目)

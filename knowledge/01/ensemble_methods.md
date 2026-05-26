# Cycle 01 で採用する Ensemble 手法 (TTA + mean+max blending)

> 出典: [days 7th place writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/days-7th-place-solution) の "4. ensemble" セクション
> 関連: `baseline_proposal.md`

## 1. Test-Time Augmentation (TTA) — 検索ランキング・スライス TTA

### アイデア
同じ question に対して、retrieval の **上位 0 位 (= 最も類似)** を含めつつ、その他のスライスを変えた 4 通りの context を作って推論し、結果をまとめる。

### スライス定義 (記事より)
```
TTA1: [0, 1, 2, 3, 4, 5]       ← top 6 を素直に
TTA2: [0, 6, 7, 8, 9, 10]      ← 0位 + rank 6-10 (中位)
TTA3: [0, 11, 12, 13, 14, 15]  ← 0位 + rank 11-15
TTA4: [0, 16, 17, 18, 19, 20]  ← 0位 + rank 16-20 (下位)
```
ここの数字は **retrieval の rank** (0 が最類似)。

### なぜ効くのか
- top-K だけだと「retrieval が拾えなかった重要情報」をモデルに渡せない
- 下位スライスを混ぜることで **retriever のノイズ・取りこぼし** を補完
- 1 つの context だけだと過適合しがちな予測に **分散** をもたらす
- 推論回数は 4 倍に増えるが、Kaggle T4×2 でも納まる範囲

### 実装イメージ
```python
TTA_SLICES = [
    [0, 1, 2, 3, 4, 5],
    [0, 6, 7, 8, 9, 10],
    [0, 11, 12, 13, 14, 15],
    [0, 16, 17, 18, 19, 20],
]

predictions = []
for tta_id, ranks in enumerate(TTA_SLICES):
    context_chunks = [retrieved[r] for r in ranks]
    context_str = " ".join(context_chunks)
    pred_df = model.predict(question, context_str)  # df: (id, A, B, C, D, E)
    pred_df["tta_id"] = tta_id
    predictions.append(pred_df)

all_preds = pd.concat(predictions, ignore_index=True)
```

## 2. Multi-Model Ensemble — 3 モデル × 4 (model, retrieval, dataset) 組合せ

3 種類の DeBERTa v3 large 系モデルを基に、retrieval (gte-base / e5-base) と訓練データ (all / without 3,7,8,9 / without 10) を組合せて **4 つの instance** を作り、それぞれを TTA 推論。

| Combo | Base model | Retrieval | Train data subset |
|---|---|---|---|
| C1 | microsoft/deberta-v3-large | gte-base | all |
| C2 | OpenAssistant/reward-deberta-v3-large-v2 | e5-base | without 3,7,8,9 |
| C3 | deepset/deberta-v3-large-squad2 | gte-base | without 10 |
| C4 | ... | ... | ... |

(具体的 4 通りは記事に明示なし、組合せから best 4 を選定したと推察)

## 3. Aggregation — mean + max blending (本手法のキー)

### 数式
全予測 (`n_test × n_tta × n_models` 行) を id でグループ化:

$$
\text{score}_q(\text{label}) = \underbrace{\text{mean}_q(\text{label})}_{\text{安定性}} + \underbrace{\text{max}_q(\text{label})}_{\text{強い予測を伸ばす}}
$$

### 実装
```python
# df: index は無し、列に id, A, B, C, D, E (各選択肢の確率)
df_agg = df.groupby("id").mean() + df.groupby("id").max()
top3 = df_agg[["A", "B", "C", "D", "E"]].apply(
    lambda row: " ".join(row.nlargest(3).index), axis=1
)
submission = pd.DataFrame({"id": df_agg.index, "prediction": top3})
```

### なぜ "mean + max" が単純平均より良いか
| 集計 | 性質 |
|---|---|
| `mean` のみ | 1 つのモデルが強く支持しても他が中庸だと薄まる |
| `max` のみ | 1 つのモデルが偶然高スコアを出すと突出してノイズに弱い |
| `mean + max` | 「**多数決の平均** + **強い 1 票のボーナス**」両方が反映される |

確率の和なので、softmax で正規化される設計のモデル出力に対しては実質的に **重み付き平均 + α** と等価。順位付け (top-3 選択) には正規化は不要なので、mean+max を直接 sort して上位 3 つ取り出せば OK。

### 注意点
- 各モデルの出力スケールが揃っている前提（softmax 後の確率なら OK）
- もし logits を混ぜるなら sigmoid/softmax してから集計すべき
- top-3 の **順序** はそのまま MAP@3 に影響するので、aggregated score の降順で 3 つ取る

## 4. Cycle 01 簡略版での適用方針

リソース制約から、Cycle 01 完了基準としては:

| 要素 | フル (記事) | Cycle 01 簡略版 |
|---|---|---|
| Models | 3 base × 4 combo = 4 ensemble | **1 model** (microsoft/deberta-v3-large) のみ |
| Retrieval | gte-base + e5-base | **e5-base** のみ |
| TTA | 4 slice | **2 slice** (TTA1, TTA2) |
| Aggregation | mean + max | **mean + max** (これは記事通り) |

つまり最初は **1 model × 2 TTA = 2 予測** を mean+max で集計するだけ。これだけでも記事の核心 (TTA + mean+max) を再現できる。

Cycle 02 以降で:
- Retrieval を gte-base 追加 → 2 retrieval × 2 TTA = 4 予測
- Base model を 3 種類に拡張 → 6 予測
- TTA を 4 slice に拡張 → 12 予測

と段階的に増やす。

## 関連

- ベースライン全体: `baseline_proposal.md`
- 代替 (もし重すぎたら): `alternative_tfidf_baseline.md`
- Output フォーマット規約: `../09_output_format.md`
- 全体トレンド: `../01_overview_trends.md`

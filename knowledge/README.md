# knowledge/ — 知識ベース

Kaggler 手法 / Cycle 専用調査 / 人間調査メモを集約するディレクトリ。**3 つの階層**で使い分ける。

## 階層

```
knowledge/
  NN_<title>.md          ← Cycle 横断で参照する一般知識（番号 prefix で並び制御）
  NN/                    ← Cycle NN 専用の調査・提案文書
    baseline_proposal.md
    ensemble_methods.md
    alternative_<name>.md   ← 採用しなかったが将来戻る可能性のある案
  search/                ← 人間が手動で調べた調査結果
    NN_<title>.md        ← 連番 prefix で並び制御
```

## 使い分け

| 用途 | 置き場 | 例 |
|---|---|---|
| **Cycle NN に強く紐づく**（採用案・別案・ensemble 詳細・hyperparam tuning 結果など） | `knowledge/NN/<title>.md` | `knowledge/01/baseline_proposal.md` |
| **Cycle を跨いで参照する** 一般知識（kaggler 手法サマリ、auth/submission 手順、output_format 等） | `knowledge/<NN>_<title>.md` (直下、番号 prefix で並びを制御) | `knowledge/09_output_format.md` |
| **採用しなかったが将来戻る可能性**がある案 | `knowledge/NN/alternative_<name>.md` | `knowledge/01/alternative_tfidf_baseline.md` |
| **人間が手動で調べた調査結果**（Claude 出力ではなくユーザー収集メモ、Web 記事の写し、外部資料の要約など） | `knowledge/search/NN_<title>.md` (`00_title.md` 形式、連番で並び制御) | `knowledge/search/00_ensemble_pipeline_origin_validity.md` |

## `knowledge/search/` Q&A 運用

`knowledge/search/<file>.md` の **内容について追加質問** を受けた場合の運用:

1. **新規ファイルを作らない** — 既存の質問対象ファイル末尾に追記する
2. ファイル末尾に `## Q&A 履歴` セクションがあればそこに、なければ新設してから追記する
3. 追記フォーマット:
   ```
   ### Q<連番> (YYYY-MM-DD): <質問の要約タイトル>

   **発問**:
   > <ユーザーの発問をそのまま引用>

   **回答**:
   <本文>
   ```
4. 質問対象ファイルが不明確な場合は、どのファイルへの追加質問かをユーザーに確認してから追記する

---

## レポート / 振り返り文書のスコア記載・考察規約（必須）

`report/NN_score_report.md`、`knowledge/NN/cycle*_retrospective*.md`、その他 **Cycle のスコアを扱う全文書** は、以下の 5 要件を **必ず満たすこと**。`val` と `LB` の数字を並べて「ギャップは 0.08」とだけ書く類の浅い書き方は禁止。

### 用語定義

| 用語 | 定義 |
|---|---|
| **ベースラインスコア** | **Kaggle テストデータに似たデータ** に対するスコア。具体的には公式 train.csv を hold-out / KFold 分割した val 集合、または同分布のサンプル (公式 train から派生した GPT 生成 MCQ 等) に対する MAP@3。**自分でラベルが見える領域で測れる信号**。 |
| **Kaggle スコア** | Kaggle に submit して得た Public LB / Private LB スコア。**自分でラベルが見えない領域** (公式 test 200 行 + 隠し test) に対する MAP@3。 |

### 必須要件 5 つ

#### R1. ベースラインスコアを必ず記載する

- val MAP@3 (または同等の「Kaggle テストに似たデータでの MAP@3」) を **数値で明記**
- val 集合のサイズ / 由来 / split 方法を 1 行で記載 (例: 「公式 train.csv 200 行を 80/20 split, val=40」)
- **「同分布の信号がそもそも何か」を明確にする**

#### R2. Kaggle スコアの特徴を記載する

単に数値を載せるだけでなく、**Kaggle 採点データの性質**を明示する:

- 採点対象の母集団 (例: Public LB は test 200 行のうち 公開 ~50%、Private LB は残り + 隠し test)
- 取り得る値域、サンプル数、評価指標の挙動 (MAP@3 のスケール、本コンペでの分散の目安)
- **Public と Private の差が何を意味するか** (overfit シグナル, 分布シフト等)

#### R3. スコアが上下するデータ特性を列挙する

「どんなデータに対してスコアが上がるか / 下がるか」を **具体的なデータ特徴で**書く。最低 3 項目以上、推測ではなく実際の誤答ログ / val の誤りパターン / 正解選択肢の特徴等から抽出すること。

例 (Cycle 01 m2 の場合に書くべき内容):
- ✅ 上がる: 短い prompt (≤100 chars) + 1 語の選択肢 (well-known physics terms 等)
- ⚠️ 下がる: 数値・年代・固有名詞を問う設問 (parametric memory に無い long-tail)
- ⚠️ 下がる: 選択肢間の編集距離が小さい設問 (e.g. 同じ用語の異なる定義)
- ⚠️ 下がる: 数式 / 公式の記号が含まれる設問 (tokenizer 段で情報損失)

→ "なんとなく難しい設問で落ちる" は不可。**データの観測可能な特徴に基づくこと**。

#### R4. Kaggle スコアとベースラインスコアを比較し、データ単位で考察する

R1 と R2 の数値を並べた上で、**両者がなぜ違うのか**を**データレベルで**論じる。観点:

- val の分布と Kaggle test の分布がどう違うか (サイズ、難易度、トピック偏り)
- val で当てられて Kaggle で落ちる設問群の共通点は何か (R3 の分類に紐付ける)
- val で落ちて Kaggle で当たる設問群の共通点 (運要素 / val が小さい場合のノイズ)
- **「val が test より甘い / 辛い」と判定し、その方向を説明する**

#### R5. Kaggle スコアとベースラインスコアの **差 (Δ)** を明示する

- `Δ = Kaggle - baseline` を **数値で明示**。符号必須 (大抵負だが、たまに正)
- Public Δ / Private Δ を別々に出す (両方ある場合)
- Δ の値域に対する解釈 (e.g. 「Δ = -0.11 は val が test より +0.11 楽観的」)
- **次 Cycle で Δ を詰めるための具体的な仮説**を 1 つ以上書く (なぜ Δ が出ているかの仮説 → 検証可能な実験案)

### 最小テンプレ (R1-R5 をこの順で記述)

```markdown
## スコア考察 (R1-R5)

### R1. ベースラインスコア
- val MAP@3: 0.7958 (公式 train.csv 200 行を 80/20 split、val=40 行)

### R2. Kaggle スコアの特徴
- 採点母集団: test 200 行 (Public ≈ 公開 50%, Private ≈ 残り + 隠し test)
- Public LB: 0.682480
- Private LB: 0.714337
- Public < Private → 公開 50% の方が分布として辛い (例外的)

### R3. スコアが上下するデータ特性
- 上がる: ...
- 下がる: ...
- 下がる: ...

### R4. データレベル比較考察
- val 集合と Kaggle test の分布差: ...
- val で当てられて Kaggle で落ちる設問群: ...
- val で落ちて Kaggle で当たる設問群: ...

### R5. スコア差 (Δ)
- Δ_Public = 0.682480 − 0.7958 = **−0.1133**
- Δ_Private = 0.714337 − 0.7958 = **−0.0815**
- 解釈: val は test より **+0.08–0.11** 楽観的
- Δ を詰める仮説: ...
- 検証実験案: ...
```

### NG 例 (これらは不可)

- ❌ 「val/LB ギャップ ≈ 0.08-0.10」とだけ書いて R3-R5 が無い
- ❌ 「test の方が難しい設問が多い」と書くだけで、**どんな特徴の設問が難しいか**を示さない
- ❌ Public/Private の片方しか書かない (両方あれば両方出す)
- ❌ Δ を計算式だけで済ませて、**解釈・仮説・検証実験案**が無い

### 既存文書への遡及適用

R1-R5 規約は **Cycle 01 結果と考察 (`knowledge/02/cycle01_2_results_and_analysis.md`) を含む既存の浅い考察文書** に遡及適用する（この文書は適用済み）。次の Cycle 開始前に更新済みかをチェック。

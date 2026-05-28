# 00 アンサンブルの根拠と実践 — Kaggle LLM Science Exam における設計判断

> Scope: ユーザー発問駆動の調査メモ (`knowledge/search/`)。Cycle 横断の参考資料。
> 具体的な実装は [`knowledge/01/ensemble_methods.md`](../01/ensemble_methods.md) を参照。
> 目的: 「なぜアンサンブルか」「学術的に問題ないか」「上位チームはどう設計したか」を一望できるようにする。
>
> 元の発問:「Science Exam LLM でモデルをアンサンブルにしている人たちは、どのようにアンサンブルのパイプラインやモデルの選定をしたか？ そもそもアンサンブルにする発想はどこから得たか？ アンサンブルはアカデミック的に問題ないか？」
> 読み手の理解負荷を下げるため、**起源 → 妥当性 → 実践** の順で並べ替えている。

---

## Q1. アンサンブルにする発想はどこから来たのか

### 1.1 機械学習側の系譜

アンサンブルは Kaggle 文化の産物ではなく、**1990 年代から続く古典的アイデア**である。

| 年 | できごと | 何を提示したか |
|---|---|---|
| 1996 | Breiman, *Bagging Predictors* | ブートストラップ平均で分散を減らす |
| 1997 | Freund & Schapire, *AdaBoost* | 弱学習器を逐次重み付けで強学習器化 |
| **2000** | **Dietterich, *Ensemble Methods in Machine Learning*** | **アンサンブルが効く 3 つの理由を整理** (後述) |
| 2001 | Breiman, *Random Forests* | 木のアンサンブルが事実上の標準分類器に |
| 2009 | Netflix Prize 優勝 ("BellKor's Pragmatic Chaos") | **線形 blending された数百モデルの集合が金字塔的な勝者**となり、Kaggle 文化の原型となる |
| 2010s〜 | Kaggle 全盛期 | "Stacking" / "Blending" が事実上のテンプレートに |
| 2023 | この LLM Science Exam | **LLM 時代でもアンサンブルが勝つ**ことを再確認 |

### 1.2 Dietterich (2000) の 3 理由 — なぜ効くか

直接の理論的根拠としていまでも引かれる枠組み。

1. **Statistical (統計的)**
   同じ訓練データに対して同じくらい良い仮説が複数ある時、1 つを選ぶより平均する方が「真の仮説から遠い hypothesis を引いてしまう確率」を下げられる。
2. **Computational (計算的)**
   勾配降下などの局所探索は局所最適に落ちる。異なる初期値・データ順から学習した複数モデルを平均すれば、より良い大域近似が得られる。
3. **Representational (表現的)**
   真の関数が単一モデルクラスの仮説空間に入っていなくても、複数モデルの**重み付き和**として表現できる場合がある。

### 1.3 bias-variance 分解 — もう一つの言い方

期待誤差 = bias² + variance + noise。

- **平均化系 (bagging, blending)** は variance を下げる
- **boosting** は bias を下げる
- **多様なモデルの加算** は両方下げ得る

2023 年の arXiv 2301.03962 *A Unified Theory of Diversity in Ensemble Learning* は、これまで曖昧だった「diversity (モデル多様性)」が **bias-variance 分解の隠れた次元**として現れることを示した。つまり「多様性こそが効く理由」というのは経験則だけでなく、理論側でも整理が進んでいる。

### 1.4 Kaggle で広まった経路

Kaggle 文化に「とにかく ensemble」が定着したのは Netflix Prize 経由が大きい。

- 単一モデルの限界打破 → 異なる前処理 / 特徴 / アルゴリズムを足す
- Public LB と Private LB の差を縮めるための **保険** として有効
- 上位 1% の細かい差を取りに行く段階では、新しいモデル 1 個より「既存ベストの blending」の方が安価で確実

このため LLM Science Exam のような新型コンペでも、上位陣はほぼ全員アンサンブルを採用している（後述）。

---

## Q2. アンサンブルはアカデミック的に問題ないか

### 2.1 結論：問題ない（むしろ標準手法）

アンサンブルは **査読付き ML 論文 / 標準教科書 / 産業システムすべてで認められた手法**。Dietterich, Breiman, Schapire, Zhi-Hua Zhou(『Ensemble Methods』) など、ensemble だけで複数の名著がある。

NeurIPS / ICML / ACL 等のトップ会議でも、ensemble baseline は常に出てくるし、SOTA 主張の多くは「アンサンブル前提」で書かれている (例: GLUE / SuperGLUE 上位、ImageNet 上位、医療診断モデル等)。

### 2.2 ただし「論文では弱く扱われる」場面はある

完全にニュートラルというわけではなく、次のような **論調の差**がある。

| 場面 | 扱われ方 |
|---|---|
| Kaggle / 産業 | **歓迎**。精度が最優先。 |
| 新規アーキテクチャ提案論文 | **嫌われる場合あり**。「単一モデルの能力か、ensemble の力か区別不能」になるため、本文では single-model 結果で勝負させ、ensemble は付録扱い。 |
| 効率重視 (TinyML, on-device) | **慎重**。推論コストが N 倍になる。 |
| 因果推論・統計的検定 | **要注意**。何個も走らせて best を取ると multiple testing 問題が出る → CV 設計が必要。 |

つまり「**アンサンブル自体が悪い**」ではなく、「**アンサンブルで上がった分を新規手法のおかげと誤解する**」のがダメ、ということ。

### 2.3 倫理・再現性の論点

- **再現性**: ensemble は構成要素が多いほど再現しにくい。論文では各 base model と blend 重みを公開すべき。
- **計算コスト**: モデル数 × 推論コスト。環境負荷の観点でも近年議論あり。
- **データリークの risk**: スタッキングは out-of-fold で組まないと簡単にリークする (Kaggle で頻発の失敗パターン)。

### 2.4 LLM 時代特有の論点

- LLM 出力の確率を ensemble するのは MCQ など離散ラベルでは健全。
- 一方、自由生成テキストの ensemble は意味的に困難（"voting" は naive すぎる）。LLM-as-judge や Self-Consistency 等の派生形が研究中。
- 2024-2025 年の arXiv にも `Dipper: Diversity in Prompts for Producing LLM Ensembles` や `Harnessing Consistency for Robust Test-Time LLM Ensemble` など、**LLM 時代のアンサンブル理論を再構築する論文**が増えている。

**まとめ**: 学術的にアンサンブルそのものは完全に正統。ただし「何の勝因か」を分離できる検証設計と、再現可能性の担保が論文化する場合の必須条件。Kaggle コンペでは特段の問題なし。

---

## Q3. LLM Science Exam 上位チームのアンサンブル設計

### 3.1 全体像 — どの上位チームも 4 つの軸で多様性を作っている

writeup を横断すると、上位陣の多様性源は次の 4 軸に整理できる。

```
┌──────────────────────────────────────────────────────────┐
│ 多様性軸     例                                             │
├──────────────────────────────────────────────────────────┤
│ ① モデル本体  DeBERTa v3 large / Mistral 7B / Llama 2 70B  │
│ ② Retriever  BM25 (sparse) / e5 / gte / SFR-Embedding (dense) │
│ ③ コーパス    Wikipedia 2023-06 / 2023-07 / cirrussearch  │
│ ④ 訓練データ  radek 6.5k / MMLU / cdeotte 自家生成 / subset  │
└──────────────────────────────────────────────────────────┘
            ↓
   モデル毎の出力 (確率/ロジット)
            ↓
   集約 (mean / mean+max / XGBRanker / cascade)
            ↓
        top-3 を MAP@3 で提出
```

「1 軸だけ揺らす」ではなく **複数軸を同時に揺らす** のが共通点。これにより 各モデルが**異なる種類の誤りをする**ようになり、blending 効果が最大化される。

### 3.2 各上位チームの実装

#### 1st place — H2O LLM Studio

- **構成**: 7B モデル × 5 + 13B モデル × 1 = **6 モデル ensemble**
- **多様化の源**:
  - retriever: MTEB leaderboard 上位の **異なる embedding モデル**
  - corpus: **異なる Wikipedia dump** バージョン
  - retrieved chunk 数を変える
- **学習**: LoRA で全 linear 層を fine-tune。**binary classification 化** ((context, question, ONE answer) → 「これが正解の確率」)。
- **集約**: 6 モデル出力を ensemble (具体的 weighting は不詳、likely 平均/重み付け)
- **設計判断の根拠**: Kaggle の 10B 制約があるため、**1 個の超大モデルが置けない**。「複数の小モデルを多様化で束ねる」方が制約下で最強、という判断。

#### 2nd place — @solokin (個人参加)

- **構成**: **DeBERTa v3 + Mistral** の異種アーキ ensemble
- **多様化の源**: アーキテクチャの**質的な違い** (encoder MCQ 型 vs decoder LM 型)
- **retrieval**: 文単位に分割 → overlap chunk → **BM-25 (Apache Lucene)** → **DeBERTa v3 reranker** で再ランキング
- **集約**: **カスタム XGBRanker でブレンド** ← ここが他チームと違う特徴。
  - 各モデルの確率を特徴量として GBDT を再学習し、最終 ranking を出す
  - 単純平均ではなく**学習された重み**で結合 → stacking の一種
- **設計判断の根拠**: 単独で勝つには「異質モデル同士の組み合わせ」が必要、と判断したのが伺える。

#### 3rd place — @podpall

- **構成**: 多数の小型 LM + **70B モデル 2 つ**
- **特徴**: ensemble の規模を物量で押す方向。各モデルの specialty は不詳。

#### 4th place — Preferred Scantron

- **構成**: Llama 2 7B (zero-shot 検証用) + **fine-tuned DeBERTa v3 large** をメインに、複数 reranker を組合せ
- **多様化の源**: 複数の reranking 指標 — Elasticsearch score / **edit distance** / semantic search
- **設計判断の根拠**: 70B クラスを置かなくとも、retrieval 多様化 + DeBERTa fine-tune で十分競争力があると判断。実際 4 位を獲得。

#### 5th place — Preferred おしゃべりんぼう

- **構成**: Mistral 7B + Llama 2 70B (QLoRA)
- **特徴的設計**: **Cascade ensemble** (= 3-stage 推論)
  1. 簡単な問題 → Mistral 7B が答える
  2. 難問のみ → Llama 2 70B に回す
- **設計判断の根拠**: Kaggle の**推論 9 時間制限**下で、全問に 70B を回すと間に合わない。
  単純平均アンサンブルではなく、**「どのモデルに任せるか」を動的選択**するルーティングで時間と精度を両立。

#### 7th place — days (Cycle 01 で参考にしている解法)

- 詳細: `knowledge/01/ensemble_methods.md`
- **構成**: DeBERTa v3 large 3 種 × retrieval 2 種 × subset データ 3 種 から **4 instance** + **TTA 4 slice**
- **集約**: **mean + max** (単純な平均だけでなく「max ボーナス」も足すアグリ)
- 上記他チームと比べシンプル・低コスト寄り。だからこそ Cycle 01 ベースラインに採用。

### 3.3 共通する「モデル選定の論理」

上位 writeup から逆算すると、選定基準はだいたい次のように働いている。

1. **強い single model を最低 1 個確保する**
   ↓
2. **その強い model に対して "間違え方が違う" model を足す**
   - 異なる backbone (encoder MCQ vs decoder LM)
   - 異なる retriever (sparse BM25 vs dense embedding)
   - 異なる corpus (dump 違い、cirrussearch 等)
3. **同じ backbone でも train subset を変えて N copy 作る** (overfitting 軽減)
4. **集約は単純平均から始め、効くなら learned blender (XGBRanker, logistic stacking) へ**

「とにかく多くのモデル」ではなく、**「ベスト single + ベストと違う誤りをする model」** という選び方が共通している。これは Q1 の "diversity が bias-variance 分解の隠れた次元" の理論と一致する。

### 3.4 集約手法のスペクトル

| 集約手法 | 説明 | 採用チーム例 | 特徴 |
|---|---|---|---|
| **単純平均** | 確率を平均 | 多くの中位陣 | 実装が最簡。スケール一致が前提 |
| **mean + max** | 平均に max を足す | 7th (days) | 強い 1 票のボーナス。シンプルだが効く |
| **重み付き平均** | CV で重み最適化 (optuna 等) | 中位上位の常套 | 軽い stacking。over-fit に注意 |
| **XGBRanker / GBDT** | 各モデル確率を特徴量に GBDT 再学習 | **2nd solokin** | 高度な stacking。OOF 設計が必須 |
| **Cascade / Router** | 難易度で異なるモデルへ振り分け | **5th Preferred** | 推論コスト最適化と精度の両立 |

### 3.5 Cycle 横断の指針 (本プロジェクトへの示唆)

上記から本プロジェクトで採るべき方針:

1. **Cycle 01**: まず 1 model + TTA + mean+max (= 7th place の simplified)。集約手法とパイプラインの土台を作る。
2. **Cycle 02 以降**: 「軸を 1 つ追加するごとに CV を測る」。
   - +retriever (gte 追加) → CV 上がる？
   - +backbone (OpenAssistant reward / deepset squad2) → 上がる？
   - +corpus (cirrussearch) → 上がる？
3. **過剰アンサンブルの注意**: モデル数 × TTA × retriever が積で増える。Kaggle T4 ×2 の 9h 制限内に収まるよう、**追加 1 個ごとに CV / 推論時間の両方を記録**して、`report/NN_score_report.md` に追記する。
4. **stacking 化** (XGBRanker 等) は Cycle 03 以降。base model が落ち着いてから。

---

## 関連メモ

- 実装詳細 (TTA + mean+max): [`knowledge/01/ensemble_methods.md`](../01/ensemble_methods.md)
- Cycle 01 ベースライン案: [`knowledge/01/baseline_proposal.md`](../01/baseline_proposal.md)
- 上位 Kaggler 個別手法: [`02_kaggler_h2o_llm_studio.md`](../02_kaggler_h2o_llm_studio.md), [`03_kaggler_solokin.md`](../03_kaggler_solokin.md), [`04_kaggler_podpall.md`](../04_kaggler_podpall.md), [`05_kaggler_preferred_scantron.md`](../05_kaggler_preferred_scantron.md), [`06_kaggler_preferred_oshaberinbo.md`](../06_kaggler_preferred_oshaberinbo.md)
- 他の解法概観: [`07_other_notable_approaches.md`](../07_other_notable_approaches.md)
- 全体トレンド: [`01_overview_trends.md`](../01_overview_trends.md)

## 主要出典

- Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning*. MCS 2000.
- Breiman, L. (1996). *Bagging Predictors*. Machine Learning 24.
- Breiman, L. (2001). *Random Forests*. Machine Learning 45.
- Wood, D. et al. (2023). *A Unified Theory of Diversity in Ensemble Learning*. arXiv:2301.03962.
- Hippocampus's Garden — *Kaggle Competition Report: LLM Science Exam* (上位 5 解法の総括 writeup)
- Kaggle Discussion 各上位 writeup (1st H2O / 2nd solokin / 4th Preferred Scantron / 5th Preferred おしゃべりんぼう / 7th days)

---

## Q&A 履歴

ユーザーの追加発問とそれに対する回答をここに追記する（CLAUDE.md 「knowledge/search Q&A 運用」規約参照）。

### Q1 (2026-05-28): 回帰モデル 3 つをどうアンサンブルする？分布の重み付け結合は一般的？

**発問**:
> 統計の回帰モデルが 3 つあった場合、どのようにアンサンブルする？モデルの出力値を平均にする？統計的なモデルで、複数の分布を組み合わせたいときは、それぞれのモデルの出力値に重みをつけて予測値とすることが一般的？

**結論（短答）**:
- 最も基本: **単純平均** `ŷ = (y1 + y2 + y3) / 3`
- より一般的: **重み付き平均** `ŷ = Σ wi yi` (`Σ wi = 1`, 通常 `wi ≥ 0`)
- 「重みをつけて予測値とする」は **統計学・予測理論で完全に標準的**。Bates & Granger (1969) 以降 50 年以上の蓄積がある。
- 確率分布そのものを結合するなら **linear opinion pool** か **Bayesian Model Averaging (BMA)**。

#### 手法スペクトル

| 手法 | 式 | いつ使う | 理論的根拠 |
|---|---|---|---|
| 単純平均 | `ŷ = (1/N) Σ yi` | 各モデルの精度が同程度・誤差が比較的独立 | 誤差無相関なら variance が `σ²/N` に縮む |
| 重み付き平均 | `ŷ = Σ wi yi`, `Σ wi = 1` | 精度に差がある | Bates & Granger (1969) |
| 逆分散重み | `wi ∝ 1/σi²` | 各モデルの予測誤差分散が推定できる | GLS 解。誤差無相関仮定下で MSE 最小 |
| Stacking | `ŷ = f(y1, y2, y3)` (`f` は meta-learner) | OOF 予測が用意できる | Wolpert (1992) |
| Median / robust 集約 | `ŷ = median(yi)` | 1 モデルが外れ値を出しうる | ロバスト統計 |
| Linear opinion pool | `p(y) = Σ wi pi(y)` | **予測分布**を結合したい | 混合分布。多峰になり得る |
| Logarithmic opinion pool | `p(y) ∝ Π pi(y)^wi` | 予測分布を結合し sharpening したい | Product of Experts |
| Bayesian Model Averaging | `wi = P(Mi \| D)` | モデル選択の不確実性を扱いたい | 事後確率での重み付け |

#### 回帰特有のポイント

1. **理論的に重み付き平均が "best linear combination"**
   Bates & Granger (1969) は「複数予測の組み合わせが平均的に best single 予測より良い」ことを示した。誤差無相関で各 `σi²` が既知なら `wi ∝ 1/σi²` が MSE 最小。
2. **`Σ wi = 1`, `wi ≥ 0` を制約に置くのが一般的** (= convex combination)。制約を外す (自由 OLS) と負の重みが出て、解釈困難・過学習に振れやすい。
3. **モデル間の誤差相関**が高いほど利得が減る。多様性が鍵 → 本ファイル本編 Q1 の "diversity が hidden dimension" と同じ話。
4. **bias-variance 分解**: 平均は **variance を下げる** が **bias は下がらない** (各モデルが同じバイアスを持つ場合)。だから「異なる誤り方をするモデル」を混ぜる意味がある。

#### 「複数の分布を組み合わせたい」への直接回答

**はい、重み付き結合は統計学で標準**。具体的に何を結合するかで手法が分かれる:

- **点推定値**を結合 → 重み付き平均 `ŷ = Σ wi yi`
- **予測分布**を結合 → 2 系統が代表的:
  - **Linear opinion pool**: `p(y) = Σ wi pi(y)` ← 混合分布、多峰性を保持
  - **Logarithmic opinion pool**: `p(y) ∝ Π pi(y)^wi` ← 積、sharpening (Product of Experts)
- **重みの決め方** (代表的なもの):
  - 事後確率 (BMA): `wi = P(Mi | D)`
  - 検証データで CRPS / log-likelihood を最大化
  - 過去予測誤差からの逆分散

#### 実用上の意思決定フロー

```
回帰モデル 3 つの出力 → どうアンサンブルする？
 │
 ├─ 各モデルの精度が同程度          → 単純平均 (実装最簡)
 ├─ 精度差がある & validation が取れる → 検証 MSE 最小化で w を学習 (Σwi=1, wi≥0 制約推奨)
 ├─ 各モデル出力が予測分布            → linear opinion pool (混合) または log pool (sharpen)
 ├─ 各モデルの事後確率 P(Mi|D) が出る  → BMA
 └─ 非線形な結合が効きそう & OOF 予測ある → Stacking (Ridge / GBDT を meta-learner に)
```

#### 注意点

- **検証データで重みを決めるなら out-of-fold (OOF) 必須**。base model 訓練と重み学習を同じデータでやると過学習。
- **重みを自由学習** (制約なし OLS) すると負重み・極端な大重みが出やすい → `Σwi=1, wi≥0` を入れる。
- **モデルの予測スケールを揃える**こと (正規化 / 標準化された target で訓練しているか確認)。
- LLM Science Exam の `mean + max` 集約 ([`../01/ensemble_methods.md`](../01/ensemble_methods.md)) は **離散ラベルの確率** に対する集約で、回帰の連続値ではあまり見ない設計（MAP@3 評価に特化したヒューリスティック）。

#### 主要出典

- Bates, J. M. & Granger, C. W. J. (1969). *The Combination of Forecasts*. Operational Research Quarterly.
- Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks.
- Hoeting, J. A. et al. (1999). *Bayesian Model Averaging: A Tutorial*. Statistical Science.
- Genest & Zidek (1986). *Combining Probability Distributions: A Critique and an Annotated Bibliography*. (linear / log opinion pool の整理)

### Q2 (2026-05-28): LLM のアンサンブル学習で、平均をとるのは一般的？

**発問**:
> LLMの場合、アンサンブル学習で、平均をとるのは一般的？

**結論（短答）**:
- **タスクの出力形態で大きく分かれる**:
  - 分類 / MCQ / スコアリング（**離散ラベル + 確率**）→ **平均は超一般的、事実上の標準**
  - 自由生成（**文章生成**）→ **生成文を「平均」することは不可能**。代わりに「複数生成 → 投票 / 選別 / 再生成」が使われる
- LLM Science Exam の context では (= MCQ + softmax 確率) → **平均は一般的かつ妥当**。実際 7th `mean + max`、1st H2O の binary likelihood 平均、4th Preferred Scantron など、上位陣はほぼ全員確率平均系を使っている。

#### 出力形態別の整理

| LLM の出力形態 | 平均は使われる？ | 代表的なアンサンブル手法 |
|---|---|---|
| MCQ の正解確率 (本コンペ含む) | **○ 標準** | softmax 確率の平均、`mean+max`、重み付き平均、stacking (XGBRanker) |
| 分類タスクのロジット / 確率 | **○ 標準** | logit / probability averaging |
| 回答候補へのスコアリング (likelihood) | **○ 標準** | log-likelihood の平均 |
| 自由生成テキスト | **× 直接平均は不可** | Self-Consistency、Majority Vote、LLM-as-Judge、Reranker、LLM-Blender |
| Token-level next-token 分布 | △ 条件付きで可 | logit averaging（**同一トークナイザ・同一語彙が前提**、研究レベル） |
| 検索 / RAG の候補集合 | ○ ランク融合系 | Reciprocal Rank Fusion (RRF)、BM25 + dense の merge |
| 報酬モデル (RLHF) | ○ よく使う | 複数 RM の平均で reward hacking 緩和 |

#### 「LLM 文章生成での非平均ensemble」が一般的な理由

文章は連続値ベクトルではないので **「2 つの出力の中間」が定義できない**。  
→ 平均の代わりに次が主流:

1. **Self-Consistency** (Wang et al., 2022)
   同じモデルから温度付きで N 個サンプル → 最終答えだけを多数決。  
   "ensemble of one model with N stochastic runs"。CoT 系で大幅に精度向上。
2. **Majority Vote / Plurality Vote**
   複数モデル / 複数プロンプトの最終答えを単純多数決。
3. **LLM-as-Judge / Selection**
   別の LLM (or 同じ LLM) に「最も良い回答を選べ」と問う。
4. **Verifier / Reranker**
   学習済み verifier (報酬モデル等) に候補をスコアリングさせて top-1 を取る。
5. **LLM-Blender** (Jiang et al., 2023)
   ペアワイズ比較 + GenFuser で複数 LLM 出力を融合する専用フレームワーク。
6. **Mixture-of-Agents (MoA)** (Together AI, 2024)
   複数 LLM を**層に積む**: 下位層の出力を context に入れて上位層が再生成。
7. **Logit-level averaging**
   同じ vocab を持つ同系統モデル (e.g., LLaMA 系列の異 fine-tune) なら、次トークン分布を平均してから sample / argmax 可能。実用例は限定的。

#### MCQ / 確率出力での平均 — 本コンペでの実例

本ファイル本編 Q3 で挙げた通り、上位陣はほぼ全員「**softmax 確率の (重み付き) 平均 or 平均+α**」で集約している:

| 順位 | 集約手法 | 平均ベース？ |
|---|---|---|
| 1st H2O | 6 model の予測平均 (binary likelihood) | ○ |
| 2nd solokin | XGBRanker による stacking | △ (学習された非線形結合、平均の発展形) |
| 4th Preferred Scantron | 複数 reranker / 複数 context の予測集約 | ○ |
| 5th Preferred おしゃべりんぼう | Cascade (難易度で振り分け) | × (ルーティング型) |
| 7th days | TTA × model の **mean + max** | ○ |

→ MCQ + 確率出力という設定では **平均は最も基本かつ強い手法** と言える。

#### 実用上の注意（LLM 特有）

1. **温度 / softmax の鋭さがモデルごとに違う** → 単純平均する前に temperature scaling や calibration を入れた方が良いことがある（特に LLM は train objective が異なれば確率スケールがバラつく）。
2. **論理確率 (log-probability) で平均すべきか確率で平均すべきか**:
   - 確率の算術平均 = linear opinion pool（混合分布的、保守的）
   - log の算術平均 = 幾何平均 = log opinion pool（積で sharpening、自信が強い側に寄る）
   - MCQ 確率では算術平均が無難。確信の強い少数意見を重視したいなら log 系。
3. **生成系を絡める場合** (e.g., LLM が候補を再ランクする hybrid)、必ず「確率を集約する層」と「生成を集約する層」を分ける。両者を混ぜると意味が崩れる。
4. **大コスト**: LLM N 個を毎クエリ叩くのは非現実的な場面が多い → Self-Consistency (1 model × N sample) や cascade routing で実質的な ensemble にする工夫が主流になりつつある。

#### まとめ

- **「LLM の出力 = 確率/スコア」のタスクでは平均は一般的かつ標準**。本コンペの上位陣も大半がこれ。
- **「LLM の出力 = 文章」のタスクでは平均は使えない**。Self-Consistency / 多数決 / Judge / Reranker / MoA が代替手段。
- 本プロジェクト (LLM Science Exam) は前者の世界 → Cycle 01 の `mean + max` 集約は妥当。

#### 主要出典

- Wang, X. et al. (2022). *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models*. arXiv:2203.11171.
- Jiang, D. et al. (2023). *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion*. ACL 2023.
- Together AI (2024). *Mixture-of-Agents Enhances Large Language Model Capabilities*. arXiv:2406.04692.
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR.

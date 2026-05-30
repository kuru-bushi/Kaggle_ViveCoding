# Cycle 01: 提出モデルの概要と特性

> **このドキュメントのシリーズ**（Cycle 01 振り返り → Cycle 02 計画、4 分割）:
> **(1) 本ファイル＝提出モデル概要** → (2) [[cycle01_2_results_and_analysis]] 結果と考察 → (3) [[cycle01_3_challenges]] 課題 → (4) [[cycle02_4_model_plan]] 02 構成案
>
> 本ファイルの守備範囲: **Cycle 01 で提出した 3 モデルが「何者」で、なぜ性能差が出る構造だったか**。スコアそのものの分析は (2) に分離。

最終更新: 2026-05-30（ensemble 提出 ref 53157358 を反映。`kaggle competitions submissions kaggle-llm-science-exam` で実機照会済）

---

## 1. 提出 4 件の対応表

単独 3 件（m1/m2/m3）は **コード本体 (`01_v1_*.py`) が完全に同一**で、違うのは `kernel-metadata.json` の `dataset_sources` 1 行だけ（backbone に対応する Kaggle Dataset を指す）。**この 3 件は「同じパイプラインに違う脳みそを差し込んだ」対照実験**。4 件目の ensemble は、その 3 backbone を 1 Notebook 内で順次学習し、softmax 確率を **mean+max blending** で集約したもの。

| # | submit/ dir | Kaggle kernel slug | sub ref | submit date | backbone / 構成 | val MAP@3 | Public LB | Private LB | wall (Kaggle T4) |
|---|---|---|---|---|---|---|---|---|---|
| m1 | `submit/01_m1_microsoft/` | `kunihiro1997/llm-science-exam-01-v1-baseline` | 53093495 | 2026-05-27 | `microsoft/deberta-v3-large` (plain) | 0.4708 | 0.388056 | 0.378399 | ~3.4 min |
| **m2** | `submit/01_m2_openassistant/` | `…-01-v1-m2-openassistant-reward` | 53113415 | 2026-05-28 | `OpenAssistant/reward-model-deberta-v3-large-v2` | **0.7958** | 0.682480 | 0.714337 | ~3.4 min |
| m3 | `submit/01_m3_deepset/` | `…-01-v1-m3-deepset-squad2` | 53113423 | 2026-05-28 | `deepset/deberta-v3-large-squad2` | 0.6458 | 0.592176 | 0.605449 | ~3.4 min |
| **ens** | `submit/01_ensemble_m1m2m3/` | `…-01-v1-ensemble-m1m2m3-meanmax` | 53157358 | 2026-05-29 | m1+m2+m3 **mean+max blending** (days 7th 流) | 0.7958 | **0.684144** | **0.714858** | ~10 min (3 model 逐次) |

> ⚠️ スコアの**意味・考察 (R1–R5)** はこの表では扱わない。Public/Private LB の母集団の説明、val=40 のサンプリング誤差、過学習シグナルの読み方、ensemble の純効果は **→ [[cycle01_2_results_and_analysis]]** に集約。ここは「どのモデルがどんな素性か」だけ。

- Public LB に乗っているのはこの 4 件（`kaggle competitions submissions kaggle-llm-science-exam` で実機照会済、2026-05-30 時点）。
- **ensemble は最良 single (m2) を Public +0.0017 / Private +0.0005 で僅かに上回り、ボード最良を更新**。ただし val は m2 と同値 0.7958（後述のとおり、ほぼ無力な m1 を含んでも `max` 項が強モデルの確信を保持するため下がらなかった）。
- 詳細・コード差分: `submit/01_*/`, `report/01_score_report.md`（ensemble の R1–R5 考察あり）, `report/01_submissions_summary.md`。

---

## 2. 共通の土台 — DeBERTa-v3-large + MCQ head

3 モデルとも **アーキテクチャは同じ DeBERTa-v3-large**（約 300M パラメータ）。違うのは「事前学習で何を経験したか（重みの初期値）」だけ。まずこの共通土台を理解する。

### 📘 用語: DeBERTa-v3 とは

BERT 系（Transformer encoder）の改良版で、Microsoft が開発。素の BERT に対し v3 では主に 2 点が効いている:

1. **Disentangled attention（分離注意）**: 通常の Transformer は「単語の意味ベクトル」と「位置情報」を足し合わせて 1 本のベクトルにするが、DeBERTa は **内容ベクトルと相対位置ベクトルを別々に持ち**、両者の相互作用を別々に計算する。これで「どの語が」「どの距離にあるか」を分けて捉えられ、文の構造把握が上がる。
2. **ELECTRA 流の事前学習 (Replaced Token Detection)**: BERT は「文の一部を [MASK] して当てる」(Masked LM) が、DeBERTa-v3 は別の小モデルが置き換えた偽トークンを **「この語は本物か偽物か」を全トークンで判定**する事前学習を使う。マスクした 15% だけでなく全トークンが学習信号になるのでサンプル効率が高い。

→ 本コンペで DeBERTa-v3-large が定番なのは、(a) 300M と軽く Kaggle の 9h/T4 制約に収まる、(b) encoder 型は「文を読んで分類する」MCQ タスクに向く、(c) 上位陣の実証（4 位が DeBERTa-v3-large で Public 0.92+）があるため。出典: [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) §E.1。

### 📘 用語: MCQ head（5 択をどう解くか）

本コンペは「設問 + 5 つの選択肢 → 正解 1 つ」を選ぶ多肢選択 (Multiple-Choice QA)。DeBERTa を MCQ に使う標準の形は:

```
選択肢ごとに 1 本ずつ入力を作る:
  入力 i = [CLS] 設問文 [SEP] 選択肢_i [SEP]    (i = A..E、計 5 本)
        ↓ DeBERTa encoder（5 本を 1 バッチで通す）
  各入力 i から [CLS] 位置のベクトル → 線形層 → スカラー score_i
        ↓ 5 つの score を softmax
  最も高い score の選択肢が予測 1 位
```

- 「head」= encoder の上に載せる **小さな出力層**（ここでは score_i を出す線形 1 層）。事前学習済み backbone は流用し、この head は本コンペの 200 行で新規に学習する。
- 提出は MAP@3 なので **softmax 上位 3 つ**を順位つきで出す（指標の詳細は (2) 参照）。

→ 3 モデルで違うのは backbone の重みだけ。**MCQ head も学習設定（epochs 3, fp16, max_length 384 等）も完全に共通**。だから本節以降の性能差は純粋に「事前学習の質」に帰着できる。

### 学習設定と実測時間（3 モデル共通 / Kaggle T4 実測）

epochs・時間も 3 モデルで揃えてある（対照実験の条件）。共通設定: **3 epoch / fp16 / batch=2 × grad_accum=8 / max_len train=256・infer=384 / 公式 train 160 行で学習・val 40 行で評価・test 200 行を推論**。実測内訳（出典: `report/01_score_report.md` §学習時間）:

| # | Model | epochs | model load | epoch1 | epoch2 | epoch3 | 学習合計 (train) | 推論 (infer) | wall |
|---|---|---|---|---|---|---|---|---|---|
| m1 | `microsoft/deberta-v3-large` | 3 | 8.5s | 46.9s | 47.5s | 49.8s | **144.2s** (2.4 min) | 38.0s | 3.5 min |
| **m2** | `OpenAssistant/reward-model-deberta-v3-large-v2` | 3 | 3.1s | 45.6s | 45.0s | 46.0s | **136.6s** (2.3 min) | 33.8s | 3.3 min |
| m3 | `deepset/deberta-v3-large-squad2` | 3 | 3.6s | 46.6s | 47.1s | 49.0s | **142.7s** (2.4 min) | 34.9s | 3.4 min |

- **学習時間はほぼ横並び（136–144 秒）**: 同一アーキ・同一 epoch・同一 160 行なので当然。性能差（val 0.47〜0.80）は学習時間ではなく **backbone の事前学習の質**から来ている、を裏づける。
- **3 本トータル wall ≈ 10.2 min**（直列）。Kaggle Notebook の 9h 上限に対し **40 倍以上の余裕** → Cycle 02 で retrieval、Cycle 03 で TTA/ensemble を重ねても時間制約には当たらない。

---

## 2.5. 系譜と事前学習データ（要約）

**3 モデルは同じ出発点 `microsoft/deberta-v3-large` を共有する**。m1 はその素のまま、m2・m3 は**第三者がその素の重みをベースに中間タスクで追加 fine-tune** したもの（＝「m2・m3 は m1 をベースに学習」と言える。ただし派生元は HF 公式の素の重みで、本コンペで我々が fine-tune した後の m1 ではない）。

```
microsoft/deberta-v3-large（共通の出発点 = m1 そのもの）
   ├─ そのまま ........................... m1
   ├─ + 選好データで追加学習(reward) ...... m2 (OpenAssistant が作成)
   └─ + SQuAD2 で追加学習(抽出型QA) ....... m3 (deepset が作成)
```

| # | 中間学習で見たデータ（差を生んだ部分） | 種類 | 中間学習時の head |
|---|---|---|---|
| m1 | **なし**（汎用事前学習のみ） | 自己教師あり(RTD) | – |
| m2 | **人間の選好比較ペア**（OpenAssistant 収集） | fine-tune | reward(回帰) |
| m3 | **SQuAD2.0**（Wikipedia ベース読解QA） | fine-tune | QA(span抽出) |

- **本コンペの fine-tune データは 3 モデル共通**（公式 train 200 行 / MCQ head）。中間学習の head は捨て、MCQ head を新規に学習する。
- **m1（素の deberta-v3-large）の事前学習データ = 160GB の汎用テキスト 5 種**（HF カード「160GB data as DeBERTa V2」＋ DeBERTaV3 論文 arXiv:2111.09543 Table 8 で確認）:
  **Wikipedia / BookCorpus / OpenWebText / STORIES / CC-News**。特定タスク用データではなく生テキストの自己教師あり学習(RTD)のみ。

→ 各 backbone の「型」と性能差の詳細考察は次節 §3。

---

## 3. 3 つの backbone の素性 — どんな「中間事前学習」を経たか

ここが Cycle 01 の核心。3 つは同じ DeBERTa-v3-large を出発点にしつつ、**fine-tune の前段でそれぞれ別の "中間タスク" を経験している**。

### 📘 用語: 中間事前学習 (intermediate / further pre-training)

「汎用事前学習済みモデル」を、本番タスクで fine-tune する**前に**、本番に近い別タスクで一度鍛えておくこと。少量の本番データ（ここでは 200 行）しか無いとき、中間タスクで得た「判断の型」が転移して効くことがある。本コンペはまさにこれが効いた事例。

### m1 — `microsoft/deberta-v3-large` (plain / 素のまま)

- 中間事前学習: **なし**。汎用事前学習（前述の RTD）だけを終えた素の状態。
- 性格: 「文の一般的な意味」は分かるが、「どちらの答えが良いか選ぶ」「文書を根拠に答える」といった**タスク特有の判断の型を持っていない**。
- 結果: 200 行 × 3 epoch の fine-tune だけでは MCQ head と判断の型を同時に獲得しきれず、3 つの中で最下位（val 0.47 / Public 0.39）。

### m2 — `OpenAssistant/reward-model-deberta-v3-large-v2` ⭐ 最強

- 中間事前学習: **preference / reward model**（選好モデル）。人間が「どちらの回答が良いか」を比較した大量のペアデータで「良い回答 > 悪い回答」のスコア付けを学習している。
- 📘 **「reward model」と「RLHF」の関係（誤解しやすい点）**: reward model は RLHF（人間のフィードバックによる強化学習）の **一部品**。RLHF 全体は「(1) 人間の選好データで報酬モデルを訓練 → (2) その報酬を使って本体 LLM を強化学習で更新」という 2 段だが、**m2 は (1) の報酬モデル単体**であって、RLHF ループ全体を回したモデルではない。「人間の選好比較で訓練されたスコアリングモデル」と理解するのが正確。
- なぜ MCQ に刺さるか: MCQ は本質的に「5 つの候補を相対比較して一番良いものを選ぶ」タスク。reward model が中間学習で得た **「候補をスコアで順位づけする型」がそのまま MCQ head の仕事と同型**。だから 200 行でも一気に立ち上がる（val 0.7958 / Public 0.682、m1 に圧勝）。

### m3 — `deepset/deberta-v3-large-squad2` (QA fine-tune 済み)

- 中間事前学習: **SQuAD2 で抽出型 QA**。「文章（context）を読み、質問の答えに当たるスパンを文中から抜き出す」タスク。SQuAD2 は「答えが文中に無い」ケースも含むので、**根拠の有無を判断する型**も持つ。
- なぜ中位か: 「文章を根拠に答える型」は本コンペ（Wikipedia 由来設問）と親和性が高いが、Cycle 01 では **まだ retrieval を入れていない**ので、肝心の「読むべき文章 (context)」を与えていない。QA の型を持っているのに材料が無い状態 → m1 よりは強いが m2 には及ばない（val 0.6458 / Public 0.592）。
  - → この「m3 は context を与えれば化ける可能性」が Cycle 02 で retrieval を入れる動機の一つ。詳細 → [[cycle01_3_challenges]]。

---

## 3.5. 4 件目の提出 — 3 モデル ensemble（mean+max blending）

単独 3 件のあと、同じ 3 backbone を束ねた ensemble を 1 件提出した（ref 53157358, 2026-05-29）。**submit した中身（構成）**は次のとおり:

- **コード**: `submit/01_ensemble_m1m2m3/01_v1_ensemble_m1m2m3.py`。1 つの Notebook 内で m1→m2→m3 を**逐次** fine-tune（モデル間で `del` + `torch.cuda.empty_cache()` を挟み T4 16GB に収める）。学習設定は単独 3 件と同一。
- **共通 split**: 3 モデルとも **同一の 80/20 split（SEED=42）** を共有 → val を揃えて比較可能にしている。
- **集約方式 = mean+max blending（days 7th 流、`knowledge/01/ensemble_methods.md`）**: 各モデルの softmax 確率を id・選択肢ごとに集約する。

  ```
  score(id, option) = mean_models(softmax_prob) + max_models(softmax_prob)
  ```
  - `mean` は 3 モデルの平均（多数決的に安定させる）、`max` は「どれか 1 モデルが強く確信した選択肢」を拾う項。両者の和を取る。
  - 単純平均ではなく days 7th place の構成をそのまま再利用（ユーザー指示 2026-05-29）。

- **この run 内の val 内訳**: per-model = deepset 0.6458 / OpenAssistant **0.7958** / microsoft **0.4042**（単独 submit 時の m1 val 0.4708 とズレるのは fresh 学習 + val=40 のノイズ）。ensemble(mean)=0.7917、**ensemble(mean+max)=0.7958**。
- **結果**: **Public 0.684144 / Private 0.714858**。最良 single (m2) を Public +0.0017 / Private +0.0005 で僅かに上回り、ボード最良を更新。
  - **示唆**: ほぼランダムに近い m1（val 0.4042）を混ぜても落ちなかったのは、`max` 項が強モデル（m2）の確信を保持するため。単純 `mean`(0.7917) は弱モデルに薄まって mean+max に劣後した → 弱モデルを含む blending では **mean だけより mean+max の方が頑健**、という実証。
  - スコアの R1–R5 考察は **→ [[cycle01_2_results_and_analysis]] §7**、一次記録は `report/01_score_report.md`「3 モデル ensemble」節。

> 📌 補足: この ensemble は当初「Cycle 03 で本格化」と位置づけていたが、Cycle 01 のうちに mean+max 版を 1 本提出済み。Cycle 03 で狙うのは **retriever/corpus/backbone の多様性を増やした上での** ensemble であり、本件はその前段の最小版（[[cycle01_3_challenges]] 打ち手 H 参照）。

---

## 4. 本ファイルから言える観察事実

この節は、§2-3 で見た **「3 モデルは同じ箱（パイプライン）に違う脳（backbone）を差し込んだ対照実験」** という構図から、**モデルの素性について何が読み取れるか**をまとめる（スコアの分布・LB ギャップそのものの分析は (2) に分離）。各項目を「**観察事実 → なぜそう言えるか → 意味**」の 3 段で展開する。

### 観察 1: これは変数を 1 つに絞った「きれいな対照実験」になっている

- **観察事実**: 3 件の Public LB は m1 = 0.388 / m3 = 0.592 / m2 = 0.682。最大で約 **0.29 の差**がついた。
- **なぜそう言えるか**: §1・§2 のとおり、3 件は **コード本体・ハイパラ（epochs 3, fp16, max_length 384 …）・MCQ head・学習データ（200 行）・epoch 数まで完全に同一**で、唯一違うのは `dataset_sources` が指す **backbone の重み**（＝事前学習で何を経験したか）だけ。実験で動かした変数が 1 つしかないので、生じた差の原因をその 1 変数（事前学習の質）に**限定して断定できる**。これが「対照実験」の意味。
- **意味**: Cycle 01 で最もスコアを動かしたレバーは、モデル内部の工夫や学習設定ではなく **「出発点の backbone をどれにするか」という選択そのもの**だった。同じ労力でも、まず良い backbone を選ぶことが最優先だと分かる。

### 観察 2: 学習データが 200 行しかない領域では、素の backbone は力を出せない

- **観察事実**: 中間事前学習を経ていない素の m1 が 3 つの中で最下位（val 0.47 / Public 0.39）。reward / QA を経た m2・m3 が大きく上回った。
- **なぜそう言えるか**: 本コンペの学習データは **200 行しかない**。素の m1 は、この 200 行だけで「MCQ の解き方（head の重み）」と「候補を比較して良い方を選ぶ判断の作法」を**同時にゼロから覚えねばならず、量が足りない**。一方 m2・m3 は、その「判断の作法」を中間事前学習（後述）で**既に身につけている**ので、200 行を head の微調整だけに使えて立ち上がりが速い。
- **意味**（＝元の "少量データ regime" の言い換え）: 「regime（レジーム）」とは **「データ量がどの領域にあるか」** を指す言葉。**データが少ない領域**では、モデル自体を細工するより **「最初から良い作法を持った backbone を選ぶ」方が圧倒的に効く**。逆に学習データが潤沢な領域なら、この差は縮む可能性がある（＝Cycle 01 の結論は「データが 200 行と少ないからこそ」のもの、という留保つき）。

### 観察 3: 中間学習で得た「型」が本番タスクと形が似ているほど強い

- **観察事実**: 強さの順は **m2（reward）> m3（QA）> m1（なし）**。
- **なぜそう言えるか**: ここでの **「判断の型」とは、中間事前学習で身についた "問題の解き方の作法"** のこと（§3 参照）。その作法が**本番タスク（5 択を比べて 1 つ選ぶ MCQ）とどれだけ形が似ているか**で強さが決まった:
  - **m2 = reward model**: 「複数の候補を相対比較してスコアで順位づけする」作法。これは MCQ の「5 つを比べて一番を選ぶ」と**ほぼ同じ形（同型）** → 最強。
  - **m3 = QA（SQuAD2）**: 「文章を根拠に答えを抜き出す」作法。本番と方向性は近いが、Cycle 01 では肝心の根拠文（context）を retrieval で**与えていない**ため作法を活かしきれず**割引** → 中位。
  - **m1 = なし**: そもそも作法を持っていない → 最下位。
- **意味**: backbone は「有名・高性能そう」で選ぶのではなく、**「中間タスクの形が本番タスクと同型か」**で選ぶべき。これが Cycle 02 で backbone を m2 に固定する根拠であり、また m3 に context（retrieval）を与えれば伸びる余地がある、という見立ての根拠でもある。

→ この 3 つの観察から導く Cycle 02 への具体的な打ち手（backbone を m2 に固定する等）は **→ [[cycle01_3_challenges]]** の打ち手表で扱う。

---

## 関連

- [[cycle01_2_results_and_analysis]] — 本 3 モデルのスコア考察 (R1–R5)
- [[cycle01_3_challenges]] — ここから導くボトルネックと打ち手
- [[cycle02_4_model_plan]] — Cycle 02 の構成案（m2 固定 + retrieval）
- `report/01_score_report.md`, `report/01_submissions_summary.md` — Cycle 01 結果の一次記録
- [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) — DeBERTa+RAG の定量根拠
- `knowledge/01/baseline_proposal.md` — days 7th 派生の元案

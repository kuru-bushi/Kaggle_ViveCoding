# Cycle 01: 提出モデルの概要と特性

> **このドキュメントのシリーズ**（Cycle 01 振り返り → Cycle 02 計画、4 分割）:
> **(1) 本ファイル＝提出モデル概要** → (2) [[cycle01_2_results_and_analysis]] 結果と考察 → (3) [[cycle01_3_challenges]] 課題 → (4) [[cycle02_4_model_plan]] 02 構成案
>
> 本ファイルの守備範囲: **Cycle 01 で提出した 3 モデルが「何者」で、なぜ性能差が出る構造だったか**。スコアそのものの分析は (2) に分離。

最終更新: 2026-05-29（`cycle01_retrospective_and_v2_plan.md` を 4 分割して作成）

---

## 1. 提出 3 件の対応表

3 件とも **コード本体 (`01_v1_*.py`) は完全に同一**。違うのは `kernel-metadata.json` の `dataset_sources` 1 行だけで、それが backbone（事前学習済みの重み）に対応する Kaggle Dataset を指す。**つまりこの 3 件は「同じパイプラインに違う脳みそを差し込んだ」対照実験**になっている。

| # | submit/ dir | Kaggle kernel slug | sub ref | submit date | backbone (HF repo) | val MAP@3 | Public LB | Private LB | wall (Kaggle T4) |
|---|---|---|---|---|---|---|---|---|---|
| m1 | `submit/01_m1_microsoft/` | `kunihiro1997/llm-science-exam-01-v1-baseline` | 53093495 | 2026-05-27 | `microsoft/deberta-v3-large` (plain) | 0.4708 | 0.388056 | 0.378399 | ~3.4 min |
| **m2** | `submit/01_m2_openassistant/` | `…-01-v1-m2-openassistant-reward` | 53113415 | 2026-05-28 | `OpenAssistant/reward-model-deberta-v3-large-v2` | **0.7958** | **0.682480** | **0.714337** | ~3.4 min |
| m3 | `submit/01_m3_deepset/` | `…-01-v1-m3-deepset-squad2` | 53113423 | 2026-05-28 | `deepset/deberta-v3-large-squad2` | 0.6458 | 0.592176 | 0.605449 | ~3.4 min |

> ⚠️ スコアの**意味・考察 (R1–R5)** はこの表では扱わない。Public/Private LB の母集団の説明、val=40 のサンプリング誤差、過学習シグナルの読み方は **→ [[cycle01_2_results_and_analysis]]** に集約。ここは「どのモデルがどんな素性か」だけ。

- Public LB に乗っているのはこの 3 件のみ（`kaggle competitions submissions kaggle-llm-science-exam` で実機照会済）。
- multi-model ensemble は **未提出**（Cycle 03 へ）。
- 詳細・コード差分: `submit/01_*/`, `report/01_score_report.md`, `report/01_submissions_summary.md`。

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

## 4. 本ファイルから言える観察事実

元文書 §1-2 のうち、**「モデルの素性」に関する観察**を抜粋（スコアの分布・ギャップの分析は (2) へ）:

1. **同一コード・同一ハイパラ・違うのは backbone weights のみ** → 性能差は完全に「事前学習の質」由来と断定できる、きれいな対照実験になっている。
2. **200 行 train の 3 epoch fine-tune では plain backbone (m1) の能力を引き出せない**。中間事前学習（reward / QA）を経たものが圧倒。少量データ regime では「事前に良い判断の型を持っているか」が決定的。
3. **判断の型の "形" が本番タスクと近いほど強い**: reward model（候補を順位づけ＝MCQ と同型, m2）＞ QA（根拠から抽出, 材料未供給で割引, m3）＞ なし (m1)。

→ この観察から導く Cycle 02 への含意（backbone を m2 に固定する等）は **→ [[cycle01_3_challenges]]** の打ち手表で扱う。

---

## 関連

- [[cycle01_2_results_and_analysis]] — 本 3 モデルのスコア考察 (R1–R5)
- [[cycle01_3_challenges]] — ここから導くボトルネックと打ち手
- [[cycle02_4_model_plan]] — Cycle 02 の構成案（m2 固定 + retrieval）
- `report/01_score_report.md`, `report/01_submissions_summary.md` — Cycle 01 結果の一次記録
- [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) — DeBERTa+RAG の定量根拠
- `knowledge/01/baseline_proposal.md` — days 7th 派生の元案

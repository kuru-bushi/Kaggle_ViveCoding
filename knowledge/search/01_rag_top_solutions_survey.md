# 01 RAG 上位解法サーベイ — Kaggle LLM Science Exam

> ⚠️ **本ファイルは Claude が Web 調査して作成**したメモである。`knowledge/search/` は通常「ユーザー手動の調査メモ」だが、本回はユーザー指示により Claude が代行している。Kaggle Discussion ページは WebFetch で取得できなかった (HTML が JS 描画 / reCAPTCHA) ため、**二次出典 (まとめブログ・GitHub・他者引用) に依存**している箇所が多い。一次ソース (Kaggle Discussion) で再検証が必要な数字には `[要再検証]` を付けた。
>
> 元の発問: 「このコンペティションで RAG を使っている人たちのナレッジについてまとめて。…RAG で精度が向上するかの原因も調査しておいて。」
>
> 本ファイルの守備範囲: **誰が・どう RAG を組んだか** (上位 6 + 注目解法の構成詳細)。
> 関連: 数値根拠 → [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)、因果分析 → [`03_rag_why_it_works.md`](03_rag_why_it_works.md)、Cycle 横断の概観 → [`../01_overview_trends.md`](../01_overview_trends.md)。

---

## 0. なぜこのコンペは "RAG が刺さる" 設計だったのか

問題設定そのものが retrieval 前提だった。これが本コンペの本質。

| 設定 | 含意 |
|---|---|
| 問題は **GPT-3.5 が Wikipedia 記事を元に生成** した MCQ (5 択)。topic は Wikipedia から抽出。 | 正解の根拠は **必ず Wikipedia のどこかにある**。"open book" は出題者の意図でもある。 |
| 提供される train set は **わずか 400 問** | この量で MCQ 全分野を fine-tuning だけでカバーするのは不可能 → 外部知識ベース必須。 |
| Code Competition (Notebook submission), Internet OFF | 外部 API 不可。Wikipedia dump を Kaggle Dataset としてマウントする運用が事実上の標準。 |
| 評価指標: **MAP@3** (公式) — Hippocampus Garden の記事は誤って "MAP@5" と表記しているが、`overview/evaluation.md` の通り **MAP@3** が正。 | 上位 1 つを当てればフル得点 (1.0)、2 位 0.5、3 位 0.333、圏外 0。retrieval が悪いと "正解が選択肢にないのに当てに行く" 状態になり破綻する。 |

→ Hippocampus's Garden ([URL](https://hippocampus-garden.com/kaggle_llm/)) は **"high-quality dataset (for training AND for retrieval) could be a key to winning"** と総括している。

---

## 1. 上位解法の RAG 構成一覧 (要約表)

| 順位 | チーム | コーパス | Retriever | チャンク | Reranker | 推論モデル |
|---|---|---|---|---|---|---|
| 1 | **H2O LLM Studio** | 複数の Wikipedia dump (filter なし) | **Dense 多モデル** (MTEB leaderboard 由来、title+chunk 連結を embed、PyTorch matmul で線形検索) | article chunk | (明示なし) | **Mistral 7B + Llama 2 70B** (QLoRA 4bit, xFormers memory-efficient attention)、3-stage cascade (易→難で大モデル) |
| 2 | **@solokin** | `graelo/wikipedia/20230601.en` (単一) | **BM25** (pyserini / Apache Lucene 実装) | **sentence 単位の overlapping chunks** | **DeBERTa v3 reranker** | DeBERTa 系 + LLM |
| 3 | **@podpall** | (詳細不明) | 多段パイプライン (small LM 群 + 2× 70B) | — | — | 多モデル ensemble (Kaggle 9h 制限内に詰め込み) |
| 4 | **Preferred Scantron** | Wikipedia | **Elasticsearch (sentence-wise keyword retrieval)** | sentence | **3 種 rerank**: ES score / **edit distance** / semantic search score | **DeBERTa v3 large 単体** (~300M)、token 長 512 → 1280 で段階的向上 |
| 5 | **Preferred おしゃべりんぼう** | Wikipedia | **BM25 (pyserini) + Dense (sentence/paragraph embed)** | sentence / paragraph | (明示なし) | Mistral 7B + Llama 2 70B、3-stage 推論 |
| 6 | **@rbiswasfc** | Wikipedia (FAISS 化) | **3 段構成: retriever (`e_topic`, bge 系) + ranker (`e_ranker`) + reader (`r_delta`, spanwise)** | — | dedicated ranker | A100 / A6000 で学習 |

→ **共通項**: 全員 Wikipedia ベース、ほぼ全員が **複数 retriever を混ぜる**、上位ほど reranker を明示的に置く。
→ **分岐点**: 1st/5th は LLM (Mistral/Llama) に重心、4th は DeBERTa に重心。ただし **どちらも 0.92+ に届く** ので、retrieval 品質さえあればモデルサイズは決定要因ではない。

---

## 2. 各解法の "RAG 設計判断" を深掘り

### 2.1 1st place — H2O LLM Studio

- **コーパス多様化**: 複数の Wikipedia dump を併用。当初は "science 系記事だけにフィルタ" を試したが retriever が irrelevant 文書を自動で無視するため不要と判断。
  - 含意: retriever が十分強ければ corpus を絞る必要はない (再現性確保のためコーパス固定の方が大事)。
- **検索の実装**: title + chunk を連結して embed → クエリ embed と **PyTorch 行列積一発**。FAISS や ANN を使わず、シンプルでスケールする matmul を選択。
  - 含意: index 構築が単純な分、デバッグ容易・再現性高い。70B 系を 9h 内に推論する制約があるので "retrieval は早く堅く済ます" 戦略。
- **embedding source**: MTEB leaderboard 上位の複数モデルを試して採用。単一モデル依存を避けている。
- **MCQ モデル化**: 「contexts + question + 1 つの選択肢」を入力に **二値分類** (この選択肢は正解か?)。各設問で 5 回推論し softmax で順位付け。
  - 利点: 1 つの正解ラベルしかない MCQ を、**5 つの独立した二値ラベル**として LoRA 学習でき、データ拡張が利く。
- **3-stage cascade 推論**: 易問は Mistral 7B、難問だけ Llama 2 70B。9h 制限を超えない時間配分。

### 2.2 2nd place — @solokin

- **single corpus**: `graelo/wikipedia/20230601.en` 一本に絞り、retriever を BM25 (Apache Lucene 系) で固定。
- **sentence 単位の overlapping chunks**: 文ごとに切り、隣接窓を重ねる。BM25 は短いクエリ × 短い chunk で精度が出やすい。
- **DeBERTa v3 reranker**: BM25 で上位 N を取った後、cross-encoder で再順位付け。これが 1st と並ぶ "上位の必須要素"。
- **教訓**: 「シンプルなコーパス + 強いリランカー」で 2 位まで到達可能。多コーパス必須ではない。

### 2.3 3rd place — @podpall

- 公開情報が薄い。「多数の small LM + 2× 70B を 1 つの Kaggle Notebook (9h) に詰め込む」という工夫が中心と報じられている。
- RAG 詳細は要再検証 [要再検証]。

### 2.4 4th place — Preferred Scantron

- **billion-parameter モデルを使わず DeBERTa v3 large (~300M) で 4 位** に入った特異な解法。
- **retrieval は Elasticsearch** で sentence-wise。
- **3 種の reranking スコア合成**:
  1. Elasticsearch のスコア (BM25 派生)
  2. **edit distance** (Levenshtein)
  3. **semantic search score** (dense embedding 類似度)
  - 3 つを混ぜることで keyword 一致 / 表層一致 / 意味一致 を全部押さえる。
- **token 長を 512 → 1024 → 1280 と伸ばすと public score が steady に向上**。これは「retrieved context をどれだけ詰め込めるか」が直接効くことの実証。
- 含意: **大きいモデルより、コンテキスト品質 × 長さ**。

### 2.5 5th place — Preferred おしゃべりんぼう

- pyserini BM25 + dense (sentence + paragraph 両粒度を embed) の **hybrid retrieval**。
- 推論側は 1st 同様 Mistral 7B + Llama 2 70B、3-stage cascade。
- 1st との差は corpus 多様化の浅さと言われている (詳細は要再検証)。

### 2.6 6th place — @rbiswasfc ([GitHub](https://github.com/rbiswasfc/llm-science-exam))

- **3 段構造を明示**: retriever → ranker → reader、各々別モデル。
  - retriever: `e_topic` (config 名)、bge 系 embedding が示唆される
  - ranker: `e_ranker` で再順位付け
  - reader: **spanwise** (`train_r_delta.py`) — MCQ を span 抽出問題として解く 2 段学習 (大規模 MCQ で pretrain → 難問で specialization)
- ハード: A100 40GB or A6000 48GB、128GB RAM
- 含意: **学習段階での難易度カリキュラム** (easy → hard) が retrieval 系でも有効。

### 2.7 注目の中位・公開解法

#### Lizhecheng02 ([GitHub](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam)) — 最も詳細な ablation を公開

- **3 種 retriever 併用**: (a) Wikipedia ページ全体抽出 (b) TF-IDF (c) sentence transformer
- **270K → ~6M rows へクラスタ拡張**
- **embedding model ablation 表** (Public LB):
  | model | LB |
  |---|---|
  | gte-small (採用) | **0.851** |
  | all-MiniLM-L6 | 0.843 |
  | all-MiniLM-L12 | 0.837 |
  | bge-small-en | 0.822 |
- top-k 設定: **10 ページ × 30 sentences**
- 最終: Public 0.930 / Private 0.923–0.924

→ embedding モデル選択だけで Public LB が 0.030 ぶれる。**retriever の embedder は ablation 必須**。

#### Chris Deotte (Top 100) — [Top 100 Solution - Fast RAPIDS TF-IDF RAG](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/chris-deotte-top-100-solution-fast-rapids-tf-idf-r) [要再検証]

- **TF-IDF を RAPIDS cuML で GPU 化** → Kaggle の 2× T4 で高速に retrieval 完結。
- 「dense は不要、BM25/TF-IDF だけで top 100 ライン (~0.92) に届く」というメッセージ。
- 詳細数値は Kaggle Discussion で WebFetch 取得不可、要再検証。

#### Zero-shot 70B + RAG ([LB 0.836 公開 Notebook](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620)) [要再検証]

- **fine-tuning なし**、70B をそのまま使い RAG だけ被せて Public LB 0.836。
- "RAG 単独でも 0.83 まで届く" ベンチマーク的存在。fine-tune 込みのフルパイプは 0.92+。

---

## 3. クロスチーム比較で見える "勝ち筋"

| 設計判断 | 採用率 (上位 6) | 観察 |
|---|---|---|
| Wikipedia をコーパスにする | 6/6 | 全員。問題が Wikipedia 由来なので必然。 |
| BM25 / sparse retrieval を含む | 4-5/6 | keyword 一致に強い。基底として置くチーム多数。 |
| dense embedding retrieval を含む | 4-5/6 | semantic 補完。MTEB / bge / gte 系。 |
| **Hybrid (sparse + dense)** | ~5/6 | 単独より一貫して強い。 |
| **明示的な reranker (cross-encoder or rule-based)** | 4/6 (上位ほど高頻度) | top-100 と top-10 を分ける要素。 |
| 複数 corpus / dump を併用 | 1st のみ明示 | 必須ではない。再現性とのトレードオフ。 |
| 大型 LLM (70B) を使う | 1, 3, 5 位 | 4 位 (DeBERTa 300M) でも 0.92 圏に到達できることが反例。 |
| Ensemble | 6/6 | 単一モデルで勝った例なし。詳細は [`../search/00_ensemble_pipeline_origin_validity.md`](00_ensemble_pipeline_origin_validity.md)。 |

---

## 4. 上位 vs 中位 vs 下位の決定的な差は何か

Hippocampus Garden の総括と上位ブログを総合すると:

1. **retrieval の多様性** (BM-25 / ES / BERT family / LLM embed を複数混ぜる) があるか
2. **reranking layer** を入れているか (単純な top-k だけでは irrelevant 文書が混ざる)
3. **訓練・retrieval 両用の高品質データセット** を自前で拡張したか (e.g. @radek1 の 6.5k curated、150k GPT-3.5 generated)
4. **context 長を最大限活用するモデル設計** (DeBERTa を 1280 tokens まで伸ばす、Llama の 4K を埋める)

→ いずれも "モデルそのもの" より **データ・retrieval パイプライン**の問題。本コンペは **データセントリック側に勝因が偏ったタイプの NLP コンペ** と位置づけられる。

---

## 5. 本プロジェクト Cycle への含意

- **Cycle 01** (days 7th place 解法、`knowledge/01/baseline_proposal.md`) は既に RAG+DeBERTa 構成。本サーベイの観察 (retriever 多様化 / reranker 必須) と整合。
- **Cycle 02** (`knowledge/02/dataset_improvements.md`) で cirrussearch 版 Wikipedia dump に切替予定 → 数値欠落の修正だが、**corpus 多様化** という勝因要素にも該当する。
- 将来 Cycle 候補: (a) BM25 + dense の hybrid retriever 化、(b) cross-encoder reranker の導入、(c) embedding モデルの ablation (gte-small / bge-small / e5-small) は本サーベイで効果サイズが最大級。

---

## 出典 (信頼度ランク付き)

**Tier 1 (一次ソースに近い・自分で本文確認済み)**
- [Lizhecheng02 GitHub README](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam) — ablation 表を直接掲載
- [rbiswasfc GitHub README](https://github.com/rbiswasfc/llm-science-exam) — 6th place コード構造
- [Open Book LLM Science Exam Kaggle Notebook (jjinho)](https://www.kaggle.com/code/jjinho/open-book-llm-science-exam) — 直接本文取得は失敗、外部引用経由

**Tier 2 (二次まとめ・1 人の整理に依存)**
- [Hippocampus's Garden — Kaggle Competition Report: LLM Science Exam](https://hippocampus-garden.com/kaggle_llm/) — 上位 5 整理 (MAP@5 と誤記あり、本来は MAP@3)

**Tier 3 (Kaggle Discussion 経由・WebFetch 不可で間接引用)** [要再検証]
- [1st place discussion 446511](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446511)
- [3rd place discussion (@podpall)](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/podpall-3rd-place-solution-update-code-links)
- [LB 0.836 Zero-shot 70B + RAG discussion 440620](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620)
- [Chris Deotte Top 100 writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/chris-deotte-top-100-solution-fast-rapids-tf-idf-r)

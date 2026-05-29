# Cycle 02: Wikipedia Retrieval 設計 — A 案の詳細

> Cycle 02 の改善案 A（retrieval 注入）について、コーパス → chunk → embedder → index → 注入方式 → 評価の各段で
> 選択肢を比較し、**v1 で採用する 1 系統** と **v2 で試す代替案** を確定する。
> 全体方針と他案との位置付けは [[cycle02_4_model_plan]]、dump 切替詳細は [[dataset_improvements]]。

最終更新: 2026-05-28

---

## 0. なぜ retrieval が要るか（再掲）

- Cycle 01 m2 (OpenAssistant reward) で Public 0.682 まで来たが、上位陣 0.92+ には程遠い。差の主因は **問題文の固有名詞・年代・公式を backbone が暗記できないこと**。
- 上位陣の共通解は **retrieval で Wiki 短文 chunk を context として注入**、MCQ head が「context と選択肢の整合性」を判定する形に変える。
- これにより `MCQ(question, options, context) → top-3` というタスク再定式化が成立し、CV +0.10〜0.20、LB 同等の上昇が見込める。
- 上位陣の解法分析: `knowledge/search/01_rag_top_solutions_survey.md`, `knowledge/01/baseline_proposal.md`。

---

## 1. コーパス選定

### 1-1. 候補

| コーパス | サイズ (圧縮) | 形式 | カバレッジ | 数値・公式の保持 |
|---|---|---|---|---|
| **`jjinho/wikipedia-20230701`** (Cycle 01 想定だった方) | 数 GB (Kaggle Dataset) | parquet | en 全体 (整形済) | △ 数値欠落報告あり |
| **`cirrussearch` 月別 dump** | ~40-80GB | json.gz | en 全体 (cirrussearch 投入用) | ◎ 整形漏れが少ない |
| **`graelo/wikipedia` (HF)** | ~20GB | parquet | en 全体, dump 由来 | ○ |
| **`wikipedia-stem-subset` (要作成)** | ~5-10GB | parquet | STEM カテゴリのみ | (元コーパス次第) |

### 1-2. 採用

- **v1: cirrussearch の 1 か月 dump を STEM カテゴリで filter → ~5-10GB parquet**
  - 理由: dump 切替 (B 案) と一体で進められる。STEM filter で WSL の DL 帯域・ローカルディスクの制約を回避。
  - filter 方法: cirrussearch JSON の `category` フィールドに `STEM` 関連カテゴリ (Physics, Chemistry, Biology, Mathematics, Computer science, Engineering, Astronomy, ...) を持つページのみ抽出。
- **v2 候補**: 全 en dump（filter なし）、`graelo/wikipedia` 比較

### 1-3. リスクと対応

- DL 帯域: 並行作業中の WSL2 mirrored mode で解決を狙う。失敗時は Windows aria2c 経由。
- ディスク: STEM filter で 5-10GB に収まる想定。失敗時は streaming で読み込み → embedding を chunk 単位に書き出す。

---

## 2. Chunk 戦略

### 2-1. 候補

| 戦略 | chunk 長 | overlap | 推論コスト | 検索質 |
|---|---|---|---|---|
| **days 7th 派生** | 90 word | 3 sentence | 中 | 高 (短文 → 局所一致) |
| 段落分割 (`\n\n` で split) | 可変 (50-500 word) | 0 | 高 (長文 chunk) | 中 (長文に薄まる) |
| sentence 単位 | 1 sentence | 0 | 高 (chunk 数膨大) | 高 (細粒度) |
| 256 token sliding | ~200 word | 50 token | 中 | 中 |

### 2-2. 採用

- **v1: days 7th 派生 (90 word + 3 sentence overlap)**
  - 理由: Cycle 01 の baseline_proposal で既に検討済、上位陣で頻出、短文 chunk は科学設問の局所事実引用に強い
  - 実装: `nltk.sent_tokenize` で文分割 → 90 word 制限内に詰める → overlap として末尾 3 文を次 chunk の先頭に付与
- **v2 候補**: 256 token sliding（context が長文寄りの設問で v1 が弱ければ）

---

## 3. Embedder 選定

### 3-1. 候補（MTEB benchmark + Kaggle 慣行）

| Embedder | dim | retrieval MTEB | サイズ | inference 速度 | 備考 |
|---|---|---|---|---|---|
| `intfloat/e5-base-v2` | 768 | ~50 | 110M | 高 | 上位 Kaggler 多数採用 |
| `intfloat/e5-large-v2` | 1024 | ~52 | 335M | 中 | base より +0.5-1.0pt |
| `BAAI/bge-base-en-v1.5` | 768 | ~53 | 110M | 高 | 2024 ベンチで強い |
| `BAAI/bge-large-en-v1.5` | 1024 | ~54 | 335M | 中 | large の有力 |
| `thenlper/gte-base` | 768 | ~50 | 110M | 高 | days 7th が採用 |
| `BAAI/bge-m3` | 1024 | dense+sparse 両対応 | 568M | 中 | 多機能、Cycle 05 想定 |

### 3-2. 採用

- **v1: `BAAI/bge-base-en-v1.5`**
  - 理由: 2024+ ベンチで e5-base 同等以上、サイズも同等、Cycle 01 完了時点の SOTA に近い base モデル
- **v2 候補**: `BAAI/bge-large-en-v1.5`（query 数 × 計算量が足りれば）, `intfloat/e5-large-v2`
- HF mirror DL 帯域問題は別タスクで解決中（並行作業）

### 3-3. クエリ整形

- bge / e5 ともに **query prefix が必要**。
  - bge: query には `"Represent this sentence for searching relevant passages: " + q` を付ける（v1.5 系で推奨）
  - e5: `"query: " + q` / `"passage: " + p` の prefix を付ける
- query 設計: `f"{question} {choice_A} {choice_B} {choice_C} {choice_D} {choice_E}"`（5 択全部を query に混ぜる）
  - 代替: question のみで検索、選択肢ごとに別 query — Cycle 02 v2 で実験

---

## 4. Index 構築

### 4-1. 候補

| Index | 構築時間 | 検索時間 (top-5) | 精度 | メモリ |
|---|---|---|---|---|
| FAISS `IndexFlatIP` (brute force) | 0 | 大（N に線形） | 100% | dim × N × 4byte |
| FAISS `IVF1, PQ` | 中 | 小 | ~95-99% | ~1/8 of Flat |
| FAISS `HNSW` | 中 | 小 | ~99% | 中 |
| FAISS GPU index | 速 | 速 | 99% | GPU メモリ依存 |

### 4-2. 採用

- **v1: FAISS `IndexFlatIP`** (STEM filter で N ≈ 数百万 chunks, dim 768 → メモリ 5-10GB)
  - 理由: 構築 0 秒、精度 100%、retrieval pipeline の初期段階では recall を最大化したい
  - inner product (cosine 相当、bge/e5 とも cosine 想定)
- **v2: IVF or HNSW**（v1 で N が 1000 万を超えて重くなったら切替）

### 4-3. 永続化

- `data/tmp/faiss/cycle02_v1.index` に保存（gitignore 済）
- Kaggle Notebook に乗せる時は Kaggle Dataset としてアップロード（index + chunk text + id mapping 一式）

---

## 5. Context 注入方式

### 5-1. 候補

| 方式 | 形式 | MCQ への影響 | コメント |
|---|---|---|---|
| **prefix concat** | `"[CTX] {top5_concat}\n\n[Q] {q}\n[A] {opt}"` を 5 並列で MCQ に渡す | 統一 prefix で 5 回入力 → 5 ロジット | days 7th 採用、安定 |
| per-option context | 選択肢ごとに専用 retrieval を走らせ、各 (q, opt) ペアに別 context | 計算 5×、精度上限高い | Cycle 02 v2 候補 |
| context-as-separate | DeBERTa の token_type で context 区別 | tokenizer 拡張が必要 | 効果不明、見送り |
| chain-of-thought CoT prefix | "Let's think..." で LLM 風 prefix | 効果不安定 | 見送り |

### 5-2. 採用

- **v1: prefix concat (1 retrieval × 5 choice MCQ)**
  - tokenizer 例:
    ```python
    enc = tokenizer(
        [f"[CTX] {ctx}\n\n[Q] {q}"] * 5,
        [str(opt) for opt in [A, B, C, D, E]],
        truncation=True, max_length=512, padding="max_length",
    )
    ```
  - max_length=512 は context 余裕分（Cycle 01 は 384、本 Cycle は context 入る分拡大）
- **v2 候補**: per-option context（精度上限を取りに行く Cycle 02 v2）

---

## 6. 評価プロトコル

### 6-1. 内部 (ローカル) 評価

| 指標 | 計算方法 | 用途 |
|---|---|---|
| **retrieval recall@5** | top-5 chunk が「正答に必要な情報を含むか」を 50 問サンプリングで人手判定 | retrieval 単体の質を測る |
| **val MAP@3 (context あり)** | 80/20 split で MCQ head fine-tune → 評価 | E2E の質 |
| **val MAP@3 (context なし)** | 同 split で context 抜きで学習・評価 | retrieval Δ の測定 |

### 6-2. Kaggle 提出評価

- **Public LB**: retrieval あり vs なしの 2 提出
- **Private LB**: 結果が出てから retrieval Δ を確定

### 6-3. ablation 計画

| ablation | 比較対象 | 目的 |
|---|---|---|
| (a) retrieval あり vs なし | v1 ↔ Cycle 01 m2 再現 | A 単独の Δ |
| (b) cirrussearch vs jjinho (v2) | 別 dump で v1 を再走 | B 単独の Δ |
| (c) bge-base vs e5-base (v2) | embedder 入替え | embedder の感度 |
| (d) top-3 vs top-5 vs top-10 (v2) | retrieval 数を sweep | recall/precision tradeoff |

---

## 7. v1 構成サマリ（採用）

| layer | 採用 |
|---|---|
| corpus | cirrussearch enwiki 1 か月 dump, STEM filter |
| chunker | 90 word + 3 sentence overlap |
| embedder | `BAAI/bge-base-en-v1.5` (query prefix 付き) |
| index | FAISS IndexFlatIP, dim 768 |
| retrieval | top-5 chunks per row, query = question + 5 choices concat |
| context 注入 | prefix concat 1 retrieval × 5 choice |
| MCQ backbone | `OpenAssistant/reward-model-deberta-v3-large-v2` (Cycle 01 m2) |
| max_length | train 384 / infer 512 |
| epochs | 3 |
| TTA / ensemble | なし |
| ablation | retrieval あり/なしの 2 サブ |

## 8. ベスト案 考察（retrieval 単体）

- 上記 v1 は「**retrieval pipeline の最小 working setup**」を志向。各 layer で v2 余地を残しているのは意図的で、Cycle 02 が回り終えてから「どこを伸ばすと最も効くか」を切り分けやすくするため。
- 上位陣 (0.92+) との残差を埋めるには **v2 で top-k 増、re-ranker 追加、per-option context** あたりが効くと推測。これは Cycle 02 v2 / Cycle 03 / Cycle 04 に分割する。
- **v1 で目指す LB**: 0.78-0.85 (m2 0.68 + retrieval +0.10-0.17)。これが達成できれば Cycle 02 のスコープは成功。
- 達成できなかった場合の打ち手:
  1. retrieval recall@5 を測る → 50% 未満なら corpus / embedder を見直す（chunk が短すぎるか、STEM filter が厳しすぎる）
  2. ablation で context あり/なしの Δ が小さい (<0.05) なら、context 注入方式 (5-1) を per-option context へ
  3. 全部試した結果改善しない場合は backbone 多様化 (Cycle 03) に進む

## 関連

- [[cycle02_4_model_plan]] — 02 の全体方針 (本案の上位)
- [[dataset_improvements]] — cirrussearch dump 詳細 (本案の corpus 層)
- `knowledge/01/baseline_proposal.md` — days 7th 派生のもう少しコンパクトな retrieval 設計
- `knowledge/01/ensemble_methods.md` — Cycle 03 で再利用する mean+max ensemble
- `knowledge/search/01_rag_top_solutions_survey.md` — 上位陣 RAG パターン調査
- `knowledge/search/02_rag_accuracy_quantitative_impact.md` — retrieval 改善幅の定量データ
- `knowledge/search/03_rag_why_it_works.md` — retrieval がなぜ効くかの理論
- `knowledge/search/04_rag_build_and_iterate.md` — RAG 構築のイテレーション方法

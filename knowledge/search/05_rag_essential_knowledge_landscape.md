# 05 RAG 関連で知っておくべき知識 — 全体地図

> ⚠️ **本ファイルは Claude が Web 調査して作成**したメモ。`knowledge/search/` は通常ユーザー手動メモだが、本回はユーザー指示により Claude が代行。
>
> 元の発問: 「RAG関連で知っておくべき知識はある？それを新しいファイルにまとめて。」
>
> 守備範囲: これまでの 01-04 で扱いきれなかった **RAG 全般の概念地図** (基礎用語、派生形、失敗モード、評価ツール、最新動向)。**辞書 + 地図** として参照することを想定。各項目は 1-2 段落で概要を押さえ、深く知りたい時は外部リンクへ。
>
> 関連:
> - 上位解法 → [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)
> - 数値根拠 → [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)
> - なぜ効くか → [`03_rag_why_it_works.md`](03_rag_why_it_works.md)
> - 作り方・改善 → [`04_rag_build_and_iterate.md`](04_rag_build_and_iterate.md)
> - LLM への入り方・フォーマット・取捨選択 → [`06_how_llm_consumes_rag.md`](06_how_llm_consumes_rag.md)

---

## §0. 全体マップ — どこに何があるか

```
┌─────────────────────────────────────────────────────────────────┐
│              RAG パイプラインの構成要素                          │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  Index 構築   │ Query 加工    │  Retrieve    │  Generate         │
│   (offline)   │  (online)     │  + Rerank    │   (online)        │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ §2 Chunking  │ §4 Query     │ §3 Embedding │ §6 Variants       │
│ §5 Vector DB │ rewriting    │ §3 BM25/Dense│ (Self-RAG等)      │
│              │ (HyDE,       │ §3 RRF       │ §7 Failure modes  │
│              │  Multi-Q)    │ §3 Reranker  │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
                          ↓ 全体を横断
            §8 Evaluation (RAGAS / 層別評価)
            §9 RAG vs Long-Context 論争
            §10 用語集 (glossary)
```

---

## §1. 必須の基礎概念 6 つ

### 1.1 Parametric vs Non-parametric memory
- LLM 重みに焼き込まれた知識 (パラメトリック) vs 外部 corpus から retrieval する知識 (非パラメトリック)
- RAG は **2 種類のメモリを分業** させる設計思想 ([Lewis et al. 2020](https://arxiv.org/abs/2005.11401))
- 詳細: [`03_rag_why_it_works.md`](03_rag_why_it_works.md) §A

### 1.2 Sparse vs Dense retrieval

| | Sparse (BM25 / TF-IDF) | Dense (embedding) |
|---|---|---|
| 表現 | 単語のスパースベクトル | 数百-数千次元の連続ベクトル |
| 一致 | **lexical (語彙) 一致** | **semantic (意味) 類似** |
| 強い | 固有名詞、希少語、term match | 言い換え、同義表現 |
| 弱い | 言い換え (vocabulary mismatch) | OOV 数値・記号 |
| index | inverted index (Lucene) | ANN index (FAISS, HNSW) |

→ 本コンペ上位陣は **両方使う** (hybrid)。詳細は [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md) §3。

### 1.3 Bi-encoder vs Cross-encoder

| | Bi-encoder | Cross-encoder |
|---|---|---|
| 構造 | query と doc を独立に encode → 内積 / cosine | query+doc を 1 つの input にして scoring |
| 速度 | **超高速** (事前 encode 可能) | **遅い** (毎回 forward) |
| 精度 | 中 | **高** (相互作用を見る) |
| 用途 | **retrieval (top-100 など大量検索)** | **reranking (top-100 → top-10)** |

→ 「bi-encoder で大量に絞る → cross-encoder で精緻に並べ直す」がカスケードの定石。本コンペ 2nd / 4th 位の DeBERTa reranker は cross-encoder。

### 1.4 Chunk

- 元文書を retrieval 単位 (通常 100-1000 tokens) に分割した断片
- chunk の **粒度** と **境界** が retrieval 品質を直接決める (§2)

### 1.5 Embedding

- テキストを固定長ベクトルに変換したもの
- 同じ意味のテキストはベクトル空間で近接 (cosine 類似)
- 選定は MTEB leaderboard が事実上の標準 (§3)

### 1.6 Reranker

- retrieval で得た top-k を、より精密なモデルで再順位付けする層
- 通常は cross-encoder (DeBERTa, MonoT5, bge-reranker など)
- **本コンペ上位陣の必須要素** (上位 6 中 4 チームが明示採用)

---

## §2. Chunking 戦略 — 6 種類とその使い分け

| 戦略 | 何をする | 適性 | 注意 |
|---|---|---|---|
| **Fixed-size** | N tokens / N chars で機械的に切る (overlap あり) | プロトタイプ、文書が均質 | 文の途中で切れる |
| **Sentence / Paragraph** | 自然境界で切る | **文章型 corpus 全般**、本コンペ 2nd 位採用 | 短い文 / 長い段落で粒度バラつき |
| **Recursive** | 構造単位 (\n\n → \n → . → " ") で再帰分割 | LangChain の default、heterogeneous 文書 | 推奨スタート点 |
| **Semantic** | embedding 類似度で境界を決定 | 同一トピックがまとまる | **NAACL 2025 で fixed 200-word に負けると報告**、計算コスト見合わない場面多い |
| **Parent-Document (small-to-large)** | 子 chunk で retrieve → 親 chunk を context に渡す | 短い chunk で精度、長い context で生成 | 実装複雑 |
| **Late Chunking** | 全文を embed してから chunk 単位の vector を取り出す | header / 代名詞解決が必要な文書 | 2024 後発、長文 embedder 必要 |

→ **本コンペでは sentence + overlapping window が定番**。Recursive も無難。Semantic chunking は ROI 低い (上記論文)。

参考: [Chunking Strategies for RAG (SurePrompts)](https://sureprompts.com/blog/chunking-strategies-for-rag)、[Best Chunking Strategies for RAG 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)

---

## §3. Retrieval の中身 — Embedding / BM25 / Fusion / Rerank

### 3.1 Embedding model の landscape (MTEB 2025-2026)

MTEB ([Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)) が事実上の標準ベンチマーク。

| Tier | 代表モデル | サイズ | 特徴 |
|---|---|---|---|
| **Top (closed)** | Gemini Embedding 001, OpenAI text-embedding-3-large | API | 最高精度、API コスト |
| **Top (open, large)** | NV-Embed (NVIDIA) ~7B, Qwen3-Embedding-8B, QZhou-Embedding | 7-8B | GPU 必須、ローカル可 |
| **Mid (open, medium)** | bge-m3, e5-large-v2, gte-large | 300M-1B | バランス良 |
| **Small (open)** | **bge-small-en-v1.5, gte-small, e5-small, all-MiniLM-L6-v2** | 22-100M | **Kaggle 9h 制約に最適** |

**本コンペでの実績**: gte-small が最良 (Lizhecheng02 ablation: 0.851 vs bge-small 0.822 [MAP@3])。
→ 「大きい embedder が常に良いとは限らない」: タスク domain + chunk 長との相性が重要。

### 3.2 BM25 — sparse retrieval の標準

- 1994 年 Robertson et al. の改良 TF-IDF
- スコア式: `BM25(q, d) = Σ IDF(qi) · (f(qi,d)·(k1+1)) / (f(qi,d) + k1·(1-b+b·|d|/avgdl))`
- 典型値: `k1=1.2, b=0.75`
- 実装: pyserini (Apache Lucene wrapper), rank_bm25 (pure Python), Elasticsearch / OpenSearch
- **強い**: 固有名詞、希少語、term-overlap な query
- **弱い**: 言い換え (the cat sat / the feline rested は別物扱い)

### 3.3 Hybrid retrieval — Reciprocal Rank Fusion (RRF)

複数 retriever の結果を統合する **業界デフォルト** 手法。

**式**: `RRF_score(d) = Σ 1 / (k + rank_i(d))`

- `k = 60` が経験的デフォルト
- **score の正規化不要** (rank だけ使う) — BM25 score (unbounded) と cosine (-1〜1) を直接比較する罠を回避
- **OpenSearch / Elasticsearch / Azure AI Search / MongoDB Atlas / Weaviate のデフォルト hybrid 手法**

```python
def rrf(rankings_list, k=60):
    scores = {}
    for ranks in rankings_list:  # 各 retriever の結果 (doc_id 順)
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

→ 本コンペ Cycle 01 のフル構成で BM25 + Dense を統合する時の第一選択。

### 3.4 Reranker

| 手法 | 種類 | コスト | 精度 |
|---|---|---|---|
| BM25 only | rule-based | 極小 | 中 |
| Edit distance | rule-based (Levenshtein) | 小 | 中 (本コンペ 4th 採用) |
| MonoT5 / MonoBERT | cross-encoder | 中 | 高 |
| **bge-reranker-base / large** | cross-encoder | 中 | **高 (open default)** |
| **DeBERTa v3 reranker** | cross-encoder | 中 | **高 (本コンペ 2nd 採用)** |
| Cohere Rerank | API | 中 (API) | 最高クラス |

→ retriever で top-100 → reranker で top-10 が標準。

---

## §4. Query 側の工夫 — クエリ表現を改善する

ユーザーの query 文字列をそのまま検索するのではなく、retrieval しやすい形に変換する。

| 技法 | 何をするか | 効くケース |
|---|---|---|
| **Query expansion** | synonyms / 関連語を追加して長くする | OOV が多い domain、略語 |
| **HyDE** (Hypothetical Document Embeddings) | LLM に「この質問への回答」を仮生成 → その回答を embed して検索 | query と doc の **表現ギャップ** が大きい時 |
| **Multi-query** | 1 query を LLM で N 個の言い換えに展開 → それぞれ検索 → 統合 | 短い / 曖昧な query |
| **Step-back prompting** | LLM に「より抽象的な親概念の質問」を生成させ、両方で検索 | 階層的知識が必要な query |
| **Sub-question decomposition** | multi-hop query を sub-query に分解 | 複合質問 (本コンペでは稀) |

→ **本コンペでは MCQ なので query = question + 5 options。query 側の余地は中程度** だが、option を query に混ぜることで HyDE 的効果は得られる (上位陣は option も retrieval query に含める)。

参考: [Refining RAG: Advanced Query Strategies (Towards AI)](https://pub.towardsai.net/refining-rag-advanced-query-strategies-prompt-mastery-and-precise-evaluation-2ed031be920e)

---

## §5. Vector DB / Index の選択肢

> 本コンペは Code Competition (internet OFF) なので **vector DB サーバーは使えない**。FAISS / 自前 matmul が現実解。ただし一般知識として整理。

| DB / Lib | 種類 | プロダクション規模 | 特徴 |
|---|---|---|---|
| **FAISS** (Meta) | library | 大 | GPU 対応、Kaggle 標準。**本プロジェクトでの第一選択** |
| **Annoy** (Spotify) | library | 中 | tree-based、read-only 寄り |
| **HNSWlib** | library | 大 | graph-based、SOTA レベルの ANN |
| **ChromaDB** | embedded DB | 小-中 | プロト向け、~50M vectors で頭打ち |
| **pgvector** | PostgreSQL ext | 中 | RDB と統合、moderate 規模に最適 |
| **Qdrant** | server | 大 | Rust 実装、低遅延 |
| **Milvus / Zilliz** | server | 超大 | エンタープライズ規模 |
| **Pinecone** | managed | 大 | SaaS、楽だが課金 |
| **Weaviate** | server | 大 | hybrid 標準装備 |

→ **本コンペ用には: 1st place の "PyTorch matmul 一発" or FAISS IndexFlatIP**。ANN すら不要なケース (chunk 数 ~6M、embedding 384 次元なら GPU で全件 matmul で間に合う)。

参考: [Vector Stores for RAG Comparison 2025 (Glukhov)](https://www.glukhov.org/post/2025/12/vector-stores-for-rag-comparison/)

---

## §6. RAG の派生形 — 進化系統樹

2024-2025 にかけて多くの派生形が登場。本コンペでは vanilla RAG で十分だが、知識として:

| 名前 | 何が違うか | 効くシーン |
|---|---|---|
| **Vanilla RAG** | retrieve → 結合 → 生成 | 本コンペ標準 |
| **Self-RAG** | LLM が「retrieve すべきか」「自己批評で出力を採点」を学習 | 設問により retrieval 不要を判断 |
| **CRAG (Corrective RAG)** | retrieval 品質を別モデルで評価、低品質なら web 検索 / 訂正 | corpus に答えが無いケース対応 |
| **Adaptive RAG** | query 複雑度に応じて strategy を切替 | 単純 / 複雑が混在するタスク |
| **HyDE-based RAG** | query → 仮回答 → 仮回答で検索 (§4 参照) | query-doc 表現ギャップ大 |
| **Multi-query RAG** | query を N 言い換え → 各々で retrieve → fusion | 短く曖昧な query |
| **GraphRAG** (Microsoft) | corpus を knowledge graph 化し、graph 上で retrieval | 関係性中心のクエリ、要約 |
| **FLARE** (Forward-Looking Active Retrieval) | 生成途中で「次の文に retrieval 必要か」を判断 | 長文生成 (本コンペは MCQ なので不要) |
| **LongRAG** | chunk を巨大 (4K-8K tokens) に、retrieved unit を少数に | long-context LLM 前提 |
| **Agentic RAG** | RAG を tool として LLM agent が使う、複数 retrieval 反復 | 複雑な multi-step QA |
| **RAFT (Retrieval-Augmented Fine-Tuning)** | retrieval を訓練時にも混ぜる + distractor を入れて頑健化 | 本コンペ上位陣の発想に近い |

→ **本コンペ向け**: Vanilla RAG + (任意で) HyDE / Multi-query / RAFT 流の訓練。Self-RAG / GraphRAG / Agentic は ROI が低い。

参考: [Beyond Vanilla RAG: 7 Modern RAG Architectures (DEV)](https://dev.to/naresh_007/beyond-vanilla-rag-the-7-modern-rag-architectures-every-ai-engineer-must-know-4l0c)、[Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/html/2501.09136v4)

---

## §7. 既知の失敗モード — 知っておかないと事故る

### 7.1 Lost in the Middle ([Liu et al. 2024, TACL](https://aclanthology.org/2024.tacl-1.9.pdf))

- LLM は **context の先頭と末尾に注意が偏り、中央の情報を取りこぼす**
- U 字型の精度曲線: 同じ relevant 情報でも先頭 / 末尾なら使えるが、中央だと無視される
- **対策**: 
  - retrieved chunk のうち **最重要を先頭または末尾に置く**
  - context を短くする (§7.2)
  - "lost in the middle" を直す独自手法 (Found in the Middle 等) は研究段階

### 7.2 Long-context degradation ([arXiv 2510.05381](https://arxiv.org/abs/2510.05381))

- **retrieval が完璧でも** context が長いだけで 13.9-85% の精度低下
- 「relevant token だけマスクして注意を絞らせても」劣化が起きる → LLM 自身の限界
- **対策**: minimum sufficient context、不要な chunk は捨てる、reranker を必ず通す

### 7.3 Context distraction / noise sensitivity ([Yoran et al. 2024 ほか](https://arxiv.org/pdf/2601.07226))

- retrieved context に irrelevant or 矛盾文書が混ざると **base 精度より悪化**することがある
- NoisyBench (2025) では SOTA 推論モデル (Gemini 2.5 Pro) で **最大 80% の精度低下**を観測
- **対策**: 
  - reranker を必ず入れる
  - retrieved 件数を絞る (top-3〜5 程度)
  - 訓練時に hard negatives (似ているが間違いの distractor) を混ぜて頑健化 (RAFT 流)

### 7.4 Retrieval index / chunk text のズレ

- chunk 化を途中で再起動して embedding と text の対応が崩れる古典バグ
- 症状: 検索結果が無関係、PCA で outlier
- **対策**: index 構築後に [chunk_id, embedding, text] を一括 dump して visual sanity check

### 7.5 Stale corpus

- corpus の dump 日付以降の情報は答えられない
- **対策**: corpus version を実験 ID に紐付けて記録、定期更新

### 7.6 RAG data poisoning / leakage

- 攻撃者が corpus に悪意ある文書を混入させると **5 docs で 90% trigger 成功**などの報告 (PoisonedRAG)
- 本コンペは公開 Wikipedia なので問題ないが、production では脅威
- **対策**: corpus の origin 検証、document-level access control

---

## §8. 評価フレームワーク

### 8.1 RAGAS ([arXiv 2309.15217](https://arxiv.org/abs/2309.15217))

reference-free に RAG を評価する **業界事実上標準**。LLM-as-a-judge ベース。

| メトリクス | 何を測る | 計算方法 |
|---|---|---|
| **Faithfulness** | 生成回答が retrieved context に **根拠付けられているか** | claim を抽出 → 各 claim が context にサポートされる率 |
| **Answer Relevancy** | 回答が **質問に答えているか** | 回答から逆向きに質問を生成し、元 query との類似度 |
| **Context Precision** | 上位 chunk のうち relevant な割合 | LLM で各 chunk を relevance 判定 |
| **Context Recall** | 必要な情報のうち retrieve できた割合 | ground-truth answer → 各文の retrievability 判定 |

→ **本コンペには直接適用しにくい** (MCQ 評価で十分) が、retrieval 品質単体の診断には有用。

### 8.2 他の評価ツール

- **TruLens**: faithfulness / groundedness の trace
- **ARES**: 自動 + 人手 hybrid 評価
- **DeepEval**: pytest 風の RAG テスト framework
- **MTEB Retrieval task**: embedding 単体の retrieval 精度 (nDCG@10)

### 8.3 本プロジェクト向け推奨

- **層別評価** (recall@k / MRR / MAP@3) を必須実装 ([`04_rag_build_and_iterate.md`](04_rag_build_and_iterate.md) §6)
- **failure mode 4 分類** (retrieval miss / reranker miss / reader miss / 問題困難) で誤答ログ
- RAGAS は overkill、Cycle 後半で導入するか判断

---

## §9. Long-context LLM vs RAG — "RAG is dead?" 論争

2024 後半以降、Gemini 2.5 Pro / Claude 4 Opus / GPT-4.1 が **1M tokens** に到達し「もう RAG いらないのでは」議論が活発。

### 9.1 当事者の主張

| | Long-context 派 | RAG 派 |
|---|---|---|
| 主張 | 1M tokens あれば corpus 全部入る | corpus は 1M tokens じゃ足りない (Wikipedia 全体は数十 B tokens) |
| Cost | API 料金が token 数に比例 → 高い | retrieve した分しか送らない → 安い |
| Latency | 数十秒〜分 | ~1 秒 |
| 更新 | 毎回 corpus 全部送り直し | corpus 更新は index 差替えだけ |
| 精度 | "needle in haystack" は良い、推論は劣化 (§7.2) | retrieval 失敗で天井あるが安定 |

### 9.2 現実的な結論

- **小さい / 静的な corpus** → long-context が楽
- **大きい / 動的な corpus** → RAG が依然優位
- **コスト / latency 制約あり** → RAG
- **本コンペ**: corpus = Wikipedia (10s of GB) なので **RAG 一択**。1M tokens に入らない。

参考: [RAG vs Long-Context LLMs (Databricks Blog)](https://www.databricks.com/blog/long-context-rag-performance-llms)、[Stop Chasing Million-Token Context Windows (Medium)](https://medium.com/@reliabledataengineering/stop-chasing-million-token-context-windows-youre-solving-the-wrong-problem-696b8ba881d7)

---

## §10. 用語集 (Glossary)

> アルファベット順。本コンペで使う頻度の高いものから。

- **ANN (Approximate Nearest Neighbor)**: 近似最近傍探索。FAISS / HNSW / Annoy 等。
- **BM25**: 確率モデルベースの sparse retrieval 標準。
- **Bi-encoder**: query / doc を独立 encode する embedder。retrieval 用。
- **Chunk**: retrieval の単位。100-1000 tokens が一般的。
- **Closed-book / Open-book**: 外部資料なし / あり。RAG は open-book QA に該当。
- **Cosine similarity**: ベクトル間の角度。dense retrieval の標準距離。
- **Cross-encoder**: query+doc を 1 input にして scoring。reranker 用。
- **Dense retrieval**: embedding ベースの retrieval。semantic 一致に強い。
- **Distractor**: MCQ の誤答選択肢 / retrieval で誤って混入する irrelevant 文書。
- **Embedding**: テキストの固定長ベクトル表現。
- **FAISS**: Meta 製の ANN ライブラリ。本コンペ標準。
- **Grounding**: 出力を具体的 source に紐付けること。hallucination 抑制の鍵。
- **Hallucination**: モデルがそれらしく見えるが事実でない出力をすること。
- **Hard negative**: retrieval / 分類で「正解に近いが間違い」の難しい否定例。訓練に混ぜると頑健化。
- **HNSW (Hierarchical Navigable Small World)**: graph-based ANN。SOTA レベル。
- **HyDE**: 仮回答を生成して embed → 検索する手法。
- **Hybrid retrieval**: sparse + dense の併用。
- **Index**: retrieval の事前計算データ構造 (inverted index, ANN index)。
- **In-context learning**: prompt 内の例で学習せず推論する能力。
- **Knowledge cutoff**: モデル学習データの最新時点。
- **Long-tail knowledge**: 出現頻度が低い事実。RAG が特に効く領域。
- **MAP@K (Mean Average Precision @ K)**: 本コンペの評価指標。
- **MCQ (Multiple Choice Question)**: 多肢選択問題。本コンペの形式。
- **MTEB**: 標準 embedding ベンチマーク。
- **Non-parametric memory**: 外部 corpus に保持された知識。
- **Parametric memory**: モデル重みに焼き込まれた知識。
- **Pooling**: token embeddings から文 embedding を作る集約方法 (mean / max / cls)。
- **Pyserini**: Anserini (Lucene) の Python wrapper、BM25 標準実装。
- **Query expansion**: クエリに関連語を足して検索改善。
- **RAG**: Retrieval-Augmented Generation。本ファイルの主題。
- **RAGAS**: RAG 評価フレームワーク。
- **Recall@K**: 正解が top-K に含まれる率。retriever の標準メトリクス。
- **Reranker**: retrieved top-N を再順位付けする層。通常 cross-encoder。
- **RRF (Reciprocal Rank Fusion)**: hybrid 検索の標準統合手法。`1/(k+rank)` を sum。
- **Sentence Transformers**: 文 embedding の標準ライブラリ。
- **Sparse retrieval**: BM25 / TF-IDF 系の term-based 検索。
- **TF-IDF**: 古典的な term weighting。BM25 の前身。
- **Top-K**: retrieval で取得する文書数。
- **Vector DB**: embedding を格納・検索するデータベース。

---

## §11. 本プロジェクトで採用 / 不採用の判断

| 概念 | 採用 (Cycle 01) | 採用 (将来 Cycle 候補) | 不採用 |
|---|---|---|---|
| BM25 (pyserini) | ✓ | | |
| Dense embedding (gte-small) | ✓ | | |
| Hybrid + RRF | | ✓ (Cycle 01 フル) | |
| Cross-encoder reranker (DeBERTa) | | ✓ (Cycle 01 フル) | |
| Sentence-level chunking | ✓ | | |
| Parent-document retrieval | | ✓ (Cycle 03+ 候補) | |
| HyDE / Multi-query | | ✓ (Cycle 03+ 候補) | |
| FAISS (or PyTorch matmul) | ✓ | | |
| Self-RAG / CRAG / Agentic RAG | | | × (overkill) |
| GraphRAG | | | × (corpus 性質と不一致) |
| RAGAS | | ✓ (診断用に Cycle 02+) | |
| Long-context (1M tokens) で RAG 置換 | | | × (Wikipedia は入らない) |

---

## 出典 (信頼度ランク)

**Tier 1 (一次論文・公式ドキュメント)**
- [Lewis et al. 2020 — RAG (arXiv 2005.11401)](https://arxiv.org/abs/2005.11401)
- [Liu et al. 2024 — Lost in the Middle (TACL)](https://aclanthology.org/2024.tacl-1.9.pdf)
- [Context Length Alone Hurts (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381)
- [RAGAS paper (arXiv 2309.15217)](https://arxiv.org/abs/2309.15217)
- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/html/2501.09136v4)
- [MTEB Leaderboard (HuggingFace)](https://huggingface.co/spaces/mteb/leaderboard)

**Tier 2 (まとめ記事 / 業界ブログ)**
- [Beyond Vanilla RAG (DEV Community)](https://dev.to/naresh_007/beyond-vanilla-rag-the-7-modern-rag-architectures-every-ai-engineer-must-know-4l0c)
- [Chunking Strategies for RAG (SurePrompts)](https://sureprompts.com/blog/chunking-strategies-for-rag)
- [Best Chunking Strategies 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Reciprocal Rank Fusion Explained (BigData Boutique)](https://bigdataboutique.com/blog/reciprocal-rank-fusion-how-it-works-and-when-to-use-it)
- [Vector Stores for RAG Comparison 2025 (Glukhov)](https://www.glukhov.org/post/2025/12/vector-stores-for-rag-comparison/)
- [Long Context RAG (Databricks)](https://www.databricks.com/blog/long-context-rag-performance-llms)
- [Refining RAG: Advanced Query Strategies (Towards AI)](https://pub.towardsai.net/refining-rag-advanced-query-strategies-prompt-mastery-and-precise-evaluation-2ed031be920e)

**プロジェクト内クロスリファレンス**
- [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)
- [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)
- [`03_rag_why_it_works.md`](03_rag_why_it_works.md)
- [`04_rag_build_and_iterate.md`](04_rag_build_and_iterate.md)

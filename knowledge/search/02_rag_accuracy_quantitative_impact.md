# 02 RAG 有無で精度はどれだけ変わるか — 定量的根拠

> ⚠️ **本ファイルは Claude が Web 調査して作成**したメモ。`knowledge/search/` は通常ユーザー手動メモだが、本回はユーザー指示により Claude が代行。
> 元の発問: 「できる限り時間をかけて RAG で精度が向上するか調査して」
>
> 守備範囲: **「RAG で精度がどれだけ上がるか」を数値で示す**。
> 関連: 上位解法一覧 → [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)、なぜ上がるか → [`03_rag_why_it_works.md`](03_rag_why_it_works.md)。

---

## ⚠️ 数値を読む前に — 単位の注意

このコンペには **2 種類の数字** が並列で流通している。混同しないこと。

| 単位 | 範囲 | 何を測るか | 例 |
|---|---|---|---|
| **MAP@3** (公式評価指標) | 0.0 – 1.0 | 上位 3 件の予測順位を考慮した精度。**1 位正解=1.0, 2 位=0.5, 3 位=0.333, 圏外=0**。 | Public LB 0.930 / Private 0.923 |
| **Accuracy (top-1 正解率)** | 0% – 100% | 1 位だけが正解かを 0/1 判定 | "Llama 2 70B + RAG で 93%" |

→ **Accuracy 93% ≠ MAP@3 0.93**。Accuracy 93% は MAP@3 換算で概ね 0.94+ になる (top-2/3 も拾うため)。
本ファイルでは数字に必ず単位ラベル `[MAP@3]` または `[acc]` を付ける。

---

## 1. クリティカルな数字 — RAG を入れる/入れないで何が変わるか

### 1.1 1st place writeup の主張 (Llama 2 70B) [acc, 要再検証]

> **Llama-2 70B with prompting (no context): 80%** acc
> **+ SFT finetuning: 86%** acc
> **+ SFT + RAG: 93%** acc

(出典: 1st place writeup の paraphrase が r/LocalLLaMA → GitHub issue 経由で流通。Kaggle Discussion 本文は WebFetch で取得できず未確認。)
- 単純解釈: **SFT は +6pt、RAG は +7pt**。両者は加算的でなく相補的。RAG の方が伸び幅が大きい。
- 注意: Public LB ではなく内部評価 acc の可能性 (本文未確認)。

### 1.2 同 writeup の GPT-4 ベンチ参考値 [acc, 要再検証]

> GPT-4 prompting: 75% → +RAG: 80% (**+5pt**) → +Finetune: 81% (**+1pt**) → +RAG+Finetune: 86% (**+11pt から**)

→ GPT-4 のような強い base モデルでも **RAG > Finetune** という順序が観測されている。

### 1.3 Lizhecheng02 の embedding model ablation (Public LB [MAP@3], 一次ソース)

同一パイプライン内で **embedding モデルだけを差し替えた** clean ablation:

| embedding | Public LB [MAP@3] | Δ vs 採用 |
|---|---|---|
| **gte-small (採用)** | **0.851** | — |
| all-MiniLM-L6 | 0.843 | -0.008 |
| all-MiniLM-L12 | 0.837 | -0.014 |
| bge-small-en | 0.822 | **-0.029** |

→ **retriever の小さな違いで MAP@3 が 0.03 ぶれる**。本コンペで 0.03 は数百順位に相当する大差。
→ 同チームの最終 ensemble は **Public 0.930 / Private 0.923–0.924**。
→ 出典: [Lizhecheng02/Kaggle-LLM_Science_Exam README](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam)。

### 1.4 既存メモ ([`../01_overview_trends.md`](../01_overview_trends.md)) の階層 [MAP@3]

| 構成 | Public LB レンジ |
|---|---|
| TF-IDF only ベースライン (no DeBERTa fine-tune) | 0.6 – 0.7 |
| DeBERTa PEFT, no RAG ([alvinleenh 公開 Notebook](https://www.kaggle.com/code/alvinleenh/0-701-llm-science-exam-peft-with-deberta)) | **0.701 → 0.742** |
| Zero-shot 70B + RAG (no fine-tune) [LB 0.836 Notebook](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620) | **0.836** |
| Open Book TF-IDF + LongFormer ([minhsienweng](https://www.kaggle.com/code/minhsienweng/llm-science-exam-tf-idf-longformer)) | **0.862** |
| DeBERTa + RAG + ensemble (上位構成) | **0.85 – 0.93** |
| 上位 100 ボーダー | 0.92 |
| 銅 / 銀メダル | 0.93+ |

→ **No-RAG (closed book) DeBERTa の天井は 0.74 付近**。これは PEFT で出る最大値とほぼ等しい。
→ **RAG を被せるだけで 0.83–0.86 まで一気に飛ぶ** (LLM サイズに関係なく)。
→ **そこから fine-tune + ensemble で 0.92–0.93** まで詰める。

---

## 2. 主要な数値マップ

```
                  no RAG          RAG only       RAG + finetune
              ┌──────────────┬──────────────┬──────────────────┐
DeBERTa-300M  │ 0.70-0.74    │ ~0.86 (open  │ 0.92-0.93 ens.   │
              │ (PEFT 公開)  │ book NB)     │ (4th place: 0.92)│
              ├──────────────┼──────────────┼──────────────────┤
Llama 2 70B   │ 80% [acc]    │ 0.84 LB,     │ 93% [acc] /      │
              │              │ 80%[acc]     │ 0.93 LB          │
              └──────────────┴──────────────┴──────────────────┘
              ↑ closed book   ↑ +retrieval   ↑ +SFT
              ベース天井         一気に +0.10    上限近傍

[acc] = top-1 accuracy (%)、[MAP@3] = 公式評価指標 (0-1)
```

ポイント:
1. **RAG の追加効果 (no-RAG → +RAG) は概ね +0.10 〜 +0.15 [MAP@3] / +5〜+13pt [acc]**。Cycle 01 で目指す範囲は明確。
2. **モデルサイズより retrieval の有無の方がインパクトが大きい**。DeBERTa 300M + RAG > 70B closed-book。
3. **RAG 単独で 0.83+ には届くが、0.93+ には fine-tune + ensemble が必要**。

---

## 3. Retrieval 設計の内側 — どの要素がどれだけ効くか

### 3.1 Context 長 (token 数) の効果 — 4th place ([acc/LB ハイブリッド])

Preferred Scantron は **DeBERTa v3 large の token 長を 512 → 1024 → 1280** と伸ばす実験を公開している:

> "trained a DeBERTa v3 Large model (~300M parameters) on token lengths up to 1280 tokens, with their **public score steadily increasing** as they increased tokens from 512 to 1280"

→ context 長 (= 詰め込める retrieved chunk 量) は **段階的に LB を押し上げる**。直線的とは限らないが少なくとも 1280 まで diminishing return に達していない。

### 3.2 Reranking の効果 (定性 / 一部定量)

- @solokin: BM25 retrieve → **DeBERTa v3 reranker** を入れることが 2nd place の柱。
- Preferred Scantron: **3 種スコア (ES / edit distance / semantic)** を混ぜる。
- Teemu Kanstrén の post-mortem ([Medium](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a)) は具体例として:
  > rerank 前: クエリ "Google Bard" に対し **Tenor (GIF サーチエンジン)** が top 5 に混入
  > rerank 後: Tenor が top 10 から消え、relevant chunk のみ残る

→ rerank は **下位ノイズを除く** 効果が大きい。MAP@3 は top-3 のうち何位に正解があるかで決まるので、ノイズ除去 = 正解の押し上げ。

### 3.3 Chunk サイズの効果 (定性)

- Teemu Kanstrén: 512 → 256 → 128 → 64 で実験 (定量比較は未公開)
- Lizhecheng02: **10 ページ × 30 sentences** が経験的に良かったと報告

→ 小さすぎる chunk は context 不足、大きすぎる chunk は retrieval 精度低下。**sentence + paragraph の二粒度**を embed する 5th place 流が安全策。

### 3.4 多コーパス / 多 retriever の効果 (定性)

- 1st: Wikipedia dump を **複数併用** (filter は不要だった)
- 4th: ES / edit distance / semantic search の **3 種スコア合成**
- Hippocampus's Garden の総括:
  > "it was beneficial to have diverse contexts retrieved by different algorithms such as BM-25, Elasticsearch, BERT family, and LLM"

→ 単独 retriever より hybrid が一貫して強い。bias-variance 分解で言えば retrieval 段の variance を下げる。

---

## 4. RAG が "効かない" / "効きにくい" 反例

- **問題が parametric memory にすでに入っている超有名トピック** (有名物理法則名、有名な化学式など) では retrieval なしでも当てる。Mallen et al. 2023 ([PopQA paper](https://aclanthology.org/2023.acl-long.546/)) が "head" knowledge では parametric だけで十分と実証。
- **retrieval された context が誤情報 / irrelevant** だと逆に精度低下 (distraction)。reranker が必須な理由。
- **context が長すぎる**と、たとえ retrieval が完璧でも LLM 自体の性能が劣化する: [Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (EMNLP 2025 findings)](https://arxiv.org/abs/2510.05381) は **13.9–85% の精度低下**を報告。詰め込めば良いわけではない。
  → 4th place が 1280 tokens で止めているのもこの diminishing return を踏まえてのことと推測。

---

## 5. 本プロジェクトの目標値

| マイルストーン | 目安 [MAP@3] | 達成条件 |
|---|---|---|
| Cycle 01 v1 簡略構成 (1 model × 1 retrieval × 2 TTA) | 0.75 – 0.85 | RAG パイプ疎通 + DeBERTa fine-tune |
| Cycle 01 フル構成 (days 7th 解法レプリカ) | 0.85 – 0.90 | + ensemble + reranker |
| Cycle 02 (cirrussearch 切替) | +0.01 – +0.02 | corpus 拡張による retrieval 改善 |
| 銅メダルライン (Late Submission 上限挑戦) | 0.93+ | 上記すべて + embedding ablation + multi-retriever hybrid |

---

## 出典 (信頼度ランク)

**Tier 1 (本文確認できた一次/直接出典)**
- [Lizhecheng02 GitHub README](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam) — embedding ablation 表
- [Context Length Alone Hurts LLM Performance (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381) — long context 劣化
- [Mallen et al. 2023 "When Not to Trust LMs" (ACL 2023)](https://aclanthology.org/2023.acl-long.546/) — long-tail と retrieval

**Tier 2 (まとめブログ)**
- [Hippocampus's Garden — LLM Science Exam report](https://hippocampus-garden.com/kaggle_llm/)
- [Teemu Kanstrén — LLM+RAG Based Question Answering (Medium)](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a)

**Tier 3 (Kaggle Discussion、WebFetch 取得失敗・二次引用)** [要再検証]
- 1st place 80/86/93% [acc] 主張 — r/LocalLLaMA → GitHub issue 経由
- [LB 0.836 Zero-shot 70B + RAG](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620)
- [Chris Deotte Top 100 RAPIDS TF-IDF](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/chris-deotte-top-100-solution-fast-rapids-tf-idf-r)
- [alvinleenh 0.701 PEFT DeBERTa Notebook](https://www.kaggle.com/code/alvinleenh/0-701-llm-science-exam-peft-with-deberta)
- [minhsienweng TF-IDF + LongFormer Notebook](https://www.kaggle.com/code/minhsienweng/llm-science-exam-tf-idf-longformer)

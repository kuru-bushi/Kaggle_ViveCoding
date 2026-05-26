# 07. その他の注目アプローチ (上位以外の参考解法)

## 10th place — Lizhecheng02 (DeBERTa v3 + 3-pronged retrieval)

### 概要

DeBERTa v3 ×4 checkpoints + Llama 7B のアンサンブル。Public 0.930 / Private 0.923。

### 主要テクニック

- **3 系統の retrieval**:
  - FAISS index で Wikipedia page extraction
  - Cluster-based 拡張 retrieval (270K → ~6M rows)
  - TF-IDF と sentence-transformer のハイブリッド
- **Sentence transformer 比較**:
  - `all-MiniLM-L6` → 0.843
  - `gte-small` → **0.851 ✅ 採用**
  - `bge-small-en` → 0.822
- **Adversarial Weight Perturbation (AWP)** をエポック 0.5 から有効化
- **データ augmentation**: 150K GPT-3.5 + 99K + 60K + 17K MMLU + 70K 公開 = 303K 使用
- 8× A100 で 4 日学習
- Optuna で **ensemble weight 最適化** (500 サンプル検証)

### 教訓

- **複数 sentence-transformer の事前比較** が retrieval 性能差を可視化 (0.022 = 順位 50 位レベルの差)
- AWP は安定して効く（adversarial robustness は MCQ にも有効）
- データ stratification + 多 fold 学習で over-fitting を抑制

### リポジトリ

- [Lizhecheng02/Kaggle-LLM_Science_Exam](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam)

---

## 91st place (Silver) — 490CAD (DeBERTa distillation + 270k context)

### 概要

DeBERTa の知識蒸留 + 270k context aug で銀メダル (top 4%)。

### 主要テクニック

- **`train_deberta_distillation.py`** — 大モデル teacher から蒸留
- **270K context training**: 標準より長い context を意図的に学習
- **多 checkpoint ensemble** (model_0914 / 0920 / 1002 等) + feature combining
- LLaMA 2 7B/13B + Wikipedia LoRA も探索（最終採用は DeBERTa 中心）

### 教訓

- **蒸留** は学習データが少ない時 (200 row 問題) に効きやすい
- 270K context は DeBERTa では希少な事例（普通は 512〜1024）
- 多 checkpoint avg は最も低コストな ensemble

### リポジトリ

- [490CAD/LLM4Science](https://github.com/490CAD/LLM4Science)

---

## 公開 ベースライン Notebook — "Top 100 Solution - Fast RAPIDS TF-IDF RAG"

### 概要

**TF-IDF + RAPIDS GPU 加速** だけで Public LB Top 100 圏内に到達した公開 Notebook。

### 主要テクニック

- RAPIDS (`cuML.TfidfVectorizer`) で Wikipedia を高速ベクトル化
- 2× T4 GPU を並列利用
- 学習不要（pure retrieval + similarity scoring）

### 教訓

- **TF-IDF だけでも 0.85+** に届く設計が存在 = retrieval が支配的なタスク
- 本プロジェクト **Cycle 01 ベースライン** はこのアイデアを CPU-TF-IDF で小型化する

### リンク

- [Top 100 Solution - Fast RAPIDS TF-IDF RAG (Discussion)](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446318)

---

## 50th place — "4 anchors and 1 captain"

### 概要

DeBERTa 系 4 + 別系統 1 のアンサンブル。詳細は writeup 参照。

### リンク

- [50th place writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/4-anchors-and-1-captain-50th-place-solution-for-th)

---

## 共通する公開リソース

| リソース | 用途 |
|---|---|
| `radek1` の 6.5k GPT-3.5 dataset | 1st place を含む多くの上位陣が使用 |
| `cdeotte` の MMLU 整形 | データ多様化 |
| 各種 270K cleaned Wikipedia chunk | RAG コーパス |
| Pyserini / Lucene / Elasticsearch | BM25 backbone |
| MTEB leaderboard 系 embedding | dense retrieval baseline |

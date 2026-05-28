# Cycle 01 — EDA (created: 2026-05-28)

## データソース

| 配置 | 由来 | 行数 | 用途 |
|---|---|---|---|
| `data/train/train.csv` | **Kaggle 公式** (`kaggle competitions download`) | 200 | Cycle 01 v1 学習 |
| `data/test/test.csv` | **Kaggle 公式** (`answer` 列なし) | 200 (visible) | Cycle 01 v1 推論 → submission.csv |
| `data/test/sample_submission.csv` | **Kaggle 公式** | 200 | 提出フォーマット参照 |
| `data/train_augmented/train_hf_g_ronimo.csv` | HF `g-ronimo/kaggle_llm_science_exam` | 5,400 | Cycle 01 v2 以降の augmentation |
| `data/test_augmented/test_hf_g_ronimo.csv` | 同上 (`answer` 列あり) | 600 | Cycle 01 v2 以降の追加 val |

> 注: 公式 test (200) は visible 分のみ。Kaggle Code Competition 形式では private test (~数千行) が hidden で、Notebook 提出経由でのみアクセス可。本プロジェクトの CSV 提出は **Public LB** に対する評価。

## 公式 train.csv 概観

- カラム: `id, prompt, A, B, C, D, E, answer` (8 列)
- 形式: 5 択 (A〜E) の Multiple Choice、`answer` は正解ラベル
- サイズ: **200 行** (極小 — 上位陣は augmentation 必須と判断)

### ラベル分布

| Label | 件数 | 割合 |
|---|---|---|
| A | 37 | 18.5% |
| B | 48 | 24.0% |
| C | 44 | 22.0% |
| D | 38 | 19.0% |
| E | 33 | 16.5% |

→ B/C にやや偏る。ランダム MAP@3 ベースラインは ~0.45。

### サンプル設問 (公式 test の冒頭)

> id=0: "Which of the following statements accurately describes the impact of Modified Newtonian Dynamics (MOND) on the observed 'missing baryonic mass' discrepancy in galaxy clusters?"

> id=1: "Which of the following is an accurate definition of dynamic scaling in self-similar systems?"

→ **物理・数学・自然科学の高度な専門問題**。一般教養レベルではない。Wikipedia の特定記事の内容を知識として保持するモデルでないと厳しい。

## 補助データ (g-ronimo, augmented)

- 5400 train + 600 test (両方 `answer` 付き)
- カラム同一
- 内容: 一般 Wikipedia 系 (人名・地名・組織等の事実問題が多い)
- **公式の専門的科学問題とは設問分布が異なる** → そのまま訓練しても官報科学に転移しない可能性

## 観察された課題

| # | 課題 | 影響 | 対策 (Cycle) |
|---|---|---|---|
| 1 | 公式 train が **200 行のみ** | 過学習リスク大、CV 不安定 | augment (Cycle 01 v2) |
| 2 | 設問が **超専門** で Wiki 知識前提 | retrieval なしでは限界 | Wikipedia retrieval (Cycle 01 v2) |
| 3 | augmented データ (g-ronimo) は分布が違う | 単純 concat は逆効果かも | 慎重に mix (Cycle 02) |
| 4 | 公式 test の private LB は CSV 提出だと不明 | LB 値 = Public のみ | Notebook 提出は将来検討 |

## 解決策候補 (Cycle ロードマップ)

| 案 | 期待効果 (MAP@3) | 担当 Cycle |
|---|---|---|
| Cycle 01 v1: DeBERTa v3 large MCQ on 公式 200 行 (現サイクル) | ~0.50-0.65 | **このサイクル** |
| Cycle 01 v2: Wikipedia retrieval (e5-base + FAISS) を context として注入 | +0.10-0.20 | 直後 |
| Cycle 02: cirrussearch wiki dump へ切替 | +0.02-0.05 | 次 |
| Cycle 03: DeBERTa v3 large 3 種 ensemble + TTA (4 slice) | +0.01-0.03 | 後 |
| augment: g-ronimo + radek1 6.5k 混合学習 | +0.02-0.04 | Cycle 02 と並行 |

## 参考 Notebook / Discussion

- [days 7th place writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/days-7th-place-solution) — 採用ベース手法
- [Hippocampus's Garden competition report](https://hippocampus-garden.com/kaggle_llm/) — 全体傾向
- [g-ronimo HF dataset](https://huggingface.co/datasets/g-ronimo/kaggle_llm_science_exam) — augmentation data
- 関連メモ:
  - `knowledge/01/baseline_proposal.md` — Cycle 01 採用案
  - `knowledge/search/00_ensemble_pipeline_origin_validity.md` — アンサンブルの起源・学術妥当性・上位 5 解法のパイプライン比較（**Cycle 02 以降で多様性軸を追加する根拠**）

## Cycle 01 v1 の判断

- **学習データ**: 公式 train.csv (200 行) のみ
- **検証**: KFold 5 (160/40 split) または --single-fold (smoke)
- **推論**: 公式 test.csv (200 行) → `submission.csv`
- augmented データ (g-ronimo 5400) は Cycle 01 v2 以降で活用

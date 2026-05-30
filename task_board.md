# Task Board — Kaggle LLM Science Exam

> **このファイルは Claude が中断後に作業を再開するための単一の真実の源 (single source of truth)**。
> プロジェクト全体のロードマップ + 現サイクルの詳細計画 + 進捗 + 再開手順を集約する。
> 作業を進めるたびに必ず更新すること（Done に移す、In Progress を更新する）。
> 関連: 詳細規約は `CLAUDE.md`、Cycle 01 採用案は `knowledge/01/baseline_proposal.md`。

---

## 🎯 現在のフォーカス

**Block C — Cycle 01 v1: 3 モデル単独 + ensemble submit 完了 (2026-05-29)**

| # | Model / 構成 | Public LB | Private LB | submission ref |
|---|---|---|---|---|
| m1 | microsoft/deberta-v3-large (plain) | 0.388056 | 0.378399 | 53093495 |
| m2 | OpenAssistant reward-model-deberta-v3-large-v2 | 0.682480 | 0.714337 | 53113415 |
| m3 | deepset/deberta-v3-large-squad2 | 0.592176 | 0.605449 | 53113423 |
| **ens** | **m1+m2+m3 mean+max blending (days 7th)** | **0.684144** | **0.714858** | **53157358** |

- m1 は 2026-05-27、m2/m3 は 05-28、ensemble は 05-29 提出。`kaggle competitions submit -k <kernel> -v 1` 形式。
- OpenAssistant reward が圧勝 (LB +0.29 vs plain)、reward pretraining が MCQ にも有効。
- **ensemble (mean+max) が最良 single (m2) を Public +0.0017 / Private +0.0005 で上回りボード最良**。ほぼランダムの microsoft を含めても max 項が強モデルの確信を保持し引き下げ無し（単純 mean は劣後）。val (mean+max)=0.7958。
- val/LB 相関は強い (順位完全一致)。学び・R1-R5 考察は `report/01_score_report.md` 参照。

**次の打ち手** (Cycle 01 v2 以降):
- Wikipedia retrieval (e5-base + FAISS) を context 注入 → 上限を引き上げる（固有名詞・数値設問の取りこぼしが現状の主なボトルネック）
- ensemble に retrieval/TTA 軸を足して多様性を増やす（Cycle 03）

最終更新: 2026-05-29

---

## 🗺️ 全体ロードマップ (Cycle 単位)

| Cycle | スコープ | 採用手法 (要点) | 期待 CV |
|---|---|---|---|
| **01** | **Retrieval + Models + Ensemble** (dataset は元 wiki dump 利用) | days 7th place 派生：90-word chunk + FAISS + e5-base/gte-base + DeBERTa v3 large + TTA + mean+max | 0.80-0.85 (フル) / 0.70+ (簡略) |
| **02** | **Dataset 改良** | wiki dump を `jjinho/wikipedia-20230701` から **cirrussearch** に変更（数値欠落の解決）。詳細は `knowledge/02/dataset_improvements.md` | +0.02-0.05 期待 |
| **03** | **モデル多様化** | DeBERTa v3 large の 3 種 (microsoft / OpenAssistant reward / deepset squad2) を全部走らせて ensemble へ追加 | +0.01-0.03 |
| **04** | **データ拡張** | @radek1 6.5k / cdeotte MMLU / synthetic by Claude Opus 4.7 等で train データ増強 | +0.02-0.04 |
| **05** | **2026 視点での近代化** | ModernBERT 8K 長 context、bge-reranker-v2-m3、Qwen 3 8B (unsloth LoRA) を比較 | 検討 |

> 各 Cycle 完了時に `report/NN_score_report.md` を更新し、本ボードの「Done」へ移動。
> 計画は柔軟に変更可能（Cycle 01 結果次第で Cycle 02 以降を見直す）。

---

## 📊 進捗

### ✅ Done

- [x] **Block A**: プロジェクト基盤 (uv env, CLAUDE.md, hook, 認証スクリプト, folder skeleton) — commit `724ac6b` (2026-05-26)
- [x] **Block B**: 調査と知識整理 (overview/4 + knowledge/直下 11 + knowledge/01/3 + knowledge/02/1 + task_board + CLAUDE.md 更新) — commit `bae654f` (2026-05-27)
- [x] **Block B 追補**: `knowledge/search/00_ensemble_pipeline_origin_validity.md` 追加（ensemble の起源・学術妥当性・上位 5 解法のパイプライン比較）。併せて `knowledge/search/` フォルダ規約を CLAUDE.md に明文化 — 2026-05-28
- [x] **Block B 追補2**: RAG×LLM 学習教材を `knowledge/search/` に追加 — `07_rag_llm_relationship_for_beginners.md`（RAG と LLM の関係を図中心・初心者向けに解説、読む順番ガイド付き）+ `08_rag_ensemble_design_and_selection.md`（RAG 付き複数 LLM のアンサンブル設計・モデル選定基準、LLM Ensemble survey arXiv 2502.18036 の before/during/after 分類・RAG-Fusion・MoA・LTRR で補強）。既存 05/06/00 を土台に再構成 — 2026-05-30
- [x] Kaggle CLI 認証 (kaggle.json 配置 + .env 作成、`kaggle competitions list` 動作確認)
- [x] ML 依存追加 (torch 2.12 +cu130 / transformers 5.9 / accelerate 1.13 / sentencepiece、CUDA OK)
- [x] HF g-ronimo mirror から train/test parquet → CSV 変換 (train 5400 / test 600 行)
- [x] `report/01_eda.md` 作成 (データ概観、ラベル分布、テキスト長、サンプル、課題)
- [x] `report/01_score_report.md` 雛形 (実行後にスコア追記)
- [x] `01_train_pred.py` 実装 (DeBERTa v3 large MCQ、 KFold CV、--single-fold 対応、score report 自動更新)
- [x] `scripts/submit.py` 実装 (1日1回ガード、dry-run、--force、timestamp 記録)
- [x] `scripts/download_data.sh` 実装

### ✅ Done (Block C 追加)

- [x] Kaggle 公式データ取得 (新アカウント `kunihiro1997`、train 200 / test 200)
- [x] 学習スクリプトに loss 可視化 / checkpoint / resume / epoch ログを実装 (ローカル用)
- [x] CLAUDE.md に「学習時の loss 可視化」「チェックポイント・再開」「Submit code 規約」「学習前確認」規約を明記
- [x] `kaggle/01_v1_baseline.py` + `kernel-metadata.json` 作成 (no-checkpoint, fp16, T4 想定)
- [x] m1 (`microsoft/deberta-v3-large`) push & submit (2026-05-27、Public 0.388 / Private 0.378)
- [x] m2 (`OpenAssistant/reward-model-deberta-v3-large-v2`) Notebook 作成・push・submit (2026-05-28、Public 0.682 / Private 0.714)
- [x] m3 (`deepset/deberta-v3-large-squad2`) Notebook 作成・push・submit (2026-05-28、Public 0.592 / Private 0.605)
- [x] `report/01_score_report.md` を 3 モデル比較スコア表で更新
- [x] **学び**: Code Competition では `kaggle competitions submit -k <kernel> -v <ver> -f submission.csv` 形式が必須（CSV 直接 upload は 400 Bad Request）
- [x] `submit/` ディレクトリ規約整備 (CLAUDE.md にも追記)、提出 3 ファイルを `submit/01_m{1,2,3}_*/` に保存
- [x] `report/01_submissions_summary.md` 作成（提出ファイル別の結果まとめ）
- [x] **3 モデル ensemble (mean+max blending)** Notebook 作成・push・submit (2026-05-29、Public 0.684144 / Private 0.714858、ref 53157358)。`submit/01_ensemble_m1m2m3/` にスナップショット保存、`report/01_score_report.md` に R1-R5 考察追記
- [x] HF DL 高速化: `hf_transfer` 追加 + `scripts/prefetch_hf_model.py` 新設 + `01_train_pred.py` で env 設定
  - 計測: WSL2 → HF CDN の素の curl 速度 2.6 MB/s、hf_transfer 経由でも 3.5 MB/s 程度。**根本原因は WSL2 NAT 経路の帯域**で、ライブラリ層では大きく変えられないことが判明（次の打ち手は Windows 側 DL or aria2c 等）

### 🟡 TODO

- [x] commit (`01 …` 形式) — Block C 一括

### 🔵 TODO (Cycle 01 v2)

- [ ] Wikipedia subset (HF `graelo/wikipedia/20230601.en` または STEM filter) 取得
- [ ] 90-word chunk 分割 → e5-base 埋め込み → FAISS index 構築
- [ ] context 付き MCQ 学習 + 推論
- [ ] CV / LB を Cycle 01 v1 と比較

### 🔵 TODO (後で: Cycle 02 — dataset 改良)

- [ ] cirrussearch wiki dump 取得 (詳細: `knowledge/02/dataset_improvements.md`)
- [ ] 既存 chunk 化パイプラインを cirrussearch 入力に対応
- [ ] FAISS index 再構築
- [ ] Cycle 01 の他要素はそのまま流用、retrieval 部分のみ差し替え
- [ ] Cycle 01 比のスコア差を `report/02_score_report.md` に記録
- [ ] commit (`02 cirrussearch dataset ...`)

### 🔵 TODO (後で: Cycle 03+)

- [ ] Cycle 03: DeBERTa v3 large 3 種 + gte-base 追加 + TTA を 4 slice にフル化
- [ ] Cycle 04: 公開 augmentation データ統合
- [ ] Cycle 05: ModernBERT / Qwen 3 比較

---

## 🔄 中断 → 再開手順

セッションが切れた / Claude を再起動した時:

1. **このファイル** (`task_board.md`) の「現在のフォーカス」「In Progress」を読む
2. `report/NN_score_report.md` (最新 Cycle) で前回スコアと「次の打ち手」を確認
3. `knowledge/NN/baseline_proposal.md` で採用手法を確認
4. `session/YYYY-MM-DD_<sid>.md` で直前の対話ログを必要に応じて参照（ローカルのみ、gitignore）
5. In Progress の最初の未完項目から再開
6. 作業中に方針を変更したら **このファイルを必ず更新**

---

## 📝 メモ (現サイクル特有の留意)

- **Cycle 01 から dataset 改良 (cirrussearch 切替) は除外** (2026-05-27 ユーザー指示)。Cycle 01 はあくまで days 手法のうち retrieval / models / ensemble に集中。
- RTX 3080 10GB 制約下では、Cycle 01 フル構成 (3 モデル × 2 retrieval × 4 TTA) は学習・推論に数日必要 → **まず簡略版 (1 model × 1 retrieval × 2 TTA) で疎通確認**
- 1 サイクル完結ごとに commit、コミット message prefix は `NN <内容>` 形式
- 提出は 1 日 1 回まで (`scripts/submit.py` で自動ガード)

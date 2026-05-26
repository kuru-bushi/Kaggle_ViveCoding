# Task Board — Kaggle LLM Science Exam

> **このファイルは Claude が中断後に作業を再開するための単一の真実の源 (single source of truth)**。
> プロジェクト全体のロードマップ + 現サイクルの詳細計画 + 進捗 + 再開手順を集約する。
> 作業を進めるたびに必ず更新すること（Done に移す、In Progress を更新する）。
> 関連: 詳細規約は `CLAUDE.md`、Cycle 01 採用案は `knowledge/01/baseline_proposal.md`。

---

## 🎯 現在のフォーカス

**Block B (調査・知識整理)** が大筋完了し、commit 待ち。
次は **Block C (Cycle 01 実装)** に着手予定（着手前にユーザー確認）。

最終更新: 2026-05-27

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

### 🚧 In Progress

- [ ] **Block B**: 調査と知識整理
  - [x] overview/ 4 ファイル (competition / data / evaluation / rules)
  - [x] knowledge/ 直下 11 ファイル (01_overview_trends 〜 11_kaggle_submission)
  - [x] knowledge/01/ 3 ファイル (baseline_proposal / ensemble_methods / alternative_tfidf_baseline)
  - [x] knowledge/02/ 1 ファイル (dataset_improvements)
  - [x] task_board.md (本ファイル)
  - [x] CLAUDE.md 更新 (knowledge/NN/ 規約 + task_board.md 参照)
  - [ ] **commit 待ち** (ユーザー承認後)

### 🟡 TODO (次にやる: Block C = Cycle 01 実装)

- [ ] kaggle 認証 (.env 編集 → `bash scripts/setup_kaggle.sh`)
- [ ] `scripts/download_data.sh` 作成 → 公式 train.csv / test.csv 取得
- [ ] `report/01_eda.md`: EDA (data 概観、ラベル分布、テキスト長、サンプル)
- [ ] Wikipedia subset 取得 (HF `graelo/wikipedia/20230601.en` または STEM only) → `data/tmp/wiki/`
- [ ] 90-word chunk 分割スクリプト (`knowledge/01/ensemble_methods.md` 参照)
- [ ] e5-base 埋め込み + FAISS index 構築 (gte-base は Cycle 03 へ繰越)
- [ ] `microsoft/deberta-v3-large` fine-tune (max_len=256, fp16, accum=8) — 簡略版 1 モデルのみ
- [ ] TTA 推論 (まず 2 slice: `[0,1-5]`, `[0,6-10]` のみ)
- [ ] Ensemble (`mean + max`) → `submission.csv` 生成
- [ ] ローカル MAP@3 CV 計算
- [ ] `report/01_score_report.md` 自動更新
- [ ] `scripts/submit.py` 実装 + 1日1回ガード
- [ ] Kaggle 提出 → LB 確認
- [ ] commit (`01 baseline ...` 形式、必要なら分割)

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

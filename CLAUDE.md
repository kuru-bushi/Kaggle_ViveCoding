# CLAUDE.md — Kaggle LLM Science Exam プロジェクト規約

このファイルは Claude が **毎セッション必ず参照** するプロジェクト規約。元要件は `requirements.md`。

## プロジェクト概要

- Kaggle コンペ「LLM - Science Exam」(2023 終了) に **Late Submission で練習** として取り組む
- 目的: Claude による Kaggle 実践と知見の蓄積
- 評価指標: **MAP@3** (Mean Average Precision @ 3)

## フォルダ規約

| ディレクトリ | 用途 | 命名 |
|---|---|---|
| `overview/` | Kaggle 公式情報の写し（概要・データ・評価・ルール） | `competition.md`, `data.md`, `evaluation.md`, `rules.md` |
| `knowledge/` | 上位 Kaggler 手法、出力フォーマット、Kaggle 操作手順 | `01_` 〜 番号 prefix |
| `session/` | 会話ログ（hook が自動生成） | `YYYY-MM-DD_live.md`, `YYYY-MM-DD_<sid>.md` |
| `data/{train,test}/` | Kaggle データ（.gitignore 済） | Kaggle CLI が配置 |
| `data/tmp/` | 中間ファイル | `last_submission.txt` 等 |
| `models/` | モデル定義 / 学習済重み | `NN_<model_name>.py` |
| `report/` | サイクルごとの EDA / スコア記録 | `NN_eda.md`, `NN_score_report.md` |
| `scripts/` | 補助スクリプト | `setup_kaggle.sh`, `download_data.sh`, `submit.py` |
| ルート | サイクル実行スクリプト | `NN_train_pred.py` |

`NN` はサイクル番号 (`01`, `02`, ...).

## サイクル定義

1サイクル = **データ分析 → 課題抽出 → 解決策提示 → クリティカルな批評 → 選定 → 実装 → report 作成 → Kaggle 提出 → commit**

- サイクルごとに段階的に複雑化（最初はシンプルなベースライン、いきなりアンサンブルしない）
- サイクル番号で実行スクリプト・レポート・コミットメッセージを紐付ける

## 出力フォーマット規約

詳細は `knowledge/09_output_format.md`（調査結果を反映して確定）を参照。要約:

- **EDA レポート** (`report/NN_eda.md`): 統計、欠損、ラベル分布、テキスト長、サンプル、課題リスト
- **スコアレポート** (`report/NN_score_report.md`): 実験 ID / モデル / 前処理 / ハイパラ / CV score / LB score / 推論時間 / 学び / 次の打ち手 / mermaid フロー図 / 参考 Notebook リンク
- **実行スクリプト** (`NN_train_pred.py`): 冒頭メタデータコメント (`# Cycle NN / Model: <name> / Created: YYYY-MM-DD`)、末尾で CV を表示し `report/NN_score_report.md` を自動更新

## コミット規約

- 1サイクル完了ごとに必ず commit
- コミットメッセージの **先頭にサイクル番号** を付ける (例: `01 baseline TF-IDF`, `02 DeBERTa fine-tune`)
- 変更が大きい時は分割 (`01 project skeleton` / `01 EDA` / `01 baseline submission` 等)
- **commit 前に必ず機密チェック**:
  ```bash
  git diff --cached | grep -iE '(api[_-]?key|secret|password|kaggle_key|token=)' && echo "WARNING: possible secret found, ABORT" || echo "OK: no obvious secret"
  ```
  ヒットしたら commit を中止し `.gitignore` を更新する

## Kaggle 認証・提出

- 認証手順: `knowledge/10_kaggle_auth.md`
- 初回のみ `.env` に `KAGGLE_USERNAME` / `KAGGLE_KEY` を記入 → `bash scripts/setup_kaggle.sh`
- 提出: `uv run python scripts/submit.py --file submission.csv --message "01 baseline ..."`
- **提出は 1 日 1 回まで**（Kaggle ルール）。`scripts/submit.py` が `data/tmp/last_submission.txt` で 24h ガード

## Python 環境

- すべての Python 実行は **`uv run python ...`**
- 依存追加は `uv add <pkg>`
- venv は `.venv/`（.gitignore 済）
- 初回: `uv sync`

## セキュリティ規約

- `.env`, `~/.kaggle/kaggle.json` は絶対 commit しない（.gitignore 済）
- 新ファイル作成時、機密が含まれそうな拡張子・名前があれば `.gitignore` を追記
- コード中に API キー / 個人情報を直書きしない（環境変数経由で読む）

## task_board.md の運用

- セッション開始時に `task_board.md` を確認し、In Progress / TODO の続きから着手
- セッション終了時に Done / In Progress / TODO を更新

## session ログ（自動）

- Claude Code の **Stop / UserPromptSubmit hook** が `session/` に自動追記
- `session/YYYY-MM-DD_live.md`: ユーザー発話のリアルタイム log（中断時の即時保全用）
- `session/YYYY-MM-DD_<session_id>.md`: ターン終了時の完全ログ（ツール呼び出し含む）
- 中断後はこれを読めば文脈を復元できる
- **`session/*.md` は `.gitignore` 済（ローカルのみ）** — commit しない

## 作業開始時のチェックリスト

1. `task_board.md` で現状把握
2. 最新の `report/NN_score_report.md` で前サイクルの結果と「次の打ち手」を確認
3. `knowledge/` の関連メモを参照
4. 該当サイクルを進める

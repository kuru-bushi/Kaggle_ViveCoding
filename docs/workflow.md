# Workflow — サイクル運営と中断復元

このドキュメントは **プロジェクト全体の進め方** を定義する。サイクル定義 / `task_board.md` / session ログ / commit 規約をまとめる。

## サイクル定義

1 サイクル = **データ分析 → 課題抽出 → 解決策提示 → クリティカルな批評 → 選定 → 実装 → report 作成 → Kaggle 提出 → commit**

- サイクルごとに段階的に複雑化（基本方針：最初はシンプル、いきなりアンサンブルしない）
- ただしユーザー指示がある場合はその優先
- サイクル番号で実行スクリプト・レポート・コミットメッセージを紐付ける

### 現行 Cycle 計画

| Cycle | スコープ | 詳細 |
|---|---|---|
| 01 | Retrieval + Models + Ensemble（元 wiki dump 使用） | `knowledge/01/baseline_proposal.md` |
| 02 | Dataset 改良（cirrussearch wiki dump） | `knowledge/02/dataset_improvements.md` |
| 03+ | モデル多様化 / データ拡張 / モダン化 | `task_board.md` ロードマップ参照 |

サイクル番号で実行スクリプト (`NN_train_pred.py`)・レポート (`report/NN_*.md`)・コミットメッセージ (`NN <内容>`) を紐付ける。

### Cycle-specific 文書の置き場規約

- **Cycle NN に強く紐づく** (採用案・別案・ensemble 詳細・hyperparam tuning 結果など) → `knowledge/NN/<title>.md`
- **Cycle を跨いで参照する** 一般知識 → `knowledge/<NN>_<title>.md` (直下、番号 prefix で並びを制御)
- 採用しなかったが将来戻る可能性がある案 → `knowledge/NN/alternative_<name>.md`
- **人間が調べた調査結果** → `knowledge/search/NN_<title>.md`

詳細は [`knowledge/README.md`](../knowledge/README.md) を参照。

## task_board.md — 単一の真実の源

**`task_board.md` (リポジトリルート) は本プロジェクトの "single source of truth"**。
作業計画（全 Cycle ロードマップ + 現サイクル詳細）と進捗（Done / In Progress / TODO）を 1 ファイルに集約しており、**処理が中断されても、このファイルだけ読めば作業再開できる**ように設計されている。

- **セッション開始時**: 必ず `task_board.md` を最初に開き、「現在のフォーカス」と「In Progress」を確認
- **作業中**: タスク完了 / 計画変更があれば、その場で `task_board.md` を更新
- **セッション終了時**: 「現在のフォーカス」「次のアクション」を最新化
- 中断 → 再開手順は `task_board.md` 末尾の「🔄 中断 → 再開手順」セクション参照

## 中断 → 再開手順

セッションが切れた / Claude を再起動した時:

1. `task_board.md` の「現在のフォーカス」「In Progress」を読む
2. `report/NN_score_report.md` (最新 Cycle) で前回スコアと「次の打ち手」を確認
3. `knowledge/NN/baseline_proposal.md` で採用手法を確認
4. `session/YYYY-MM-DD_<sid>.md` で直前の対話ログを必要に応じて参照（ローカルのみ、gitignore）
5. In Progress の最初の未完項目から再開
6. 作業中に方針を変更したら **`task_board.md` を必ず更新**

## 作業開始時のチェックリスト

1. **`task_board.md` を必ず最初に開く** — 現在のフォーカス / 進捗 / 次のアクションを把握
2. 最新の `report/NN_score_report.md` で前サイクルの結果と「次の打ち手」を確認
3. `knowledge/NN/` (該当 Cycle 専用) と `knowledge/` 直下の関連メモを参照
4. 該当サイクルを進める
5. **作業中・終了時に `task_board.md` を更新**

## session ログ（自動）

- Claude Code の **Stop / UserPromptSubmit hook** が `session/` に自動追記
- `session/YYYY-MM-DD_live.md`: ユーザー発話のリアルタイム log（中断時の即時保全用）
- `session/YYYY-MM-DD_<session_id>.md`: ターン終了時の完全ログ（ツール呼び出し含む）
- 中断後はこれを読めば文脈を復元できる
- **`session/*.md` は `.gitignore` 済（ローカルのみ）** — commit しない

## コミット規約

- 1 サイクル完了ごとに必ず commit
- コミットメッセージの **先頭にサイクル番号** を付ける (例: `01 baseline TF-IDF`, `02 DeBERTa fine-tune`)
- 変更が大きい時は分割 (`01 project skeleton` / `01 EDA` / `01 baseline submission` 等)
- **commit 前に必ず機密チェック**:
  ```bash
  git diff --cached | grep -iE '(api[_-]?key|secret|password|kaggle_key|token=)' && echo "WARNING: possible secret found, ABORT" || echo "OK: no obvious secret"
  ```
  ヒットしたら commit を中止し `.gitignore` を更新する

# Kaggle ViveCoding — LLM Science Exam

Claude Code で Kaggle「LLM - Science Exam」コンペ（2023 終了）に **Late Submission で練習**として取り組むプロジェクト。

- 目的: Claude による Kaggle 実践と知見の蓄積
- 評価指標: **MAP@3** (Mean Average Precision @ 3)
- 元要件: [`requirements.md`](./requirements.md)

## フォルダ構成

| ディレクトリ | 用途 | 詳細 |
|---|---|---|
| `overview/` | Kaggle 公式情報の写し（概要・データ・評価・ルール） | — |
| `knowledge/` | Kaggler 手法 / Cycle 専用調査 / 人間調査メモ | [`knowledge/README.md`](./knowledge/README.md) |
| `report/` | サイクルごとの EDA とスコア記録 | [`report/README.md`](./report/README.md) |
| `kaggle/` | Kaggle Notebook (Submit 用コード) と `kernel-metadata.json` | [`kaggle/README.md`](./kaggle/README.md) |
| `submit/` | Kaggle に提出した Notebook ソースのスナップショット保管 | [`submit/README.md`](./submit/README.md) |
| `scripts/` | セットアップ・データ取得・提出スクリプト | [`scripts/README.md`](./scripts/README.md) |
| `models/` | モデル定義 / 学習済重み (`NN_<model_name>.py`) | — |
| `data/{train,test}/` | Kaggle データ（.gitignore 済、Kaggle CLI が配置） | — |
| `data/tmp/` | 中間ファイル (`last_submission.txt` 等) | — |
| `session/` | 会話ログ（hook が自動生成、`.gitignore` 済） | — |
| `docs/` | プロジェクト規約（workflow / coding conventions） | [`docs/workflow.md`](./docs/workflow.md) / [`docs/conventions.md`](./docs/conventions.md) |
| `task_board.md` | **全 Cycle ロードマップ + 進捗の単一情報源** | [`task_board.md`](./task_board.md) |
| `CLAUDE.md` | Claude Code 用挙動ルール | [`CLAUDE.md`](./CLAUDE.md) |
| ルート | サイクル実行スクリプト (`NN_train_pred.py`) | — |

## セットアップ

### 1. uv インストール（未インストール時のみ）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # またはシェル再起動
```

### 2. 依存解決
```bash
uv sync
```

### 3. Kaggle 認証
```bash
cp .env.example .env
# .env に KAGGLE_USERNAME と KAGGLE_KEY を記入
bash scripts/setup_kaggle.sh
```

API トークンは <https://www.kaggle.com/settings> → "Create New Token" から取得。

### 4. データダウンロード
```bash
bash scripts/download_data.sh
```

## サイクル実行

```bash
uv run python 01_train_pred.py
```

サイクル定義と運営手順は [`docs/workflow.md`](./docs/workflow.md) を参照。

## 提出

Code Competition では **kernel 経由必須**。詳細は [`kaggle/README.md`](./kaggle/README.md) と [`scripts/README.md`](./scripts/README.md)。

```bash
~/.local/bin/uv run kaggle kernels push -p kaggle/<dir>/
~/.local/bin/uv run kaggle competitions submit \
  -c kaggle-llm-science-exam \
  -k <user>/<kernel-id> -v <version> -f submission.csv \
  -m "submission message"
```

## Claude Code で起動

```bash
claude --dangerously-skip-permissions
```

詳しい Claude 挙動規約は [`CLAUDE.md`](./CLAUDE.md) を参照。

## ドキュメント一覧

- **入門** — このファイル
- **進め方** — [`docs/workflow.md`](./docs/workflow.md) （サイクル運営、task_board、commit、session）
- **コーディング規約** — [`docs/conventions.md`](./docs/conventions.md) （出力フォーマット、loss 可視化、checkpoint/resume、Python 環境、セキュリティ）
- **進捗管理** — [`task_board.md`](./task_board.md) （単一情報源）
- **Claude 用挙動ルール** — [`CLAUDE.md`](./CLAUDE.md)
- **要件原本** — [`requirements.md`](./requirements.md)

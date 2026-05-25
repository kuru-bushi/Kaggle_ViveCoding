# Kaggle ViveCoding — LLM Science Exam

Claude Code で Kaggle「LLM - Science Exam」コンペに Late Submission で取り組むプロジェクト。

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

## 提出（1日1回ガード）

```bash
uv run python scripts/submit.py --message "01 baseline TF-IDF"
```

## Claude Code で起動

```bash
claude --dangerously-skip-permissions
```

詳しい規約は [CLAUDE.md](./CLAUDE.md)、元要件は [requirements.md](./requirements.md) を参照。

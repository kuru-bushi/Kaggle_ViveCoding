# 10. Kaggle CLI 認証手順 (永続ログイン)

## 目的

- Claude Code から `kaggle competitions download` / `kaggle competitions submit` を自動実行
- 認証情報は `.env` に置き、毎回ログイン不要にする
- `~/.kaggle/kaggle.json` 生成は `scripts/setup_kaggle.sh` で自動化

## 初回セットアップ (一度だけ)

### Step 1. Kaggle API トークン取得

1. ブラウザで <https://www.kaggle.com/settings> を開く
2. 「API」セクションの「**Create New Token**」をクリック
3. `kaggle.json` がダウンロードされる（中身は `{"username":"...","key":"..."}` の JSON）

### Step 2. `.env` に転記

```bash
cp .env.example .env
```

`.env` をエディタで開き、`kaggle.json` の値を転記:

```ini
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key_here
```

> `.env` は `.gitignore` 済み。絶対に commit しないこと。

### Step 3. セットアップスクリプト実行

```bash
bash scripts/setup_kaggle.sh
```

これにより `~/.kaggle/kaggle.json` が **0600 パーミッション** で作成される。
以後は Kaggle CLI が自動で認証する（永続ログイン）。

### Step 4. 動作確認

```bash
uv run kaggle competitions list | head
```

参加コンペ一覧が表示されれば成功。

## トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `403 Forbidden` | `kaggle.json` のパーミッションが緩い → `chmod 600 ~/.kaggle/kaggle.json` |
| `401 Unauthorized` | API キー失効。Kaggle Settings で再生成して `.env` を更新 |
| `Could not find kaggle.json` | `bash scripts/setup_kaggle.sh` 実行を忘れていないか |
| 提出時 `You have not accepted the rules` | コンペページを開いて Rules を承諾 (1 回だけ手動) |

## セキュリティチェック

`.env` と `kaggle.json` は **絶対 commit しない**:

```bash
git status | grep -E '\.env|kaggle\.json' && echo "STOP: secret about to be committed" || echo "OK"
```

`.gitignore` のセクション `# --- secrets ---` で除外済み。

## 環境変数経由でも動く

`scripts/setup_kaggle.sh` を使わず、`uv run` 実行時に環境変数を渡すことも可能:

```bash
KAGGLE_USERNAME=... KAGGLE_KEY=... uv run kaggle competitions list
```

→ ただし毎回入力が面倒なので、`setup_kaggle.sh` 経由の永続化を推奨。

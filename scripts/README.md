# scripts/ — 補助スクリプト

セットアップ・データ取得・提出 etc. を補助する shell / python スクリプト群。

## ファイル一覧

| ファイル | 用途 |
|---|---|
| `setup_kaggle.sh` | `.env` の `KAGGLE_USERNAME` / `KAGGLE_KEY` を `~/.kaggle/kaggle.json` に配置し、Kaggle CLI を使えるようにする |
| `download_data.sh` | Kaggle 公式 train / test データを `data/{train,test}/` に取得 |
| `submit.py` | competition への submit（1 日 1 回ガード + 24h ガード付き）。Kaggle Code Comp 用には kernel-based submit (`-k -v -f`) を推奨 |
| `prefetch_hf_model.py` | HuggingFace から model weights を事前ダウンロード（`hf_transfer` で高速化）。WSL2 環境で NAT 経由の素 curl が遅い場合の対策 |

## 認証

初回のみ:
```bash
cp .env.example .env
# .env に KAGGLE_USERNAME / KAGGLE_KEY を記入
bash scripts/setup_kaggle.sh
```

API トークンは <https://www.kaggle.com/settings> → "Create New Token" から取得。

## データ取得

```bash
bash scripts/download_data.sh
```

## 提出

### Late Submission (CSV 直接 upload) — 通常 Code Comp では使えない

```bash
uv run python scripts/submit.py --message "01 baseline ..."
```

- `submit.py` は `data/tmp/last_submission.txt` で **24h ガード**（1 日 1 回まで）
- `--force` で bypass
- **Code Competition では CSV 直接 upload が 400 で弾かれる**ため、本コンペでは kernel-based submit を使う（[`../kaggle/README.md`](../kaggle/README.md) 参照）

### Code Competition (kernel-based) — 本コンペで使う形式

```bash
~/.local/bin/uv run kaggle competitions submit \
  -c kaggle-llm-science-exam \
  -k <user>/<kernel-id> \
  -v <version> \
  -f submission.csv \
  -m "submission message"
```

## HF Model の事前 DL（任意）

WSL2 環境では HF CDN への素の curl が 2-3 MB/s で遅い。`hf_transfer` を使うと多少高速化:

```bash
uv run python scripts/prefetch_hf_model.py --model microsoft/deberta-v3-large --out data/tmp/deberta_v3_large
```

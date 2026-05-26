# 11. Kaggle 提出手順 (自動化 + 1日1回ガード)

## 概要

- 提出は `scripts/submit.py` を介して行う
- **1 日 1 回まで** の自主制限 (Kaggle 共通制限は 5 回/日だが、無駄打ち防止)
- 直前の提出時刻を `data/tmp/last_submission.txt` に記録、24h 経過まで拒否

## 事前準備

- `bash scripts/setup_kaggle.sh` 済 (`knowledge/10_kaggle_auth.md` 参照)
- `submission.csv` がプロジェクトルートに存在
- 一度だけ Kaggle 公式ページで Rules 承諾済み (<https://www.kaggle.com/competitions/kaggle-llm-science-exam/rules>)

## 通常提出

```bash
uv run python scripts/submit.py --message "01 baseline TF-IDF"
```

- `--file <path>` で別ファイルも指定可（既定: `submission.csv`）
- `--message <text>` で Kaggle 上のコメント
- 成功すると `data/tmp/last_submission.txt` に ISO タイムスタンプを記録

## Dry-run (実提出なし)

```bash
uv run python scripts/submit.py --message "test" --dry-run
```

→ 提出コマンドを表示するだけで送信しない。スクリプト挙動確認用。

## 提出履歴確認

```bash
uv run kaggle competitions submissions kaggle-llm-science-exam
```

直近の提出と Public LB スコアを表示。

## エラー対処

| エラー | 原因 / 対処 |
|---|---|
| `Daily submission limit (1) reached for our self-enforced cap` | `data/tmp/last_submission.txt` を確認、24h 経過まで待つ。緊急時は `--force` (要追加実装) |
| `403 You have not accepted the rules` | Kaggle 公式ページで Rules 承諾 |
| `400 submission.csv format invalid` | カラム名 (`id`, `prediction`) と prediction の `"A B C"` 形式を確認 |
| `Submission still being scored` | Kaggle 側で処理中。`kaggle competitions submissions ...` で確認 |

## コミット規約との連動

- 提出メッセージは **commit message と同じ prefix** にする（例: `01 baseline TF-IDF`）
- 提出後、`report/NN_score_report.md` の `Public LB` 列を更新して commit
- LB 値が出てから commit (Public LB は数分以内に出る)

## Late Submission の特記事項 (2026 時点)

- メダル付与なし（リーダーボードは赤字表示）
- スコア自体は出るので、自分の手法を上位陣ベンチマークと比較可能
- Public/Private split は当時のまま、過学習評価にも使える

## サンプル: scripts/submit.py 仕様 (Block C で実装)

```text
usage: submit.py [-h] [--file FILE] [--message MESSAGE] [--dry-run] [--competition COMP]

options:
  --file FILE          CSV path (default: submission.csv)
  --message MESSAGE    Submission message (required unless --dry-run)
  --dry-run            Print command without sending
  --competition COMP   Kaggle slug (default: kaggle-llm-science-exam)

exit codes:
  0: success
  1: rate limit (24h not elapsed)
  2: file not found / invalid
  3: kaggle CLI error
```

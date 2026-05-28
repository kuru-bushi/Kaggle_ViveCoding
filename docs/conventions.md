# Coding & Output Conventions

実装と成果物のフォーマット規約。スクリプトを書く時 / レポートを書く時に従う。

## 出力フォーマット規約

詳細は `knowledge/09_output_format.md`（調査結果を反映して確定）を参照。要約:

- **EDA レポート** (`report/NN_eda.md`): 統計、欠損、ラベル分布、テキスト長、サンプル、課題リスト
- **スコアレポート** (`report/NN_score_report.md`): 実験 ID / モデル / 前処理 / ハイパラ / CV score / LB score / 推論時間 / 学び / 次の打ち手 / mermaid フロー図 / 参考 Notebook リンク / 学習曲線への参照
- **実行スクリプト** (`NN_train_pred.py`): 冒頭メタデータコメント (`# Cycle NN / Model: <name> / Created: YYYY-MM-DD`)、末尾で CV を表示し `report/NN_score_report.md` を自動更新

詳細は [`report/README.md`](../report/README.md) を参照。

## 学習時の loss 可視化（必須、ローカル学習スクリプト）

すべての学習スクリプト (`NN_train_pred.py`) は **学習中の loss / メトリクスを可視化する**こと。最低 2 系統で残す:

| 系統 | 出力場所 | 用途 |
|---|---|---|
| **TensorBoard event files** | `models/tmp_NN_fold{F}/runs/` (HF Trainer の `report_to=["tensorboard"]` で自動生成) | 学習中のリアルタイム監視 |
| **静的 PNG** | `report/img/NN_loss_fold{F}.png` | commit 可能、score report から参照可能 |
| **生 JSON ログ** | `report/img/NN_log_history_fold{F}.json` | 後日 re-plot / 分析用 (Trainer.state.log_history のダンプ) |

- **学習開始時に TensorBoard サーバーを自動起動して URL を表示する** こと (空きポート検出して `http://localhost:<port>/` を banner 表示)
- `score_report.md` 内で PNG への相対リンクを必ず含める (`![loss](img/NN_loss_fold0.png)`)
- TensorBoard 以外の方法 (wandb, neptune 等) を使う場合も最低 PNG + JSON を残す
- 学習中に外から監視したい時は別シェルで `~/.local/bin/uv run tensorboard --logdir=models/` でも可

## チェックポイント・再開（必須、ローカル学習のみ）

**ローカル学習スクリプト** (`NN_train_pred.py` 等) は中断耐性のため次を満たす:

1. **デフォルト 5 epoch ごとに checkpoint 保存**
   - HF Trainer: `save_strategy="steps"` + `save_steps = steps_per_epoch * save_every_epochs` (default `save_every_epochs=5`)
   - `save_total_limit=2` でディスク節約
   - 学習終了時には必ず `checkpoint-final/` も追加保存（短い run でも end-of-training を残す）
   - cadence は `--save-every-epochs N` で変更可
2. **再開は `--resume` フラグで実施**
   - `trainer.train(resume_from_checkpoint=True)` で最新の `checkpoint-*` を自動読込
   - checkpoint が無い場合は警告を出して fresh start
3. **epoch 終了時のログに必ず epoch 番号と経過時間を含める**
   - 例: `[epoch  1.00] epoch_elapsed= 312.4s total_elapsed=  5.2min step=13`
   - 実装: `TrainerCallback` で `on_epoch_end` フックを使う (Cycle 01 では `EpochTimerCallback`)

> Kaggle Notebook (Submit 用コード) では checkpoint / resume を付けない。詳細は [`kaggle/README.md`](../kaggle/README.md) を参照。

## Python 環境

- すべての Python 実行は **`uv run python ...`**
- 依存追加は `uv add <pkg>`
- venv は `.venv/`（.gitignore 済）
- 初回: `uv sync`

## セキュリティ規約

- `.env`, `~/.kaggle/kaggle.json` は絶対 commit しない（.gitignore 済）
- 新ファイル作成時、機密が含まれそうな拡張子・名前があれば `.gitignore` を追記
- コード中に API キー / 個人情報を直書きしない（環境変数経由で読む）

# report/ — サイクルごとの EDA とスコア記録

サイクル単位で生成する **データ理解 (EDA)** と **実験結果 (Score Report)** の置き場。

## ファイル命名

- `NN_eda.md` — Cycle NN の EDA レポート
- `NN_score_report.md` — Cycle NN の実験結果レポート
- `NN_submissions_summary.md` — Cycle 内で複数提出した場合のサマリ (任意)
- `img/NN_loss_fold{F}.png` — 学習曲線 PNG
- `img/NN_log_history_fold{F}.json` — Trainer.state.log_history のダンプ

`NN` はサイクル番号 (`01`, `02`, ...)。

## EDA レポート (`NN_eda.md`) の含めるべき項目

- 統計サマリ (行数 / 列数 / 型)
- 欠損値の有無
- ラベル分布（カテゴリ別カウント）
- テキスト長分布（prompt / options それぞれ）
- 代表的なサンプル数行
- 観察された課題リスト（次の Cycle で打ち手の対象になり得るもの）

## スコアレポート (`NN_score_report.md`) の含めるべき項目

1. **実験 ID** (例: `01_v1_deberta_v3_large_mcq_nowiki`)
2. **モデル / パイプライン** (mermaid フロー図推奨)
3. **ハイパーパラメータ表**
4. **スコア表** (CV / Public LB / Private LB / submission ref)
5. **学習時間 / 推論時間 / コスト**
6. **学び**
7. **課題** (スコア面 / 実装・運用面)
8. **次の打ち手**
9. **批評 (Critique)**
10. **参考にした Notebook / Kaggler** リンク
11. **学習曲線への参照** (`![loss](img/NN_loss_fold0.png)`)
12. **Reproducibility** セクション (再現コマンド)

## 学習曲線の運用

ローカル学習スクリプトでは下記 2 系統を必ず出力する（規約: [`docs/conventions.md`](../docs/conventions.md)）:

- 静的 PNG: `img/NN_loss_fold{F}.png` (train/eval loss + eval metric)
- 生 JSON: `img/NN_log_history_fold{F}.json`
- TensorBoard event は `models/tmp_NN_fold{F}/runs/` (commit しない)

> Kaggle Notebook では `report_to="none"` で raw stdout に頼ってよい。

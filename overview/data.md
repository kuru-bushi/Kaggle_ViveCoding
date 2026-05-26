# Data Description

> 出典: Hippocampus's Garden、複数の上位 solution、HuggingFace のミラー、公開 GitHub。
> 実物のサイズや列は `kaggle competitions download -c kaggle-llm-science-exam` 後に確認すること。

## ファイル

| ファイル | 概要 |
|---|---|
| `train.csv` | 訓練データ。**約 200 行のみ**（極小） |
| `test.csv` | テストデータ。約数千行（大半は private LB） |
| `sample_submission.csv` | 提出フォーマット |

## カラム (train.csv / test.csv 共通)

| カラム | 説明 |
|---|---|
| `id` | 設問 ID |
| `prompt` | 設問文 (英語) |
| `A` | 選択肢 A |
| `B` | 選択肢 B |
| `C` | 選択肢 C |
| `D` | 選択肢 D |
| `E` | 選択肢 E |
| `answer` | 正解ラベル (A〜E) — **train のみ** |

## sample_submission.csv

| カラム | 例 |
|---|---|
| `id` | `0`, `1`, ... |
| `prediction` | 上位 3 つの選択肢ラベルをスペース区切り `"A B C"` |

## データソースとサイズ感の注意

- 元データ生成: **GPT-3.5 が Wikipedia 記事を読んで作成**した 5 択問題
- **train.csv は 200 件しかない** → 上位陣はほぼ全員、自前で **追加データを GPT-3.5 等で大量生成**（6k〜数十k）
- 公開された augmented dataset:
  - `radek1` の 6.5k GPT-3.5 dataset (1st place が使用)
  - `cdeotte` の MMLU 系
  - 各種公開 notebook の 60k〜150k
- `test.csv` の大部分は private LB スコア計算用（CV と LB のギャップを意識）

## 外部データ (RAG 用)

- Wikipedia ダンプを「Kaggle Dataset」としてアップロードして使うのが標準
- 上位ノートで定番: `graelo/wikipedia/20230601.en`（Hugging Face）
- 270k や 6M ページ規模の事前処理済 dump が公開されている
- Inference notebook 内では offline で読み込む

## 関連メモ

- 詳細な調査・トレンドは `knowledge/01_overview_trends.md`
- ベースライン手法は `knowledge/07_other_notable_approaches.md` 参照

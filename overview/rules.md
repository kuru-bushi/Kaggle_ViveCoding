# Competition Rules (要約)

> 公式: <https://www.kaggle.com/competitions/kaggle-llm-science-exam/rules>
> 以下は一般的な Kaggle Code Competition の慣行と本コンペで明示されていた制約のまとめ。

## 提出形式

- **Code Competition**（Notebook submit 型）
- Kaggle Notebook 上で `submission.csv` を生成し、それが採点される
- 推論専用 Notebook（学習は別 Notebook、重みを Dataset としてアップロード）が一般的

## 計算リソース制限

- GPU: **T4 ×2** または **P100** ×1（推論時）
- 実行時間: **9 時間以内**
- インターネット: **OFF**（外部 API・モデルダウンロード不可）

→ 学習済モデル・Wikipedia 等の外部データは Kaggle Dataset として事前アップロード必須。

## 提出回数制限

- Kaggle 共通: **1 日 5 回まで**
- 本プロジェクト規約: **自主的に 1 日 1 回まで**（無駄打ちを避ける）
- 最終提出: 2 ノートブックを Final 指定

## 外部データの扱い

- 公開・無料の外部データは利用可（要 Discussion での共有）
- 商用 API 経由でラベル生成・ファインチューニングは可
- 上位陣は GPT-3.5 / GPT-4 で訓練データ大量生成済み

## チーム

- チーム 5 人まで
- マージ期限あり

## Late Submission の扱い (2026 時点)

- リーダーボード採点は継続（赤字スコア表示）
- メダル付与なし
- 練習・ベンチマークとして引き続き使える

# Kaggle - LLM Science Exam: Competition Overview

> **注意**: Kaggle 公式ページは reCAPTCHA でプログラムからの直接取得が制限されているため、本ファイルは公開された
> Discussion / 解法レポート (Hippocampus's Garden、GitHub 上位 solution) を主出典として再構成した二次情報。
> 確定値は必ず Kaggle 本家 (<https://www.kaggle.com/competitions/kaggle-llm-science-exam>) で確認すること。

## タスク

- 科学に関する **5択 (A〜E) の Multiple Choice Question** に答える
- 各設問の問題文と 5 つの選択肢が与えられ、**最も正解らしい上位 3 つ** を順位付きで提出する
- 設問は **GPT-3.5 で生成された** Wikipedia ベースの科学問題 (物理・化学・生物・天文 等)
- 設問の元になった Wikipedia 記事は与えられない → **検索 (RAG) が事実上必須** という設計

## ホスト・スポンサー

- **ホスト**: Kaggle（Will Lifferth, Walter Reade）
- **賞金総額**: 約 USD 50,000（実値は公式参照）

## タイムライン (推定)

- 開始: 2023 年 7 月頃
- 終了: 2023 年 10 月 11 日
- 参加チーム: 2,600+ チーム

> 2026-05-26 時点では **既に終了済み**。Late Submission として提出可能。リーダーボード上位入りは不可。

## 制約 (重要)

- **Kaggle Notebook 実行制限**: GPU 9 時間以内 (T4 ×2 または P100)
- **モデルサイズ実質上限**: ~10B params (上記時間制約から)
- **インターネット禁止** (推論時)：Wikipedia 等の外部知識はデータセットとしてアップロードして利用
- 提出は **1日5回まで** (Kaggle 共通) → 本プロジェクトでは安全側に **1日1回** に自主制限

## 主な参考リンク

- 公式: <https://www.kaggle.com/competitions/kaggle-llm-science-exam>
- レポート: [Hippocampus's Garden — Kaggle Competition Report](https://hippocampus-garden.com/kaggle_llm/)
- 1st place: [Team H2O LLM Studio writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/team-h2o-llm-studio-1st-place-solution)
- 3rd place: [podpall writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/writeups/podpall-3rd-place-solution-update-code-links)

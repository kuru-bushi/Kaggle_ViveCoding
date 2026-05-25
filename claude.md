## Kaggle - LLM Science Exam
- LLM Science Exam に取り組む。要件について記載する。
- 要件(元)
  - claude.mdとsettings.jsonを作成すること
  - AIが途中で何をしたかわかるような出力フォーマットを検討する。簡潔に、かつ、詳細に書くこと。
  - サイクル: データの分析→課題の抽出→解決策の提示→解決策が妥当かクリティカルに批評→選定→実装→reportの作成→kaggleに提出
  - サイクルことに構成を複雑にすること。例えば、アンサンブルでモデルを組む際には最初から複雑にしすぎないこと。
  - kaggleのサイトの情報は、一フォルダ内にまとめる。(overview)必ず公式のウェブサイトから、コンペの概要、データについて説明、scoreの説明、そのほかの情報などをまとめること。
  - データディレクトリにまとめる(data)
  - 各サイクルのscoreや推論、学習に関わることはreportディレクトリに必ず記載すること。また、edaの結果や課題、解決策もreportディレクトリにまとめる。使ったモデルやフローはmermaid図にして記載すること。参考にしたkagglerのNoteがあれば記載すること。
  - kaggle への提出を自動化すること。ただし、kaggleへの提出は1日1回にすること。
  - フォルダの構成例
```
overview
data
- train
- test
- tmp
models
- 01_llm_models.py
report
- 01_eda.md
- 01_score_report.md
01_train_pred.py
```

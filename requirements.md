## Kaggle - LLM Science Exam
- LLM Science Exam に取り組む。要件について記載する。
- 要件(元)
  - claudueにkaggleをさせるために、最適なclaude.mdとsettings.jsonを作成すること。
  - AIが途中で何をしたかわかるような出力フォーマットを検討してほしい。簡潔に、かつ、詳細に書くこと。全国のkagglerがどのようにしているか調査して、markdownにまとめて。
  - フォルダの構成例
```
overview
knowledge
session
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
task_board.md
```
  - kaggleのサイトの情報は、一フォルダ内にまとめる。(overview)必ず公式のウェブサイトから、コンペの概要、データについて説明、scoreの説明、そのほかの情報などをまとめること。
  - データディレクトリにまとめる(data)
  - 各サイクルのscoreや推論、学習に関わることはreportディレクトリに必ず記載すること。また、edaの結果や課題、解決策もreportディレクトリにまとめる。使ったモデルやフローはmermaid図にして記載すること。参考にしたkagglerのNoteがあれば記載すること。
  - knowledgeにデータ分析、kagglerのノートをみてわかったことを簡潔に書くこと。
  - task_board.md にclaudeが中断しても良いように、完了したタスクと、途中のタスク、未着手のタスクを書くこと
  - サイクルごとのスコアの出力は 01_train_pred.py にのような形式で書くこと。ローカルで検証して
  - session フォルダにチャットの履歴を書くこと。
  - サイクル: データの分析→課題の抽出→解決策の提示→解決策が妥当かクリティカルに批評→選定→実装→reportの作成→kaggleに提出
  - サイクルことに構成を複雑にすること。例えば、アンサンブルでモデルを組む際には最初から複雑にしすぎないこと。
  - 1サイクルが終わったらcommitすること。commit のコメントのコメントの最初には 01 などサイクルを記載すること。変更が大きくなった場合は コミットを分けること。
  - kaggle への提出を自動化すること。自動化した自動化したただし、kaggleへの提出は1日1回にすること。kaggleへの提出手順はknowledgeフォルダにまとめること。
  - kaggle へのログインを省力化したい。常にログインしている状態で問題ないので、ログイン方法を検討すること。個人情報やセキュリティに問題がある情報はファイルを確認して、.gitignoreに記載すること。個人情報などはenvに記載するとする。
  - commit 前に個人情報がかかれていないか確認して、.gitignore に書くこと
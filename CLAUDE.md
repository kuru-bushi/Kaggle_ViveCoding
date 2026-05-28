# CLAUDE.md — Claude Code 用挙動ルール

このファイルは **Claude Code 専用の挙動ルール**。プロジェクトの一般ドキュメント（フォルダ規約、出力フォーマット、commit 規約、Python 環境等）は `README.md` と `docs/` 配下にあり、本ファイルでは `@` import で取り込んでいる。

> プロジェクト要件原本: `requirements.md`

## 取り込みドキュメント（@ import）

@README.md
@docs/workflow.md
@docs/conventions.md
@knowledge/README.md
@report/README.md
@kaggle/README.md
@scripts/README.md
@submit/README.md

---

## Claude 専用挙動ルール

ここから下は **Claude にだけ伝えたい挙動指令**。人間ドキュメントには載っていない / 載せても意味がない内容のみ。

### 学習を開始する前の確認（必須）

`NN_train_pred.py` などの学習スクリプトを **実行する前に必ずユーザーに確認** すること。確認する内容:

1. **目的はどちらか**:
   - (a) **Kaggle に提出する** (Code Competition → Kaggle Notebook 経由で Private LB を取りに行く)
   - (b) **ローカルで loss / CV 信号を取得する** (実装の妥当性チェック・スコア感の把握)
   - 両方やる場合もあり (b → a の順)
2. 採用する構成 (簡略 / フル) と所要時間 (RTX 3080 での想定) を提示
3. ユーザーが OK と返したら起動。それまでは **絶対に勝手に長時間 GPU プロセスを起動しない**

> 理由: 学習は GPU 時間 / ディスク / メモリを長時間占有する。前提合意のないまま走らせると、ユーザーの想定とズレた成果物 (LB に出ない CSV、目的と違う構成 etc.) を作って時間を無駄にする。Cycle 01 v1 の進行で実例あり (CSV 提出は Public LB の visible test 200 行しか採点されない、Code Competition は Notebook 提出が本筋)。

### Submit する前の確認（必須）

`kaggle kernels push --accelerator <GPU>` で submit する直前に、必ず以下をユーザーに確認:

1. **使用する GPU は `NvidiaTeslaT4` で良いか？** (デフォルト推奨だが、P100 / TPU を選びたい特別な理由がある場合は別)
2. 提出メッセージ (例: "01 v1 baseline DeBERTa MCQ")
3. 1 日提出上限 (Kaggle 5/日, 自主 1/日) に達していないか — `data/tmp/last_submission.txt` で確認
4. **そのサイクルの「提出予定ファイル × 提出状況」表を必ず提示する** — 詳細は次節

→ ユーザーが OK と返したら push + submit。それまでは push しない（Code Competition は kernel commit のたびに quota を消費する）。

### サイクルごとの提出ファイルは複数（必須提示の表）

**前提**: 1 サイクル (`NN` = 01, 02, ...) には複数の提出ファイル（backbone 違い、retrieval 有無、ensemble 構成違い、TTA 設定違い等）が含まれる。Cycle 01 では実際に m1/m2/m3 の **3 ファイル**を別々に提出している。今後の Cycle も同様に複数提出が基本。

**Claude のルール**: そのサイクルで 1 件でも push / submit する直前には、ユーザーに必ず以下の形式で **そのサイクルの全提出予定ファイルの状況表** を提示してから承認を待つこと。

```
## Cycle NN 提出状況

| # | submit/ dir | 説明 (backbone / retrieval / ensemble / TTA …) | 提出状況 | sub ref | Public LB | Private LB |
|---|---|---|---|---|---|---|
| f1 | submit/NN_<purpose>/ | <説明> | ✅ 提出済 | <ref> | 0.xxx | 0.xxx |
| f2 | submit/NN_<purpose>/ | <説明> | 🟡 これから提出 | -    | -     | -     |
| f3 | submit/NN_<purpose>/ | <説明> | ⬜ 未提出 (このセッション外) | - | - | - |
```

- 「提出予定ファイル」は `knowledge/NN/` の採用案 (例: `knowledge/02/cycle01_retrospective_and_v2_plan.md`) と `submit/NN_*/` の現状から導く
- 「提出状況」は `kaggle competitions submissions <comp>` で実機照会した結果と突き合わせる（記憶や docs だけに頼らない）
- これから push するファイルは 🟡 にして、ユーザーがその行で OK と言ったら push
- 提出後は表を更新し、commit メッセージにも反映

> 理由: 「単発で 1 ファイル提出」という頭で push すると、サイクル全体の進捗把握が壊れる（残り何件か / どれが ensemble か / どれが ablation 対照か が見えなくなる）。表を毎回出すことで「全体計画 vs 現実」のズレが即座に検出できる。Cycle 01 の振り返りで「提出ファイル一覧と backbone の対応が取れない」事故が起きた実例あり (2026-05-28)。

### 作業開始時のチェックリスト

1. **`task_board.md` を必ず最初に開く** — 現在のフォーカス / 進捗 / 次のアクションを把握
2. 最新の `report/NN_score_report.md` で前サイクルの結果と「次の打ち手」を確認
3. `knowledge/NN/` (該当 Cycle 専用) と `knowledge/` 直下の関連メモを参照
4. 該当サイクルを進める
5. **作業中・終了時に `task_board.md` を更新**

詳細は `docs/workflow.md` の「中断 → 再開手順」「task_board.md — 単一の真実の源」を参照。

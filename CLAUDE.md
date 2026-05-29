# CLAUDE.md — Claude Code 用挙動ルール

このファイルは **Claude Code 専用の挙動ルール**。プロジェクトの一般ドキュメント（フォルダ規約、出力フォーマット、commit 規約、Python 環境等）は `README.md` / `docs/` / 各ディレクトリの `README.md` にある。

> **`@import` は使わない。** 各ドキュメントは「いつ読むか」を下の【フォルダ別ガイド】に書いてあるので、**その状況になったら必要なファイルを `Read` で読む**（常時 context に載せると中盤の内容が埋もれるため）。
>
> プロジェクト要件原本: `requirements.md`

---

## 常時厳守の挙動ルール（最優先・lazy load しない）

以下は取り返しのつかない／外部影響を伴うため、**毎回必ず本体ルールとして守る**（ファイル参照に逃がさない）。

### 0. 作業開始時チェックリスト

1. **`task_board.md` を必ず最初に開く** — 現在のフォーカス / 進捗 / 次のアクションを把握（単一情報源）
2. 最新の `report/NN_score_report.md` で前サイクルの結果と「次の打ち手」を確認
3. `knowledge/NN/`（該当 Cycle 専用）と `knowledge/` 直下の関連メモを参照
4. 該当サイクルを進める
5. **作業中・終了時に `task_board.md` を更新**

### 1. Python 実行は必ず `uv run`

- すべての Python 実行は **`~/.local/bin/uv run python ...`**（素の `python` を使わない）
- 依存追加は `uv add <pkg>`、初回は `uv sync`

### 2. commit 前の機密チェック（必須）

- `.env` と `~/.kaggle/kaggle.json` は **絶対に commit しない**（.gitignore 済）。コード中に API キー / 個人情報を直書きしない（環境変数経由）
- **commit 前に必ず**次を実行し、ヒットしたら commit を中止して `.gitignore` を更新:
  ```bash
  git diff --cached | grep -iE '(api[_-]?key|secret|password|kaggle_key|token=)' && echo "WARNING: possible secret found, ABORT" || echo "OK: no obvious secret"
  ```
- commit は 1 サイクル完了ごと。メッセージ先頭にサイクル番号（例: `01 baseline TF-IDF`）。`session/*.md` は commit しない

### 3. 学習スクリプトを実行する前の確認（必須）

`NN_train_pred.py` などの学習スクリプトを **実行する前に必ずユーザーに確認**:

1. **目的はどちらか**:
   - (a) **Kaggle に提出する** (Code Competition → Kaggle Notebook 経由で Private LB を取りに行く)
   - (b) **ローカルで loss / CV 信号を取得する** (実装の妥当性チェック・スコア感の把握)
   - 両方やる場合もあり (b → a の順)
2. 採用する構成 (簡略 / フル) と所要時間 (RTX 3080 での想定) を提示
3. ユーザーが OK と返したら起動。それまでは **絶対に勝手に長時間 GPU プロセスを起動しない**

> 理由: 学習は GPU 時間 / ディスク / メモリを長時間占有する。前提合意のないまま走らせると、ユーザーの想定とズレた成果物 (LB に出ない CSV、目的と違う構成 etc.) を作って時間を無駄にする。Cycle 01 v1 で実例あり (CSV 提出は Public LB の visible test 200 行しか採点されない、Code Competition は Notebook 提出が本筋)。

### 4. Submit する前の確認（必須）

`kaggle kernels push --accelerator <GPU>` で submit する直前に、必ず以下をユーザーに確認:

1. **使用する GPU は `NvidiaTeslaT4` で良いか？** (デフォルト推奨。P100 / TPU を選ぶ特別な理由がある場合のみ別)
2. 提出メッセージ (例: "01 v1 baseline DeBERTa MCQ")
3. 1 日提出上限 (Kaggle 5/日, 自主 1/日) に達していないか — `data/tmp/last_submission.txt` で確認
4. **そのサイクルの「提出予定ファイル × 提出状況」表を必ず提示する**（次項）

→ ユーザーが OK と返したら push + submit。それまでは push しない（Code Competition は kernel commit のたびに quota を消費する）。

### 5. サイクルごとの提出ファイルは複数（必須提示の表）

**前提**: 1 サイクル (`NN` = 01, 02, ...) には複数の提出ファイル（backbone 違い、retrieval 有無、ensemble 構成違い、TTA 設定違い等）が含まれる。Cycle 01 では実際に m1/m2/m3 の **3 ファイル**を別々に提出している。今後の Cycle も同様に複数提出が基本。

そのサイクルで 1 件でも push / submit する直前には、ユーザーに必ず以下の形式で **そのサイクルの全提出予定ファイルの状況表** を提示してから承認を待つ:

```
## Cycle NN 提出状況

| # | submit/ dir | 説明 (backbone / retrieval / ensemble / TTA …) | 提出状況 | sub ref | Public LB | Private LB |
|---|---|---|---|---|---|---|
| f1 | submit/NN_<purpose>/ | <説明> | ✅ 提出済 | <ref> | 0.xxx | 0.xxx |
| f2 | submit/NN_<purpose>/ | <説明> | 🟡 これから提出 | -    | -     | -     |
| f3 | submit/NN_<purpose>/ | <説明> | ⬜ 未提出 (このセッション外) | - | - | - |
```

- 「提出予定ファイル」は `knowledge/NN/` の採用案と `submit/NN_*/` の現状から導く
- 「提出状況」は `kaggle competitions submissions <comp>` で実機照会した結果と突き合わせる（記憶や docs だけに頼らない）
- これから push するファイルは 🟡 にして、ユーザーがその行で OK と言ったら push
- 提出後は表を更新し、commit メッセージにも反映

> 理由: 「単発で 1 ファイル提出」という頭で push すると、サイクル全体の進捗把握が壊れる。表を毎回出すことで「全体計画 vs 現実」のズレが即座に検出できる。Cycle 01 の振り返りで「提出ファイル一覧と backbone の対応が取れない」事故が起きた実例あり (2026-05-28)。

### 6. HF / モデル重みのダウンロード — 最速の方法（必ずこれを使う）

WSL2 環境では多くの DL 経路が rate limit / NAT で 1–4 MB/s に張り付く。実測比較の結果、**最速かつ最も安定するのは `scripts/prefetch_hf_model.py` 経由の `snapshot_download` (Xet / hf_transfer + parallel workers + safetensors 優先)** 1 択。

```bash
~/.local/bin/uv run python scripts/prefetch_hf_model.py \
    --repo <hf-user>/<model-name> \
    --dest data/tmp/<model-name> \
    --workers 8
```

- `--workers 8` で並列ファイル取得、Xet/hf_transfer で 1 ファイル内チャンク並列も同時に効く
- safetensors を優先取得。`.safetensors` が無いリポは自動で `.bin` にフォールバック
- 既に dest にウェイトがあれば skip。`--force` で再 DL
- スクリプトが `HF_XET_HIGH_PERFORMANCE=1` / `HF_HUB_ENABLE_HF_TRANSFER=1` を setdefault するのでユーザー設定不要
- **Kaggle Dataset / コンペデータ**: `kaggle datasets download -d <ref> -p <dest> --unzip` または `kaggle competitions download -c <comp> -p <dest>`
- **やってはいけない**: 単発 `curl`/`wget` で巨大 weight を取る（遅く resume も無い）、Windows 側 `curl.exe` 経由（`/mnt/c/` 越し write が overhead）
- 高頻度に必要なら `.env` の `HF_TOKEN` にアクセストークンを置く（rate-limit 解除、スクリプトが自動使用）

---

## フォルダ別ガイド（各フォルダの役割 ＋ いつ何を読む/書くか）

`@import` の代わりに、**下表の「いつ読む/書くか」の状況に入ったら、対応するファイルを `Read` で読んでから作業する**。詳細ルール（必須項目・フォーマット）は各ファイル側にあるので、ここでは内容を再掲しない。

| フォルダ / ファイル | 役割（何を置くか） | いつ読む / 書くか |
|---|---|---|
| `task_board.md` | 全 Cycle ロードマップ + 進捗の単一情報源 | **毎セッション開始時に必ず最初に開く**。作業中・終了時に更新（中断復元の起点） |
| `requirements.md` | プロジェクト要件原本 | スコープ・ゴールに迷ったとき |
| `README.md` | プロジェクト入門・フォルダ構成・セットアップ手順 | 全体像 / セットアップ / 提出コマンドの形を確認するとき |
| `docs/workflow.md` | サイクル定義・現行 Cycle 計画・task_board 規約・session ログ・**commit 規約** | サイクルを開始/完了するとき、中断から再開するとき、commit 規約の詳細を確認するとき |
| `docs/conventions.md` | 出力フォーマット・**loss 可視化(必須)**・**checkpoint/resume(必須)**・Python 環境・セキュリティ | **学習スクリプト (`NN_train_pred.py`) を書くとき**（loss 可視化と checkpoint の必須ルールあり、書く前に必ず読む）、EDA/score report のフォーマットを確認するとき |
| `knowledge/README.md` | knowledge 階層規約・search の Q&A 運用・**score 記載の R1–R5 必須規約** | knowledge にメモを置くとき、**score report / 振り返り文書を書くとき**（R1–R5 の必須考察規約があるので書く前に必ず読む） |
| `knowledge/NN_<title>.md` | Cycle 横断で参照する一般知識 | kaggler 手法・auth/submission 手順など汎用知識を参照するとき |
| `knowledge/NN/` | Cycle NN 専用の採用案・別案・ensemble 詳細・hyperparam 結果 | 該当 Cycle の手法を確認/設計するとき、提出予定ファイルを導くとき |
| `knowledge/search/` | 人間が手動で調べた調査メモ | 既存調査を参照するとき。**その内容への追加質問は新規ファイルを作らず既存ファイル末尾に追記**（運用は `knowledge/README.md` 参照） |
| `report/README.md` | EDA / score report に含めるべき項目・学習曲線の運用 | report を書く前に項目を確認するとき |
| `report/NN_eda.md` / `report/NN_score_report.md` | Cycle NN の EDA / スコア記録 | 前サイクルの結果と「次の打ち手」を確認するとき、結果を書くとき |
| `kaggle/README.md` | Submit 用 Notebook 規約・選択可能 GPU・kernel-based submit 手順 | **Kaggle Notebook を書く / push / submit するとき**（書く前に必ず読む） |
| `submit/README.md` | 提出 Notebook スナップショット規約・提出履歴 | Kaggle に提出した後、Notebook を `submit/NN_<purpose>/` に保存するとき |
| `scripts/README.md` | setup / download / submit / prefetch スクリプトの一覧と使い方 | セットアップ・データ取得・提出補助スクリプトを使うとき |
| `models/` | モデル定義 / 学習済重み (`NN_<model_name>.py`) | 学習スクリプトからモデルを読み書きするとき |
| `data/{train,test}/` | Kaggle データ（.gitignore 済、Kaggle CLI が配置） | データ参照時（直接 commit しない） |
| `data/tmp/` | 中間ファイル（`last_submission.txt` 等） | 提出ガード・中間生成物を扱うとき |
| `session/` | 会話ログ（hook 自動生成、.gitignore 済） | 中断後に直前の文脈を復元したいとき（**commit しない**） |

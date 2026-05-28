# CLAUDE.md — Kaggle LLM Science Exam プロジェクト規約

このファイルは Claude が **毎セッション必ず参照** するプロジェクト規約。元要件は `requirements.md`。

## プロジェクト概要

- Kaggle コンペ「LLM - Science Exam」(2023 終了) に **Late Submission で練習** として取り組む
- 目的: Claude による Kaggle 実践と知見の蓄積
- 評価指標: **MAP@3** (Mean Average Precision @ 3)

## フォルダ規約

| ディレクトリ | 用途 | 命名 |
|---|---|---|
| `overview/` | Kaggle 公式情報の写し（概要・データ・評価・ルール） | `competition.md`, `data.md`, `evaluation.md`, `rules.md` |
| `knowledge/` | 上位 Kaggler 手法、出力フォーマット、Kaggle 操作手順（**Cycle 横断**で参照する一般知識） | `01_` 〜 番号 prefix の md |
| `knowledge/NN/` | **Cycle NN 専用の調査・提案文書**（採用案・別案・ensemble 手法詳細など） | `knowledge/01/baseline_proposal.md` 等 |
| `knowledge/search/` | **人間が調べた調査結果**を記載（Claude の出力ではなくユーザー手動の調査メモ） | `00_title.md` (連番 prefix + 内容タイトル) |
| `session/` | 会話ログ（hook が自動生成） | `YYYY-MM-DD_live.md`, `YYYY-MM-DD_<sid>.md` |
| `data/{train,test}/` | Kaggle データ（.gitignore 済） | Kaggle CLI が配置 |
| `data/tmp/` | 中間ファイル | `last_submission.txt` 等 |
| `models/` | モデル定義 / 学習済重み | `NN_<model_name>.py` |
| `report/` | サイクルごとの EDA / スコア記録 | `NN_eda.md`, `NN_score_report.md` |
| `scripts/` | 補助スクリプト | `setup_kaggle.sh`, `download_data.sh`, `submit.py`, `prefetch_hf_model.py` |
| `submit/` | **Kaggle に提出したファイルのスナップショット保管** (Notebook 本体 + `kernel-metadata.json` 一式)。Cycle 番号 + 用途で `NN_<purpose>/` に分け、提出後は中身を書き換えない（履歴として残す）。次に Kaggle へ提出するファイルは Cycle **02** から `submit/02_*/` に置く | `submit/01_m1_microsoft/01_v1_baseline.py` 等。詳細: `submit/README.md` |
| ルート | サイクル実行スクリプト | `NN_train_pred.py` |

`NN` はサイクル番号 (`01`, `02`, ...).

## サイクル定義

1サイクル = **データ分析 → 課題抽出 → 解決策提示 → クリティカルな批評 → 選定 → 実装 → report 作成 → Kaggle 提出 → commit**

- サイクルごとに段階的に複雑化（基本方針：最初はシンプル、いきなりアンサンブルしない）
- **ただしユーザー指示がある場合はその優先**:
  - **Cycle 01**: days 7th place 解法のうち **Retrieval / Models / Ensemble** の 3 改良を実装 (元 wiki dump 使用)
    - 詳細: `knowledge/01/baseline_proposal.md`、ensemble: `knowledge/01/ensemble_methods.md`
    - 軽量 fallback: `knowledge/01/alternative_tfidf_baseline.md`
  - **Cycle 02**: **Dataset 改良** (cirrussearch wiki dump への切替) — 詳細: `knowledge/02/dataset_improvements.md`
  - 以降の計画は `task_board.md` を参照
- サイクル番号で実行スクリプト・レポート・コミットメッセージを紐付ける

### Cycle-specific 文書の置き場規約

- **Cycle NN に強く紐づく** (採用案・別案・ensemble 詳細・hyperparam tuning 結果など) → `knowledge/NN/<title>.md`
- **Cycle を跨いで参照する** 一般知識 (kaggler 手法サマリ、auth/submission 手順、output_format 等) → `knowledge/<NN>_<title>.md` (直下、番号 prefix で並びを制御)
- 採用しなかったが将来戻る可能性がある案は `knowledge/NN/alternative_<name>.md` で保存
- **人間が調べた調査結果**（ユーザー手動収集のメモ、Web 記事の写し、外部資料の要約など） → `knowledge/search/NN_<title>.md` (`00_title.md` 形式、連番で並びを制御)

### `knowledge/search/` Q&A 運用

`knowledge/search/<file>.md` の **内容について追加質問** を受けた場合:

1. **新規ファイルを作らない** — 既存の質問対象ファイル末尾に追記する
2. ファイル末尾に `## Q&A 履歴` セクションがあればそこに、なければ新設してから追記する
3. 追記フォーマット:
   ```
   ### Q<連番> (YYYY-MM-DD): <質問の要約タイトル>

   **発問**:
   > <ユーザーの発問をそのまま引用>

   **回答**:
   <本文>
   ```
4. 質問対象ファイルが不明確な場合は、どのファイルへの追加質問かをユーザーに確認してから追記する

## 出力フォーマット規約

詳細は `knowledge/09_output_format.md`（調査結果を反映して確定）を参照。要約:

- **EDA レポート** (`report/NN_eda.md`): 統計、欠損、ラベル分布、テキスト長、サンプル、課題リスト
- **スコアレポート** (`report/NN_score_report.md`): 実験 ID / モデル / 前処理 / ハイパラ / CV score / LB score / 推論時間 / 学び / 次の打ち手 / mermaid フロー図 / 参考 Notebook リンク / 学習曲線への参照
- **実行スクリプト** (`NN_train_pred.py`): 冒頭メタデータコメント (`# Cycle NN / Model: <name> / Created: YYYY-MM-DD`)、末尾で CV を表示し `report/NN_score_report.md` を自動更新

## 学習時の loss 可視化（必須）

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

## 学習を開始する前の確認（必須）

`NN_train_pred.py` などの学習スクリプトを **実行する前に必ずユーザーに確認** すること。確認する内容:

1. **目的はどちらか**:
   - (a) **Kaggle に提出する** (Code Competition → Kaggle Notebook 経由で Private LB を取りに行く)
   - (b) **ローカルで loss / CV 信号を取得する** (実装の妥当性チェック・スコア感の把握)
   - 両方やる場合もあり (b → a の順)
2. 採用する構成 (簡略 / フル) と所要時間 (RTX 3080 での想定) を提示
3. ユーザーが OK と返したら起動。それまでは **絶対に勝手に長時間 GPU プロセスを起動しない**

> 理由: 学習は GPU 時間 / ディスク / メモリを長時間占有する。前提合意のないまま走らせると、ユーザーの想定とズレた成果物 (LB に出ない CSV、目的と違う構成 etc.) を作って時間を無駄にする。Cycle 01 v1 の進行で実例あり (CSV 提出は Public LB の visible test 200 行しか採点されない、Code Competition は Notebook 提出が本筋)。

## チェックポイント・再開（必須、**ローカル学習のみ**）

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

## Submit 用コード (Kaggle Notebook) の規約

**Kaggle 提出向け Notebook (`kaggle/NN_*.ipynb` または `kaggle/NN_*.py`) は、ローカル用とは別物として扱う:**


| 項目 | 内容 |
|---|---|
| Checkpoint 保存 | **付けない** (Notebook は 1 ラン完結、scoring は fresh re-run なので checkpoint 不要) |
| `--resume` | 不要 |
| Model weights | 既存の **public Kaggle Dataset を input にマウント** して読込 (internet OFF 制約下で動作するように)。Notebook 内 DL は避ける |
| 実行時間 | **最短設計**を優先。Kaggle 9h GPU 制限内に余裕を持って収める |
| 命名 | `kaggle/NN_<purpose>.ipynb` または `.py` (例: `kaggle/01_v1_baseline.py`) |
| metadata | 同フォルダに `kernel-metadata.json` を配置、`kaggle kernels push` で submit |
| Visibility | デフォルト **private** |
| Output | `/kaggle/working/submission.csv` |
| 単一 GPU | `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` を torch import 前に設定 (DataParallel のメモリ二重消費を防ぐ) |

> ローカル学習でモデルが完成したら、その重みを Kaggle Dataset 化して Notebook で読み込む構成 (inference-only Notebook) も可。Cycle 02 以降で検討。

### Kaggle Notebook で選択可能な GPU と特性

| GPU | `machine_shape` 値 | VRAM | Compute Cap. | bf16 | 強い場面 |
|---|---|---|---|---|---|
| **NVIDIA T4 x2** | `NvidiaTeslaT4` | 16GB × 2 = 32GB | sm_75 (Turing) | ❌ | Inference, 中-小 model (fp16), 並列推論、デフォルト推奨 |
| **NVIDIA P100** | `NvidiaTeslaP100` | 16GB × 1 (HBM2) | sm_60 (Pascal) | ❌ | 大バッチ・vision 中心、**ただし最新 PyTorch (sm_70+) と非互換で SDK_no_kernel_image エラー多発、現実的には避ける** |
| **TPU v3-8** | `Tpu1VmV38` | 128GB HBM | — | ✅ | TPU 最適化済 (Flax/JAX or torch_xla) フレームワーク使用時のみ |

選定基準 (Kaggler の慣行):
1. **デフォルトで `NvidiaTeslaT4` を選ぶ** — sm_75 で最新 PyTorch とほぼ全互換、2 GPU で並列処理も可能
2. P100 はメモリ帯域 (HBM2) が魅力だが **2024年以降の PyTorch wheel が sm_60 を切り捨てた** ため、in-notebook で `torch` を使う場合は P100 を避ける (TF/JAX 等なら別途検討)
3. fp16 を使うなら T4 (TensorCore 利用可)。bf16 が欲しい場合は A100 が必要だが Kaggle にはない → fp16 で代用
4. TPU は xla 等への移植コストが高く、ピンポイントに最適化された Notebook 以外では使われない

→ **本プロジェクト Cycle 01-N は `NvidiaTeslaT4` 固定**。詳細経緯は memory `reference-kaggle-cli-gpu-spec` 参照。

## Submit する前の確認（必須）

`kaggle kernels push --accelerator <GPU>` で submit する直前に、必ず以下をユーザーに確認:

1. **使用する GPU は `NvidiaTeslaT4` で良いか？** (デフォルト推奨だが、P100 / TPU を選びたい特別な理由がある場合は別)
2. 提出メッセージ (例: "01 v1 baseline DeBERTa MCQ")
3. 1日提出上限 (Kaggle 5/日, 自主 1/日) に達していないか — `data/tmp/last_submission.txt` で確認

→ ユーザーが OK と返したら push + submit。それまでは push しない（Code Competition は kernel commit のたびに quota を消費する）。

## コミット規約

- 1サイクル完了ごとに必ず commit
- コミットメッセージの **先頭にサイクル番号** を付ける (例: `01 baseline TF-IDF`, `02 DeBERTa fine-tune`)
- 変更が大きい時は分割 (`01 project skeleton` / `01 EDA` / `01 baseline submission` 等)
- **commit 前に必ず機密チェック**:
  ```bash
  git diff --cached | grep -iE '(api[_-]?key|secret|password|kaggle_key|token=)' && echo "WARNING: possible secret found, ABORT" || echo "OK: no obvious secret"
  ```
  ヒットしたら commit を中止し `.gitignore` を更新する

## Kaggle 認証・提出

- 認証手順: `knowledge/10_kaggle_auth.md`
- 初回のみ `.env` に `KAGGLE_USERNAME` / `KAGGLE_KEY` を記入 → `bash scripts/setup_kaggle.sh`
- 提出: `uv run python scripts/submit.py --file submission.csv --message "01 baseline ..."`
- **提出は 1 日 1 回まで**（Kaggle ルール）。`scripts/submit.py` が `data/tmp/last_submission.txt` で 24h ガード

## Python 環境

- すべての Python 実行は **`uv run python ...`**
- 依存追加は `uv add <pkg>`
- venv は `.venv/`（.gitignore 済）
- 初回: `uv sync`

## セキュリティ規約

- `.env`, `~/.kaggle/kaggle.json` は絶対 commit しない（.gitignore 済）
- 新ファイル作成時、機密が含まれそうな拡張子・名前があれば `.gitignore` を追記
- コード中に API キー / 個人情報を直書きしない（環境変数経由で読む）

## 🔑 中断復元の単一情報源: `task_board.md`

**`task_board.md` (リポジトリルート) は本プロジェクトの "single source of truth"**。
作業計画（全 Cycle ロードマップ + 現サイクル詳細）と進捗（Done / In Progress / TODO）を1ファイルに集約しており、**処理が中断されても、このファイルだけ読めば作業再開できる**ように設計されている。

- **セッション開始時**: 必ず `task_board.md` を最初に開き、「現在のフォーカス」と「In Progress」を確認
- **作業中**: タスク完了 / 計画変更があれば、その場で `task_board.md` を更新
- **セッション終了時**: 「現在のフォーカス」「次のアクション」を最新化
- 中断 → 再開手順は `task_board.md` 末尾の「🔄 中断 → 再開手順」セクション参照

## session ログ（自動）

- Claude Code の **Stop / UserPromptSubmit hook** が `session/` に自動追記
- `session/YYYY-MM-DD_live.md`: ユーザー発話のリアルタイム log（中断時の即時保全用）
- `session/YYYY-MM-DD_<session_id>.md`: ターン終了時の完全ログ（ツール呼び出し含む）
- 中断後はこれを読めば文脈を復元できる
- **`session/*.md` は `.gitignore` 済（ローカルのみ）** — commit しない

## 作業開始時のチェックリスト

1. **`task_board.md` を必ず最初に開く** — 現在のフォーカス / 進捗 / 次のアクションを把握
2. 最新の `report/NN_score_report.md` で前サイクルの結果と「次の打ち手」を確認
3. `knowledge/NN/` (該当 Cycle 専用) と `knowledge/` 直下の関連メモを参照
4. 該当サイクルを進める
5. **作業中・終了時に `task_board.md` を更新**

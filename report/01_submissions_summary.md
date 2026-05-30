# Cycle 01 — 提出ファイル別 結果サマリ

> 各提出 (`submit/01_*/`) について、出した実物の構成と Kaggle 上の結果を 1 ファイルずつ整理する。
> 比較・批評・次手は `report/01_score_report.md` を参照。

最終更新: 2026-05-30（ensemble 提出 ref 53157358 を反映。`kaggle competitions submissions kaggle-llm-science-exam` 実機照会済）

## 概要 (4 提出の集計)

| # | dir | backbone / 構成 | val MAP@3 | Public LB | Private LB | sub ref | submit date |
|---|---|---|---|---|---|---|---|
| m1 | `submit/01_m1_microsoft/` | `microsoft/deberta-v3-large` | 0.4708 | 0.388056 | 0.378399 | 53093495 | 2026-05-27 |
| **m2** | `submit/01_m2_openassistant/` | `OpenAssistant/reward-model-deberta-v3-large-v2` | **0.7958** | 0.682480 | 0.714337 | 53113415 | 2026-05-28 |
| m3 | `submit/01_m3_deepset/` | `deepset/deberta-v3-large-squad2` | 0.6458 | 0.592176 | 0.605449 | 53113423 | 2026-05-28 |
| **ens** | `submit/01_ensemble_m1m2m3/` | m1+m2+m3 **mean+max blending** | 0.7958 | **0.684144** | **0.714858** | 53157358 | 2026-05-29 |

単独 3 件 (m1/m2/m3) の共通構成: T4 ×1 / fp16 / 80-20 split / 3 epoch / batch=2 × grad_accum=8 / `AutoModelForMultipleChoice` / **retrieval なし / TTA なし / ensemble なし**。
4 件目 (ens) は同じ 3 backbone を 1 Notebook で逐次学習し、softmax 確率を mean+max で集約（retrieval/TTA はまだ無し）。**ボード最良は ens の Public 0.684144 / Private 0.714858**。

---

## m1 — microsoft/deberta-v3-large (plain)

- **submit dir**: `submit/01_m1_microsoft/`
- **kernel**: `kunihiro1997/llm-science-exam-01-v1-baseline`
- **dataset_sources**: `radek1/deberta-v3-large-hf-weights`
- **submission ref**: `53093495`
- **submit date**: 2026-05-27
- **submit message**: `01 v1 baseline DeBERTa v3 large MCQ no-wiki`
- **val MAP@3**: 0.4708
- **Public LB**: 0.388056
- **Private LB**: 0.378399
- **wall time on Kaggle T4**: ~3.4 min

**意図**: 素の DeBERTa v3 large を MCQ head だけ付けて fine-tune。retrieval なしのパイプライン疎通確認 + ベースライン獲得。

**所見**:
- 200 行 train では plain backbone を引き出しきれず、当初予想 (CV 0.55-0.70) を下回る。
- val (0.47) と Public LB (0.39) で約 0.08 のギャップ → train/val 同分布と test の分布差を反映。

---

## m2 — OpenAssistant/reward-model-deberta-v3-large-v2 ★最強

- **submit dir**: `submit/01_m2_openassistant/`
- **kernel**: `kunihiro1997/llm-science-exam-01-v1-m2-openassistant-reward`
- **dataset_sources**: `nags98/openassistantreward-model-deberta-v3-large-v2`
- **submission ref**: `53113415`
- **submit date**: 2026-05-28
- **val MAP@3**: 0.7958
- **Public LB**: **0.682480** (m1 比 +0.294)
- **Private LB**: **0.714337** (m1 比 +0.336)
- **wall time on Kaggle T4**: ~3.4 min

**意図**: RLHF preference 判定で学習済の reward model が、MCQ「どの選択肢が正答らしいか」を選ぶタスクに転用できないかの検証。

**所見**:
- m1 (plain) より Public で **+0.294**、Private で **+0.336** という大差。
- 「人間の好む応答を選ぶ」訓練が「5 択から正答を選ぶ」MCQ にも強く転移することを実証。
- Private > Public のため過学習リスクは低そう。

---

## m3 — deepset/deberta-v3-large-squad2

- **submit dir**: `submit/01_m3_deepset/`
- **kernel**: `kunihiro1997/llm-science-exam-01-v1-m3-deepset-squad2`
- **dataset_sources**: `katwooo/deberta-v3-large-squad2`
- **submission ref**: `53113423`
- **submit date**: 2026-05-28
- **val MAP@3**: 0.6458
- **Public LB**: 0.592176 (m1 比 +0.204)
- **Private LB**: 0.605449 (m1 比 +0.227)
- **wall time on Kaggle T4**: ~3.4 min

**意図**: SQuAD2 で QA fine-tune 済の汎用「読解 → 回答選択」能力が MCQ にも効くかの検証。

**所見**:
- m1 と m2 の中間。中間 pretrain (QA) は寄与するが、reward 系ほど MCQ には強くない。
- Private > Public の傾向は m2 と同様で、汎化は概ね保たれている。

---

## ens — 3 モデル ensemble (mean+max blending) ★ボード最良

- **submit dir**: `submit/01_ensemble_m1m2m3/`
- **code**: `01_v1_ensemble_m1m2m3.py`
- **kernel**: `kunihiro1997/llm-science-exam-01-v1-ensemble-m1m2m3-meanmax`
- **title**: `LLM Science Exam 01 v1 ensemble m1m2m3 meanmax`
- **dataset_sources**: `radek1/deberta-v3-large-hf-weights` + `nags98/openassistantreward-model-deberta-v3-large-v2` + `katwooo/deberta-v3-large-squad2`（3 backbone 同時マウント）
- **submission ref**: `53157358`
- **submit date**: 2026-05-29
- **submit message**: `01 v1 ensemble m1+m2+m3 mean+max blending (days 7th, val MAP@3=0.7958)`
- **val MAP@3**: 0.7958（mean+max）／ 参考: ensemble(mean)=0.7917、run 内 per-model = m1 0.4042 / m2 0.7958 / m3 0.6458
- **Public LB**: **0.684144**（m2 単独比 +0.001664）
- **Private LB**: **0.714858**（m2 単独比 +0.000521）
- **wall time on Kaggle T4**: ~10 min（3 モデルを逐次学習、間で `del`+`empty_cache`）

**構成（submit した中身）**: 1 Notebook 内で m1→m2→m3 を順次 fine-tune（学習設定は単独 3 件と同一、SEED=42 の共通 80/20 split）。各モデルの softmax 確率を id・選択肢ごとに **`score = mean_models(prob) + max_models(prob)`**（days 7th place の集約、`knowledge/01/ensemble_methods.md`）で集約し top-3 を提出。

**意図**: 「強い single (m2) を確立したうえで、まず最小コストの ensemble で上積みを測る」。多様性軸（retriever/corpus/backbone）の本格追加は Cycle 02→03。

**所見**:
- **最良 single (m2) を Public +0.0017 / Private +0.0005 で僅かに上回りボード最良を更新**。ただし上積みは僅少。
- val は m2 と同値 0.7958（val=40 では m2 の正答行を ensemble も当て、それ以上を測れない）。
- ほぼ無力な m1（val 0.4042）を混ぜても下がらなかったのは `max` 項が m2 の確信を保持するため。単純 `mean`(0.7917) は弱モデルに薄まって劣後 → **弱モデル込みでは mean+max が頑健**。
- R1–R5 の詳細考察は `report/01_score_report.md`「3 モデル ensemble」節、および `knowledge/01/cycle01_2_results_and_analysis.md` §7。

---

## 横断的な学び

1. **val ↔ LB 相関は強い**: val 順位 (m2 > m3 > m1) と LB 順位が完全一致、スケール感も保たれている → 以降の Cycle で val を意思決定に使える。
2. **中間 pretrain の質が支配的**: 同じ DeBERTa v3 large でも、何で前学習されているかで Public LB が ±0.3 動く。バックボーン選定は最重要レバー。
3. **コード本体は完全共通**: `_find_model_dir()` で Kaggle Dataset を auto-detect する設計のおかげで、提出間で違うのは `kernel-metadata.json` の `dataset_sources` 1 行のみ。次 Cycle 以降のモデル切替コストも低い。
4. **Kaggle CLI で Code Competition に出す形式**: `kaggle competitions submit -c <comp> -k <kernel> -v <ver> -f submission.csv -m "..."` が正解（CSV を直接 -f に渡す形式は 400）。`scripts/submit.py` でも同形式に追従。

## 次の打ち手

`report/01_score_report.md` 「次の打ち手」セクション参照。
要点: 3 モデルの mean+max ensemble は **提出済み（ens, 上積み僅少）**。次は Cycle 02 で Wikipedia retrieval を m2 に注入し long-tail 設問を取りに行く（暗記モデルの ensemble だけでは伸びないことが ens で判明）。Cycle 03 で retriever/corpus/backbone の多様性を増やした上での ensemble に拡張。

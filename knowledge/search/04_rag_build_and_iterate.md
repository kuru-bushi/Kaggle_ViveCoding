# 04 RAG の作り方・思想・継続的改善

> ⚠️ **本ファイルは Claude が Web 調査 + 既存ナレッジ統合で作成**したメモ。`knowledge/search/` は通常ユーザー手動メモだが、本回はユーザー指示により Claude が代行。
>
> 元の発問: 「RAG の作り方、思想、どのようにすれば継続的に改善できるかについて新しいファイルにまとめて。」
>
> 守備範囲:
> - **§1-2 思想**: なぜ RAG なのか、設計原則
> - **§3-5 作り方**: 最小構成 → 標準構成 → 高度構成への段階的構築
> - **§6-8 継続的改善**: 評価設計、診断、改善ループの回し方
>
> 関連:
> - 上位解法の具体例 → [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)
> - 数値根拠 → [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)
> - なぜ効くかの理論 → [`03_rag_why_it_works.md`](03_rag_why_it_works.md)
> - Cycle 01 採用案 → [`../01/baseline_proposal.md`](../01/baseline_proposal.md)
> - ensemble 設計 → [`../01/ensemble_methods.md`](../01/ensemble_methods.md)

---

# Part I. 思想

## §1. RAG の根本思想 — "知識を学習しない、参照する"

RAG (Retrieval-Augmented Generation) は **2 つの記憶を分業させる**アーキテクチャ。

| 記憶の種類 | 担う場所 | 容量 | 更新コスト | 強い領域 |
|---|---|---|---|---|
| **Parametric memory** (パラメトリック) | モデル重み | 有限 (~B params) | **再学習が必要、高コスト** | 文法、推論、一般常識、head knowledge |
| **Non-parametric memory** (非パラメトリック) | 外部 corpus + index | **事実上無限** (TB級) | **ファイル差替え、低コスト** | 固有名詞、数値、long-tail facts、最新情報 |

→ Lewis et al. 2020 ([arXiv 2005.11401](https://arxiv.org/abs/2005.11401)) が示した枠組み。
→ 「**全部覚えさせる**」のではなく「**必要な時に必要なだけ引いてくる**」というのが RAG 思想の核。

### 1.1 なぜ "分業" が合理的か

3 つの根拠:

1. **長尾分布**: 世界知識は **べき乗則分布** (Zipf). 上位の頻出事実だけ覚えるコストは小、長尾全てを覚えるコストは爆発的。**頭はパラメトリック、尾は retrieval** が最適配分。
2. **更新コスト**: 学習データに含まれない / 訓練後に更新された事実は、再学習なしには反映できない。corpus 差替えなら即座。
3. **検証可能性 (provenance)**: 出力の根拠を **具体的な文書として提示できる**。hallucination 検知が容易。

### 1.2 RAG ≠ "ただの search + LLM"

混同しがちな点を明確に:

| 誤解 | 実態 |
|---|---|
| "RAG は単に search 結果を prompt に貼るだけ" | retrieval 品質と reader の融合が本質。reranker / context format / 訓練時の context 注入も含む。 |
| "embedding さえ良ければ retrieval は解決" | sparse (BM25) と dense (embedding) は **異なる失敗モード**を持つ。両方使うのが上位の常識。 |
| "context は長ければ長いほど良い" | [Context Length Alone Hurts (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381) が示す通り、長すぎると **retrieval が完璧でも** 13.9-85% 精度低下。 |

---

## §2. RAG 設計の 5 原則

上位 Kaggler の writeup と RAG 文献を統合した経験則:

### 原則 1: **Recall を上流で、Precision を下流で**

```
retriever (high recall, top-100) → reranker (high precision, top-10) → reader (final answer)
```

- retriever は「正解 chunk を **拾い損ねない**」ことを優先 → top-k を大きめ (k=50-200)
- reranker は「上位 N に **ノイズを混ぜない**」ことを優先 → cross-encoder で精緻に
- reader は厳選された context だけで判断

→ 1 段で全部やろうとすると失敗する。本コンペ上位の 2nd / 4th / 6th 位がいずれも明示的に reranker を分けている理由。

### 原則 2: **Sparse と Dense は補完関係**

| | Sparse (BM25 / TF-IDF) | Dense (embedding) |
|---|---|---|
| 強い | 固有名詞、希少語、正確な term match | 言い換え、同義表現、文脈類似 |
| 弱い | 言い換え (vocabulary mismatch) | OOV (out of vocabulary) の数値・記号 |
| コスト | index 構築は速い | embedding 計算が重い |

→ 単独より hybrid (union or weighted sum) が一貫して強い。実装は **RRF (Reciprocal Rank Fusion)** が単純で堅牢。

### 原則 3: **Context は "minimum sufficient"**

- 短すぎる: 正解の根拠が含まれない → retrieval 不全
- 長すぎる: noise 混入、attention 希薄化、推論精度低下

→ 4th place が 512 → 1024 → 1280 まで段階的に上げた経験則は **diminishing return を見ながら止める**こと。盲目的に max まで詰めない。

### 原則 4: **Retrieval は学習時にも入れる (train-test consistency)**

- 推論時だけ context を渡し、訓練時は context なしで fine-tune すると **distribution shift** が発生。
- 訓練データにも retrieved context を含めて fine-tune するのが上位陣の標準。Lizhecheng02 は「**retrieve したコンテキストを訓練データの context カラムに入れて fine-tune**」を明示。

### 原則 5: **Pipeline 全体で 1 つの ablation 単位**

- "embedding model を変えた" の効果を測る時、reranker / reader を固定すること。
- Lizhecheng02 の embedding ablation 表 (gte-small 0.851 / bge-small 0.822) は同一 pipeline 内の clean ablation だから意味がある。
- 複数変更を同時にしない (ML 実験の基本だが RAG では retriever × reranker × reader の組合せ爆発に注意)。

---

# Part II. 作り方 — 段階的構築

RAG は **3 段階で組む** のが王道。いきなり最終形に挑むと診断不能になる。

```
Step 1: minimal RAG    →   Step 2: standard RAG   →   Step 3: advanced RAG
(疎通 / sanity check)      (中位ライン到達)           (上位狙い)
```

## §3. Step 1 — Minimal RAG (疎通用)

**目的**: パイプライン全体が動くことの確認。スコアは度外視。

### 構成

```
Wikipedia (subset 50k-100k 記事)
    ↓ chunk (sentence単位、overlap なし、512 tokens 上限)
    ↓ TF-IDF index (sklearn でローカルに作る)
    ↓ retrieve top-5
    ↓ 各 chunk を結合 → context
    ↓
[CLS] question [SEP] context [SEP] option [SEP] → DeBERTa-base → 二値 logit
    ↓ 5 選択肢を順位付け → 上位 3 を MAP@3 提出
```

### 期待スコア [MAP@3]

- ~ **0.55 – 0.70** (closed book DeBERTa-base が 0.6 付近、TF-IDF だけ被せて少し上)

### このステップの目的

- end-to-end で submission.csv が作れるか
- retrieval 結果が極端に壊れていないか (目視で 5 件チェック)
- 訓練ループが動くか / OOM が出ないか

→ **Cycle 01 の v1 簡略構成** ([`../01/baseline_proposal.md`](../01/baseline_proposal.md)) がこのレイヤーに該当。

### よくある落とし穴

1. **embedding と chunk text のインデックスがズレる** ([Teemu Kanstrén 報告](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a) の実例)
   - 対策: index 構築後に PCA / UMAP で可視化して outlier を確認
2. **訓練時に context を渡していない**
   - 対策: 訓練データ生成スクリプトで retrieved context をテスト時と同じく付与
3. **MAP@3 と Accuracy を混同**して評価
   - 対策: 評価コードは公式の formula ([`overview/evaluation.md`](../../overview/evaluation.md)) を直接実装

---

## §4. Step 2 — Standard RAG (中位 ~0.85 ライン)

**目的**: 上位 30% (Public LB ~0.85) に届かせる。

### 構成

```
Wikipedia full dump (jjinho/wikipedia-20230701 → 6M chunks)
    ↓ sentence overlapping chunks (window=3 sentences, stride=2)
    ↓
┌─── BM25 (pyserini)     ──→ top-100 ──┐
└─── Dense (gte-small)   ──→ top-100 ──┤
                                          ├──→ RRF fusion → top-50
                                          ↓
                          DeBERTa-v3-large cross-encoder reranker
                          (question + chunk → relevance score)
                                          ↓ top-10
    ↓ 各 chunk concat (1280 tokens 上限) → context
    ↓
DeBERTa v3 large MCQ head, fine-tuned on (train 400 + 50k augmented)
    ↓ 5 選択肢 logit → softmax → 順位
```

### 追加要素 (Step 1 比)

1. **Dense retrieval 追加** (gte-small が本コンペでは最良 — Lizhecheng02 ablation で +0.029)
2. **Hybrid fusion** (RRF が簡単・堅牢)
3. **Cross-encoder reranker** (これが効く理由は [`03_rag_why_it_works.md`](03_rag_why_it_works.md) §D)
4. **Context 長拡張** (512 → 1280, 4th place の経験則)
5. **訓練データ拡張** (radek 6.5k + GPT-3.5 generated 50k+ など)

### 期待スコア [MAP@3]

- Public **0.83 – 0.88**
- Cycle 01 フル構成 ([`../01/baseline_proposal.md`](../01/baseline_proposal.md) days 7th 解法レプリカ) がここ

### 設計判断の根拠

- **なぜ gte-small か**: Lizhecheng02 ablation (0.851 vs bge-small 0.822) で本コンペ最良。MTEB の retrieval task 平均でも上位。
- **なぜ DeBERTa reranker か**: 2nd place (@solokin) が採用。同じ DeBERTa を reader と reranker で共有すると訓練データを使い回せる。
- **なぜ RRF か**: weight tuning 不要。各 retriever の rank だけ使う `RRF(d) = Σ 1/(k+r_i(d))` で十分。

---

## §5. Step 3 — Advanced RAG (上位 0.92+ ライン)

**目的**: 銅メダルライン (Public 0.93+ [MAP@3]) を狙う。

### 構成

```
Multiple Wikipedia dumps (jjinho + cirrussearch + graelo)  [Cycle 02 で着手]
    ↓ multi-granularity chunks (sentence + paragraph + section の 3 階層)
    ↓
┌─── BM25 (pyserini)            ──→ top-200 ──┐
├─── Dense gte-small             ──→ top-200 ──┤
├─── Dense bge-small (multi-embedder)         ─┤
└─── Edit distance (4th place)   ──→ top-100 ──┤
                                                ├──→ RRF / learned fusion → top-50
                                                ↓
                          DeBERTa cross-encoder reranker (multi-fold)
                                                ↓ top-15
    ↓
Multi-model ensemble:
  - DeBERTa v3 large (300M)        ← 4th place
  - Mistral 7B QLoRA + binary cls  ← 1st place
  - Llama 2 70B QLoRA (難問のみ)   ← 1st/5th place
    ↓ weight blending
    ↓ 5 選択肢 logit → softmax → MAP@3
```

### 追加要素 (Step 2 比)

1. **Multi-corpus** (1st place の発見: filter は不要、複数 dump を素で使う)
2. **Multi-granularity chunks** (sentence + paragraph) — 異なる粒度で retrieval を強化
3. **Multi-embedder ensemble** (1st place の MTEB 上位複数モデル併用)
4. **Cascade inference** (1st place の 3-stage: 易問は小モデル、難問だけ 70B)
5. **Model ensemble** ([`../search/00_ensemble_pipeline_origin_validity.md`](00_ensemble_pipeline_origin_validity.md) §3)

### 期待スコア [MAP@3]

- Public **0.90 – 0.93+** (上位 100 圏内〜銅メダル境界)

### 注意点

- **Kaggle 9h 制約** に収まるかが現実的な制約。1st place は cascade で時短。
- **マルチコーパスは再現性管理が大変** → corpus 固定 + commit 必須。
- **embedding 計算済み Kaggle Dataset を事前に作っておく** こと。Notebook 内で全 Wikipedia を embed するのは時間切れ。

---

# Part III. 継続的改善 — 改善ループの回し方

ここからが本題。「組んだ後、どう改善し続けるか」。

## §6. 評価設計 — 何を測るか

RAG の改善は **層別評価** が必須。end-to-end の MAP@3 だけ見ていると、どこを直せば良いか分からない。

### 6.1 計測すべき 3 レイヤーのメトリクス

| 層 | メトリクス | 計測方法 | 目標値の目安 |
|---|---|---|---|
| **Retriever** | `recall@k` (正解 chunk が top-k に入る率) | 検証データ + gold passage 注釈が必要。GPT-3.5 で擬似 gold を生成 or 手動アノテーション (~50 設問) | **recall@100 ≥ 0.90** |
| **Reranker** | `nDCG@10` / `MRR@10` | 同上の検証データ | **MRR ≥ 0.50** |
| **End-to-End** | `MAP@3` (公式) | 公式評価 | 0.85 → 0.90 → 0.93 |

→ 検証データ作成は **`knowledge/search/` Q&A 運用** で手動アノテーションした 50 設問でも十分機能する。

### 6.2 ローカル評価セットの作り方

train 400 問だけだと過学習しやすい。推奨:

1. **train 400 を 80/20 split** → 320 学習 / 80 hold-out
2. **GPT-3.5 で類似 MCQ を 200 問生成** → 内部検証用 (LB と相関する)
3. 各実験は **hold-out 80 + 生成 200 = 280 問** で評価
4. 確信度の高い改善のみ submission に回す (1 日 5 回上限なので慎重に)

→ Cycle 01 で実装すべき (まだ未整備、`task_board.md` 要更新候補)。

### 6.3 失敗ケースの 4 分類

submission 後の誤答を分析する際の分類:

| ケース | 症状 | 原因 | 修正対象 |
|---|---|---|---|
| **A: retrieval miss** | top-100 にも正解 chunk が無い | corpus 不足、retriever 弱い、query 表現 mismatch | corpus 追加、hybrid 化、query expansion |
| **B: reranker miss** | top-100 にあるが top-10 に来ない | reranker 訓練データ不足、cross-encoder 弱 | reranker fine-tune、ensemble |
| **C: reader miss** | top-10 に正解 chunk あるが間違える | reader fine-tune 不足、prompt format | reader 拡張、train context format 統一 |
| **D: question 困難** | 全モデルが当てられない | parametric にも non-parametric にも無い | 諦める or 別 corpus 追加 |

→ 1 つの誤答に対して「どの層で失敗したか」を判定するスクリプトを 1 つ書いておくと診断が劇的に楽になる。

---

## §7. 改善ループ — どの順序で何を試すか

### 7.1 優先順位の経験則

効果サイズ × 実装コストで並べると:

```
1. corpus を増やす / 切り替える       (大効果, 低コスト)
   └ Cycle 02 で cirrussearch 切替が該当
2. embedding model を変える            (中-大効果, 低コスト)
   └ Lizhecheng02 で 0.029 改善実例
3. hybrid retrieval (sparse + dense)   (中効果, 中コスト)
   └ 単独 retriever から +0.01-0.02
4. reranker を入れる                    (中効果, 中コスト)
   └ ノイズ除去で +0.01-0.03
5. context 長を伸ばす                   (小-中効果, 低コスト)
   └ 512 → 1280 で steady gain (4th place)
6. train data 拡張                      (大効果, 高コスト)
   └ radek 6.5k / GPT 50k 生成
7. model ensemble                       (中効果, 高コスト)
   └ 単一 model の天井を超える
8. 大型モデルへの置換 (DeBERTa → 70B)  (中効果, 超高コスト)
   └ 9h 制約と相談
```

→ **小さい改善を素早く積む** が王道。一発で +0.05 を狙う設計変更は危険。

### 7.2 1 ループのフォーマット (= 1 つの実験)

```
[hypothesis]   gte-small を bge-base-en に置換すると recall が上がる
[change]       embedding model だけ差し替え。pipeline 他は固定
[measure]      hold-out 80 + GPT 200 の recall@100 と MAP@3
[result]       recall@100 0.91 → 0.93, MAP@3 0.851 → 0.847 (悪化)
[interpret]    recall は上がったが reranker が新しい dense 空間に適応せず逆効果
[next]         (a) reranker を bge と再 fine-tune (b) gte-small に戻す
[commit]       実験 ID とともに report/NN_score_report.md に記録
```

→ **CLAUDE.md の `report/NN_score_report.md` フォーマット** がこのループを直接サポートする設計になっている。

### 7.3 やってはいけない改善 (anti-pattern)

| パターン | なぜ駄目 | 代替 |
|---|---|---|
| 複数変更を同時に | どれが効いたか分からない | 1 変更 / 1 実験 |
| LB だけ見て CV を作らない | LB は 1 日 5 回しか叩けない、overfit する | ローカル評価セット必須 |
| reader だけ強化 | retrieval が悪いと天井がある | retrieval を先に上げる |
| Top 100 の Notebook を盲目コピー | 自分の pipeline と組合せ未検証 | 1 要素ずつ取り入れて ablation |
| corpus を頻繁に切替 | index 再構築が重い、再現性低下 | corpus は Cycle 単位で固定 |

---

## §8. プロジェクト固有の改善ロードマップ

本プロジェクト Cycle 01-N での具体的なロードマップ。

### Cycle 01 (現在進行中)

- [x] Minimal RAG 疎通 (v1 簡略構成)
- [ ] Standard RAG 到達 (フル構成: BM25 + dense + reranker + ensemble)
- [ ] embedding model ablation (gte-small / bge-small / e5-small)
- [ ] hold-out 80 + GPT 生成 200 のローカル評価セット整備
- 目標 [MAP@3]: Public **0.85+**

### Cycle 02

- [ ] corpus 切替 (jjinho → cirrussearch) — 数値欠落の修正
- [ ] multi-corpus 化 (1st place の手法を採用するか判断)
- 目標 [MAP@3]: Public **+0.02** (0.87+)

### Cycle 03 候補

- [ ] cross-encoder reranker の独立 fine-tune
- [ ] multi-granularity chunks (sentence + paragraph)
- [ ] 訓練データ拡張 (GPT-3.5 で 50k 生成)
- 目標 [MAP@3]: Public **0.90+**

### Cycle 04 候補

- [ ] Llama 2 7B QLoRA を ensemble に追加 (1st place 流の二値分類)
- [ ] cascade inference (易問は DeBERTa、難問は 70B)
- 目標 [MAP@3]: Public **0.92+** (上位 100 ライン)

→ 各 Cycle の詳細計画は `task_board.md` で管理。本ファイルはロードマップの "なぜ" を保管。

---

## §9. まとめ — 1 行ずつ

- **思想**: 知識を覚えさせない、参照させる (parametric × non-parametric 分業)
- **作り方**: minimal (TF-IDF + DeBERTa-base) → standard (hybrid + reranker) → advanced (multi-corpus + ensemble) の 3 段
- **継続改善**: 層別評価 (recall@k / nDCG / MAP@3) を必ず分けて測る、1 変更 / 1 実験、効果サイズ順に優先
- **本コンペでの上限**: Public LB 0.93+ (銅メダル境界) を目指す。retrieval 品質が天井を決める

---

## 出典 (信頼度ランク)

**Tier 1 (本文確認した一次/直接出典)**
- [Lewis et al. 2020 — RAG for Knowledge-Intensive NLP Tasks (arXiv 2005.11401)](https://arxiv.org/abs/2005.11401) — 設計思想
- [Mallen et al. 2023 — When Not to Trust LMs (ACL 2023)](https://aclanthology.org/2023.acl-long.546/) — long-tail 仮説
- [Context Length Alone Hurts (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381) — context 長の diminishing return
- [Lizhecheng02 GitHub README](https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam) — embedding ablation 数値

**Tier 2 (まとめ・ポストモーテム)**
- [Hippocampus's Garden — LLM Science Exam report](https://hippocampus-garden.com/kaggle_llm/)
- [Teemu Kanstrén — LLM+RAG QA (Medium)](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a) — pipeline バグ実例

**プロジェクト内クロスリファレンス**
- [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)
- [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)
- [`03_rag_why_it_works.md`](03_rag_why_it_works.md)
- [`../01/baseline_proposal.md`](../01/baseline_proposal.md)
- [`../01/ensemble_methods.md`](../01/ensemble_methods.md)
- [`../02/dataset_improvements.md`](../02/dataset_improvements.md)

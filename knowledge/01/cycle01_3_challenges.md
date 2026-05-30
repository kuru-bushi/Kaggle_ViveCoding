# Cycle 01: 課題（ボトルネックと打ち手）

> **このドキュメントのシリーズ**: (1) [[cycle01_1_models_overview]] 提出モデル概要 → (2) [[cycle01_2_results_and_analysis]] 結果と考察 → **(3) 本ファイル＝課題** → (4) [[cycle02_4_model_plan]] 02 構成案
>
> 本ファイルの守備範囲: **(2) のスコア考察から導かれる Cycle 01 パイプラインのボトルネックを特定し、打ち手を「効果見込み × 実装コスト」で並べて取捨選択する**。具体的な構成は (4) に分離。

最終更新: 2026-05-29（`cycle01_retrospective_and_v2_plan.md` を 4 分割して作成）

---

## 0. なぜ retrieval が次の一手なのか（LLM 学習用の前提）

Cycle 01 の最大の課題は **「retrieval が無い」** こと。なぜそれがボトルネックなのかを概念から理解する。

### 📘 用語: parametric memory vs retrieval（暗記 vs 検索）

- **parametric memory（パラメトリック記憶）**: 事前学習でモデルの**重み（パラメータ）の中に圧縮して焼き込まれた知識**。「暗記」に近い。Cycle 01 の 3 モデルはこれだけに頼って答えていた（= closed book、参考書を見ずに記憶だけで解く）。
- **retrieval（検索＝non-parametric memory）**: テスト時に**外部の文書集合（ここでは Wikipedia dump）から関連箇所を引いてきて、設問と一緒にモデルに渡す**こと。「参考書を開いて解く」(open book) に相当。引いた文章を context として入力に足す。

**具体例（同じ設問で closed-book と open-book を比べる）**:

> 設問: *"In what year was the element radium discovered?"*
> 選択肢: (a) 1898 (b) 1911 (c) 1869 (d) 1923 (e) 1905

- **closed-book（Cycle 01 の 3 モデル）**: m2 は重みの中の曖昧な記憶だけで判断するため、"1898" と "1911"（Curie のノーベル賞年）で迷って外しうる。年号のような long-tail fact は parametric memory で曖昧になりやすい。
- **open-book（Cycle 02 で入れる retrieval）**: Wikipedia "Radium" 記事から *"Radium was discovered in 1898 by Marie and Pierre Curie."* という一文を引いて設問の前に貼る → モデルは **根拠文と照合して (a) 1898 を選べる**。これが「参考書を開いて解く」状態。

### 📘 なぜこのコンペで retrieval が必然なのか

[`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) の整理（一次ソース確認済み）によると、本コンペは構造的に open-book 向きだった:

1. **問題設計**: 設問は Wikipedia の science 記事から GPT-3.5 が生成。**正解の根拠文 (gold passage) が必ず Wikipedia のどこかに存在する** → 引ければ当たる構造。
2. **データ量**: train が 200〜400 問しか無く、fine-tune だけで全科学分野をカバーするのは情報理論的に無理 → 外部知識の参照が事実上必須。
3. **long-tail knowledge**: Mallen et al. 2023 (PopQA) は、エンティティの popularity 別に parametric only vs retrieval を比較し、**無名 (tail) のエンティティでは parametric が ~15% まで落ち、retrieval で 45–50% に回復**することを示した（[`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) §C）。科学の細部・固有名詞・数値はまさに tail。
4. **grounding**: retrieval された根拠文があると「もっともらしいが誤りの選択肢 (distractor)」に引きずられにくくなり hallucination が減る（Lewis et al. 2020）。

定量的には [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) が **「no-RAG の DeBERTa の天井は Public ~0.74、RAG を被せるだけで 0.83–0.86 に跳ね、fine-tune+ensemble で 0.92–0.93」** という階層を示している。我々の m2 は no-RAG で 0.68 なので、**retrieval は最大の単一レバー**。

### 📘 注意: retrieval は「入れれば必ず勝つ」ではない

同じ調査が反例も挙げている（(4) のリスク設計の根拠）:
- retrieval が irrelevant/誤情報を渡すと逆に精度低下（distraction）→ reranker でノイズ除去が要る。**具体例**: 上の radium 設問で、検索が誤って別元素 "radon"（綴りが近い）の記事 chunk を引くと、無関係な年号が context に混ざり、closed-book なら当てられた行をかえって外す。
- context を**長くしすぎると**、retrieval が完璧でも LLM 自体の性能が劣化（13.9–85% の劣化報告, arXiv 2510.05381）→ 必要最小限だけ渡す。**具体例**: 正解の一文だけ渡せば当たる設問に、top-50 chunk（数千 word）を丸ごと貼ると、肝心の一文が埋もれて attention が拡散し精度が落ちる。
- head（超有名）知識では parametric で足りるので gain は出ない。**具体例**: "What is gravity?" のような頻出概念は m2 が暗記済みなので、retrieval を足しても上がらない（落ちはしないが手間が無駄）。

→ だから Cycle 02 では「とりあえず大量に詰める」ではなく **適量を狙って入れ、効果を ablation で測る**設計にする（(4)）。

---

## 1. パイプライン上のボトルネック仮説（3 つ）

(2) [[cycle01_2_results_and_analysis]] のスコア考察を踏まえ、Cycle 02 で潰すべきボトルネックを 3 つに整理。

- **① backbone の事前知識の差がそのままスコア差になっている**: m2 ≫ m3 ≫ m1 の差 ≒「事前知識を選択肢に転写する能力」の差。Wikipedia 由来の固有名詞・年代・公式・定数を訓練時に見せていないので、知識質の差が直接スコアに出る。
  - (2) §4 R3 の「下がる設問特性」(数値・年代・固有名詞) と整合。
- **② retrieval が無いことが上限を決めている**（最大のボトルネック）: MCQ の選択肢を判別するには問題文と Wiki テキストの突き合わせが必要だが、現状は backbone の暗記頼り。**具体例**: §0 の radium 設問のように「根拠文さえ引ければ確実に当たる」行を、closed-book では曖昧な暗記で取りこぼしている。定量的には §0 の通り no-RAG 天井 ~0.74 に対し我々の m2 は 0.68 で止まっている。
  - (2) §6 R5 の仮説 H2（long-tail 設問で落ちている）と直接対応。Cycle 02 v1 の「context あり/なし」ablation で検証可能。
- **③ val/LB ギャップ ≈ 0.08–0.11 で意思決定の信号が不安定**: (2) §5 R4 で「val=40 のサンプリングノイズ + 楽観バイアス + Public 50% の分布偏り」の合成と分解した。
  - 詰める手: (a) KFold val=200 化（仮説 H1）、(b) retrieval 導入（H2）、(c) train 増量（Cycle 04 候補）。

---

## 2. 打ち手の優先順位（効果見込み × 実装コスト）

ボトルネック①②③から導かれる打ち手を一覧化:

| # | 打ち手 | 期待 ΔLB | 実装コスト | 依存 | コメント |
|---|---|---|---|---|---|
| **A** | Wikipedia retrieval (e5-base + FAISS) を MCQ に context 注入 | **+0.10–0.20** | 中 | dump 取得・index 構築 | **Cycle 02 の主役**。ボトルネック②に直撃。詳細 → [[wikipedia_retrieval_plan]] |
| **B** | dataset を cirrussearch に切替（数値・公式の欠落解消） | +0.02–0.05 | 中 | A と同時運用 | A と組み合わせ前提。詳細 → [[dataset_improvements]] |
| **C** | core backbone を **m2 (OpenAssistant reward)** に固定 | (再現性) | 低 | A の前段 | ボトルネック①の結論。Cycle 01 で確定済の知見をそのまま採用 |
| D | train データ拡張 (radek1 6.5k + cdeotte MMLU + synthetic) | +0.03–0.06 | 中 | (none) | Cycle 04 に回す。retrieval 効果と切り分け難しいので 02 では入れない |
| E | epochs 3→5–8 へ伸ばす（200 行では未収束の可能性） | +0.01–0.03 | 低 | (none) | 同 pipeline で安価に試せるので A の後に実験 |
| F | max_length 384→512 へ拡大（retrieval context 用の余裕） | +0.01–0.03 | 低 | A | A 必須。B 用にも有用 |
| G | 推論時 TTA (4 slice = 異なる context 切り出し) | +0.01–0.02 | 中 | A | Cycle 03 で本格化、02 はスキップ |
| H | m2 + m3 + m1 の ensemble (mean / max) | +0.01–0.04 | 低 | (none) | retrieval 効果と切り分け難しいので Cycle 03 に分離 |
| I | re-ranker (bge-reranker-v2-m3) を retrieval 後段に追加 | +0.02–0.04 | 中 | A | Cycle 02 v2 で検討（§0 の distraction 対策） |
| J | context-window-aware sliding (chunk ごとに推論 → 集約) | +0.02–0.05 | 中 | A | F 採用後の自然な次手、Cycle 02 v2 |
| K | bf16 → fp16 のみ T4 は持つ、precision tuning | (微) | 低 | (none) | Cycle 01 で fp16 確定済、これ以上は不要 |
| L | learning rate / warmup の hyperparam sweep | +0.00–0.01 | 中 | (none) | 効果薄い割にコスト高、後回し |

---

## 3. 取捨選択の基準

- **Cycle 02 は "なぜ効くかが明確で、効果が大きい" 1–2 個に絞る**。あれもこれも入れると効果切り分け不能になり、次 Cycle の判断材料が壊れる。
- **A と B は同時投入する**（Cycle 02 のスコープ宣言）。両方とも「retrieval 経路の質を上げる」共通テーマで内部依存もあるため、切り分けは「B あり/なしの ablation を 1 本走らせる」で達成可。
- **C（core m2）は無コストで効くので確定**（ボトルネック①の結論）。
- **D, G, H は明らかに効くが効果切り分け不能** → 別 Cycle（03, 04）に分離。
- **E, F は A の副作用として必要なら入れる**（特に F は retrieval context を切らないために実質必須）。

### 取捨選択を一言で

> Cycle 01 で「最強 backbone は m2」と確定した（①）。残る最大のレバーは retrieval の不在（②）なので、**A（retrieval）+ B（cirrussearch）+ C（m2 固定）+ F（max_length 拡大）を同時投入し、context あり/なしの ablation で retrieval の純効果を測る**。val 信号の不安定さ（③）は KFold val=200 化で同時に潰す。

→ これを具体的なパイプライン・spec・リスク設計に落としたものが **→ [[cycle02_4_model_plan]]**。

---

## 関連

- [[cycle02_4_model_plan]] — 本ファイルの打ち手 A/B/C/F を具体化した構成案
- [[cycle01_2_results_and_analysis]] — ボトルネックの根拠となるスコア考察
- [[wikipedia_retrieval_plan]] — 打ち手 A の設計詳細
- [[dataset_improvements]] — 打ち手 B（cirrussearch 切替）の詳細
- [`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) — retrieval が効く因果（§0 の根拠）
- [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) — RAG の定量インパクト
- [`../search/00_ensemble_pipeline_origin_validity.md`](../search/00_ensemble_pipeline_origin_validity.md) — 「強い single → 多様性」の戦略順序の根拠

# 03 なぜ RAG で精度が上がるのか — 因果分析

> ⚠️ **本ファイルは Claude が Web 調査して作成**したメモ。`knowledge/search/` は通常ユーザー手動メモだが、本回はユーザー指示により Claude が代行。
> 元の発問: 「なぜ RAG で精度が向上するかの原因も調査しておいて。」
>
> 守備範囲: **「なぜ」RAG が効くのか** — 理論的根拠 + 本コンペ固有要因 + 失敗パターン。
> 関連: 上位解法一覧 → [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md)、数値根拠 → [`02_rag_accuracy_quantitative_impact.md`](02_rag_accuracy_quantitative_impact.md)。

---

## 結論先出し — RAG が効く理由は 5 層に分けられる

| 層 | 原因 | エビデンス |
|---|---|---|
| **(A) 問題設計** | 設問が **Wikipedia から GPT-3.5 が生成** している。正解の根拠が必ず Wikipedia に存在する。 | [`01_rag_top_solutions_survey.md`](01_rag_top_solutions_survey.md) §0、Hippocampus 記事 |
| **(B) データ量** | train は **400 問** だけ。fine-tune だけで全分野をカバーすることが不可能 → 外部知識参照が必須。 | Hippocampus 記事の "key to winning" 引用 |
| **(C) Long-tail knowledge** | LLM の parametric memory は **頻出 (head) 知識**に偏り、稀な事実 (long tail) は記憶していない。RAG は long tail を retrieval で補う。 | Mallen et al. 2023 (PopQA) |
| **(D) Grounding (接地)** | parametric だけだと **hallucination**。retrieval は出力を verifiable な source に縛り、誤答を減らす。 | Lewis et al. 2020 (RAG 原論文)、医療 RAG ベンチ |
| **(E) 計算リソース節約** | 同じ精度に到達するのに **小さいモデル + 大きい retrieval** の方が GPU/学習データを節約できる。 | DeBERTa 300M + RAG が 70B closed-book に勝つ |

以下、層ごとに掘り下げる。

---

## A. 問題設計レベル — このコンペは "open book by design" だった

### A.1 設問生成プロセス

公式 [`overview/competition.md`] と Kaggle 説明を総合すると、本コンペの設問は次の手順で生成された:

```
Wikipedia の science 系記事 → topic を抽出 → GPT-3.5 に "この記事に基づいて 5 択問題を作れ" → MCQ
```

→ **設問のうち正解の根拠が含まれる文 (the gold passage) は元の Wikipedia 記事のどこかに必ず存在する**。
→ retrieval さえ完璧なら、極論 generation を介さずに **answer = retrieve → score** だけで MAP@3 = 1.0 が達成できる構造。
→ これは Open-Domain QA の一種であり、Lewis et al. 2020 ([RAG paper, arXiv 2005.11401](https://arxiv.org/abs/2005.11401)) が示した「retrieval + reader」枠組みがそのまま刺さる。

### A.2 Wikipedia は LLM の学習データに含まれている (が、不完全)

GPT-3.5 / Llama 2 / DeBERTa pretrain も Wikipedia を含む。だから **closed-book でもある程度は当たる** (70B で 80% [acc])。
ただし:

1. **量子化された (圧縮された) 形で記憶**されるため、長尾の固有名詞や数値はロストしやすい。
2. **訓練時点以降の更新** (2023 編集) は反映されない。
3. **複数記事を跨ぐ知識統合** (compositional) は苦手。

→ retrieval で **その記事本文を直接渡せば**、圧縮ロスも統合の手間もスキップできる。

---

## B. データ量レベル — 400 問では fine-tune だけで分野をカバーできない

### B.1 train 400 問の制約

- 全 5 択 = 2,000 個の "選択肢 × ラベル" の組。
- 物理 / 化学 / 生物 / 地学 / 数学… の全分野に分散すると、**各分野せいぜい数十問**。
- これだけで DeBERTa に「物理学全般を覚えさせる」のは情報理論的に無理。
- 上位陣はだから **GPT-3.5 で 150k 問を自動生成** (data augmentation)、さらに **MMLU を 100k 追加** 等で水増ししている。

### B.2 RAG は "学習データを増やすコスト" を回避する

外部 corpus を引っ張ってくれば、訓練データを増やさずに **テスト時の知識** を増やせる:

- **訓練データ拡張**: 100k 問追加 → 学習時間 × データ生成コスト
- **RAG**: Wikipedia dump をマウント → ゼロ学習でテスト時に丸ごと使える

→ Kaggle のように **9h 制約 / GPU 制約** がある環境では、RAG の方がコスト効率良い。

---

## C. Long-tail knowledge — Mallen et al. 2023 の発見

### C.1 PopQA 論文 ([When Not to Trust Language Models, ACL 2023](https://aclanthology.org/2023.acl-long.546/))

Mallen らは **エンティティの popularity (Wikipedia の page view)** を軸に、parametric LLM と RAG の精度を比較した。

| エンティティ | LLM (parametric only) | LLM + retrieval |
|---|---|---|
| head (高頻度・有名) | 高 | 同等 |
| **tail (低頻度・無名)** | **15%** | **45–50%** |

→ **長尾知識では retrieval が 3 倍以上効く**。
→ Science Exam の設問は (有名な記事ではなく) **科学の細部や固有名詞** を問うことが多く、**典型的な long-tail QA**。

### C.2 これが Kaggle の数字とどう整合するか

| 対象 | RAG なし [acc] | RAG あり [acc] | Δ |
|---|---|---|---|
| Llama 2 70B (1st place 主張) | 80% | 93% | **+13pt** |
| GPT-4 (同 writeup 引用) | 75% | 80% | +5pt |
| PopQA tail (Mallen) | 15% | 45-50% | **+30pt** |

→ Mallen ほど劇的ではないが (Science Exam は完全 long-tail ではないため)、**同じメカニズムが働いている**と理解できる。
→ "RAG が無いと取りこぼす設問群" = 訓練時には覚えきれなかった長尾科学知識。

---

## D. Grounding — hallucination を抑える

### D.1 Lewis et al. 2020 の主張

オリジナル RAG 論文 [arXiv 2005.11401](https://arxiv.org/abs/2005.11401) の核心:

> "RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline."

→ 同じ生成モデルでも、retrieval を介すると **factual** な出力が増える。
→ MCQ は free-form generation ではないが、**正解選択肢を選ぶ判断** が hallucination の影響を受ける (もっともらしいが間違った選択肢に引きずられる)。
→ retrieved context が「正解選択肢を裏付ける文」を含んでいれば、誤誘導されにくい。

### D.2 MCQ における具体的なメカニズム

MCQ 5 択で RAG が効く瞬間を分解すると:

1. **正解選択肢の語が context にも出現** → モデルが lexical overlap で正解を識別できる (semantic も lexical も両方)。
2. **distractor (誤答) が context と矛盾** → モデルは「context にない主張は怪しい」と学習している (fine-tune 時のシグナル)。
3. **正解と distractor の差分が context で明示される** (例: "X is Y" と書いてあれば、"X is Z" の選択肢は弾ける)。

→ これらは parametric memory にも入っているが、**RAG は実行時に "今この設問用に" 適切な根拠を取り出せる**。

### D.3 1st place の二値分類アーキテクチャ

H2O LLM Studio は MCQ を「(context, question, 各選択肢) → 0/1 の二値分類」に分解した。これは **NLI (Natural Language Inference) タスクと同型** で、retrieved context が **premise**、選択肢が **hypothesis** になる。NLI は grounding が最も明示的に機能するタスク形式。

---

## E. 計算リソース — 小モデル + RAG > 大モデル closed-book

### E.1 Preferred Scantron の存在証明

- **DeBERTa v3 large (~300M) で 4 位、Public LB 0.92+ [MAP@3]**
- 一方 70B 系 closed-book は 80% [acc] (= MAP@3 換算で 0.85 程度)

→ **70B のパラメータに詰め込んだ知識 < 300M + Wikipedia dump 全文**。
→ retrieval は **knowledge を非パラメトリックに保存** することで、容量当たり効率が圧倒的に良い。

### E.2 Kaggle 9h 制約との相性

- 70B を 9h で全推論 → QLoRA 4bit / 並列推論など極限技が必須
- 300M + retrieval → retrieval を事前計算可能、推論は軽い

→ **Code Competition での実装可能性そのものが RAG 寄り**に偏る。

### E.3 一般化された主張

Wang et al. 2024 [Fine-Tuning vs. RAG for Less Popular Knowledge (arXiv 2403.01432)](https://arxiv.org/html/2403.01432v3):

> "Fine-tuning improves accuracy of the base model but does not reach the effectiveness of RAG on the base model, with optimal performance achieved by integrating both fine-tuning and RAG."

→ RAG > Fine-tune > Base、ただし **RAG + Fine-tune が最強**。
→ Kaggle の 80% → 86% → 93% [acc] 進行も完全にこの順序。

---

## F. 反証 — RAG が "効かない" / "逆効果" になる条件

完全性のため、RAG が機能しない条件も整理する:

### F.1 retrieval が誤情報を渡す (Distraction)

retrieved context が irrelevant or 矛盾していると、**LLM はそれに引きずられて誤答する**。
- 対策: reranker を必須化 (上位 6 中 4 チームが実装)、複数 retriever の合議

### F.2 context が長すぎる (Long-Context Degradation)

[Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381):

> "Even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%–85%) as input length increases."

→ 「relevant token だけマスクして注意を絞っても」性能劣化が観測される。
→ つまり long context は **LLM 自身の限界** であり retrieval 品質の問題ではない。
→ 対策: **necessary minimum** だけ渡す。Preferred Scantron が 1280 tokens で止めた合理性。

### F.3 head knowledge では gain なし

Mallen 2023 の通り、**有名・頻出知識では parametric だけで足りる**。
- 例: "What is gravity?" のような設問では RAG なしでも当たる。
- このコンペの設問の多くは long-tail 側 (固有名詞・数値・専門用語) なので有効。

### F.4 retrieval index と inference の不整合 (実装バグ)

Teemu Kanstrén ([Medium post-mortem](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a)) は **chunk サイズ変更時に embedding と text の対応がズレ**、PCA で初めて気付いたと報告。

> "two outlier dots ... we didn't notice until afterward."

→ retrieval パイプは **常にエンドツーエンドで検証** すべき。

---

## G. 結論 — なぜ "効く" のかを一文で

> **本コンペの設問は Wikipedia 由来であり、long-tail 科学知識を問い、train は 400 問しかない。LLM の parametric memory は long tail を覚えきれず、fine-tune だけでは知識量が足りない。RAG は外部 Wikipedia dump を非パラメトリック・メモリとして接続することで、これら 3 つの制約を同時に解消する。同時に retrieval は出力を verifiable source に grounding し、hallucination を減らす。**

すなわち RAG は **任意の改善テクニックではなく、本コンペの問題構造に対応した必然解**だった。1st〜6th まで全員が RAG を採用したのは偶然ではない。

---

## 出典 (信頼度ランク)

**Tier 1 (本文確認した一次・準一次ソース)**
- [Lewis et al. 2020 — RAG for Knowledge-Intensive NLP Tasks (arXiv 2005.11401)](https://arxiv.org/abs/2005.11401)
- [Mallen et al. 2023 — When Not to Trust Language Models (ACL 2023)](https://aclanthology.org/2023.acl-long.546/)
- [Wang et al. 2024 — Fine-Tuning vs. RAG for Less Popular Knowledge (arXiv 2403.01432)](https://arxiv.org/html/2403.01432v3)
- [Context Length Alone Hurts LLM Performance (arXiv 2510.05381)](https://arxiv.org/abs/2510.05381)

**Tier 2 (まとめブログ / ポストモーテム)**
- [Hippocampus's Garden — LLM Science Exam report](https://hippocampus-garden.com/kaggle_llm/)
- [Teemu Kanstrén — LLM+RAG Based Question Answering (Medium)](https://medium.com/data-science/llm-rag-based-question-answering-6a405c8ad38a)

**Tier 3 (二次引用、要再検証)**
- 1st place "80/86/93% [acc]" — Kaggle Discussion 本文未取得

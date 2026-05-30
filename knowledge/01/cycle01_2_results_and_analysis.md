# Cycle 01: 結果と考察（R1–R5）

> **このドキュメントのシリーズ**: (1) [[cycle01_1_models_overview]] 提出モデル概要 → **(2) 本ファイル＝結果と考察** → (3) [[cycle01_3_challenges]] 課題 → (4) [[cycle02_4_model_plan]] 02 構成案
>
> 本ファイルの守備範囲: **3 提出のスコアを [`../README.md`](../README.md) の R1–R5 規約に従ってデータレベルで考察する**。モデルの素性は (1)、そこから導く打ち手は (3) に分離。

最終更新: 2026-05-30（§7 に ensemble (mean+max, ref 53157358) の R1–R5 を追記。実機照会済）

> 規約と本セクションの対応: R1 (§2) / R2 (§3) / R3 (§4) / R4 (§5) / R5 (§6)。

---

## 0. まず指標とスコアの種類を理解する（LLM 学習用の前提）

### 📘 用語: MAP@3（このコンペの評価指標）

Mean Average Precision @ 3。**各設問の正解はちょうど 1 つ**なので、一般的な MAP の定義は単純化され、実質「**予測上位 3 件のうち何位に正解が来たか**」で決まる:

| 正解が来た順位 | その行のスコア |
|---|---|
| 1 位 | **1.0** |
| 2 位 | 0.5 (= 1/2) |
| 3 位 | 0.333 (= 1/3) |
| 4 位以下（圏外） | 0.0 |

これを全行で平均したものが MAP@3。値域 0.0–1.0。
- 直感: 「自信のある順に 3 つ出す。1 位で当てれば満点、外して 2・3 位で拾えば部分点、3 つとも外せば 0」。
- だから **2 位・3 位の保険を上手に使う**ことが地味に効く（top-1 accuracy とは別物）。`[acc]`(top-1 正解率) と混同しないこと。出典: [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) 冒頭の単位注意。

**具体例（1 設問のスコア）**: 正解が選択肢 **C** の設問で、予測を自信順に並べたとき:

| 出した予測（自信が高い順） | 正解 C の位置 | この行のスコア |
|---|---|---|
| `[C, A, B]` | 1 位 | **1.0** |
| `[A, C, B]` | 2 位 | **0.5** |
| `[A, B, C]` | 3 位 | **0.333** |
| `[A, B, D]` | 圏外（上位 3 に C 無し） | **0.0** |

**具体例（複数行の平均）**: 仮に 4 設問でこの 4 パターンが 1 つずつ起きたら、MAP@3 = (1.0 + 0.5 + 0.333 + 0.0) / 4 = **0.458**。「1 位で当てた行の割合」とは別物で、2・3 位で拾った部分点も効いていることが分かる。

### 📘 用語: val / Public LB / Private LB の 3 種類のスコア

| スコア | 採点データ | ラベルが見えるか | 役割 |
|---|---|---|---|
| **val（ベースラインスコア）** | 公式 train.csv を分割した hold-out | **見える**（自分で採点） | 提出せず手元で意思決定に使う信号 |
| **Public LB** | 公式 test 200 行の **公開 ~50%**（≈100 行） | 見えない | 提出するたびに即返る暫定スコア |
| **Private LB** | 残り 50% + 隠し test | 見えない | 最終順位を決める本番スコア（通常コンペ終了まで隠される） |

- なぜ Public/Private を分けるか: **Public への過学習を防ぐため**。見える側だけ最適化しても隠し側で崩れれば最終順位は伸びない。「Public で良くて Private で落ちる」現象を shake down と呼ぶ。
- このプロジェクトは**終了済みコンペの Late Submission（練習）**なので、提出すれば Public/Private 両方の値が即返る特殊状況。

---

## 1. スコア再掲（提出物との対応は (1) を参照）

| Model | val MAP@3 | Public LB | Private LB | sub ref |
|---|---|---|---|---|
| m1 `microsoft/deberta-v3-large` | 0.4708 | 0.388056 | 0.378399 | 53093495 |
| **m2 `OpenAssistant/reward-model-deberta-v3-large-v2`** | **0.7958** | 0.682480 | 0.714337 | 53113415 |
| m3 `deepset/deberta-v3-large-squad2` | 0.6458 | 0.592176 | 0.605449 | 53113423 |
| **ens** m1+m2+m3 mean+max blending | 0.7958 | **0.684144** | **0.714858** | 53157358 |

> R2–R6（§2–§6）は単独 3 モデルの考察。**4 件目の ensemble (mean+max) の R1–R5 考察は §7 にまとめた**（一次記録は `report/01_score_report.md`「3 モデル ensemble」節）。

---

## 2. R1: ベースラインスコア

| Model | val MAP@3 (公式 train 80/20, val=40) |
|---|---|
| m1 | 0.4708 |
| **m2** | **0.7958** |
| m3 | 0.6458 |

- **val 集合の由来**: 公式 train.csv (200 行) を seed=42 で 80/20 split。val=40 行、train=160 行。
- **val が "Kaggle テスト似" である理由**: 公式 train.csv と公式 test.csv は **同じ生成プロセス** (GPT-3.5 が Wikipedia の science 記事から 5 択を生成) で作られている。分布的に最も近い同分布信号。出典: [`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) §A.1。
- **val のサイズ制約（📘 サンプリング誤差）**: 40 行は小さい。MAP@3 は行ごとのスコアの平均なので、母比率の推定と同じく **サンプル数が少ないほど値がブレる**。二項分布近似で **95%CI は概算 ±0.07**。これは「真の実力が同じでも、引いた 40 行次第で val が ±0.07 揺れうる」という意味で、**val スコア単体を 0.01 単位で比べるのは危険**。
  - **具体例**: m2 の val 0.7958 は、ざっくり 40 行中およそ 32 行を 1 位正解できた水準（32/40 = 0.80）。もし同じ実力のまま**別の 40 行**を引いていたら、1 位正解は概ね **29〜35 行**に揺れ、val は **約 0.72〜0.88** に散らばりうる。
  - **だから何が言えるか**: m2 (0.796) と m3 (0.646) の **0.15 差**は誤差幅 (±0.07) を超えるので実力差として信用できる。一方、もし 2 案が val 0.796 vs 0.78 のような **0.01〜0.02 差**だったら、それは誤差に埋もれるので val だけで優劣を決めてはいけない。
  - → この小ささが後述 R5 の Δ の大半を説明しうる（仮説 H1）。

---

## 3. R2: Kaggle スコアの特徴

| Model | Public LB | Private LB | sub ref |
|---|---|---|---|
| m1 | 0.388056 | 0.378399 | 53093495 |
| **m2** | **0.682480** | **0.714337** | 53113415 |
| m3 | 0.592176 | 0.605449 | 53113423 |

採点母集団の構造（§0 の再掲＋本コンペ固有値）:
- Public LB: 公式 test 200 行のうち公開 ~50%（≈100 行）に対する MAP@3。Late Submission でも値が返る。
- Private LB: 残り 50% + 隠し test 部分。Late Submission ではコンペ終了後に解禁。
- 値域 0.0–1.0。本コンペ上位陣は 0.92+（[`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md)）。我々の m2 は 0.68 でまだ大きな伸びしろ。

3 モデルに共通する特徴（**実数で確認**）:
- **Public < val が一貫**: 例えば m2 は val 0.796 → Public 0.682（−0.11）、m3 は val 0.646 → Public 0.592（−0.05）、m1 も val 0.471 → Public 0.388（−0.08）と、3 モデルとも val より Public が低い。理由: val=40 は train と**同じ生成プロセス**から引いた「見慣れた 40 行」なのに対し、test=200 はトピックがより広く未見 → 一貫して test の方が辛い。
- **m2/m3 で Public < Private、m1 で Public > Private**: m2 は Public 0.682 **<** Private 0.714、m3 は Public 0.592 **<** Private 0.605。つまり test 200 行のうち「公開 50%（＝Public 採点側）」に、**たまたま辛い設問が多めに**入っていたと読める。一方 m1 だけは Public 0.388 **>** Private 0.378 と逆転しているが、差は 0.01 と小さく**ノイズ範囲**（m1 は plain backbone で全体に低位）。
- **m2 で Private がむしろ val に近い**: Private 0.714 と val 0.796 の差は **0.08** で、Public との差 **0.11** より小さい。採点行数が Public（≈100 行）より Private（残り + 隠し test で多い）の方が多いぶん、val=40 の「たまたま当たった分の上振れ（楽観バイアス）」がならされて、より実力値に近づくため。

### 📘 過学習シグナルの読み方（Public vs Private）

一般に **Public ≫ Private** なら「見える側にチューニングしすぎた過学習」を疑う。今回はむしろ **m2/m3 で Public ≤ Private** = 隠し側の方が良い → **過学習リスクは低く、健全に汎化できている**。これは「200 行という少量での fine-tune が、見えない test に対して悪さをしていない」良い兆候。

---

## 4. R3: スコアが上下するデータ特性

> ⚠️ **現時点では誤答ログの粒度分析が未実施**。以下は (a) 上位陣 writeup と本コンペ設計に基づく **仮説**、(b) val=40 と test=200 の構成差から推定した **構造的特徴**。Cycle 02 で実際の誤答ログを公式 train.csv 200 行で取り、本セクションを **検証データで上書きする TODO**。

**仮説: スコアが上がる設問特性**
- **頻出概念を問う設問**（例: "What is gravity?", "Define DNA"）: parametric memory（モデルの重みに焼き込まれた知識）に強く入っているため retrieval 無しでも当たる。Mallen et al. 2023 の "head knowledge" に対応（[`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) §F.3）。
- **選択肢間の意味距離が大きい設問**: 正解と distractor（誤答選択肢）が明らかに別カテゴリ。**具体例**: 正解 "photosynthesis" に対し distractor が "gravity" / "mitosis" / "erosion" のように**別分野の単語**で並ぶ設問。reward model (m2) はこの「明らかに場違いな答え」を弾く preference 判定が強い。
- **prompt が短く言い換えが少ない設問**: tokenizer が崩しにくい、attention が拡散しない。
- **reward model pretrain (m2) で見たような対話/選好フォーマットに近い設問**: m2 が m1/m3 に圧勝する説明要因。

**仮説: スコアが下がる設問特性**（= retrieval で救えると期待する設問群）
- **数値・年代・固有名詞を問う設問**（例: "When was radium discovered?" → 答え 1898 のような年号）: long-tail facts。Mallen et al. 2023（PopQA, [`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md) §C）の通り parametric memory では tail で精度が ~15% にまで落ちる。
- **選択肢間の編集距離が小さい設問**（微妙な定義違い）: **具体例**: 正解 "covalent bond" に対し distractor が "coordinate covalent bond" / "covalent bonding" / "polar covalent bond" のように **1〜2 語しか違わない**並び。retrieval した context との突き合わせが無いと、表記が似ているだけに判別不能。
- **数式・記号・化学式を含む設問**（**具体例**: "E = mc²" の意味や "H₂SO₄ + NaOH →" の生成物を問う設問）: DeBERTa の tokenizer が `²` や下付き数字・記号を細かく分割 → 情報損失。
- **複数エンティティを関連付ける設問**（X と Y の関係。**具体例**: "What is the relationship between entropy and the second law of thermodynamics?"）: single-hop で答えられず、parametric では文脈統合に失敗。
- **200 行 train で見ていない特殊用語を含む設問**（**具体例**: "endoplasmic reticulum" のような専門語が設問・選択肢に出る）: domain-specific terminology の未学習。

**val=40 と test=200 の構造的差**
- val=40 は公式 train 由来のサブセットで、train で「同じ生成プロセスのサンプル」を見ているので overfit 気味の信号。
- test=200 は同分布だが未見、かつトピック幅が広い → 上記「下がる」特性の比率が高い設問群を含む確率が高い。

---

## 5. R4: データレベル比較考察

**val と Kaggle test の分布的関係**
- 同一の公式生成プロセス（GPT-3.5 from Wikipedia）→ **分布シフト自体は小さい**。ImageNet → 別ドメインのような大きな shift ではない。
- 違いは主に **サンプリング誤差** と **トピック幅**:
  - val=40 のサンプル誤差 95%CI ≈ ±0.07
  - test=200 はトピックが広く、val に無い long-tail 設問を含む確率が高い

**val で当てられて Kaggle で落ちる設問群（推定）**
- val 40 行のうち m2 が当てた ~32 行（val 0.7958 ≈ 32/40）。Kaggle Public で当てた割合は ~68%。
- 落ちている差分の中身（仮説）: §4 R3 で挙げた「数値・固有名詞・編集距離小・記号・関係性」設問。
- val では train と同じトピックが偶然サンプルされた可能性 → "見たことある分野の易しい設問"が val に多く混ざっている可能性。

**val で落ちて Kaggle で当たる設問群（推定）**
- val=40 のサンプリングノイズで「val で偶然外れたが test 全体では当てている」設問群が m2 Private で +0.06 効いている可能性が高い（0.7143 vs Public 0.6825）。
- これは **val=40 という小ささそのものの問題** であり、KFold 化（5-fold で val=200）すれば消える。

**val が test より「甘い」/「辛い」の判定**
- m2 で **val (0.7958) > Public (0.6825) > Private (0.7143)** の関係。
- 単純な「val 甘め」ではなく、**val ≈ Private + 楽観バイアス、Public は更に辛い 100 行を引いた** という構造。
- 解釈: m2 は test 全体（Public + Private）では Private 寄りの実力（~0.71）、Public LB が更に辛い分布から引かれたためギャップが大きく見える。

---

## 6. R5: スコア差 (Δ)

| Model | Δ_Public = Public − val | Δ_Private = Private − val |
|---|---|---|
| m1 | 0.388056 − 0.4708 = **−0.0827** | 0.378399 − 0.4708 = **−0.0924** |
| **m2** | 0.682480 − 0.7958 = **−0.1133** | 0.714337 − 0.7958 = **−0.0815** |
| m3 | 0.592176 − 0.6458 = **−0.0536** | 0.605449 − 0.6458 = **−0.0404** |

**Δ の値域に対する解釈**
- 全モデルで **Δ < 0** → val は test より一貫して **楽観的**。
- m2 で |Δ_Public| が最大 (−0.11) だが、これは「m2 の絶対値が高いほど分布差が拡大して見える」（MAP@3 の上限 1.0 に近づくほど僅差の見え方が大きくなる）効果も含む。
- Δ_Private (−0.08 〜 −0.04) は val=40 の 95%CI (±0.07) と **同オーダー**。**Δ_Private はサンプリングノイズの可能性が高い**。
- Δ_Public は val=40 + Public 公開 50%（≈100 行）の二重サンプリング誤差 + Public 側分布の偏りを含むため、より大きく出ている。

**Δ を詰めるための仮説と検証実験案**

| 仮説 | 検証実験 | Cycle |
|---|---|---|
| H1: val=40 のノイズが Δ の大半を占める | val を **5-fold KFold (val=200/fold)** に変える。Δ が縮まれば仮説支持 | Cycle 02 v1（低コスト、すぐ試せる） |
| H2: test には val に無い long-tail 科学設問が多く、retrieval なしの暗記モデルが落ちている | Cycle 02 v1 の **「context あり/なし」2 サブ提出** で、context 注入後の Δ が縮まれば仮説支持 | Cycle 02 v1 |
| H3: m2 の Public が異常に辛いのは Public 50% の抽出バイアス（検証不能、実質ノイズ） | （実験不可、Δ_Private のみを意思決定に使う） | — |
| H4: 数値・固有名詞設問で誤答が集中している | 公式 train 200 行で m2 を再評価 → 誤答行を `prompt` 長・数値含有 (`re.search(r'\d')`)・選択肢間 Levenshtein 距離で分類 | Cycle 02 v1 前段の前処理タスク |

**次 Cycle での Δ ターゲット**
- Cycle 02 v1 で **Δ_Public を −0.05 以下に縮める**（retrieval 導入 + KFold val=200 化で同時達成を狙う）。
- 失敗時の切り分け: KFold 化単独で Δ が縮まなければ retrieval 効果は別軸、retrieval 単独で縮まなければ val 設計の問題、と判定。
- → これらの仮説を打ち手に落とす整理は **→ [[cycle01_3_challenges]]**、実験の具体構成は **→ [[cycle02_4_model_plan]]**。

---

## 7. 4 件目の提出 — 3 モデル ensemble (mean+max) の R1–R5

単独 3 モデルのあとに提出した ensemble（ref 53157358, 2026-05-29）の考察。**構成（submit した中身）は [[cycle01_1_models_overview]] §3.5**、一次記録は `report/01_score_report.md`「3 モデル ensemble」節。ここは R1–R5 の要点のみ。

### R1（ベースライン）
- ensemble (mean+max) val MAP@3 = **0.7958**（公式 train 200 行を SEED=42 で 80/20 split、3 モデル共通の val=40）。
- この run 内 per-model val: deepset 0.6458 / OpenAssistant **0.7958** / microsoft **0.4042**。ensemble(mean)=0.7917、ensemble(mean+max)=**0.7958**。
- ⚠️ val=0.7958 は m2 単独と**同値** → val 上は ensemble の上乗せがゼロに見える（val=40 では m2 が当てる 40 行を ensemble も当て、それ以上は測れない）。

### R2（Kaggle スコア）
- Public **0.684144** / Private **0.714858**。
- Public < Private（差 **+0.0307**）。m2 単独でも同方向（+0.0319）で、**この test 集合では Public 採点側に難設問が偏在**する性質が ensemble でも一貫。

### R3（上下するデータ特性）
- retrieval 無しなので**単独モデルと同じ失敗様式を継承**（数値・年代・固有名詞・記号・編集距離小で落ちる。§4 参照）。
- **ensemble 特有**: 3 モデルが揃って外す設問（知識が全モデルの parametric memory に無い）は mean+max でも救えない。多様性が「同じ知識を別表現で持つ」だけでは long-tail は埋まらない → retrieval が要る根拠（[[cycle01_3_challenges]] §0）。

### R4（データレベル比較）
- val 0.7958 は Private 0.7149 とほぼ一致するが Public 0.6841 を大きく上回る。
- val=40 が「m2 が得意な設問」に偏ってスコアが高く出やすく（m2 単独 val と同値がその証拠）、test=200 の long-tail 比率を反映できていない。

### R5（差 Δ・**ensemble の純効果**）

| 指標 | 値 |
|---|---|
| Δ_Public = Public − val | 0.684144 − 0.7958 = **−0.1117** |
| Δ_Private = Private − val | 0.714858 − 0.7958 = **−0.0809** |
| **ensemble − 最良 single (m2)** Public | 0.684144 − 0.682480 = **+0.001664** |
| **ensemble − 最良 single (m2)** Private | 0.714858 − 0.714337 = **+0.000521** |

- **ensemble の純効果は LB +0.0005〜+0.0017 と僅少**。ボード最良は更新したが、これは「同型の暗記モデルを 3 つ束ねた」だけで多様性が乏しいため。Δ_Public が Δ_Private より大きいのは §3 R2 の「Public 側に難設問偏在」と整合。
- **示唆**: ほぼ無力な m1（val 0.4042）を混ぜても下がらなかったのは `max` 項が m2 の確信を保持したから。単純 `mean`(0.7917) は弱モデルに薄まり劣後 → 弱モデル込みなら **mean+max が頑健**。ただし**大きく伸ばすには backbone 暗記の多様化では足りず、retriever/corpus 軸の多様性（Cycle 02→03）が必要**（[[cycle01_3_challenges]] 打ち手 H）。

---

## 関連

- [[cycle01_1_models_overview]] — 本スコアを出した 3 モデルの素性
- [[cycle01_3_challenges]] — R1–R5 から導くボトルネックと打ち手
- [[cycle02_4_model_plan]] — H1/H2 を検証する Cycle 02 の実験設計
- [`../README.md`](../README.md) — R1–R5 スコア記載規約の原本
- [`../search/03_rag_why_it_works.md`](../search/03_rag_why_it_works.md), [`../search/02_rag_accuracy_quantitative_impact.md`](../search/02_rag_accuracy_quantitative_impact.md) — long-tail / 定量根拠

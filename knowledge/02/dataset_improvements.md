# Cycle 02: Dataset 改良 — Wikipedia dump を cirrussearch に切替

## 概要

days 7th place 解法の 4 つの改良のうち、**"1. dataset"** に該当する部分。
Cycle 01 では計画変更により未実施（Cycle 01 は元 wiki dump で進める）。
Cycle 02 でこの差し替えのみを行い、**Cycle 01 比でのスコア差** を測定する。

## 動機

元記事より:

> Because some of the data (mainly numbers?) were missing as discussed here, I changed wiki dump from
> `https://www.kaggle.com/datasets/jjinho/wikipedia-20230701` to **cirrussearch wiki dump**.

つまり、`jjinho/wikipedia-20230701` ダンプには **数値情報の欠落** がある（HTML 由来の特殊文字や MathML 等の処理ミスと推測）。この欠落は科学設問にとって致命的（化学式・物理定数・年号などが消える）。

cirrussearch ダンプは Wikimedia 公式の Elasticsearch 投入用 dump で、**JSON 形式** かつ retain される情報が広い（infobox の構造化、テンプレ展開済等）。

## cirrussearch dump とは

- 配布元: <https://dumps.wikimedia.org/other/cirrussearch/>
- 形式: gzip 圧縮 JSON Lines (`*.json.gz`)
- 各行 = 1 Wikipedia ページの構造化ドキュメント
- フィールド例: `title`, `text` (本文整形済), `opening_text`, `headings`, `popularity_score`, `incoming_links`, `template`, etc.
- 言語別 + 月次更新（例: `enwiki-20231009-cirrussearch-content.json.gz`）
- サイズ: en 全体で **~40-80 GB 圧縮 / ~200-400 GB 展開**

## Cycle 02 実装手順

### 1. ダウンロード

```bash
# 例: 2023-10 月のフルダンプ (~40GB 圧縮)
mkdir -p data/tmp/cirrussearch
cd data/tmp/cirrussearch
wget https://dumps.wikimedia.org/other/cirrussearch/20231009/enwiki-20231009-cirrussearch-content.json.gz
```

> 注意:
> - 一気に全 wiki を扱うとローカル 1TB クラスのストレージが必要。
> - **STEM カテゴリだけ filter する** か、サイズの小さい月別 dump を試すのが現実的。
> - 数日かかる転送になる可能性 → `wget -c` でレジューム可。

### 2. JSON Lines → テキスト抽出

```python
import gzip, json
from pathlib import Path

src = Path("data/tmp/cirrussearch/enwiki-20231009-cirrussearch-content.json.gz")
out = Path("data/tmp/wiki_text/cirrussearch.parquet")
out.parent.mkdir(parents=True, exist_ok=True)

records = []
with gzip.open(src, "rt", encoding="utf-8") as f:
    for line in f:
        # cirrussearch alternates index meta + doc lines
        d = json.loads(line)
        if "title" in d and "text" in d:
            records.append({"title": d["title"], "text": d["text"]})
        if len(records) % 100_000 == 0:
            print(f"loaded {len(records)} pages")

import pandas as pd
pd.DataFrame(records).to_parquet(out)
```

### 3. Cycle 01 と同じ chunk 化を再適用

`knowledge/01/baseline_proposal.md` の 2-2 (90 word, 3 sentence overlap) をそのまま再利用。
**chunk 化スクリプトを Cycle 01 から関数化しておく**と Cycle 02 での再利用が楽。

### 4. FAISS index 再構築

- 同じ embedding model (e5-base / gte-base)
- 同じパラメータ (`nlists=1, M=64, nbits=8`)
- 出力先: `data/tmp/faiss_cirrus_e5.index`

### 5. Cycle 01 の推論 / 学習スクリプトをほぼそのまま再利用

- `02_train_pred.py` を Cycle 01 のコピーから作成
- 入力 FAISS index 名 / chunk parquet 名のみ差し替え
- 他の hyperparameter, model, TTA, ensemble はすべて Cycle 01 と同じ

### 6. スコア比較

`report/02_score_report.md` に Cycle 01 vs Cycle 02 のスコア表を作る:

| 項目 | Cycle 01 (元 dump) | Cycle 02 (cirrussearch) | 差分 |
|---|---|---|---|
| CV MAP@3 | x.xxx | x.xxx | +0.0xx |
| Public LB | x.xxx | x.xxx | +0.0xx |
| Retrieval recall@1 (250問評価) | x.xxx | x.xxx | +0.0xx |
| 推論時間 | x 秒 | x 秒 | ±x |

## 期待される効果

| 指標 | 期待差 | 根拠 |
|---|---|---|
| Retrieval recall (250 ChatGPT 設問評価) | +0.05-0.10 | 数値欠落解消 → 検索ヒット率向上 |
| MAP@3 (LB) | +0.02-0.05 | retrieval 改善が context 質向上 → scorer 精度向上 |
| 学習所要時間 | 変化なし (~) | dataset 形式のみで学習 pipeline は同じ |
| ディスク容量 | **+200-400GB** | 全 cirrussearch dump 展開時 |

## リスクと対応

| リスク | 対応 |
|---|---|
| ダウンロード時間 (40GB) | `wget -c` でレジューム、`screen` で 1 晩放置 |
| ディスク容量 | STEM カテゴリで filter、または gzip のまま streaming 処理 |
| メモリ (parquet 化時) | chunk 単位で書き出し (例: 100k pages ごと) |
| FAISS index 再構築時間 | GPU で embedding 並列化、`nlists=1` は CPU でも分単位 |
| Cycle 01 比較の公平性 | 学習 seed, hyperparameter, model checkpoint は完全に固定する |

## 参考リンク

- [Wikimedia cirrussearch dumps](https://dumps.wikimedia.org/other/cirrussearch/)
- [Cirrus Search データ形式の説明 (MediaWiki)](https://www.mediawiki.org/wiki/Extension:CirrusSearch)
- [元 dump `jjinho/wikipedia-20230701` (問題のある方)](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701)
- 関連: `../01/baseline_proposal.md` (4 つの改良の文脈), `../01_overview_trends.md` (全体傾向)

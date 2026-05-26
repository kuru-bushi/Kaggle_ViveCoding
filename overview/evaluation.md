# Evaluation Metric: MAP@3

## 定義

**Mean Average Precision @ 3 (MAP@3)** — 各設問につき上位 3 件の予測を提出し、正解がどの順位にあるかでスコア化。

## スコアリング (1 設問あたり)

各設問の正解は 1 つだけ (relevant = 1) なので、AP@3 は次の通り単純化される:

| 正解の予測順位 | 1 設問あたりの AP@3 |
|---|---|
| 1 位 (最初に当てる) | **1.000** |
| 2 位 | 0.500 |
| 3 位 | 0.333 |
| 4 位以下 (top 3 圏外) | 0.000 |

全設問の AP@3 を平均したものが **最終 MAP@3 スコア** (0.0 〜 1.0)。

## 数式

$$
\text{MAP@3} = \frac{1}{N}\sum_{q=1}^{N}\sum_{k=1}^{\min(3,n_q)} \frac{P(k) \cdot \text{rel}_q(k)}{\min(3, R_q)}
$$

- $N$: 設問数
- $R_q$: 設問 q の正解数（このコンペでは常に 1）
- $\text{rel}_q(k)$: 順位 k の予測が正解なら 1、それ以外 0
- $P(k)$: 順位 k までの precision

## 戦略的含意

- **1 位を当てる価値が最大**（次点との差 0.5）→ モデルの信頼度が高い予測を 1 位に置く
- **top 3 から外すと 0** → 「確信が持てない時は無難な 3 つ」よりも「3 つを真剣に絞る」が効く
- 予測順序 = `argsort(probabilities)[-3:][::-1]` で OK

## 提出フォーマット

```csv
id,prediction
0,A B C
1,B D E
...
```

`prediction` は **スペース区切り 3 文字** (A〜E)。順序が成績に影響する。

## ローカル CV 用 Python スコアラー

```python
import numpy as np

def map_at_3(y_true: list[str], y_pred_top3: list[list[str]]) -> float:
    """y_pred_top3: 各設問の予測トップ3 (例: [['A','C','B'], ...])"""
    scores = []
    for true, pred in zip(y_true, y_pred_top3):
        if true in pred:
            rank = pred.index(true) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return float(np.mean(scores))
```

# 09. 本プロジェクトの出力フォーマット規約

> 上位 Kaggler のノートブック・GitHub の構成、Discussion での記述スタイルを参考に、本プロジェクト独自の規約として定義。

## 設計思想

- **再現性**: 後日同じ実験を回したとき、同じスコアが出る情報を全部残す
- **批評性**: スコアの良し悪しだけでなく「なぜ」「次に何をやるか」を明文化
- **可視性**: グラフ・mermaid 図で一目で把握できる
- **増分性**: 1 サイクル = 1 セット (EDA + score_report) で完結、過去サイクルを上書きしない

---

## 1. EDA レポート: `report/NN_eda.md`

### 必須セクション

```markdown
# Cycle NN — EDA (created: YYYY-MM-DD)

## データ概観
- train: <rows> × <cols>
- test : <rows> × <cols>
- カラム: prompt, A, B, C, D, E, (answer)

## 欠損 / 重複
| カラム | 欠損数 | 重複数 |
|---|---|---|

## ラベル分布
（train のみ。A/B/C/D/E がどれくらい偏っているか）

![label dist](img/NN_label_dist.png)

## テキスト長分布
- prompt: 平均/中央/最大/p95
- options: 平均/中央/最大/p95

![text length](img/NN_text_length.png)

## サンプル例 (難易度別 3 件)
1. 易: <prompt と answer>
2. 中: ...
3. 難: ...

## 観察された課題
- [ ] 課題 1: ...
- [ ] 課題 2: ...

## 解決策候補（次サイクルへの種）
- 案 A: ...
- 案 B: ...

## 参考 Notebook / Discussion
- [link 1](url)
```

---

## 2. スコアレポート: `report/NN_score_report.md`

### 必須セクション

```markdown
# Cycle NN — Score Report (created: YYYY-MM-DD)

## 実験 ID
`NN_<手法略称>` 例: `01_tfidf_baseline`, `02_bm25_dense_hybrid`

## モデル / パイプライン
（mermaid 図）

\`\`\`mermaid
flowchart LR
    Q[Question] --> R[Retrieval]
    ...
\`\`\`

## ハイパーパラメータ
| 項目 | 値 |
|---|---|
| seed | 42 |
| folds | 5 |
| ... | ... |

## スコア
| 指標 | CV | Public LB | Private LB |
|---|---|---|---|
| MAP@3 | 0.xxx | 0.xxx | 0.xxx (Late) |

## 推論時間 / コスト
- ローカル: <分>
- (Kaggle Notebook 想定: <時間>)

## 学び (今回判明したこと)
- ✅ ...
- ⚠️ ...

## 次の打ち手 (Next Cycle Candidates)
1. ...
2. ...

## 批評 (Critique)
（自分の手法を他手法 / SOTA と比べて率直に評価）

## 参考にした Notebook / Kaggler
- [Hippocampus's Garden](https://hippocampus-garden.com/kaggle_llm/)
- [...](...)

## Reproducibility
\`\`\`bash
uv run python NN_train_pred.py
\`\`\`
コミット: `<sha>`
```

---

## 3. 実行スクリプト: `NN_train_pred.py`

### 冒頭テンプレ (必須)

```python
"""
# Cycle NN
# Model: <手法名>
# Created: YYYY-MM-DD
# Author: Claude Code (with kuru-bushi)
# Score (CV): TBD
# Score (LB): TBD
# Notes: <一行の意図>
"""
```

### 末尾の慣行

- `if __name__ == "__main__":` で実行可能
- CV score を **stdout に明示** (`print(f"CV MAP@3: {score:.4f}")`)
- 同時に **`report/NN_score_report.md` の該当行を自動更新** (string replace で十分)
- `submission.csv` を **プロジェクトルートに保存** (`scripts/submit.py` がこれを拾う)

---

## 4. task_board.md の運用

```markdown
# Task Board

## In Progress
- [ ] Cycle 02: BM25 retrieval 導入

## TODO
- [ ] Cycle 03: DeBERTa fine-tune
- [ ] Cycle 04: ensemble

## Done
- [x] Cycle 01: TF-IDF baseline (CV: 0.65, LB: -)
```

---

## 5. session ログ (自動)

`.claude/hooks/append_session.py` が自動生成（CLAUDE.md 参照）。
**commit しない** (`.gitignore` 済み)。

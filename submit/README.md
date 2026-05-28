# submit/ — Kaggle 提出ファイルのスナップショット

このディレクトリには、**実際に Kaggle に提出した Notebook ソース + `kernel-metadata.json`** をサイクル毎にまとめて保存する（提出履歴のスナップショット）。

## 命名規約

```
submit/
  NN_<purpose>/                       # NN = Cycle 番号 (01, 02, ...)
    <code_file>.py                    # Notebook 本体
    kernel-metadata.json              # Kaggle CLI 用 metadata
```

- `NN` = サイクル番号（このサイクルで何回目の提出か、ではなく **Cycle 番号**）。
  同 Cycle 内で複数の派生を提出した場合は `NN_<purpose-a>`, `NN_<purpose-b>` のように
  `<purpose>` で枝分かれを表現する。
- 1 度 Kaggle に submit したファイルは、後で改良したくなっても **このディレクトリ内のスナップショットは書き換えない**。
  再提出するなら `submit/NN_<purpose>_v2/` 等で別ディレクトリにする。
  → 「あの時実際に提出した中身」がいつでも辿れる状態を保つ。

## 提出履歴（Cycle 01）

| Cycle | dir | submission ref | Public LB | Private LB | Date |
|---|---|---|---|---|---|
| 01 | `01_m1_microsoft/` | 53093495 | 0.388056 | 0.378399 | 2026-05-27 |
| 01 | `01_m2_openassistant/` | 53113415 | **0.682480** | **0.714337** | 2026-05-28 |
| 01 | `01_m3_deepset/` | 53113423 | 0.592176 | 0.605449 | 2026-05-28 |

詳細スコアと考察は `report/01_score_report.md` を参照。

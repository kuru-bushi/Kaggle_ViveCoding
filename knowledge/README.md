# knowledge/ — 知識ベース

Kaggler 手法 / Cycle 専用調査 / 人間調査メモを集約するディレクトリ。**3 つの階層**で使い分ける。

## 階層

```
knowledge/
  NN_<title>.md          ← Cycle 横断で参照する一般知識（番号 prefix で並び制御）
  NN/                    ← Cycle NN 専用の調査・提案文書
    baseline_proposal.md
    ensemble_methods.md
    alternative_<name>.md   ← 採用しなかったが将来戻る可能性のある案
  search/                ← 人間が手動で調べた調査結果
    NN_<title>.md        ← 連番 prefix で並び制御
```

## 使い分け

| 用途 | 置き場 | 例 |
|---|---|---|
| **Cycle NN に強く紐づく**（採用案・別案・ensemble 詳細・hyperparam tuning 結果など） | `knowledge/NN/<title>.md` | `knowledge/01/baseline_proposal.md` |
| **Cycle を跨いで参照する** 一般知識（kaggler 手法サマリ、auth/submission 手順、output_format 等） | `knowledge/<NN>_<title>.md` (直下、番号 prefix で並びを制御) | `knowledge/09_output_format.md` |
| **採用しなかったが将来戻る可能性**がある案 | `knowledge/NN/alternative_<name>.md` | `knowledge/01/alternative_tfidf_baseline.md` |
| **人間が手動で調べた調査結果**（Claude 出力ではなくユーザー収集メモ、Web 記事の写し、外部資料の要約など） | `knowledge/search/NN_<title>.md` (`00_title.md` 形式、連番で並び制御) | `knowledge/search/00_ensemble_pipeline_origin_validity.md` |

## `knowledge/search/` Q&A 運用

`knowledge/search/<file>.md` の **内容について追加質問** を受けた場合の運用:

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

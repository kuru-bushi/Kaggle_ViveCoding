# kaggle/ — Kaggle Notebook (Submit 用コード)

**Kaggle Code Competition への提出向け Notebook** を置くディレクトリ。ローカル学習スクリプト (`NN_train_pred.py`) とは別物として扱う。

## レイアウト

```
kaggle/
  NN_<purpose>.py             ← Notebook 本体（または .ipynb）
  kernel-metadata.json        ← Kaggle CLI 用 metadata
  m2_openassistant/           ← 派生 Notebook はサブディレクトリで分ける
    01_v1_m2_openassistant.py
    kernel-metadata.json
  m3_deepset/
    ...
```

## Submit 用コードの規約

| 項目 | 内容 |
|---|---|
| Checkpoint 保存 | **付けない** (Notebook は 1 ラン完結、scoring は fresh re-run なので checkpoint 不要) |
| `--resume` | 不要 |
| Model weights | 既存の **public Kaggle Dataset を input にマウント** して読込（internet OFF 制約下で動作するように）。Notebook 内 DL は避ける |
| 実行時間 | **最短設計**を優先。Kaggle 9h GPU 制限内に余裕を持って収める |
| 命名 | `kaggle/NN_<purpose>.py` (例: `kaggle/01_v1_baseline.py`) |
| metadata | 同フォルダに `kernel-metadata.json` を配置、`kaggle kernels push` で submit |
| Visibility | デフォルト **private** |
| Output | `/kaggle/working/submission.csv` |
| 単一 GPU | `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` を **torch import 前に**設定（DataParallel のメモリ二重消費を防ぐ） |

> ローカル学習でモデルが完成したら、その重みを Kaggle Dataset 化して Notebook で読み込む構成 (inference-only Notebook) も可。Cycle 02 以降で検討。

## Kaggle Notebook で選択可能な GPU

| GPU | `machine_shape` 値 | VRAM | Compute Cap. | bf16 | 強い場面 |
|---|---|---|---|---|---|
| **NVIDIA T4 x2** | `NvidiaTeslaT4` | 16GB × 2 = 32GB | sm_75 (Turing) | ❌ | Inference, 中-小 model (fp16), 並列推論、デフォルト推奨 |
| **NVIDIA P100** | `NvidiaTeslaP100` | 16GB × 1 (HBM2) | sm_60 (Pascal) | ❌ | 大バッチ・vision 中心。**ただし最新 PyTorch (sm_70+) と非互換で SDK_no_kernel_image エラー多発、現実的には避ける** |
| **TPU v3-8** | `Tpu1VmV38` | 128GB HBM | — | ✅ | TPU 最適化済 (Flax/JAX or torch_xla) フレームワーク使用時のみ |

### 選定基準

1. **デフォルトで `NvidiaTeslaT4` を選ぶ** — sm_75 で最新 PyTorch とほぼ全互換、2 GPU で並列処理も可能
2. P100 はメモリ帯域 (HBM2) が魅力だが **2024 年以降の PyTorch wheel が sm_60 を切り捨てた** ため、in-notebook で `torch` を使う場合は P100 を避ける (TF/JAX 等なら別途検討)
3. fp16 を使うなら T4 (TensorCore 利用可)。bf16 が欲しい場合は A100 が必要だが Kaggle にはない → fp16 で代用
4. TPU は xla 等への移植コストが高く、ピンポイントに最適化された Notebook 以外では使われない

→ **本プロジェクト Cycle 01-N は `NvidiaTeslaT4` 固定**。

## Submit の手順

Kaggle Code Competition では **kernel 経由必須**（CSV 直接 upload は 400 Bad Request）。

```bash
# 1. kernel push
~/.local/bin/uv run kaggle kernels push -p kaggle/<dir>/

# 2. 完了確認
~/.local/bin/uv run kaggle kernels status <user>/<kernel-id>

# 3. competition へ submit
~/.local/bin/uv run kaggle competitions submit \
  -c kaggle-llm-science-exam \
  -k <user>/<kernel-id> \
  -v <version-num> \
  -f submission.csv \
  -m "submission message"
```

- `-k`: kernel ID
- `-v`: kernel version (push のたびに 1, 2, ... と増える)
- `-f`: **kernel output 内の filename**（ローカルパスではない）

提出後の history は `~/.local/bin/uv run kaggle competitions submissions -c kaggle-llm-science-exam`。

## 提出物のスナップショット保存

実際に Kaggle に提出した Notebook + `kernel-metadata.json` は **書き換えない形で** [`submit/NN_<purpose>/`](../submit/) に保管する。詳細は [`submit/README.md`](../submit/README.md)。

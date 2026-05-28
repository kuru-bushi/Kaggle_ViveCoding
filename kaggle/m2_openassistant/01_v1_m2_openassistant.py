"""
Cycle 01 v1 — Kaggle Submission Notebook (Code Competition)

Model: microsoft/deberta-v3-large (MCQ head) — loaded from radek1's pre-uploaded dataset.
Strategy: train on official train.csv (200 rows, ~80/20 split) for 3 epochs, infer on official test.csv.

Design choices for **minimum execution time**:
- Pre-uploaded model dataset → no in-notebook download
- No checkpointing (Notebook is single-shot, fresh re-run for scoring)
- fp16 + gradient checkpointing
- No CV folds (single 80/20 split for eval signal)
- max_length: 256 train / 384 infer

Inputs (mounted by Kaggle):
- /kaggle/input/kaggle-llm-science-exam/{train.csv,test.csv,sample_submission.csv}
- /kaggle/input/deberta-v3-large-hf-weights/{config.json,pytorch_model.bin,spm.model,tokenizer_config.json}

Output:
- /kaggle/working/submission.csv

Author: Claude Code (with kuru-bushi)
"""
from __future__ import annotations

import os

# Use single GPU only to avoid DataParallel memory duplication (T4 ×2 → OOM at batch≥4)
# MUST be set BEFORE any torch / transformers import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import glob
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def _debug_inputs() -> None:
    print("=== /kaggle/input/ tree (depth 3) ===", flush=True)
    try:
        out = subprocess.run(
            ["find", "/kaggle/input", "-maxdepth", "3", "-printf", "%p (%s bytes)\n"],
            capture_output=True, text=True, timeout=10,
        )
        print(out.stdout[:3000], flush=True)
    except Exception as e:
        print(f"  (find failed: {e})", flush=True)


def _find_one(*names: str, root: str = "/kaggle/input") -> str:
    """Recursively find first file matching any of `names` under root."""
    for nm in names:
        hits = glob.glob(f"{root}/**/{nm}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {names} under {root}")


def _find_model_dir(root: str = "/kaggle/input") -> str:
    """Find directory containing config.json AND a weights file (pytorch_model.bin or safetensors)."""
    for cfg in glob.glob(f"{root}/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if os.path.exists(f"{d}/pytorch_model.bin") or glob.glob(f"{d}/*.safetensors"):
            return d
    raise FileNotFoundError("no model dir with config.json + weights under /kaggle/input/")


_debug_inputs()
MODEL_PATH = _find_model_dir()
TRAIN_CSV = _find_one("train.csv")
TEST_CSV = _find_one("test.csv")
OUTPUT_CSV = "/kaggle/working/submission.csv"
print(f"MODEL_PATH={MODEL_PATH}")
print(f"TRAIN_CSV={TRAIN_CSV}")
print(f"TEST_CSV={TEST_CSV}")

OPTIONS = ["A", "B", "C", "D", "E"]
MAX_LEN_TRAIN = 256
MAX_LEN_INFER = 384
SEED = 42
EPOCHS = 3
LR = 1e-5
BATCH_TRAIN = 2   # single T4 16GB: batch>2 + DeBERTa-large OOMs even fp16
BATCH_EVAL = 4
GRAD_ACCUM = 8   # effective batch = 16


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MCQDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, has_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_label = has_label

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        options = [str(row[o]) for o in OPTIONS]
        enc = self.tokenizer(
            [prompt] * 5,
            options,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v for k, v in enc.items()}
        if self.has_label and "answer" in row and pd.notna(row["answer"]):
            item["label"] = torch.tensor(OPTIONS.index(row["answer"]), dtype=torch.long)
        return item


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        if k == "label":
            out["labels"] = torch.stack([b["label"] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def map_at_3(y_true_idx: list[int], probs: np.ndarray) -> float:
    top3 = np.argsort(-probs, axis=1)[:, :3]
    s = []
    for i, t in enumerate(y_true_idx):
        if t in top3[i]:
            r = list(top3[i]).index(t) + 1
            s.append(1.0 / r)
        else:
            s.append(0.0)
    return float(np.mean(s))


def compute_metrics_fn(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return {"map@3": map_at_3(labels.tolist(), probs)}


class EpochTimerCallback(TrainerCallback):
    """Print epoch number and elapsed time at every epoch end (project convention)."""

    def __init__(self):
        self.t0 = None
        self.te = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        print(f"[train] begin at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.te = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.t0 is None or self.te is None:
            return
        now = time.time()
        ep, tot = now - self.te, now - self.t0
        epoch = state.epoch if state.epoch is not None else 0.0
        print(
            f"[epoch {epoch:5.2f}] epoch_elapsed={ep:6.1f}s "
            f"total_elapsed={tot/60:5.1f}min step={state.global_step}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.t0 is None:
            return
        tot = time.time() - self.t0
        print(f"[train] done in {tot/60:.1f}min ({tot:.0f}s)", flush=True)


def make_training_args(output_dir: str) -> TrainingArguments:
    """No checkpointing per Submit-code convention (`save_strategy='no'`).

    Notes:
    - gradient_checkpointing OFF: known to interact badly with fp16+DeBERTa in some HF versions
      (and unnecessary on T4 16GB for DeBERTa-large at batch=4)
    - fp16 ON: T4 supports fp16 natively (bf16 not supported on Turing)
    """
    common = dict(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.01,
        learning_rate=LR,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=5,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=SEED,
        gradient_checkpointing=False,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def main() -> None:
    set_seed(SEED)
    t_start = time.time()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"train: {train.shape}  test: {test.shape}")
    print(f"GPU available: {torch.cuda.is_available()}, count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

    print(f"loading model from {MODEL_PATH} ...")
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)
    # Ensure all params are fp32 so AMP GradScaler can unscale properly
    # (fixes "Attempting to unscale FP16 gradients" on some transformers versions)
    model = model.float()
    print(f"  model loaded in {time.time()-t:.1f}s, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, dtype: {next(model.parameters()).dtype}")

    # 80/20 split for eval signal (no CV to minimize execution time)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train))
    n_val = max(1, len(train) // 5)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    tr_df, va_df = train.iloc[tr_idx], train.iloc[val_idx]
    print(f"split: train={len(tr_df)} val={len(va_df)}")

    train_ds = MCQDataset(tr_df, tokenizer, MAX_LEN_TRAIN, has_label=True)
    val_ds = MCQDataset(va_df, tokenizer, MAX_LEN_TRAIN, has_label=True)

    args = make_training_args("/kaggle/working/tmp_train")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=[EpochTimerCallback()],
    )
    trainer.train()

    final_eval = trainer.evaluate()
    print(f"\nfinal eval: {final_eval}")
    print(f"val MAP@3: {final_eval.get('eval_map@3', float('nan')):.4f}")

    # Inference on official test
    print("\ninferring on test ...")
    t = time.time()
    test_ds = MCQDataset(test, tokenizer, MAX_LEN_INFER, has_label=False)
    raw = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(raw.predictions), dim=-1).numpy()
    print(f"  inference in {time.time()-t:.1f}s")

    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    predictions = [" ".join(OPTIONS[i] for i in row) for row in top3_idx]
    sub = pd.DataFrame({"id": test["id"], "prediction": predictions})
    sub.to_csv(OUTPUT_CSV, index=False)
    print(f"saved {OUTPUT_CSV}: {sub.shape}")
    print(f"first 3 rows:\n{sub.head(3).to_string()}")

    print(f"\n==== total wall time: {(time.time()-t_start)/60:.1f}min ====")


if __name__ == "__main__":
    main()

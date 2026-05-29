"""
Cycle 01 v1 — Kaggle Submission Notebook (Code Competition) — 3-MODEL ENSEMBLE

Ensembles the SAME three DeBERTa-v3-large variants already submitted as m1/m2/m3:
  m1: microsoft/deberta-v3-large                          (radek1/deberta-v3-large-hf-weights)
  m2: OpenAssistant/reward-model-deberta-v3-large-v2      (nags98/openassistantreward-model-deberta-v3-large-v2)
  m3: deepset/deberta-v3-large-squad2                     (katwooo/deberta-v3-large-squad2)

Aggregation = **mean + max blending of softmax probabilities** (days 7th-place construction).
  score_q(label) = mean_models(prob) + max_models(prob)   per test id, per option
  -> take top-3 options by descending score.
Reference: knowledge/01/ensemble_methods.md (days 7th place writeup, "4. ensemble").
Rationale: this is the documented Kaggler ensemble that uses these exact 3 models, so we
reuse its aggregation rather than a plain equal-weight mean (user instruction 2026-05-29).

Each model is fine-tuned fresh on official train.csv (the saved m1/m2/m3 submission CSVs hold
only top-3 letters, not probabilities, so post-hoc CSV averaging is impossible). Models are
trained/inferred sequentially with del + empty_cache between them to stay within T4 16GB.

Design choices for minimum execution time (per kaggle/README.md Submit-code convention):
- Pre-uploaded model datasets -> no in-notebook download
- No checkpointing (Notebook is single-shot; scoring is a fresh re-run)
- fp16, single GPU
- One fixed 80/20 split (SEED=42) shared across all 3 models for a comparable val signal

Inputs (mounted by Kaggle):
- /kaggle/input/kaggle-llm-science-exam/{train.csv,test.csv,sample_submission.csv}
- three model weight datasets (each: config.json + spm.model + pytorch_model.bin/safetensors)
Output:
- /kaggle/working/submission.csv

Author: Claude Code (with kuru-bushi)
"""
from __future__ import annotations

import os

# Single GPU only to avoid DataParallel memory duplication (T4 x2 -> OOM at batch>=4).
# MUST be set BEFORE any torch / transformers import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
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
        print(out.stdout[:4000], flush=True)
    except Exception as e:
        print(f"  (find failed: {e})", flush=True)


def _find_one(*names: str, root: str = "/kaggle/input") -> str:
    """Recursively find first file matching any of `names` under root."""
    for nm in names:
        hits = glob.glob(f"{root}/**/{nm}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {names} under {root}")


def _find_all_model_dirs(root: str = "/kaggle/input") -> list[str]:
    """Find every directory containing config.json AND a weights file.

    For the 3-model ensemble we expect EXACTLY 3 such dirs (one per mounted model
    dataset). Fail loud otherwise: a silent find-2 would yield a wrong ensemble that
    we cannot undo on a one-shot submission.
    """
    dirs = []
    for cfg in sorted(glob.glob(f"{root}/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if os.path.exists(f"{d}/pytorch_model.bin") or glob.glob(f"{d}/*.safetensors"):
            if d not in dirs:
                dirs.append(d)
    print(f"=== model dirs discovered ({len(dirs)}) ===", flush=True)
    for d in dirs:
        print(f"  {d}", flush=True)
    assert len(dirs) == 3, (
        f"expected exactly 3 model dirs for the ensemble, found {len(dirs)}: {dirs}. "
        "Check kernel-metadata.json dataset_sources lists all 3 model datasets."
    )
    return dirs


OPTIONS = ["A", "B", "C", "D", "E"]
MAX_LEN_TRAIN = 256
MAX_LEN_INFER = 384
SEED = 42
EPOCHS = 3
LR = 1e-5
BATCH_TRAIN = 2   # single T4 16GB: batch>2 + DeBERTa-large OOMs even fp16
BATCH_EVAL = 4
GRAD_ACCUM = 8    # effective batch = 16


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
    """No checkpointing per Submit-code convention (`save_strategy='no'`)."""
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


def train_and_predict(model_dir: str, tr_df, va_df, test_df, tag: str):
    """Fine-tune one model on tr_df; return (val_probs, test_probs) as softmax arrays."""
    print(f"\n========== [{tag}] {model_dir} ==========", flush=True)
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)
    # Ensure fp32 params so AMP GradScaler can unscale (fixes "unscale FP16 gradients").
    model = model.float()
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded in {time.time()-t:.1f}s, {n_param:.1f}M params", flush=True)

    train_ds = MCQDataset(tr_df, tokenizer, MAX_LEN_TRAIN, has_label=True)
    val_ds = MCQDataset(va_df, tokenizer, MAX_LEN_TRAIN, has_label=True)
    test_ds = MCQDataset(test_df, tokenizer, MAX_LEN_INFER, has_label=False)

    args = make_training_args(f"/kaggle/working/tmp_{tag}")
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

    val_raw = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.tensor(val_raw.predictions), dim=-1).numpy()
    test_raw = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_raw.predictions), dim=-1).numpy()

    # Free GPU memory before the next model.
    del trainer, model, tokenizer, train_ds, val_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_probs, test_probs


def main() -> None:
    set_seed(SEED)
    t_start = time.time()

    _debug_inputs()
    model_dirs = _find_all_model_dirs()
    train_csv = _find_one("train.csv")
    test_csv = _find_one("test.csv")
    output_csv = "/kaggle/working/submission.csv"
    print(f"TRAIN_CSV={train_csv}\nTEST_CSV={test_csv}", flush=True)

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    print(f"train: {train.shape}  test: {test.shape}", flush=True)
    print(f"GPU available: {torch.cuda.is_available()}, count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)", flush=True)

    # One fixed 80/20 split shared across all models (comparable val signal).
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train))
    n_val = max(1, len(train) // 5)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    tr_df, va_df = train.iloc[tr_idx], train.iloc[val_idx]
    va_labels = [OPTIONS.index(a) for a in va_df["answer"].tolist()]
    print(f"split: train={len(tr_df)} val={len(va_df)}", flush=True)

    # Train each model, collect per-model softmax probs.
    val_stack, test_stack = [], []
    for k, mdir in enumerate(model_dirs, 1):
        tag = f"m{k}"
        vp, tp = train_and_predict(mdir, tr_df, va_df, test, tag)
        val_stack.append(vp)
        test_stack.append(tp)
        print(f"  [{tag}] val MAP@3 = {map_at_3(va_labels, vp):.4f}", flush=True)

    val_stack = np.stack(val_stack, axis=0)    # (n_models, n_val, 5)
    test_stack = np.stack(test_stack, axis=0)  # (n_models, n_test, 5)

    # --- days 7th-place aggregation: mean + max over models, per id/option ---
    def mean_plus_max(stack: np.ndarray) -> np.ndarray:
        return stack.mean(axis=0) + stack.max(axis=0)

    print("\n=== validation MAP@3 ===", flush=True)
    for k in range(val_stack.shape[0]):
        print(f"  m{k+1} alone        : {map_at_3(va_labels, val_stack[k]):.4f}", flush=True)
    print(f"  ensemble (mean)   : {map_at_3(va_labels, val_stack.mean(axis=0)):.4f}", flush=True)
    print(f"  ensemble (mean+max): {map_at_3(va_labels, mean_plus_max(val_stack)):.4f}  <-- submitted", flush=True)

    # Final test predictions via mean+max blending.
    test_score = mean_plus_max(test_stack)
    top3_idx = np.argsort(-test_score, axis=1)[:, :3]
    predictions = [" ".join(OPTIONS[i] for i in row) for row in top3_idx]
    sub = pd.DataFrame({"id": test["id"], "prediction": predictions})
    sub.to_csv(output_csv, index=False)
    print(f"\nsaved {output_csv}: {sub.shape}", flush=True)
    print(f"first 3 rows:\n{sub.head(3).to_string()}", flush=True)
    print(f"\n==== total wall time: {(time.time()-t_start)/60:.1f}min ====", flush=True)


if __name__ == "__main__":
    main()

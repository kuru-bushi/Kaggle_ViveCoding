"""
# Cycle 01 v1
# Model: microsoft/deberta-v3-large (MCQ head)
# Created: 2026-05-27
# Author: Claude Code (with kuru-bushi)
# Score (CV): TBD
# Score (LB): TBD
# Notes:
#   Simplified version of "days" 7th place approach.
#   - No Wikipedia retrieval (added in Cycle 01 v2)
#   - No TTA, no ensemble across models (Cycle 03)
#   - Cycle 02 will switch wiki dump to cirrussearch
#   See: knowledge/01/baseline_proposal.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

# Engage hf_transfer (Rust, parallel-chunk) for HF downloads. Must be set before
# `huggingface_hub` / `transformers` imports. Falls back gracefully if hf_transfer
# is not installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

_LOCAL_MODEL_PATH = "data/tmp/deberta_v3_large"
MODEL_NAME = _LOCAL_MODEL_PATH if os.path.isdir(_LOCAL_MODEL_PATH) and os.path.exists(f"{_LOCAL_MODEL_PATH}/config.json") else "microsoft/deberta-v3-large"
OPTIONS = ["A", "B", "C", "D", "E"]
N_FOLDS = 5
MAX_LEN_TRAIN = 256
MAX_LEN_INFER = 384  # smaller than 786 (article spec) for v1 speed
SEED = 42


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MCQDataset(Dataset):
    """Each row: tokenize (prompt, option_k) for k in A..E -> 5 input pairs."""

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
            out[k] = torch.stack([b[k] for b in batch])  # (B, 5, L)
    return out


def map_at_3(y_true_idx: list[int], probs: np.ndarray) -> float:
    top3 = np.argsort(-probs, axis=1)[:, :3]
    scores = []
    for i, t in enumerate(y_true_idx):
        if t in top3[i]:
            rank = list(top3[i]).index(t) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def compute_metrics_fn(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return {"map@3": map_at_3(labels.tolist(), probs)}


def make_training_args(
    output_dir: str,
    epochs: int,
    steps_per_epoch: int,
    save_every_epochs: int,
) -> TrainingArguments:
    """Configure HF Trainer args.

    - save_strategy="steps" with save_steps = steps_per_epoch * save_every_epochs
      → resume-safe checkpoint at most every `save_every_epochs` epochs.
    - eval_strategy="epoch" stays, for finer-grained metric monitoring.
    - load_best_model_at_end=False (decoupled so eval/save cadence can differ).
    """
    save_steps = max(1, int(steps_per_epoch * save_every_epochs))
    common = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.01,
        learning_rate=1e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        logging_steps=5,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        seed=SEED,
        gradient_checkpointing=True,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def find_free_port(start: int = 6006, n_tries: int = 30) -> int | None:
    """Find first free TCP port starting at `start`."""
    for p in range(start, start + n_tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return None


def launch_tensorboard(logdir: str = "models/") -> tuple[subprocess.Popen | None, str | None]:
    """Spawn a TensorBoard background server. Returns (process, url) or (None, None)."""
    tb_bin = Path(".venv/bin/tensorboard")
    if not tb_bin.exists():
        return None, None
    Path(logdir).mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    if port is None:
        return None, None
    proc = subprocess.Popen(
        [str(tb_bin), "--logdir", logdir, "--port", str(port), "--bind_all", "--reload_interval", "5"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, f"http://localhost:{port}/"


class EpochTimerCallback(TrainerCallback):
    """Print epoch number + elapsed time at every epoch_end (Japanese-friendly format)."""

    def __init__(self) -> None:
        self.t_train_start: float | None = None
        self.t_epoch_start: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_train_start = time.time()
        print(f"[train] begin at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.t_epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.t_train_start is None or self.t_epoch_start is None:
            return
        now = time.time()
        ep_sec = now - self.t_epoch_start
        total_sec = now - self.t_train_start
        epoch = state.epoch if state.epoch is not None else 0.0
        print(
            f"[epoch {epoch:5.2f}] epoch_elapsed={ep_sec:6.1f}s "
            f"total_elapsed={total_sec/60:5.1f}min "
            f"step={state.global_step}",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.t_train_start is None:
            return
        total_sec = time.time() - self.t_train_start
        print(f"[train] done in {total_sec/60:.1f}min ({total_sec:.0f}s)", flush=True)


def plot_loss_curve(log_history: list[dict], fold: int, out_path: Path) -> None:
    """Save a static loss/metric curve PNG from Trainer log_history.

    Twin-axis: left = train/eval loss, right = eval MAP@3.
    Output: report/img/01_loss_fold{N}.png
    """
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_maps = [], [], []
    for e in log_history:
        if "loss" in e and "eval_loss" not in e and "epoch" in e:
            train_steps.append(e.get("step", len(train_steps)))
            train_losses.append(e["loss"])
        if "eval_loss" in e:
            eval_steps.append(e.get("step", len(eval_steps)))
            eval_losses.append(e["eval_loss"])
            if "eval_map@3" in e:
                eval_maps.append(e["eval_map@3"])

    if not train_losses and not eval_losses:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax1.plot(train_steps, train_losses, "b-", linewidth=1, label="train loss")
    if eval_losses:
        ax1.plot(eval_steps, eval_losses, "r-o", linewidth=1.5, markersize=6, label="eval loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_title(f"Cycle 01 — fold {fold} loss / MAP@3")

    if eval_maps:
        ax2 = ax1.twinx()
        ax2.plot(eval_steps, eval_maps, "g-^", linewidth=1.5, markersize=6, label="eval MAP@3")
        ax2.set_ylabel("MAP@3")
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def dump_log_history(log_history: list[dict], fold: int, out_path: Path) -> None:
    """Persist raw Trainer log_history as JSON for later re-plotting / analysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log_history, indent=2), encoding="utf-8")


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
    epochs: int,
    save_every_epochs: int,
    resume: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
    # Ensure fp32 weights so AMP GradScaler can unscale (fixes "Attempting to unscale FP16 gradients"
    # observed on HF transformers v5.x with DeBERTa + fp16 + gradient_checkpointing).
    model = model.float()
    train_ds = MCQDataset(train_df, tokenizer, MAX_LEN_TRAIN, has_label=True)
    val_ds = MCQDataset(val_df, tokenizer, MAX_LEN_TRAIN, has_label=True)
    # Calc steps_per_epoch (one optimizer step per `accum * world_batch` samples)
    effective_batch = 2 * 8  # per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch))

    output_dir = f"models/tmp_01_fold{fold}"
    args = make_training_args(output_dir, epochs, steps_per_epoch, save_every_epochs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=[EpochTimerCallback()],
    )

    # Resume detection: if a checkpoint exists in output_dir, --resume picks it up automatically.
    has_checkpoint = any(Path(output_dir).glob("checkpoint-*")) if Path(output_dir).exists() else False
    if resume and has_checkpoint:
        print(f"[resume] fold {fold}: resuming from latest checkpoint in {output_dir}", flush=True)
        trainer.train(resume_from_checkpoint=True)
    else:
        if resume and not has_checkpoint:
            print(f"[resume] fold {fold}: no checkpoint found, starting fresh", flush=True)
        trainer.train()

    # Always save the final state so end-of-training is recoverable even if save_steps > total_steps
    trainer.save_model(f"{output_dir}/checkpoint-final")
    print(f"[save] fold {fold}: final model -> {output_dir}/checkpoint-final", flush=True)

    metrics = trainer.evaluate()
    # Persist loss curves (static PNG + raw JSON) for the score report and future analysis.
    plot_loss_curve(trainer.state.log_history, fold, Path(f"report/img/01_loss_fold{fold}.png"))
    dump_log_history(trainer.state.log_history, fold, Path(f"report/img/01_log_history_fold{fold}.json"))
    return trainer, tokenizer, metrics


def predict(trainer, tokenizer, test_df: pd.DataFrame, max_len: int) -> np.ndarray:
    test_ds = MCQDataset(test_df, tokenizer, max_len, has_label=False)
    raw = trainer.predict(test_ds)
    return torch.softmax(torch.tensor(raw.predictions), dim=-1).numpy()


def update_score_report(cv_mean: float, cv_scores: list[float]) -> None:
    path = Path("report/01_score_report.md")
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    folds_str = ", ".join(f"{s:.4f}" for s in cv_scores)
    updated = re.sub(
        r"\| CV MAP@3 \| [^|]* \|",
        f"| CV MAP@3 | **{cv_mean:.4f}** (folds: {folds_str}) |",
        content,
        count=1,
    )
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-fold", action="store_true", help="smoke test: fold 0 only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--save-every-epochs", type=int, default=5,
                        help="checkpoint save cadence (default 5; satisfies project convention)")
    parser.add_argument("--resume", action="store_true",
                        help="resume each fold from latest checkpoint in models/tmp_01_fold{N}/ if exists")
    parser.add_argument("--no-tensorboard", action="store_true", help="skip launching TensorBoard server")
    args = parser.parse_args()

    set_seed(SEED)
    train = pd.read_csv("data/train/train.csv")
    test = pd.read_csv("data/test/test.csv")
    print(f"train: {train.shape}, test: {test.shape}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"checkpoint cadence: every {args.save_every_epochs} epoch(s), resume={args.resume}")

    # Launch TensorBoard server (non-blocking). Banner with URL.
    tb_proc = None
    if not args.no_tensorboard:
        tb_proc, tb_url = launch_tensorboard("models/")
        if tb_url:
            print()
            print("=" * 64)
            print(f"📊 TensorBoard URL: {tb_url}")
            print(f"   logdir: models/  (refresh every 5s)")
            print(f"   stop:   kill {tb_proc.pid}")
            print("=" * 64)
            print()
        else:
            print("[warn] TensorBoard launch skipped (binary not found or no free port)")

    n_folds = 1 if args.single_fold else args.n_folds
    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=SEED)
    cv_scores: list[float] = []
    test_probs_folds: list[np.ndarray] = []

    try:
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
            if fold >= n_folds:
                break
            print(f"\n=== Fold {fold + 1}/{n_folds} ===")
            tr_df = train.iloc[tr_idx]
            va_df = train.iloc[va_idx]
            trainer, tokenizer, metrics = train_one_fold(
                tr_df, va_df, fold, args.epochs, args.save_every_epochs, args.resume
            )
            cv_scores.append(metrics["eval_map@3"])
            print(f"fold {fold} eval map@3 = {metrics['eval_map@3']:.4f}")

            probs = predict(trainer, tokenizer, test, MAX_LEN_INFER)
            test_probs_folds.append(probs)

            del trainer, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_probs = np.mean(test_probs_folds, axis=0)
        top3_idx = np.argsort(-avg_probs, axis=1)[:, :3]
        predictions = [" ".join(OPTIONS[i] for i in row) for row in top3_idx]

        sub = pd.DataFrame({"id": test["id"], "prediction": predictions})
        sub.to_csv("submission.csv", index=False)

        cv_mean = float(np.mean(cv_scores))
        print(f"\nCV MAP@3: {cv_mean:.4f}  (folds: {[round(s, 4) for s in cv_scores]})")
        print(f"submission.csv saved ({len(sub)} rows)")
        update_score_report(cv_mean, cv_scores)
    finally:
        if tb_proc is not None and tb_proc.poll() is None:
            print(f"\n[tensorboard] still running (PID {tb_proc.pid}). Kill manually when done: kill {tb_proc.pid}")


if __name__ == "__main__":
    main()

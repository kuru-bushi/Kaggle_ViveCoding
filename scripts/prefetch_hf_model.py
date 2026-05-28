"""Pre-download a HuggingFace model snapshot fast.

Why this exists:
    `AutoModel.from_pretrained(<repo>)` で直接落とすと、デフォルトは pure-Python の
    単一接続 (1 ファイル 1 ストリーム) なので、800MB+ の重みで体感が極端に遅い。
    `hf_transfer` (Rust) を有効化し、`snapshot_download(max_workers=N)` で
    ファイル並列 + 単一ファイル内チャンク並列の両方をオンにする。

Usage:
    uv run python scripts/prefetch_hf_model.py \
        --repo microsoft/deberta-v3-large \
        --dest data/tmp/deberta_v3_large \
        [--workers 8] [--include-bin] [--revision main]

Notes:
    - デフォルトは safetensors を優先取得（あれば .bin は落とさない → DL 量を半減）
    - safetensors が無いリポジトリは自動で .bin を落とす
    - `--include-bin` で両方落とす（互換性が必要な時のみ）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

# MUST be set before importing huggingface_hub for hf_transfer backend to engage
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402


DEFAULT_ALLOW = [
    "config.json",
    "tokenizer*",
    "*.txt",
    "*.json",
    "spm.model",
    "*.safetensors",
]
BIN_PATTERNS = ["*.bin"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. microsoft/deberta-v3-large")
    ap.add_argument("--dest", required=True, help="local dir to materialise files into")
    ap.add_argument("--workers", type=int, default=8, help="parallel file workers (default 8)")
    ap.add_argument("--revision", default=None, help="git revision (default: main)")
    ap.add_argument("--include-bin", action="store_true", help="also download *.bin (default: skip if safetensors present)")
    ap.add_argument("--force", action="store_true", help="redownload even if dest already has weights")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    if not args.force and _has_weights(args.dest):
        print(f"[skip] {args.dest} already has weights. Use --force to redownload.")
        return 0

    allow = list(DEFAULT_ALLOW)
    if args.include_bin:
        allow += BIN_PATTERNS

    print(f"[prefetch] repo={args.repo} dest={args.dest} workers={args.workers}")
    print(f"[prefetch] HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.dest,
        allow_patterns=allow,
        max_workers=args.workers,
    )

    if not _has_weights(local_dir):
        print("[prefetch] no safetensors found, retrying with *.bin ...")
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=args.dest,
            allow_patterns=allow + BIN_PATTERNS,
            max_workers=args.workers,
        )

    elapsed = time.time() - t0
    total = _dir_bytes(args.dest)
    mbps = (total / 1e6) / max(elapsed, 1e-3)
    print(f"[prefetch] done in {elapsed:.1f}s, {total/1e9:.2f} GB → {mbps:.1f} MB/s avg")
    print(f"[prefetch] files:")
    for f in sorted(os.listdir(args.dest)):
        p = os.path.join(args.dest, f)
        if os.path.isfile(p):
            print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)")
    return 0


def _has_weights(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    for f in os.listdir(d):
        if f.endswith(".safetensors") or f == "pytorch_model.bin" or f.endswith(".bin"):
            return True
    return False


def _dir_bytes(d: str) -> int:
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if os.path.isfile(p):
                n += os.path.getsize(p)
    return n


if __name__ == "__main__":
    sys.exit(main())

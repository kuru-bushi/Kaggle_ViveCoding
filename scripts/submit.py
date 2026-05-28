"""Kaggle submission with 1-per-day self-imposed rate limit.

See knowledge/11_kaggle_submission.md for full docs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = "kaggle-llm-science-exam"
TIMESTAMP_FILE = Path("data/tmp/last_submission.txt")
MIN_GAP_SECONDS = 24 * 3600


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _check_rate_limit(force: bool) -> None:
    if force or not TIMESTAMP_FILE.exists():
        return
    last = datetime.fromisoformat(TIMESTAMP_FILE.read_text().strip())
    gap = (_now() - last).total_seconds()
    if gap < MIN_GAP_SECONDS:
        remaining = MIN_GAP_SECONDS - gap
        h, m = int(remaining // 3600), int((remaining % 3600) // 60)
        print(
            f"ERROR: self-imposed 1/day limit. Last submit at {last.isoformat()}. "
            f"Wait {h}h {m}m, or pass --force.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="submission.csv")
    p.add_argument("--message", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="bypass 1/day guard")
    p.add_argument("--competition", default=COMPETITION)
    args = p.parse_args()

    sub = Path(args.file)
    if not sub.exists():
        print(f"ERROR: {sub} not found", file=sys.stderr)
        sys.exit(2)

    _check_rate_limit(args.force)

    cmd = ["kaggle", "competitions", "submit", "-c", args.competition,
           "-f", str(sub), "-m", args.message]
    print("Running:", " ".join(cmd))
    if args.dry_run:
        print("(dry-run, not executed)")
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(3)

    TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMESTAMP_FILE.write_text(_now().isoformat())
    print(f"OK: submission timestamp recorded at {TIMESTAMP_FILE}")


if __name__ == "__main__":
    main()

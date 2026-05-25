#!/usr/bin/env bash
# Read KAGGLE_USERNAME / KAGGLE_KEY from .env and write ~/.kaggle/kaggle.json (0600).
# Run once after editing .env. Persistent login — no need to re-run unless credentials change.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Run: cp .env.example .env  and fill in your Kaggle credentials." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "ERROR: KAGGLE_USERNAME or KAGGLE_KEY not set in .env" >&2
  exit 1
fi

mkdir -p "$HOME/.kaggle"
umask 077
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

echo "OK: $HOME/.kaggle/kaggle.json written (mode 0600)."
echo "Test: uv run kaggle competitions list | head"

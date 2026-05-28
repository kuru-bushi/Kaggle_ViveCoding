#!/usr/bin/env bash
# Download Kaggle "LLM - Science Exam" competition data into data/{train,test}/.
# Requires: kaggle CLI authenticated (run scripts/setup_kaggle.sh first) and
#           competition rules accepted at:
#           https://www.kaggle.com/competitions/kaggle-llm-science-exam/rules

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/train data/test
echo "Downloading competition data..."
uv run kaggle competitions download -c kaggle-llm-science-exam -p data/train/

cd data/train
unzip -o kaggle-llm-science-exam.zip
rm -f kaggle-llm-science-exam.zip
cd ../..

# Move test files to data/test/
mv -f data/train/test.csv data/test/ 2>/dev/null || true
mv -f data/train/sample_submission.csv data/test/ 2>/dev/null || true

echo ""
echo "=== Layout ==="
ls -la data/train/ data/test/

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"
COMPETITION="nvidia-nemotron-model-reasoning-challenge"
SUBMISSION_ZIP="${1:-$ROOT_DIR/submissions/kienngx_root/submission_root.zip}"
MESSAGE="${2:-public tinker-adapter baseline root-level zip}"
UPLOAD_ZIP="$SUBMISSION_ZIP"

if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Kaggle CLI not found at $KAGGLE_BIN" >&2
  exit 1
fi

if [[ ! -f "$SUBMISSION_ZIP" ]]; then
  echo "Submission zip not found: $SUBMISSION_ZIP" >&2
  exit 1
fi

unzip -l "$SUBMISSION_ZIP" | sed -n '1,40p'
if [[ "$(basename "$SUBMISSION_ZIP")" != "submission.zip" ]]; then
  UPLOAD_ZIP="$(dirname "$SUBMISSION_ZIP")/submission.zip"
  ln -f "$SUBMISSION_ZIP" "$UPLOAD_ZIP"
  echo "Using Kaggle-required upload file name: $UPLOAD_ZIP"
fi
"$KAGGLE_BIN" competitions submit \
  -c "$COMPETITION" \
  -f "$UPLOAD_ZIP" \
  -m "$MESSAGE"

"$KAGGLE_BIN" competitions submissions "$COMPETITION" --page-size 10

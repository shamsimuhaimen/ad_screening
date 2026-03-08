#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 \"what changed\" \"what learned\" \"next action\" [run_dir]"
  exit 1
fi

WHAT_CHANGED="$1"
WHAT_LEARNED="$2"
NEXT_ACTION="$3"
RUN_DIR="${4:-}"
TS="$(date +"%Y-%m-%d %H:%M:%S %Z")"
SHORT_TS="$(date +"%Y%m%d_%H%M%S")"

NOTES_FILE="documents/session_notes.md"
mkdir -p documents

if [[ ! -f "$NOTES_FILE" ]]; then
  cat > "$NOTES_FILE" <<'EOF'
# Session Notes

## Checkpoints
EOF
fi

if [[ -z "$RUN_DIR" ]]; then
  if ls -d results/ad_predictor_* >/dev/null 2>&1; then
    RUN_DIR="$(ls -dt results/ad_predictor_* | head -n1)"
  else
    RUN_DIR="(none)"
  fi
fi

{
  echo ""
  echo "### $SHORT_TS"
  echo "- Time: $TS"
  echo "- What Changed: $WHAT_CHANGED"
  echo "- What Learned: $WHAT_LEARNED"
  echo "- Next Action: $NEXT_ACTION"
  echo "- Run Dir: $RUN_DIR"
} >> "$NOTES_FILE"

echo "Updated $NOTES_FILE"

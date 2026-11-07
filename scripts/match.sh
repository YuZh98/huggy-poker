#!/usr/bin/env bash
# Usage: scripts/match.sh BOT1.py BOT2.py [judge args...]
#   e.g. scripts/match.sh agent.py baseline-bots/baseline_random.py --deals 1000 --seed 7
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BOT1=$1 BOT2=$2; shift 2
P1=${P1:-5108} P2=${P2:-5109}
LOG=$(mktemp -d)

cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT
PIDS=()
PORT=$P1 $PY "$BOT1" >"$LOG/bot1.log" 2>&1 & PIDS+=($!)
PORT=$P2 $PY "$BOT2" >"$LOG/bot2.log" 2>&1 & PIDS+=($!)

$PY judge_engine.py --agent1 "$P1" --agent2 "$P2" "$@" \
  | grep -E "agent1:|Confidence|forfeit|error|RESULT" || { echo "judge failed; logs in $LOG"; exit 1; }

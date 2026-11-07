#!/usr/bin/env bash
# Usage: scripts/match.sh BOT1.py BOT2.py [judge args...]
#   e.g. scripts/match.sh agent.py baseline-bots/baseline_random.py --deals 1000 --seed 7
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BOT1=$1 BOT2=$2; shift 2
free_port() { $PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
P1=${P1:-$(free_port)} P2=${P2:-$(free_port)}
LOG=$(mktemp -d)

cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT
PIDS=()
PORT=$P1 $PY research/serve_bot.py "$BOT1" >"$LOG/bot1.log" 2>&1 & PIDS+=($!)
PORT=$P2 $PY research/serve_bot.py "$BOT2" >"$LOG/bot2.log" 2>&1 & PIDS+=($!)
for i in $(seq 1 40); do
  curl -sf "localhost:$P1/" >/dev/null && curl -sf "localhost:$P2/" >/dev/null && break; sleep 0.5
done
for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "a bot server died; logs in $LOG"; cat "$LOG"/*.log; exit 1; }; done

$PY judge_engine.py --agent1 "http://127.0.0.1:$P1" --agent2 "http://127.0.0.1:$P2" "$@" \
  | grep -E "agent1:|Confidence|forfeit|error|RESULT" || { echo "judge failed; logs in $LOG"; exit 1; }

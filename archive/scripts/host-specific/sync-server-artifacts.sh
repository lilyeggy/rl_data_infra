#!/bin/bash
# Robust resumable artifact sync for flaky single-connection transfers.
#
# Strategy: fetch each file WHOLE over the stream, repeatedly, verifying byte
# count against the remote size until a full, exact-sized copy lands. This
# avoids the corruption risk of naive partial-burst splicing.
#
# Usage: scripts/sync-server-artifacts.sh
set -u
: "${SSHPASS:?ARCHIVED script requires SSHPASS to be supplied explicitly}"
export SSHPASS
HOST="root@sh01-ssh.gpuhome.cc"
PORT=30741
SSH() { sshpass -e ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$HOST" "$@" ; }

PAIRS=(
  "/root/rivermind-data/v3-15task/raw-ndjson|tests/fixtures/pi/v3-15task"
  "/root/rivermind-data/swebench/package-v3|artifacts/swebench-v3/package-v3"
)

fetch_file() {
  local remote="$1" local="$2" total got tries=0
  total=$(SSH "stat -c %s '$remote'" 2>/dev/null | tr -d ' \n')
  [ -n "$total" ] && [ "$total" -gt 0 ] || { echo "  (no/empty remote $remote)"; return 1; }
  mkdir -p "$(dirname "$local")"
  while :; do
    SSH "cat '$remote'" > "$local.tmp" 2>/dev/null
    got=$(wc -c < "$local.tmp" 2>/dev/null || echo 0)
    tries=$((tries+1))
    if [ "$got" = "$total" ]; then
      mv "$local.tmp" "$local"
      echo "  OK $local ($got bytes) tries=$tries"
      return 0
    fi
    rm -f "$local.tmp"
    if [ "$tries" -ge 15 ]; then echo "  GIVEUP $local ($got/$total after $tries)"; return 1; fi
    sleep 2
  done
}

for pair in "${PAIRS[@]}"; do
  remote="${pair%%|*}"; local="${pair##*|}"
  echo "== syncing $remote -> $local"
  mkdir -p "$local"
  names=""
  for a in 1 2 3 4 5 6 7 8; do
    names=$(SSH "ls -1 '$remote'" 2>/dev/null) && [ -n "$names" ] && break
    sleep 4
  done
  [ -z "$names" ] && { echo "  !! could not list $remote"; continue; }
  for name in $names; do
    fetch_file "${remote}/${name}" "${local}/${name}"
  done
done
echo "SYNC_DONE"

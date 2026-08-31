#!/bin/bash
# Chunked downloader for flaky single-connection transfers.
# server: /root/rivermind-data/v3-15task/raw-ndjson -> tests/fixtures/pi/v3-15task
set -u
: "${SSHPASS:?ARCHIVED script requires SSHPASS to be supplied explicitly}"
export SSHPASS
HOST="root@sh01-ssh.gpuhome.cc"
PORT=30741
SSH_OPTS="-p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

REMOTE_DIR="/root/rivermind-data/v3-15task/raw-ndjson"
LOCAL_DIR="tests/fixtures/pi/v3-15task"
mkdir -p "$LOCAL_DIR"

# Build chunk manifest on server (list of files + sizes)
sshpass -e ssh $SSH_OPTS "$HOST" \
  "cd $REMOTE_DIR && ls -1 *.ndjson" > /tmp/v3files.list 2>/dev/null
FILES=$(cat /tmp/v3files.list)
echo "Remote files: $(echo "$FILES" | wc -l)"

for f in $FILES; do
  if [ -s "$LOCAL_DIR/$f" ]; then continue; fi
  # fetch with retries; if truncated (not complete), retry. We detect by trying
  # to fetch full file; if it lands under a size threshold we re-fetch.
  for try in 1 2 3 4 5 6 7 8; do
    sshpass -e ssh $SSH_OPTS "$HOST" "cat $REMOTE_DIR/$f" > "$LOCAL_DIR/$f" 2>/dev/null
    sz=$(wc -c < "$LOCAL_DIR/$f" 2>/dev/null || echo 0)
    rem=$(sshpass -e ssh $SSH_OPTS "$HOST" "stat -c %s $REMOTE_DIR/$f" 2>/dev/null)
    if [ "$sz" = "$rem" ] && [ "$sz" -gt 0 ]; then
      echo "OK $f ($sz)"; break
    else
      rm -f "$LOCAL_DIR/$f"; sleep 2
    fi
  done
done
echo "DONE local fixtures: $(ls "$LOCAL_DIR" | wc -l)"

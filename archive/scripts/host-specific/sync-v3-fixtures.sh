#!/bin/bash
# 后台续传 v3 fixtures：整文件 + 远程大小校验，verify 防错位。网络通时一次跑完。
: "${SSHPASS:?ARCHIVED script requires SSHPASS to be supplied explicitly}"
export SSHPASS
HOST="root@sh01-ssh.gpuhome.cc"; PORT=30741
SSH() { sshpass -e ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 "$HOST" "$@" ; }
REMOTE="/root/rivermind-data/v3-15task/raw-ndjson"
LOCAL="tests/fixtures/pi/v3-15task"
mkdir -p "$LOCAL"
names=$(SSH "ls -1 $REMOTE" 2>/dev/null)
[ -z "$names" ] && { echo "list failed"; exit 1; }
for name in $names; do
  total=$(SSH "stat -c %s $REMOTE/$name" 2>/dev/null | tr -d ' \n')
  [ -n "$total" ] && [ "$total" -gt 0 ] || continue
  for t in $(seq 1 20); do
    SSH "cat $REMOTE/$name" > "$LOCAL/$name.tmp" 2>/dev/null
    got=$(wc -c < "$LOCAL/$name.tmp" 2>/dev/null || echo 0)
    if [ "$got" = "$total" ]; then mv "$LOCAL/$name.tmp" "$LOCAL/$name"; echo "OK $name $got"; break; fi
    rm -f "$LOCAL/$name.tmp"; sleep 2
  done
done
echo "DONE pulled=$(find $LOCAL -name '*.ndjson' -size +100c | wc -l)"

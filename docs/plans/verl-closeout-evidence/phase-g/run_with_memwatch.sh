#!/usr/bin/env bash
# Run a command while sampling host memory and per-role process footprint.
#
# Ray's node-memory OOM killer ended the first stage G attempt, so the run has
# to be observable rather than diagnosed after the fact. The sampler only ever
# reads /proc entries owned by the invoking user.
#
#   MEMWATCH_OUT=/path/mem.jsonl ./run_with_memwatch.sh <command...>
set -uo pipefail

OUT=${MEMWATCH_OUT:?set MEMWATCH_OUT to the jsonl path to write}
PROBE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${MEMWATCH_PYTHON:-/home/cxr/verl-closeout/venv/bin/python}

rm -f "$OUT"
"$PY" "$PROBE_DIR/memwatch.py" "$OUT" &
SAMPLER=$!
# The sampler must outlive a failing command so the last samples are still
# written, but never the whole shell.
trap 'kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null' EXIT

"$@"
rc=$?
echo "run_with_memwatch: command exited $rc" >&2
exit $rc

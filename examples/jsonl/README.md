# JSONL v1 example

> Compatibility note (2026-08-12): this directory demonstrates the existing `RolloutRecord v1` Training View interchange. The current universal execution contract is `TraceEvent / AgentEpisode`; see [Canonical Data Contract v2](../../docs/data-contract.md). This example remains supported but is not the main capture format.

`rollouts-v1.jsonl` contains one producer-agnostic canonical record inside the
public `rollout-jsonl/v1` envelope. It is intentionally small and synthetic; it
demonstrates the interchange contract, not a real Polar capture.

Load it without Polar or Slime:

```bash
python3 -c "from src.sources import JsonlSourceAdapter; r=JsonlSourceAdapter().convert_file('examples/jsonl/rollouts-v1.jsonl'); print(r.ok, len(r.records), sorted(c.value for c in r.capabilities))"
```

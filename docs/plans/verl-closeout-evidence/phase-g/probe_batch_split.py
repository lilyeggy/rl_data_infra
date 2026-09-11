"""Show how the agent-loop batch is split before it reaches the loop.

The stage G rollout failed with FileExistsError on one episode directory,
meaning a single PiAgentLoopWorker was handed the same per-sample token twice
inside one ``asyncio.gather``. This reproduces the split (pad -> chunk -> per
sample kwargs) with the real DataProto, no GPU involved.
"""

import numpy as np
from tensordict import TensorDict

from verl.protocol import DataProto, pad_dataproto_to_divisor

NUM_WORKERS = 4


def build(rows: int) -> DataProto:
    return DataProto(
        batch=TensorDict({"input_ids": np.zeros((rows, 4), dtype=np.int64)}, batch_size=[rows]),
        non_tensor_batch={
            "pi_sample_token": np.array([f"row{r}" for r in range(rows)], dtype=object),
            "agent_name": np.array(["pi_agent"] * rows, dtype=object),
        },
    )


def report(label: str, rows: int) -> None:
    batch = build(rows)
    padded, pad_size = pad_dataproto_to_divisor(batch, NUM_WORKERS)
    print(f"--- {label}: {rows} row(s) -> padded {len(padded)} (pad_size={pad_size}) ---")
    print("  padded non_tensor:", list(padded.non_tensor_batch["pi_sample_token"]))
    for position, chunk in enumerate(padded.chunk(NUM_WORKERS)):
        tokens = list(chunk.non_tensor_batch["pi_sample_token"])
        # This mirrors AgentLoopWorker.generate_sequences: kwargs for sample i
        # is {k: v[i] for k, v in chunk.non_tensor_batch.items()}.
        seen = [chunk.non_tensor_batch["pi_sample_token"][i] for i in range(len(chunk))]
        print(f"  worker{position}: batch_len={len(chunk)} tokens={tokens} handed={seen}")
    handed = [
        chunk.non_tensor_batch["pi_sample_token"][i]
        for chunk in padded.chunk(NUM_WORKERS)
        for i in range(len(chunk))
    ]
    duplicates = sorted({t for t in handed if handed.count(t) > 1})
    print(f"  => handed {len(handed)} samples, duplicated tokens: {duplicates or 'none'}")


def main() -> int:
    report("single validation row", 1)
    report("one task x rollout.n", 4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

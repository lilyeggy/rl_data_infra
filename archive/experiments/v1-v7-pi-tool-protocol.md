# Archived experiment conclusions: v1–v7 Qwen/Pi tool-protocol SFT experiments

> Status: **ARCHIVED — not the project mainline.**
> These experiments are retained for traceability. They do **not** represent
> general agent capability gains and must not be treated as the final
> architecture center.

## What these experiments were

v1–v7 were a sequence of Qwen2.5-Coder-14B LoRA SFT runs that tried to make a
14B student imitate a Pi-based Teacher (`opencode-go/deepseek-v4-flash`) roughly
at the **Pi tool-protocol level**, with the explicit goal of getting the model to
emit a `<tool_call>` token so the Pi harness could enter its tool loop.

The teacher trajectories came from real Pi Harness runs over 9 certified
SWE-bench TRAIN tasks (see `configs/swebench-sft-splits.json`).

## The core finding (why they are archived)

- The models can learn to *schedule* a `<tool_call>` symbol, and LoRA including
  `lm_head` does move the first-token probabilities in the right direction.
- But the approach is **harness-specific**: it teaches the model a Pi/Qwen
  output grammar and Pi-specific tool vocabulary, not a general, canonical
  agent decision ability.
- Therefore it cannot support: multiple harnesses, multiple model-native
  formats, or a claim of generalizable agent capability.

Per the project decision, v6/v7 are re-labeled as a small protocol-adaptation
side experiment. The project mainline becomes the canonical
harness-neutral `Action`/`Observation` contract (see
`docs/canonical-agent-contract.md`), multi-view exporters, layered evaluation,
and generic SFT.

## Per-experiment record

| Ver | SFT package (server) | Run dir (server) | Description | Result |
|---|---|---|---|---|
| v1 | `sft-packages/deepseek-v4-flash-qwen14b-v1` | `sft-runs/qwen14b-deepseek-teacher-v1` | First teacher turn-level package | baseline |
| v2 | `sft-packages/deepseek-v4-flash-qwen14b-tools-v2` | `sft-runs/qwen14b-deepseek-teacher-tools-v2` | Added explicit tools schema in chat template | smoke |
| v3 | `sft-packages/...-tools-v3` | `sft-runs/qwen14b-deepseek-teacher-tools-v3` | Tool-schema iteration | smoke |
| v4 | `sft-packages/...-tools-v4` | `sft-runs/qwen14b-deepseek-teacher-tools-v4` | Tool-schema iteration | smoke |
| v5 | `sft-packages/...-expanded-v5` | (preflight) | Expanded to 9-task teacher set | packaged |
| v6 | `sft-packages/deepseek-v4-flash-qwen14b-expanded-v6` | `sft-runs/qwen14b-deepseek-teacher-expanded-v6` | 235 turns, 2 epochs, 94 steps, LoRA w/o lm_head | train loss 0.5718; real Pi run only emitted `<|im_start|><|im_end|>`, no tool call |
| v7 | `sft-packages/deepseek-v4-flash-qwen14b-expanded-v7` | `sft-runs/qwen14b-deepseek-teacher-expanded-v7` | LoRA target added `lm_head` (new `<tool_call>` special token frozen otherwise) | 94/94 steps, loss 0.549752; first-token `<tool_call>` prob 5.35e-10(base)→4.15e-4(v7), rank 81,899→78; STILL first token `<|im_start|>` under greedy — protocol gate NOT passed |

### First-token comparison (server evidence)

| Model | `<tool_call>` probability | `<tool_call>` rank |
|---|---|---|
| Base Qwen | ≈ 5.35e-10 | 81,899 |
| v6 | ≈ 5.06e-5 | 1,761 |
| v7 | ≈ 4.15e-4 | 78 |

`<tool_call>` here is the model-specific special token added for Pi/Qwen tool
format; this is exactly the class of model-format-specific measure that must NOT
be confused with general agent capability.

## Why v7 protocol gate still failed

The model's first greedy token remains `<|im_start|>` (the chat-template
opener), meaning the real Pi tool loop is still not entered. Moving the
`<tool_call>` probability is necessary but not sufficient; the *structured
output boundary* at the model-native / harness level must actually be reached,
and the current evidence is format-adaptation machinery, not a decision
capability.

## Governing rules applied

1. These artifacts stay under `sft-packages/` and `sft-runs/` on the server as
   immutable runs; nothing is deleted.
2. No production training path depends on v7's `lm_head` tool-token trick as the
   mainline. That trick is at most an optional adapter-level concern, decided
   per exporter/harness.
3. Future training must be built from canonical episodes via the exporters in
   `src/exporters/`, and judged by layered evaluation
   (format / action-selection / trajectory-replay / end-to-end), not by pushing
   a private token.
4. DEV (`pytest-dev__pytest-10081`) and TEST (`pallets__flask-5014`) were **never**
   used for training and remain Teacher-unseen.
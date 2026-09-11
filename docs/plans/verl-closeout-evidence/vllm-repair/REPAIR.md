# vLLM "unavailable" — root cause and repair (verified 2026-09-10T16:14Z)

## Root cause (3 layers)

1. **Broken editable install.** The shared conda env records
   `vllm 0.1.dev1+...` as an *editable* install whose target `/home/cxr1/vllm`
   **no longer exists**; `site-packages/vllm` is an empty 0-byte directory and
   the `.pth` installs a meta-path finder that raises before any path lookup.
   → `ModuleNotFoundError: No module named 'vllm'`.
2. **SM 12.x capability guard.** After restoring the import, the engine failed:
   `Failed to get device capability: SM 12.x requires CUDA >= 12.9` →
   cascading `FlashInfer requires GPUs with sm75 or higher`.
   RTX PRO 6000 is SM 12.x (Blackwell).
3. **Missing `ninja` on PATH** for vLLM's JIT step (the executable lives in the
   conda env's bin, not on the venv PATH).

## Repair (contained to this task's venv; the shared env is NOT modified)

1. `sitecustomize.py` in the task venv: drop the finder whose module name
   contains both `editable` and `vllm`, and prepend the surviving built tree
   `/home/cxr/ds4-deploy/vllm` to `sys.path`. Being `sitecustomize`, it applies
   to vLLM's helper **subprocesses** too.
2. Run with `--attention-backend FLASH_ATTN` to bypass the SM12/FlashInfer path.
3. Run with `PATH=/home/cxr/miniconda3/envs/vllm/bin:$PATH` (provides `ninja`)
   and `CUDA_HOME=/usr/local/cuda-13.0`.

## Verified outcome

`vllm-repair-smoke.json`: import ok; engine started in 88.4s; **P0 LoRA loaded
via `LoRARequest`**; 32 generated tokens with **32 aligned logprob steps**;
`VLLM-REPAIR-SMOKE-PASS` (exit 0). Evidence log: `smoke3.log`.

## Caveats (not overclaimed)

- The restored tree reports `vllm.__version__ = "0.1.dev17320+geac9e008a"`.
  verl v0.7.1 parses that as < 0.8.5 and would take version-gated legacy code
  paths; the verl-side integration is therefore **not yet proven**.
- Only single-GPU inference with LoRA was proven. Weight sync from the FSDP
  trainer and Ray-colocated rollout remain untested.

"""Measure host memory of loading the base model the way verl's FSDP actor does.

verl/workers/fsdp_workers.py takes the actor's load dtype from
`fsdp_config.model_dtype`, whose dataclass default is "fp32". This probe
reproduces that load in a standalone process so the per-rank host footprint
can be measured without spending GPU time.

Plain VmRSS is the wrong metric here: safetensors is mmap'd, so the weight
pages are file-backed and Ray's node memory monitor does not treat reclaimable
page cache as usage. Ray itself reads `/proc/<pid>/smaps_rollup`, so this probe
reports the same fields plus the host-wide MemAvailable delta.

Usage: python probe_load_rss.py <fp32|bf16> <base_model_dir> [adapter_dir]
"""

import gc
import sys

import torch
from transformers import AutoModelForCausalLM

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

_FIELDS = ("Rss:", "Pss:", "Anonymous:", "Private_Dirty:", "Private_Clean:",
           "Shared_Clean:", "Shared_Dirty:", "Swap:")


def smaps() -> dict:
    out = {}
    with open("/proc/self/smaps_rollup") as handle:
        for line in handle:
            key, sep, value = line.partition(":")
            name = f"{key.strip()}{sep}"
            if name in _FIELDS:
                out[name] = int(value.split()[0]) / 1024 / 1024
    return out


def meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "MemFree"):
                out[key] = int(value.split()[0]) / 1024 / 1024
    return out


def report(label: str) -> None:
    gc.collect()
    s, m = smaps(), meminfo()
    print(
        f"  {label:<32} rss={s.get('Rss:', 0):7.2f} pss={s.get('Pss:', 0):7.2f} "
        f"anon={s.get('Anonymous:', 0):7.2f} priv_dirty={s.get('Private_Dirty:', 0):7.2f} "
        f"priv_clean={s.get('Private_Clean:', 0):7.2f} | host_avail={m.get('MemAvailable', 0):8.2f}",
        flush=True,
    )


def main() -> int:
    dtype_key, base_model = sys.argv[1], sys.argv[2]
    adapter = sys.argv[3] if len(sys.argv) > 3 else None
    dtype = DTYPES[dtype_key]

    print(f"=== dtype={dtype_key} ===", flush=True)
    report("baseline (torch imported)")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    report("after from_pretrained")

    # fsdp_workers.py:500 runs ``actor_module.to(torch_dtype)`` after loading.
    model.to(dtype)
    report("after .to(dtype)")

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        report("after PeftModel.from_pretrained")

    del model
    report("after del model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

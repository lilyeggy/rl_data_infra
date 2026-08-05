#!/usr/bin/env python
"""Wrapper: 注册 local-spec qwen3 桥后运行 slime 的 HF->torch_dist 转换。"""
import runpy
import sys

sys.path.insert(0, "/data/day-01-workspace/scripts")
import local_qwen3_bridge  # noqa: F401  (注册 qwen3 -> LocalQwen3Bridge)

sys.argv = ["convert_hf_to_torch_dist.py"] + sys.argv[1:]
runpy.run_path(
    "/data/day-01-workspace/src/slime/tools/convert_hf_to_torch_dist.py",
    run_name="__main__",
)

# Coding VALID_SUCCESS Fixture

状态：`WAITING_FOR_SERVER_CAPTURE`

这里保存一条由锁定 Polar commit 真实运行、经过 `qwen_code` 多轮模型/工具交互、
产生 patch，并由 `swebench_harness` 正常判定 resolved 的 Coding rollout。

必须有 clean replay 证据；不得根据模型最终文本推断 success。

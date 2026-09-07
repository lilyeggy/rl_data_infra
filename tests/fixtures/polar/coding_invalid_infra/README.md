# Coding INVALID_INFRASTRUCTURE Fixture

状态：`WAITING_FOR_SERVER_CAPTURE`

这里保存 Coding rollout 的基础设施无效结果。自然故障和 synthetic fault 都允许，
但必须明确标记。`resolved` 与 `reward` 必须保持 `null`，不得映射成模型 reward=0。

Day 3 目标至少覆盖两种不同故障层，例如 verifier timeout 与 runtime prepare failure。

# MBPP SFT v1：缺失工具观察的对照版本

## 状态

归档，不再作为正式候选模型或正式评测结论。

## 远程产物

- LoRA adapter：`/home/f630/homePLUS/agent-data-plane/sft-runs/qwen14b-mbpp-train-260827`
- 原训练包：`/home/f630/homePLUS/agent-data-plane/mbpp-sft-package-260827-v3`
- 原始教师导出：`/home/f630/homePLUS/agent-data-plane/mbpp-sft-export-260827-v5`

## 归档原因

原始执行轨迹的 `steps` 保存了每个工具调用的真实结果，但训练字段
`messages` 未将这些结果按 `tool_call_id` 插入为 `role: tool` 的环境观察。
因此模型被监督为生成工具调用，却不能在训练时条件化于对应的文件内容、
命令输出或报错。该版本仅保留为“missing-observation ablation”，用于和
修复后的 action-observation-closed Agent SFT 比较。

## 修复后的准入契约

正式 SFT 数据必须满足：每一个 assistant tool call 在下一次 assistant
决策前恰好对应一条 `role: tool` 消息；其 `tool_call_id`、工具名及 canonical
`steps` 记录一致。任一条件不成立，数据不可标记为 SFT-eligible。

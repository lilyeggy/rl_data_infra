# Calculator Success Fixture

状态：`CAPTURED`（run `20260809T164046Z-calculator`，polar f0e8343a，2026-08-09/10）

捕获后关键事实：

- task/session/request ID：`summary.json $.task_id`、`$.session_id`；completion_id `msg_ff634f2fb0a0`（raw completion record 顶层，未入 fixture）；
- 所有模型请求均经过 Polar Gateway：completion record 记录 original_request（agent 侧 `model=qwen3-coder-plus`）与 transformed_request（`model=Qwen/Qwen3-4B-Instruct-2507`），request/response 差异仅 model 重写；
- harness：`qwen_code`（@qwen-code/qwen-code@0.14.5）；agent 发出一次 read_file 工具调用后退出（见 field-map note 3），未实际写入 Calculator 文件；
- evaluator：strategy `test_on_output`，`summary.json $.trajectory.metadata.evaluation.report`（empty_generation=true, resolved=false, error_eval=false, test_timeout=false），`outcome_reward=0.0` 为模型未解出（非基础设施失败）；
- prompt/output token IDs 与 logprobs：源系统直接提供——`response.json $.choices[0].input_token_ids/prompt_token_ids/token_ids`、`$.choices[0].logprobs.content`；`summary.json $.trajectory.traces[0].prompt_ids/response_ids/response_logprobs/loss_mask`；
- model/tokenizer revision：manifest `source.model_revision/tokenizer_revision`（cdbee75f…，derived from upstream-lock）；`response.metadata.weight_version=default`；
- termination：`summary.json $.status=COMPLETED`，模型 `finish_reason=stop`；
- 缺失（不得猜测）：policy_version、group_id、old_logprobs、tool_result、final_output。

原始证据：`/data/day-01-workspace/artifacts/day-02/20260809T164046Z-calculator/raw/`（服务器本地，未提交 Git）。

# Calculator Success Fixture

状态：`WAITING_FOR_SERVER_CAPTURE`

这里将保存一条由锁定 Polar commit 真实产生、且 evaluator 正常完成并判定成功的 Calculator rollout。

捕获后必须回答：

- task、rollout、session、request ID 分别位于哪个源字段；
- 所有模型请求是否经过 Polar Gateway；
- harness 是否实际读写 Calculator 文件；
- evaluator 的命令、exit code、stdout/stderr 和 reward 位于哪里；
- prompt/output token IDs 和 logprobs 是否为源系统直接提供；
- model/tokenizer/policy revision 是否明确存在；
- termination reason 与最终状态是否一致。

不得在服务器执行前创建占位 `source-manifest.json`，避免占位值被误认为真实证据。

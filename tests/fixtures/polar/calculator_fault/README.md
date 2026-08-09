# Calculator Synthetic Fault Fixture

状态：`WAITING_FOR_SERVER_CAPTURE`

这里将保存至少一条非破坏性、明确标记的 Calculator synthetic fault。优先选择错误 task 参数或可控 verifier timeout；不得通过损坏共享环境制造故障。

Manifest 必须满足：

```json
{
  "fixture_type": "calculator_fault",
  "synthetic_fault": true
}
```

README/manifest 还必须记录：

- fault 注入点和具体配置；
- 预期失败阶段；
- Polar 实际 status/error 字段；
- runtime、Harness、model backend、verifier 哪一层失败；
- 为什么它不能被当成模型的 reward=0 样本。

不得在服务器执行前创建占位 `source-manifest.json`。

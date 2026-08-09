# Polar Golden Fixtures

该目录只保存来自锁定 Polar 版本的、小型、去密、可重复检查的 source fixture。

## 目录

```text
polar/
├── source-manifest.schema.json
├── calculator_success/
├── calculator_fault/
├── coding_success/
├── coding_valid_failure/
└── coding_invalid_infra/
```

每个完整 fixture 必须包含：

```text
source-manifest.json
request.json
response.json
summary.json
normalized-logs/
README.md
```

Coding fixture 另外要求：

```text
verifier-evidence.json
patch.diff
replay.json              # success / valid failure 必需
fault-injection.json     # synthetic invalid infrastructure 建议保存
```

服务器执行前目录中只有说明和 schema，不存在伪造的 `request.json`、`response.json` 或 `summary.json`。

## 不可违反的规则

1. Raw source 文件保持原样；去密后的副本必须说明 redaction。
2. 不得从生成文本重新 tokenize 后声称它是采样 token IDs。
3. 不得用零填充缺失 old logprobs。
4. 不得从含糊状态文本猜测 reward、policy version 或 verifier outcome。
5. Synthetic fault 必须设置 `synthetic_fault=true` 并记录注入方法。
6. 超过仓库大小限制的文件只保存外部引用、大小和 SHA256。
7. Manifest 中的每个 SHA256 必须针对最终提交的去密文件计算。

## 完成顺序

```text
capture raw artifact
→ copy into a temporary staging directory
→ redact secrets and private infrastructure details
→ normalize selected logs without modifying raw copies
→ calculate sha256
→ create source-manifest.json
→ validate manifest and checksums
→ commit fixture
```

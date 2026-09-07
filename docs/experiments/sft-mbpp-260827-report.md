# MBPP SFT Experiment Report — 2026-08-27

## Scope

使用 289 条通过 verifier 的 MBPP 训练轨迹，对 Qwen2.5-Coder-14B 做 LoRA SFT（2 epochs），并在未参与训练的 MBPP holdout 与 HumanEval 上比较 base/candidate。评测口径为执行链路完整且 verifier 通过的 `ELIGIBLE`。

## Results

| Benchmark | Base | SFT candidate | Delta |
|---|---:|---:|---:|
| MBPP holdout | 40/90 (44.4%) | 68/90 (75.6%) | +31.1 pp |
| HumanEval | 30/164 (18.3%) | 101/164 (61.6%) | +43.3 pp |

Candidate 的失败分布：MBPP 为 20 FAILED、2 ERROR、1 INSUFFICIENT_EVIDENCE；HumanEval 为 60 FAILED、3 ERROR、2 INSUFFICIENT_EVIDENCE。Base 结果主要是 verifier FAILED（MBPP 50、HumanEval 134）。

## Candidate failure analysis

对 candidate 的 verifier 输出进一步分类后，MBPP 的 22 个非通过任务中有 14 个是断言失败、6 个是 `NameError`，另有 2 个 ERROR；HumanEval 的 63 个非通过任务中有 44 个是 `NameError`、13 个是断言失败、3 个是语法错误、3 个 ERROR/证据问题。HumanEval 的 `NameError` 集中表现为模型没有按题目要求定义入口函数，优先指向任务 prompt、入口函数注入或 verifier 对接问题，而不是单纯的数据量不足。

进一步抽查发现，A6000 模型服务曾将所有生成请求硬截断为 200 tokens。典型失败轨迹中的 `<tool_call>` JSON 在写入 Python 实现前被截断，导致 `solution.py` 保持模板内容，随后 verifier 报 `NameError`。该问题属于 serving 层缺陷，已修复为可配置的 4096-token 上限；正在对典型失败样本做重放验证。

因此下一轮顺序调整为：先验证 serving 修复并重放失败任务；确认失败率下降后，再扩大 MBPP 训练池。扩大数据主要用于改善 MBPP 的断言失败和边界条件覆盖，不用于掩盖协议错误。

## Artifacts

- SFT output: `sft-runs/qwen14b-mbpp-train-260827`
- Training package: `mbpp-sft-package-260827-v3`
- Post-eval log: `sft-runs/mbpp-post-sft-holdout-260827.log`
- Holdouts: `holdout-mbpp-{base,candidate}-260827`, `holdout-humaneval-{base,candidate}-260827`

## Interpretation

本轮说明 verifier 筛选后的 MBPP 轨迹可以有效提升小模型的代码任务完成率，且在 HumanEval 上有明显迁移。但这是单次实验，样本量仍小；下一轮需要保持固定 holdout，扩大训练池，并区分模型能力失败、工具/超时失败和证据链失败，避免把所有失败都归因于模型。

## Next engineering iteration

1. 修复 HumanEval 入口函数与 verifier 契约，并重放失败任务。
2. 在不改变 holdout 的前提下扩展 MBPP 训练池，先增加约 250–300 条高质量轨迹并做去重与质量分层。
3. 进行受控对照：289/新增数据量与 1/2 epochs，保留相同模型、prompt、verifier 和评测配置。
4. SFT 基线稳定后，再复用执行结果作为奖励信号开展小规模 Agentic RL 实验。

## Serving-fixed full rerun v3

v3 使用 32768-token context、动态生成预算和 logprob 非有限值清洗后完成。MBPP base 为 69/90、candidate 为 77/90；HumanEval candidate 为 133/164。HumanEval base 虽有 102 条通过，但仍有 50 条 `ERROR`，因此 HumanEval 的绝对提升（102→133）只能作为暂定信号，不能视为完全干净的 base 对照。下一步应先定位这 50 条 base 的残余错误，再决定是否扩大数据。

## Cleaned base replay conclusion

对 v3 base HumanEval 的 50 条 ERROR 重放已完成：38 条通过、10 条真实失败、2 条仍为基础设施/证据异常。合并后 base 的有效结果为 140/162（86.4%），candidate 为 133/161（82.6%）；MBPP 则为 base 69/90（76.7%）、candidate 77/90（85.6%）。因此当前 SFT 对 MBPP 有 +8.9 pp 的收益，但对 HumanEval 没有泛化收益，反而低约 3.8 pp。结论是当前 289 条 MBPP 轨迹偏窄，出现任务分布过拟合，下一轮应扩大并多样化数据，同时控制训练轮数和学习率。

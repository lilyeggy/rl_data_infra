# Fixed-workload proxy replay: 2026-08-31

This experiment isolates the Data Plane model boundary from Agent trajectory
length. Each paired arm used the same frozen prompt, model (`qwen2.5-coder-14b-instruct`), vLLM endpoint, temperature 0, seed `20260831`, `max_tokens=128`, and upstream `logprobs=true`. The only intended difference was direct vLLM access versus the Data Plane HTTP proxy with durable trace and model-evidence capture.

Three paired trials were run at each concurrency, with 12 successful requests per arm and no failures.

| Concurrency | Direct token goodput (tok/s) | Proxy + evidence token goodput (tok/s) | Proxy/direct | Mean p95 E2E delta |
| --- | ---: | ---: | ---: | ---: |
| 1 | 22.16 | 21.33 | 96.2% | +18.8 ms |
| 4 | 83.85 | 75.92 | 90.5% | +36.4 ms |
| 8 | 123.83 | 109.11 | 88.1% | -107.8 ms mean; see note |

At c8, two of three paired p95 deltas were +52.9 ms, while one direct-vLLM trial had a +429.3 ms p95 outlier. Therefore the negative mean c8 delta is **not** evidence that the proxy reduces latency. The stable conclusion is narrower: durable evidence capture preserves 88–96% of direct-vLLM completion-token goodput under this fixed workload, while adding a small positive p95 cost at c1/c4.

Raw A6000 artifacts are under `sft-runs/proxy-replay-repeats-260831/`, with the machine-readable paired summary at `paired-summary.json`. The benchmark scripts are `scripts/benchmark_proxy_replay_a6000.py`, `scripts/run_proxy_replay_a6000.sh`, and `scripts/summarize_proxy_replay.py`.

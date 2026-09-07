# Local/Pi vs Polar Rollout 工程对比

> 本报告只写入已观测证据；不同任务集不计算虚假的 paired speedup。

## Polar smoke

| 并发 | wall(s) | throughput(session/s) | reward | completion | trace | GPU mean/peak | memory peak(MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 130.0 | 0.007692307692307693 | 1.0 | 1/1 | 2 | 6.734375/99.0 | 31197.0 |
| 4 | 450.0 | 0.008888888888888889 | 1.0 | 4/4 | 8 | 7.45945945945946/100.0 | 31205.0 |
| 8 | 180.0 | 0.044444444444444446 | 1.0 | 8/8 | 16 | 9.164804469273744/100.0 | 31409.0 |

## Paired Local/Pi

| 并发 | wall(s) | throughput(session/s) | completion | p50/p95 session(s) | GPU mean/peak | memory peak(MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.7 | 0.0509 | 1/1 | 19.7/19.7 | 89.2/100.0 | 31297.0 |
| 4 | 67.3 | 0.0594 | 4/4 | 61.9/63.7 | 98.88235294117646/100.0 | 31377.0 |
| 8 | 151.6 | 0.0528 | 8/8 | 141.0/149.0 | 99.01333333333334/100.0 | 31465.0 |

## 结论

- Polar 和 Local/Pi 已在同一 smoke task 上完成 c1/c4/c8 成对测试。
- Polar session、trace 和 verifier reward 来自 session JSON；两边 GPU/显存来自 benchmark 期间的 nvidia-smi 采样。
- MBPP 200 条统计作为数据生产背景基线保留，不与 smoke throughput 混合。

```json
{
  "report_version": "rollout-comparison/v2",
  "polar": [
    {
      "benchmark": "/home/f630/homePLUS/agent-data-plane/sft-runs/polar-live-comparison-260830/benchmark-c1.log",
      "task_id": "polar-smoke-pi-20260830T070928Z",
      "wall_seconds": 130.0,
      "throughput_sessions_per_second": 0.007692307692307693,
      "reward": 1.0,
      "done": 1,
      "total": 1,
      "completion_rate": 1.0,
      "session": {
        "session_count": 1,
        "positive_reward_count": 1,
        "trace_count": 2,
        "durations_seconds": [
          124.90687542199157
        ],
        "statuses": {
          "COMPLETED": 1
        }
      },
      "resources": {
        "samples": 64,
        "gpu_mean_percent": 6.734375,
        "gpu_peak_percent": 99.0,
        "memory_peak_mib": 31197.0,
        "power_mean_watts": 53.1415625
      },
      "session_p50_seconds": 124.90687542199157,
      "session_p95_seconds": 124.90687542199157
    },
    {
      "benchmark": "/home/f630/homePLUS/agent-data-plane/sft-runs/polar-live-comparison-260830/benchmark-c4.log",
      "task_id": "polar-smoke-pi-20260830T071141Z",
      "wall_seconds": 450.0,
      "throughput_sessions_per_second": 0.008888888888888889,
      "reward": 1.0,
      "done": 4,
      "total": 4,
      "completion_rate": 1.0,
      "session": {
        "session_count": 4,
        "positive_reward_count": 4,
        "trace_count": 8,
        "durations_seconds": [
          443.1862514518434,
          161.6909504099749,
          229.5086210750742,
          81.60198619705625
        ],
        "statuses": {
          "COMPLETED": 4
        }
      },
      "resources": {
        "samples": 222,
        "gpu_mean_percent": 7.45945945945946,
        "gpu_peak_percent": 100.0,
        "memory_peak_mib": 31205.0,
        "power_mean_watts": 48.093108108108105
      },
      "session_p50_seconds": 195.59978574252455,
      "session_p95_seconds": 229.5086210750742
    },
    {
      "benchmark": "/home/f630/homePLUS/agent-data-plane/sft-runs/polar-live-comparison-260830/benchmark-c8.log",
      "task_id": "polar-smoke-pi-20260830T064629Z",
      "wall_seconds": 180.0,
      "throughput_sessions_per_second": 0.044444444444444446,
      "reward": 1.0,
      "done": 8,
      "total": 8,
      "completion_rate": 1.0,
      "session": {
        "session_count": 8,
        "positive_reward_count": 8,
        "trace_count": 16,
        "durations_seconds": [
          172.73092229512986,
          164.29948229703587,
          154.27311909315176,
          161.89101430005394,
          164.15392727893777,
          173.6365268068621,
          154.31217694294173,
          154.68989706994034
        ],
        "statuses": {
          "COMPLETED": 8
        }
      },
      "resources": {
        "samples": 358,
        "gpu_mean_percent": 9.164804469273744,
        "gpu_peak_percent": 100.0,
        "memory_peak_mib": 31409.0,
        "power_mean_watts": 49.66829608938548
      },
      "session_p50_seconds": 163.02247078949586,
      "session_p95_seconds": 172.73092229512986
    }
  ],
  "local_smoke": [
    {
      "wall_seconds": 19.66077168600168,
      "throughput_sessions_per_second": 0.05086270345695497,
      "done": 1,
      "total": 1,
      "completion_rate": 1.0,
      "session_p50_seconds": 19.65889648301527,
      "session_p95_seconds": 19.65889648301527,
      "resources": {
        "samples": 10,
        "gpu_mean_percent": 89.2,
        "gpu_peak_percent": 100.0,
        "memory_peak_mib": 31297.0,
        "power_mean_watts": 253.267
      }
    },
    {
      "wall_seconds": 67.32662300905213,
      "throughput_sessions_per_second": 0.059411861478069324,
      "done": 4,
      "total": 4,
      "completion_rate": 1.0,
      "session_p50_seconds": 61.88679188804235,
      "session_p95_seconds": 63.698007093044,
      "resources": {
        "samples": 34,
        "gpu_mean_percent": 98.88235294117646,
        "gpu_peak_percent": 100.0,
        "memory_peak_mib": 31377.0,
        "power_mean_watts": 292.41205882352943
      }
    },
    {
      "wall_seconds": 151.6114393279422,
      "throughput_sessions_per_second": 0.05276646693324801,
      "done": 8,
      "total": 8,
      "completion_rate": 1.0,
      "session_p50_seconds": 141.0155400345684,
      "session_p95_seconds": 149.04626989306416,
      "resources": {
        "samples": 75,
        "gpu_mean_percent": 99.01333333333334,
        "gpu_peak_percent": 100.0,
        "memory_peak_mib": 31465.0,
        "power_mean_watts": 292.9368
      }
    }
  ],
  "local_baseline": {
    "label": "MBPP expansion 301-500",
    "root": "/home/f630/homePLUS/agent-data-plane/mbpp-expansion-301-500-260830",
    "tasks": 200,
    "sft_eligible": 140,
    "rejected": 60,
    "insufficient_evidence": 0,
    "verifier_status": {
      "FAILED": 60,
      "PASSED": 140
    },
    "execution_validity": {
      "VALID": 200
    },
    "mean_model_calls": 4.41,
    "max_model_calls": 26,
    "generated_at": "2026-08-29T19:07:16.580207+00:00"
  },
  "paired_comparison": true,
  "comparison_note": "Local/Pi and Polar use the same deterministic smoke task, Qwen service, and concurrency levels; Polar adds rollout gateway/session orchestration."
}
```

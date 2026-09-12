import json, subprocess, sys
task_id, worktree, output = sys.argv[1], sys.argv[2], sys.argv[3]
from evalplus.data import get_mbpp_plus
from evalplus.sanitize import extract_target_code_or_empty
problems = get_mbpp_plus()
problem = problems[task_id]
code = open(worktree + "/solution.py").read()
target = extract_target_code_or_empty(code, problem["entry_point"])
prompt = problem["prompt"]
asserts = [line.strip() for line in prompt.splitlines() if line.strip().startswith("assert ")]
test_src = target + "\n\n" + "\n".join(asserts) + "\nprint('ASSERTS-PASS')\n"
proc = subprocess.run([sys.executable, "-c", test_src], capture_output=True,
                      text=True, timeout=60)
resolved = proc.returncode == 0 and "ASSERTS-PASS" in proc.stdout
json.dump({"task_id": task_id, "case_count": len(asserts),
           "passed_cases": len(asserts) if resolved else 0,
           "pass_rate": 1.0 if resolved else 0.0, "resolved": resolved,
           "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]},
          open(output, "w"))

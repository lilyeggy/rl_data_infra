import json, subprocess, sys, tempfile
from pathlib import Path
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
with tempfile.TemporaryDirectory(prefix="mbpp-verifier-") as directory:
    source = Path(directory) / "verify.py"
    source.write_text(test_src)
    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv"]
    for runtime in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(runtime).exists():
            command += ["--ro-bind", runtime, runtime]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--ro-bind", str(source), "/verify.py", "--",
                "/usr/bin/prlimit", "--as=2147483648", "--cpu=30", "--fsize=1048576",
                "/usr/bin/python3", "-I", "/verify.py"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        proc = subprocess.CompletedProcess(command, 124, "", "solution execution timeout")
    if proc.stderr.startswith("bwrap:"):
        raise RuntimeError(f"verifier sandbox failed: {proc.stderr}")
resolved = proc.returncode == 0 and "ASSERTS-PASS" in proc.stdout
json.dump({"task_id": task_id, "case_count": len(asserts),
           "passed_cases": len(asserts) if resolved else 0,
           "pass_rate": 1.0 if resolved else 0.0, "resolved": resolved,
           "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]},
          open(output, "w"))

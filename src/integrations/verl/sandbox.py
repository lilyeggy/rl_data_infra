"""Linux mount isolation for Pi; mounts only runtime files and its workspace."""
from pathlib import Path

from src.errors import ContractValidationError


def pi_bwrap_command(command, *, workspace, home, pi, proxy_token):
    pi_path = Path(pi).absolute()
    package_root = pi_path.parent.parent
    node = pi_path.parent / "node"
    if not node.is_file():
        raise ContractValidationError("private Pi runtime must provide bin/node")
    mounts = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64") if Path(p).exists()]
    mounts += [package_root, node.resolve().parent]
    args = ["/usr/bin/bwrap", "--unshare-all", "--share-net", "--die-with-parent",
            "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in mounts:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--bind", str(workspace), str(workspace),
             "--bind", str(home), str(home),
             "--setenv", "HOME", str(home),
             "--setenv", "AGENT_MODEL_PROXY_API_KEY", proxy_token,
             "--setenv", "PATH", f"{pi_path.parent}:/usr/bin:/bin",
             "--chdir", str(workspace)]
    # Network is shared solely so the child can reach this episode's localhost
    # model proxy. No host home, checkpoint, dataset or Docker socket is mounted.
    return args + ["--"] + list(command)

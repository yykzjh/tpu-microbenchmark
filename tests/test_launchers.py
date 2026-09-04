"""Launcher checks with fake SSH/Python; never contact TPU hosts."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("TEST_BASH", "/bin/bash")


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").rglob("*.sh")))
def test_shell_syntax_and_help(script):
    subprocess.run([BASH, "-n", str(script)], check=True, timeout=10)
    result = subprocess.run([BASH, str(script), "--help"], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("host_count", [2, 4])
@pytest.mark.parametrize("fail_rank", [None, "1"])
def test_ici_tree_collects_logs_and_propagates_failure(tmp_path, fail_rank, host_count):
    version = subprocess.check_output([BASH, "-c", 'echo "${BASH_VERSINFO[0]}"'], text=True)
    if int(version.strip()) < 4:
        pytest.skip("Distributed launchers require Linux/Bash 4+ (see README)")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    (repo / "src/ici").mkdir(parents=True)
    (repo / "src/ici/test_ici.py").touch()
    bin_dir = repo / ".venv/bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print('ARGS=' + json.dumps(sys.argv[1:]))\n"
        "print('[COMMPILOT_METRIC] {}')\n"
        "rank = sys.argv[sys.argv.index('--process-id') + 1]\n"
        "sys.exit(1 if os.environ.get('FAIL_RANK') == rank else 0)\n"
    )
    python.chmod(0o755)
    ssh = bin_dir / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "target = next(arg for arg in sys.argv[1:-1] if '@' in arg)\n"
        "assert target.startswith('tester@'), target\n"
        "env = dict(os.environ, NODE_SN_NAME=target.split('@')[1])\n"
        "sys.exit(subprocess.call([os.environ['TEST_BASH'], '-c', sys.argv[-1]], env=env))\n"
    )
    ssh.chmod(0o755)
    # Recursive commands resolve the same explicitly selected Bash.
    (bin_dir / "bash").symlink_to(BASH)
    hostfile = repo / "hostfile"
    hostfile.write_text("".join(f"host{rank}\n" for rank in range(host_count)))
    env = dict(os.environ, NODE_SN_NAME="host0", TEST_BASH=BASH, LC_ALL="C")
    if fail_rank is not None:
        env["FAIL_RANK"] = fail_rank
    command = [BASH, str(repo / "scripts/ici/test_multinode_ici.sh"),
               "--hostfile", str(hostfile), "--mode", "allreduce_parallel",
               "--allreduce-data-size", "4GiB",
               "--warmup", "2", "--iterations", "5", "--xprof-timing",
               "--ssh-user", "tester", "--ssh-port", "2222"]
    if host_count == 4:
        command.extend(["--block-range", "0:2,0:2,0:2"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    assert (process.returncode == 0) == (fail_rank is None), stdout + stderr
    logs = list((repo / "logs/ici").glob("*.log"))
    assert len(logs) == host_count
    for log in logs:
        content = log.read_text()
        assert f'"--process-count", "{host_count}"' in content
        if host_count == 4:
            assert '"--block-range", "0:2,0:2,0:2"' in content
        else:
            assert '"--block-range"' not in content
        assert '"--parallel"' in content
        assert '"--xprof-timing"' in content
    assert not list((repo / "logs/.tmp").iterdir())

#!/usr/bin/env python3
"""Run msopgen and build verification on a remote Ascend machine described in a local machine inventory file."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from common import soc_candidates


def parse_machine_file(path: str) -> list[dict]:
    machines: list[dict] = []
    current: dict[str, str] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                machines.append(current)
                current = {}
            continue
        if "：" not in line:
            continue
        key, value = line.split("：", 1)
        key = key.strip()
        value = value.strip()
        if key == "机器" and current:
            machines.append(current)
            current = {}
        current[key] = value
    if current:
        machines.append(current)
    return machines


def choose_machine(machine_file: str, keyword: str) -> dict:
    machines = parse_machine_file(machine_file)
    if not machines:
        raise ValueError("No machines found in machine file.")
    lowered = keyword.lower()
    for item in machines:
        if lowered in item.get("机器", "").lower():
            return item
    return machines[0]


def run_local(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def should_retry(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    combined = f"{proc.stdout}\n{proc.stderr}"
    retry_markers = (
        "Permission denied",
        "Connection closed",
        "Connection reset",
        "timed out",
        "No route to host",
        "Connection refused",
    )
    return any(marker in combined for marker in retry_markers)


def run_with_retry(cmd: list[str], attempts: int = 3, delay_seconds: float = 1.0) -> subprocess.CompletedProcess[str]:
    last = run_local(cmd)
    for _ in range(1, attempts):
        if not should_retry(last):
            return last
        time.sleep(delay_seconds)
        last = run_local(cmd)
    return last


def ssh_base(machine: dict) -> list[str]:
    return [
        "sshpass",
        "-p",
        machine["密码"],
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        f'{machine["用户名"]}@{machine["IP地址"]}',
    ]


def scp_base(machine: dict) -> list[str]:
    return [
        "sshpass",
        "-p",
        machine["密码"],
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
    ]


def run_ssh(machine: dict, command: str) -> subprocess.CompletedProcess[str]:
    return run_with_retry(ssh_base(machine) + [command])


def run_scp(machine: dict, source: str, dest: str, recursive: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = scp_base(machine)
    if recursive:
        cmd.append("-r")
    cmd.extend([source, dest])
    return run_with_retry(cmd)


def download_remote_tree(machine: dict, remote_dir: str, local_dir: Path) -> subprocess.CompletedProcess[str]:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    password = shlex.quote(machine["密码"])
    user_host = shlex.quote(f'{machine["用户名"]}@{machine["IP地址"]}')
    remote_dir_quoted = shlex.quote(remote_dir)
    local_dir_quoted = shlex.quote(str(local_dir))
    cmd = (
        f"sshpass -p {password} ssh -o StrictHostKeyChecking=no {user_host} "
        f"'cd {remote_dir_quoted} && tar -cf - .' | tar -xf - -C {local_dir_quoted}"
    )
    last = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    for _ in range(1, 3):
        if not should_retry(last):
            return last
        time.sleep(1.0)
        last = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    return last


def probe_python_deps(machine: dict) -> tuple[str, list[str]]:
    probe = run_ssh(
        machine,
        "python3 - <<'PY'\nstatus = []\nchecks = [('numpy', 'numpy'), ('decorator', 'decorator'), ('sympy', 'sympy'), ('scipy', 'scipy'), ('psutil', 'psutil'), ('protobuf', 'google.protobuf')]\nfor label, module_name in checks:\n    try:\n        __import__(module_name)\n        status.append(f'{label}:ok')\n    except Exception:\n        status.append(f'{label}:missing')\nprint(' '.join(status))\nPY",
    )
    status = probe.stdout.strip()
    missing = [name for name in ("numpy", "decorator", "sympy", "scipy", "psutil", "protobuf") if f"{name}:missing" in status]
    return status, missing


def remote_verify(
    spec_path: str,
    inspection_path: str,
    framework: str,
    soc: str,
    machine_file: str,
    machine_keyword: str,
    remote_root: str,
    download_dir: str | None,
    bootstrap_python_deps: bool,
) -> dict:
    machine = choose_machine(machine_file, machine_keyword)
    remote_session = f"{remote_root.rstrip('/')}/codex_msopgen_{Path(spec_path).stem}_{int(time.time())}"
    remote_spec = f"{remote_session}/spec.json"
    remote_inspection = f"{remote_session}/inspection.json"
    remote_patcher = f"{remote_session}/patch_msopgen_project.py"
    remote_project = f"{remote_session}/project"

    results: dict = {
        "machine": {
            "name": machine.get("机器"),
            "host": machine.get("IP地址"),
            "user": machine.get("用户名"),
            "directory": machine.get("目录"),
        },
        "remote_session": remote_session,
        "framework": framework,
        "soc": soc,
        "downloaded_project": None,
        "succeeded": False,
    }

    mkdir = run_ssh(machine, f"mkdir -p {remote_session}")
    if mkdir.returncode != 0:
        results["stderr"] = mkdir.stderr
        return results

    files_to_upload = [
        (spec_path, f'{machine["用户名"]}@{machine["IP地址"]}:{remote_spec}'),
        (inspection_path, f'{machine["用户名"]}@{machine["IP地址"]}:{remote_inspection}'),
        (str(Path(__file__).with_name("common.py")), f'{machine["用户名"]}@{machine["IP地址"]}:{remote_session}/common.py'),
        (str(Path(__file__).with_name("ascendc_templates.py")), f'{machine["用户名"]}@{machine["IP地址"]}:{remote_session}/ascendc_templates.py'),
        (
            str(Path(__file__).with_name("pytorch_integration_templates.py")),
            f'{machine["用户名"]}@{machine["IP地址"]}:{remote_session}/pytorch_integration_templates.py',
        ),
        (str(Path(__file__).with_name("patch_msopgen_project.py")), f'{machine["用户名"]}@{machine["IP地址"]}:{remote_patcher}'),
    ]
    for source, dest in files_to_upload:
        copied = run_scp(machine, source, dest)
        if copied.returncode != 0:
            results["stderr"] = copied.stderr
            return results

    status, missing_deps = probe_python_deps(machine)
    results["python_deps_status"] = status
    results["python_deps_ready"] = not missing_deps
    if missing_deps and bootstrap_python_deps:
        install = run_ssh(
            machine,
            "python3 -m pip install --user 'numpy==1.26.4' 'decorator==5.1.1' 'sympy==1.12' 'scipy==1.11.4' 'psutil==5.9.8' 'protobuf==4.25.3'",
        )
        results["bootstrap_python_deps"] = {
            "returncode": install.returncode,
            "stdout": install.stdout,
            "stderr": install.stderr,
        }
        if install.returncode != 0:
            results["stderr"] = install.stderr or "Failed to install numpy on remote machine."
            return results
        status, missing_deps = probe_python_deps(machine)
        results["python_deps_status"] = status
        results["python_deps_ready"] = not missing_deps

    last_attempt: dict | None = None
    for candidate in soc_candidates(soc):
        remote_cmd = f"""bash -lc '
set -eo pipefail
cd "{remote_session}"
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
fi
rm -rf "{remote_project}"
/usr/local/Ascend/ascend-toolkit/latest/bin/msopgen gen -i "{remote_spec}" -f "{framework}" -c "ai_core-{candidate}" -lan cpp -out "{remote_project}"
python3 "{remote_patcher}" --project "{remote_project}" --inspection "{remote_inspection}" --framework "{framework}" --strategy "ascendc-custom" >/dev/null
cd "{remote_project}"
bash build.sh > build.codex.log 2>&1
'"""
        proc = run_ssh(machine, remote_cmd)
        log = run_ssh(machine, f'test -f "{remote_project}/build.codex.log" && tail -n 120 "{remote_project}/build.codex.log" || true')
        last_attempt = {
            "candidate_soc": candidate,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "build_log_tail": log.stdout,
        }
        if proc.returncode == 0:
            results["succeeded"] = True
            results["candidate_soc"] = candidate
            results["stdout"] = proc.stdout
            results["stderr"] = proc.stderr
            results["build_log_tail"] = log.stdout
            break

    if not results["succeeded"] and last_attempt:
        results.update(last_attempt)

    if download_dir:
        download_path = Path(download_dir).expanduser().resolve()
        if download_path.exists():
            shutil.rmtree(download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        exists = run_ssh(machine, f'test -d "{remote_project}"')
        if exists.returncode == 0:
            copied = download_remote_tree(machine, remote_project, download_path)
            if copied.returncode == 0:
                results["downloaded_project"] = str(download_path)
            else:
                if download_path.joinpath("CMakeLists.txt").exists():
                    results["downloaded_project"] = str(download_path)
                    results["download_warning"] = copied.stderr
                else:
                    results["download_error"] = copied.stderr

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--soc", required=True)
    parser.add_argument("--machine-file", required=True)
    parser.add_argument("--machine-keyword", default="910B")
    parser.add_argument("--remote-root")
    parser.add_argument("--download-dir")
    parser.add_argument("--bootstrap-python-deps", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    machine = choose_machine(args.machine_file, args.machine_keyword)
    remote_root = args.remote_root or machine.get("目录") or "/tmp"
    result = remote_verify(
        args.spec,
        args.inspection,
        args.framework,
        args.soc,
        args.machine_file,
        args.machine_keyword,
        remote_root,
        args.download_dir,
        args.bootstrap_python_deps,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0 if result.get("succeeded") else 1


if __name__ == "__main__":
    raise SystemExit(main())

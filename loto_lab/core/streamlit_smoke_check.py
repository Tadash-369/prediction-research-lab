from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parent
LOTO_LAB_DIR = CORE_DIR.parent
PROJECT_ROOT = LOTO_LAB_DIR.parent


@dataclass
class SmokeTarget:
    name: str
    app_path: Path
    preferred_port: int


TARGETS = [
    SmokeTarget("top", LOTO_LAB_DIR / "apps" / "analysis_research_lab.py", 8761),
    SmokeTarget("loto6", LOTO_LAB_DIR / "apps" / "loto6_streamlit_app.py", 8762),
    SmokeTarget("loto7", LOTO_LAB_DIR / "apps" / "loto7_streamlit_app.py", 8763),
]


def free_port(preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_http_200(port: int, timeout_seconds: int) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True, "HTTP 200"
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    return False, last_error or "timeout"


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def read_tail(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def run_target(target: SmokeTarget, timeout_seconds: int) -> dict[str, str]:
    port = free_port(target.preferred_port)
    env = os.environ.copy()
    env["PRL_LIGHT_SMOKE"] = "1"
    env.setdefault("PYTHONUTF8", "1")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(target.app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]

    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW

    with tempfile.TemporaryDirectory(prefix=f"prl_smoke_{target.name}_", ignore_cleanup_errors=True) as temp_dir:
        log_path = Path(temp_dir) / "streamlit.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            try:
                ok, detail = wait_for_http_200(port, timeout_seconds)
                if process.poll() is not None and not ok:
                    detail = f"process exited with {process.returncode}; {detail}"
                return {
                    "target": target.name,
                    "port": str(port),
                    "status": "ok" if ok else "failed",
                    "detail": detail,
                    "log_tail": "" if ok else read_tail(log_path),
                }
            finally:
                terminate_process(process)


def main() -> int:
    timeout_seconds = int(os.environ.get("PRL_SMOKE_TIMEOUT", "60"))
    results = []
    for target in TARGETS:
        if not target.app_path.exists():
            results.append(
                {
                    "target": target.name,
                    "port": "-",
                    "status": "failed",
                    "detail": f"missing app: {target.app_path}",
                    "log_tail": "",
                }
            )
            continue
        result = run_target(target, timeout_seconds)
        results.append(result)
        print(f"{result['target']}: {result['status']} {result['detail']} (port {result['port']})")
        if result["log_tail"]:
            print(f"--- {result['target']} log tail ---")
            print(result["log_tail"])
            print("--- end log tail ---")

    failed = [result for result in results if result["status"] != "ok"]
    if failed:
        print(f"Streamlit smoke check failed: {len(failed)} target(s)")
        return 1
    print("Streamlit smoke check passed: all targets returned HTTP 200 in light mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

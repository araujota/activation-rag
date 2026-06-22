#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import time
from datetime import datetime, timezone


REMOTE_PROBE = r'''
import json, pathlib, time, urllib.request
root = pathlib.Path("/tmp/activation-rag-fixed-ingest-20260611-140007/activations")
dirs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
latest = max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None
health = "unavailable"
busy = "n/a"
try:
    with urllib.request.urlopen("http://127.0.0.1:28080/health", timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ll = payload["provider"]["local_llama"]
    health = str(ll.get("ready"))
    busy = str(ll.get("busy_slots"))
except Exception as exc:
    health = type(exc).__name__
print(json.dumps({
    "activation_dirs": len(dirs),
    "progress_percent": round(len(dirs) / 5225 * 100, 2),
    "latest_age_seconds": round(time.time() - latest.stat().st_mtime, 1) if latest else None,
    "server_ready": health,
    "busy_slots": busy,
}, sort_keys=True))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Heartbeat monitor for the fixed zero-token SciFact ingestion run.")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--host", default="root@vicuna-host")
    parser.add_argument("--log", default="runs/benchmarks/scifact-fixed-zero-20260611-140007.heartbeat.log")
    parser.add_argument("--cache-dir", default="runs/telemetry-cache/scifact-fixed-zero-20260611-140007")
    parser.add_argument("--out", default="runs/benchmarks/scifact-fixed-zero-20260611-140007.json")
    parser.add_argument(
        "--process-pattern",
        default="scifact-fixed-zero-20260611-140007",
    )
    args = parser.parse_args()

    log_path = pathlib.Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(args.cache_dir)
    out_path = pathlib.Path(args.out)

    while True:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        local_pids = find_processes(args.process_pattern)
        cache_count = sum(1 for path in cache_dir.rglob("*") if path.is_file()) if cache_dir.exists() else 0
        out_state = f"present:{out_path.stat().st_size}bytes" if out_path.exists() and out_path.stat().st_size else "missing"
        remote_status = probe_remote(args.host)
        append_line(
            log_path,
            f"{ts} local_pids=[{local_pids}] cache_files={cache_count} out={out_state} remote={remote_status}",
        )
        if out_state.startswith("present:"):
            append_line(log_path, f"{ts} completed output detected; stopping heartbeat")
            return
        if not local_pids:
            append_line(log_path, f"{ts} benchmark process not found; stopping heartbeat")
            return
        time.sleep(args.interval_seconds)


def find_processes(pattern: str) -> str:
    proc = subprocess.run(["ps", "ax", "-o", "pid=", "-o", "command="], text=True, capture_output=True, check=False)
    matches = []
    for line in proc.stdout.splitlines():
        if pattern in line and "monitor_scifact_fixed_zero.py" not in line:
            matches.append(" ".join(line.split()))
    return ";".join(matches)


def probe_remote(host: str) -> str:
    proc = subprocess.run(
        ["ssh", host, "python3 -c " + shlex.quote(REMOTE_PROBE)],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return proc.stderr.strip() or proc.stdout.strip() or f"ssh_exit={proc.returncode}"


def append_line(path: pathlib.Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()

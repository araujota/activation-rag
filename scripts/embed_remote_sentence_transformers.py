#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sentence Transformers embedding on a remote host via SSH.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--host", default="root@vicuna-host")
    parser.add_argument("--remote-python", default="/root/vicuna/venvs/activation-rag-embeddings/bin/python")
    parser.add_argument("--remote-script", default="/root/vicuna/activation-rag/scripts/embed_sentence_transformers.py")
    parser.add_argument("--remote-work-dir", default="/tmp/activation-rag-embeddings")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-compress-transfer", action="store_true", help="Disable gzip compression for the remote-to-local JSONL transfer.")
    args = parser.parse_args()

    remote_root = Path(args.remote_work_dir)
    run_dir = remote_root / next(tempfile._get_candidate_names())
    remote_input = run_dir / "embedding-input.jsonl"
    remote_output = run_dir / "embedding-output.jsonl"
    run_remote(args.host, f"mkdir -p {shlex.quote(str(run_dir))}")
    subprocess.run(["scp", "-q", args.input_jsonl, f"{args.host}:{remote_input}"], check=True)
    remote_command = " ".join(
        [
            shlex.quote(args.remote_python),
            shlex.quote(args.remote_script),
            shlex.quote(str(remote_input)),
            shlex.quote(str(remote_output)),
            "--model",
            shlex.quote(args.model),
            "--device",
            shlex.quote(args.device),
            "--batch-size",
            str(args.batch_size),
        ]
    )
    run_remote(args.host, remote_command)
    if args.no_compress_transfer:
        subprocess.run(["scp", "-q", f"{args.host}:{remote_output}", args.output_jsonl], check=True)
    else:
        remote_output_gz = run_dir / "embedding-output.jsonl.gz"
        local_output_gz = Path(args.output_jsonl).with_suffix(Path(args.output_jsonl).suffix + ".gz")
        run_remote(args.host, f"gzip -c {shlex.quote(str(remote_output))} > {shlex.quote(str(remote_output_gz))}")
        subprocess.run(["scp", "-q", f"{args.host}:{remote_output_gz}", str(local_output_gz)], check=True)
        with Path(args.output_jsonl).open("wb") as output_handle:
            subprocess.run(["gzip", "-dc", str(local_output_gz)], stdout=output_handle, check=True)
        local_output_gz.unlink(missing_ok=True)
    run_remote(args.host, f"rm -rf {shlex.quote(str(run_dir))}")


def run_remote(host: str, command: str) -> None:
    completed = subprocess.run(["ssh", host, command], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"remote command failed: {command}")


if __name__ == "__main__":
    main()

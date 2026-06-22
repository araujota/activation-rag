#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path


REMOTE_CODE = r'''
import json
import os
from pathlib import Path

groups_path = Path(os.environ["RERANK_GROUPS"])
scores_path = Path(os.environ["RERANK_SCORES"])
model_name = os.environ["RERANK_MODEL"]
device = os.environ.get("RERANK_DEVICE", "cuda")
batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "64"))
max_length = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
trust_remote_code = os.environ.get("RERANK_TRUST_REMOTE_CODE", "0") == "1"

from sentence_transformers import CrossEncoder

groups = [json.loads(line) for line in groups_path.read_text(encoding="utf-8").splitlines() if line.strip()]
pairs = []
keys = []
for group in groups:
    query_id = str(group["query_id"])
    query_text = str(group["query_text"])
    for candidate in group.get("candidates", []):
        pairs.append((query_text, str(candidate.get("text", ""))))
        keys.append((query_id, str(candidate["chunk_id"])))

model = CrossEncoder(model_name, device=device, max_length=max_length, trust_remote_code=trust_remote_code)
raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
scores_path.parent.mkdir(parents=True, exist_ok=True)
with scores_path.open("w", encoding="utf-8") as handle:
    for (query_id, chunk_id), score in zip(keys, raw_scores, strict=True):
        handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": float(score)}, sort_keys=True) + "\n")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sentence Transformers CrossEncoder scoring on a remote host.")
    parser.add_argument("groups_jsonl")
    parser.add_argument("scores_jsonl")
    parser.add_argument("--host", default="root@vicuna-host")
    parser.add_argument("--remote-python", default="/root/vicuna/venvs/activation-rag-embeddings/bin/python")
    parser.add_argument("--remote-work-dir", default="/tmp/activation-rag-rerank")
    parser.add_argument("--model", default="cross-encoder/ettin-reranker-150m-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    remote_root = Path(args.remote_work_dir)
    run_dir = remote_root / next(tempfile._get_candidate_names())
    remote_groups = run_dir / "groups.jsonl"
    remote_scores = run_dir / "scores.jsonl"
    run_remote(args.host, f"mkdir -p {shlex.quote(str(run_dir))}")
    try:
        subprocess.run(["scp", "-q", args.groups_jsonl, f"{args.host}:{remote_groups}"], check=True)
        command = " ".join(
            [
                f"RERANK_GROUPS={shlex.quote(str(remote_groups))}",
                f"RERANK_SCORES={shlex.quote(str(remote_scores))}",
                f"RERANK_MODEL={shlex.quote(args.model)}",
                f"RERANK_DEVICE={shlex.quote(args.device)}",
                f"RERANK_BATCH_SIZE={args.batch_size}",
                f"RERANK_MAX_LENGTH={args.max_length}",
                f"RERANK_TRUST_REMOTE_CODE={1 if args.trust_remote_code else 0}",
                shlex.quote(args.remote_python),
                "-c",
                shlex.quote(REMOTE_CODE),
            ]
        )
        run_remote(args.host, command)
        subprocess.run(["scp", "-q", f"{args.host}:{remote_scores}", args.scores_jsonl], check=True)
    finally:
        run_remote(args.host, f"rm -rf {shlex.quote(str(run_dir))}", check=False)


def run_remote(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(["ssh", host, command], text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"remote command failed: {command}")
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    return completed


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.schema import stable_hash


CAPTURE_MODES = ("full_forward", "early_stop_layer", "early_stop_layer_prefix_cache")
PROMPT_TEMPLATE_ID = "behavior_support_pair_v1"
PROMPT_TEMPLATE = """Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark behavior-prefill capture optimizations against Ettin reranking latency.")
    parser.add_argument("--groups", required=True, help="Behavior-pair candidate groups JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="speed-benchmark")
    parser.add_argument("--query-limit", type=int, default=25)
    parser.add_argument("--candidates-per-query", type=int, default=16)
    parser.add_argument("--capture-mode", action="append", choices=CAPTURE_MODES, help="Capture mode to time; default runs all modes.")
    parser.add_argument("--capture-host", default="root@vicuna-host")
    parser.add_argument("--capture-device", default="cuda")
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--remote-python", default="/mnt/disk-3tb/rmt-tail-venv-cu128/bin/python")
    parser.add_argument("--ettin-model", default="cross-encoder/ettin-reranker-150m-v1")
    parser.add_argument("--ettin-device", default="cuda")
    parser.add_argument("--ettin-batch-size", type=int, default=64)
    parser.add_argument("--ettin-max-length", type=int, default=512)
    parser.add_argument("--skip-ettin", action="store_true")
    args = parser.parse_args()

    summary = run_speed_benchmark(
        groups_path=Path(args.groups),
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
        query_limit=args.query_limit,
        candidates_per_query=args.candidates_per_query,
        capture_modes=tuple(args.capture_mode or CAPTURE_MODES),
        capture_host=args.capture_host,
        capture_device=args.capture_device,
        capture_batch_size=args.capture_batch_size,
        remote_python=args.remote_python,
        ettin_model=args.ettin_model,
        ettin_device=args.ettin_device,
        ettin_batch_size=args.ettin_batch_size,
        ettin_max_length=args.ettin_max_length,
        skip_ettin=args.skip_ettin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_speed_benchmark(
    *,
    groups_path: Path,
    out_dir: Path,
    dataset_name: str,
    query_limit: int,
    candidates_per_query: int,
    capture_modes: tuple[str, ...],
    capture_host: str,
    capture_device: str,
    capture_batch_size: int,
    remote_python: str,
    ettin_model: str,
    ettin_device: str,
    ettin_batch_size: int,
    ettin_max_length: int,
    skip_ettin: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_groups = load_jsonl(groups_path)
    groups = trim_groups(source_groups, query_limit=query_limit, candidates_per_query=candidates_per_query)
    requests = build_capture_requests(groups, dataset_name=dataset_name)
    groups_out = out_dir / "speed-groups.behavior-pair.jsonl"
    requests_out = out_dir / "speed-capture-requests.jsonl"
    write_jsonl(groups_out, groups)
    write_jsonl(requests_out, requests)

    capture_results: list[dict[str, Any]] = []
    for mode in capture_modes:
        rows_out = out_dir / f"capture-{mode}.rows.jsonl"
        timing_out = out_dir / f"capture-{mode}.timing.json"
        started = time.perf_counter()
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "capture_qwen_sae_prefill.py"),
            str(requests_out),
            str(rows_out),
            "--host",
            capture_host,
            "--remote-python",
            remote_python,
            "--device",
            capture_device,
            "--capture-execution-mode",
            mode,
            "--optimized-batch-size",
            str(capture_batch_size),
            "--timing-summary-out",
            str(timing_out),
        ]
        if mode == "early_stop_layer_prefix_cache":
            command.append("--allow-experimental-prefix-cache")
        completed = subprocess.run(command, text=True, check=False)
        elapsed = time.perf_counter() - started
        row_count = count_jsonl(rows_out)
        timing = json.loads(timing_out.read_text(encoding="utf-8")) if timing_out.exists() else {}
        capture_results.append(
            {
                "mode": mode,
                "returncode": completed.returncode,
                "wall_time_s": elapsed,
                "row_count": row_count,
                "rows_per_second_wall": row_count / elapsed if elapsed > 0 else 0.0,
                "remote_duration_s": float(timing.get("duration_s") or 0.0),
                "rows_per_second_remote": row_count / float(timing.get("duration_s")) if float(timing.get("duration_s") or 0.0) > 0 else 0.0,
                "timing": timing,
                "rows_out": str(rows_out),
                "timing_out": str(timing_out),
            }
        )
        if completed.returncode != 0:
            break

    ettin_result = None
    if not skip_ettin:
        started = time.perf_counter()
        scores = score_with_cross_encoder(
            groups,
            model_name=ettin_model,
            device=ettin_device,
            batch_size=ettin_batch_size,
            max_length=ettin_max_length,
        )
        elapsed = time.perf_counter() - started
        score_count = len(scores)
        scores_out = out_dir / "ettin-speed-scores.jsonl"
        with scores_out.open("w", encoding="utf-8") as handle:
            for (query_id, chunk_id), score in sorted(scores.items()):
                handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")
        ettin_result = {
            "model": ettin_model,
            "device": ettin_device,
            "batch_size": ettin_batch_size,
            "max_length": ettin_max_length,
            "wall_time_s": elapsed,
            "pair_count": score_count,
            "pairs_per_second_wall": score_count / elapsed if elapsed > 0 else 0.0,
            "scores_out": str(scores_out),
        }

    summary = {
        "schema_version": "activation_rag.behavior_prefill_speed_benchmark.v1",
        "groups": str(groups_path),
        "groups_out": str(groups_out),
        "requests_out": str(requests_out),
        "dataset_name": dataset_name,
        "capture_batch_size": capture_batch_size,
        "query_count": len(groups),
        "candidate_pair_count": len(requests),
        "capture_results": capture_results,
        "ettin_result": ettin_result,
    }
    write_json(out_dir / "speed-summary.json", summary)
    return summary


def trim_groups(groups: list[dict[str, Any]], *, query_limit: int, candidates_per_query: int) -> list[dict[str, Any]]:
    selected = groups[: max(0, query_limit)] if query_limit > 0 else list(groups)
    trimmed = []
    for group in selected:
        updated = dict(group)
        updated["candidates"] = list(group.get("candidates", []))[: max(1, candidates_per_query)]
        trimmed.append(updated)
    return trimmed


def build_capture_requests(groups: list[dict[str, Any]], *, dataset_name: str) -> list[dict[str, Any]]:
    prompt_hash = stable_hash(PROMPT_TEMPLATE, 32)
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        query_id = str(group["query_id"])
        query_text = str(group.get("query_text") or "")
        for candidate in group.get("candidates", []):
            chunk_id = str(candidate["chunk_id"])
            behavior_chunk_id = str(candidate.get("behavior_chunk_id") or behavior_pair_id(
                dataset_name=dataset_name,
                query_id=query_id,
                chunk_id=chunk_id,
                prompt_hash=prompt_hash,
            ))
            if behavior_chunk_id in seen:
                continue
            seen.add(behavior_chunk_id)
            prompt_text = str(candidate.get("behavior_prompt_text") or PROMPT_TEMPLATE.format(query=query_text.strip(), evidence=str(candidate.get("text") or "").strip()))
            requests.append(
                {
                    "schema_version": "activation_rag.prefill_capture_request.v1",
                    "chunk_id": behavior_chunk_id,
                    "document_id": f"behavior_pair:{dataset_name}:speed:{query_id}",
                    "capture_run_id": f"behavior-prefill-speed-{dataset_name}",
                    "provider_id": "qwen-rmt-sae-prefill",
                    "model_id": "qwen3-4b-rmt-sae",
                    "site_id": "l07_resid_pre",
                    "layer_selection_policy": "qwen3_rmt_l07_resid_pre_core245",
                    "prompt_template_id": PROMPT_TEMPLATE_ID,
                    "prompt_template_hash": prompt_hash,
                    "normalization_policy": "qwen_sae_checkpoint_mean_rms_topk64",
                    "requested_prompt_section_label": "query_candidate_behavior_prompt",
                    "prompt_section_label": "query_candidate_behavior_prompt",
                    "text": prompt_text,
                    "prompt_text": prompt_text,
                    "text_hash": stable_hash(prompt_text, 32),
                    "token_count_estimate": max(1, len(prompt_text.split())),
                    "metadata": {
                        "dataset_name": dataset_name,
                        "query_id": query_id,
                        "source_chunk_id": chunk_id,
                        "label": int(candidate.get("label", 0)),
                    },
                }
            )
    return requests


def behavior_pair_id(*, dataset_name: str, query_id: str, chunk_id: str, prompt_hash: str) -> str:
    return stable_hash(f"behavior_pair_v1\n{dataset_name}\n{query_id}\n{chunk_id}\n{prompt_hash}", 32)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_with_cross_encoder(
    groups: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[tuple[str, str], float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device, max_length=max_length)
    pairs: list[tuple[str, str]] = []
    keys: list[tuple[str, str]] = []
    for group in groups:
        query_id = str(group["query_id"])
        query_text = str(group.get("query_text") or "")
        for candidate in group.get("candidates", []):
            pairs.append((query_text, str(candidate.get("text", ""))))
            keys.append((query_id, str(candidate["chunk_id"])))
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return {key: float(score) for key, score in zip(keys, raw_scores, strict=True)}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()

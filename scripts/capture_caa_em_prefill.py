#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROW_SCHEMA = "activation_rag.caa8_prefill_capture.row.v1"
DIMENSIONS = (
    "constraint_imposed",
    "final_answer_readiness",
    "hair",
    "plan_formation",
    "repair_readiness",
    "runtime_failure_pressure",
    "stall_looping",
    "state_carryover",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture strict zero-token Qwen 8-head CAA/EM prefill telemetry.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_dir")
    parser.add_argument("--base-url", default="http://127.0.0.1:18109")
    parser.add_argument("--neutral-baseline-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--trace-retries", type=int, default=30)
    parser.add_argument("--heartbeat-every", type=int, default=100)
    args = parser.parse_args()

    summary = capture_requests(
        input_jsonl=Path(args.input_jsonl),
        output_dir=Path(args.output_dir),
        base_url=str(args.base_url).rstrip("/"),
        neutral_baseline_json=Path(args.neutral_baseline_json),
        limit=int(args.limit),
        timeout=float(args.timeout),
        trace_retries=int(args.trace_retries),
        heartbeat_every=int(args.heartbeat_every),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def capture_requests(
    *,
    input_jsonl: Path,
    output_dir: Path,
    base_url: str,
    neutral_baseline_json: Path,
    limit: int,
    timeout: float,
    trace_retries: int,
    heartbeat_every: int,
) -> dict[str, Any]:
    baseline = _load_baseline(neutral_baseline_json)
    requests = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        requests = requests[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    captured = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        out_path = output_dir / f"{request['chunk_id']}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            row = _capture_one(
                request=request,
                baseline=baseline,
                base_url=base_url,
                timeout=timeout,
                trace_retries=trace_retries,
            )
            out_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            captured += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"chunk_id": request.get("chunk_id"), "error": repr(exc)})
            if len(failures) > 20:
                raise
        if heartbeat_every > 0 and index % heartbeat_every == 0:
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(requests),
                        "captured": captured,
                        "skipped": skipped,
                        "failed": failed,
                        "elapsed_s": elapsed,
                        "rows_per_s": index / max(elapsed, 1e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "schema_version": "activation_rag.caa8_prefill_capture_summary.v1",
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "base_url": base_url,
        "requested_count": len(requests),
        "captured_count": captured,
        "skipped_count": skipped,
        "failed_count": failed,
        "failures": failures[:20],
        "elapsed_s": time.time() - started,
        "strict_zero_required": True,
        "dimensions": list(DIMENSIONS),
    }
    (output_dir / "capture-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"CAA/EM capture failed for {failed} rows; see capture-summary.json")
    return summary


def _capture_one(
    *,
    request: dict[str, Any],
    baseline: dict[str, float],
    base_url: str,
    timeout: float,
    trace_retries: int,
) -> dict[str, Any]:
    request_id = f"activation-rag-caa-{request['capture_run_id']}"
    payload = {
        "model": "qwen3-4b-q8_0",
        "messages": [{"role": "user", "content": str(request.get("prompt_text") or "")}],
        "temperature": 0,
        "max_tokens": 0,
        "max_completion_tokens": 0,
        "max_output_tokens": 0,
        "x-vicuna-provider-max-tokens-override": 0,
        "x-vicuna-prefill-only": True,
        "reasoning_budget_tokens": 0,
        "x-vicuna-provider-reasoning-budget-override": 0,
        "stream": False,
        "session_id": request_id,
        "conversation_id": request_id,
        "turn_index": 0,
        "vicuna_include_emotive_trace": False,
    }
    response = _post_json(f"{base_url}/v1/chat/completions", payload, request_id=request_id, timeout=timeout)
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    content = ""
    finish_reason = None
    if choices:
        finish_reason = choices[0].get("finish_reason")
        content = str((choices[0].get("message") or {}).get("content") or "")
    if int(usage.get("completion_tokens") or 0) != 0 or content:
        raise RuntimeError(f"strict zero-token prefill was not honored: usage={usage} content_len={len(content)}")
    trace = _fetch_trace(base_url, request_id=request_id, timeout=timeout, retries=trace_retries)
    trace_request_id = str(trace.get("request_id") or "")
    if trace_request_id != request_id:
        raise RuntimeError(f"trace request_id mismatch: expected={request_id} got={trace_request_id}")
    current = {dim: float((trace.get("final_em_v2_features") or {}).get(dim, 0.0)) for dim in DIMENSIONS}
    if not any(abs(value) > 0.0 for value in current.values()):
        raise RuntimeError("trace did not contain non-empty final_em_v2_features")
    neutral = {dim: float(baseline.get(dim, 0.0)) for dim in DIMENSIONS}
    prior = dict(neutral)
    delta_neutral = {dim: current[dim] - neutral[dim] for dim in DIMENSIONS}
    delta_current = {dim: current[dim] - prior[dim] for dim in DIMENSIONS}
    saturation = {dim: min(1.0, max(0.0, abs(current[dim]))) for dim in DIMENSIONS}
    residual_headroom = {dim: max(0.0, 1.0 - saturation[dim]) for dim in DIMENSIONS}
    positive_mass = sum(value for value in delta_neutral.values() if value > 0.0)
    negative_mass = sum(-value for value in delta_neutral.values() if value < 0.0)
    total_mass = positive_mass + negative_mass
    signed_balance = 0.0 if total_mass == 0.0 else (positive_mass - negative_mass) / total_mass
    return {
        "schema_version": ROW_SCHEMA,
        "chunk_id": str(request["chunk_id"]),
        "document_id": str(request.get("document_id") or ""),
        "capture_run_id": str(request["capture_run_id"]),
        "provider_id": "qwen-caa8-compact-prefill",
        "model_id": "qwen3-4b-q8_0",
        "site_id": "qwen3_caa8_compact_trace",
        "hook_name": "compact_emotive_trace.final_em_v2_features",
        "layer_index": -1,
        "layer_selection_policy": "qwen3_8head_production_compact_em_v2_bundle",
        "prompt_template_id": str(request.get("prompt_template_id") or "raw_chunk_v1_strict_zero_caa8"),
        "prompt_template_hash": str(request.get("prompt_template_hash") or ""),
        "normalization_policy": "qwen3_8head_neutral_baseline_delta_v1",
        "prompt_section_label": str(request.get("prompt_section_label") or ""),
        "requested_prompt_section_label": str(request.get("requested_prompt_section_label") or request.get("prompt_section_label") or ""),
        "token_start": 0,
        "token_end": int(usage.get("prompt_tokens") or 0),
        "aggregation": "compact_em_v2_final_prefill_state",
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "current_em_state": current,
        "neutral_baseline_state": neutral,
        "prior_current_state": prior,
        "delta_vs_neutral": delta_neutral,
        "delta_vs_current": delta_current,
        "saturation": saturation,
        "residual_headroom": residual_headroom,
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "total_mass": total_mass,
        "signed_balance": signed_balance,
        "sae_feature_values": {},
        "sae_delta_vs_neutral": {},
        "sae_delta_vs_current": {},
        "sae_feature_mask": {},
        "em_v2_inputs": _float_map(trace.get("final_em_v2_inputs")),
        "telemetry_valid": True,
        "invalid_reason": None,
        "provenance": {
            "strict_zero_prefill": True,
            "request_id": request_id,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "finish_reason": finish_reason,
            "final_em_v2_input_count": len(trace.get("final_em_v2_inputs") or {}),
            "prior_current_policy": "isolated_prefill_uses_neutral_baseline_as_prior",
            "text_hash": request.get("text_hash"),
        },
    }


def _load_baseline(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = payload.get("baseline_em") or payload.get("neutral_em_baseline") or payload
    return {str(key): float(value) for key, value in baseline.items() if _is_number(value)}


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if _is_number(raw)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _post_json(url: str, payload: dict[str, Any], *, request_id: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Client-Request-Id": request_id},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_trace(base_url: str, *, request_id: str, timeout: float, retries: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"request_id": request_id, "summary_only": 1})
    url = f"{base_url}/v1/emotive/trace/latest?{query}"
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=min(timeout, 10.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            trace = payload.get("trace")
            if isinstance(trace, dict) and trace.get("final_em_v2_features"):
                return trace
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"trace for request_id={request_id} did not contain final_em_v2_features")


if __name__ == "__main__":
    main()

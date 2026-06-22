#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Any


PREFILL_MARKERS = (":prefill_last:", ":post50_mean:", ":traj_prefill_mean:")
ALLOWED_PREFILL_WINDOWS = ("prefill_last", "post50_mean", "trajectory")
ALLOWED_RAW_SECTION_LABEL = "prompt"


def prefill_feature_subset(features: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in features.items()
        if any(marker in key for marker in PREFILL_MARKERS)
    }


def requested_prompt_section_label(request: dict[str, Any]) -> str:
    explicit = request.get("requested_prompt_section_label") or request.get("prompt_section_label")
    if explicit:
        return str(explicit)
    return "query" if str(request.get("document_id")) == "query" else "document_chunk"


def build_output_row(
    request: dict[str, Any],
    *,
    features: dict[str, float],
    request_id: str,
    activation_dir: str,
    response_usage: dict[str, Any],
    endpoint: str,
    error_message: str | None = None,
    raw_activation_reused: bool = False,
    manifest_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_tokens = int(response_usage.get("completion_tokens") or 0)
    prompt_tokens = int(response_usage.get("prompt_tokens") or 0)
    total_tokens = int(response_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    generation_disabled = completion_tokens == 0
    prefill_only_extracted = bool(features)
    manifest_stats = manifest_stats or {}
    semantic_section_label = requested_prompt_section_label(request)
    provenance = {
        "capture_phase": "prefill",
        "generation_disabled": generation_disabled,
        "prefill_only_extracted": prefill_only_extracted,
        "source_generation_disabled": generation_disabled,
        "source_completion_tokens": completion_tokens,
        "source_prompt_tokens": prompt_tokens,
        "source_total_tokens": total_tokens,
        "requested_prompt_section_label": semantic_section_label,
        "raw_prompt_section_label": ALLOWED_RAW_SECTION_LABEL,
        "allowed_prefill_windows": list(ALLOWED_PREFILL_WINDOWS),
        "manifest_filter": manifest_stats,
        "raw_request_id": request_id,
        "raw_activation_dir": activation_dir,
        "endpoint": endpoint,
        "raw_activation_reused": raw_activation_reused,
    }
    if error_message:
        provenance["capture_error"] = error_message
    return {
        "schema_version": "activation_rag.sidecar_prefill_capture.row.v1",
        "chunk_id": request["chunk_id"],
        "document_id": request["document_id"],
        "capture_run_id": request["capture_run_id"],
        "provider_id": request["provider_id"],
        "model_id": request["model_id"],
        "site_id": request["site_id"],
        "layer_selection_policy": request["layer_selection_policy"],
        "prompt_template_id": request["prompt_template_id"],
        "prompt_template_hash": request["prompt_template_hash"],
        "normalization_policy": request["normalization_policy"],
        "prompt_section_label": semantic_section_label,
        "requested_prompt_section_label": semantic_section_label,
        "raw_prompt_section_label": ALLOWED_RAW_SECTION_LABEL,
        "token_start": 0,
        "token_end": int(request.get("token_count_estimate") or 0),
        "aggregation": "raw_capture_prefill_summary",
        "capture_phase": "prefill",
        "generation_disabled": generation_disabled,
        "prefill_only_extracted": prefill_only_extracted,
        "current_em_state": {},
        "neutral_baseline_state": {},
        "prior_current_state": {},
        "delta_vs_neutral": {},
        "delta_vs_current": {},
        "saturation": {},
        "residual_headroom": {},
        "sae_feature_values": features,
        "sae_delta_vs_neutral": features,
        "sae_delta_vs_current": features,
        "sae_feature_mask": {key: value != 0.0 for key, value in features.items()},
        "telemetry_valid": bool(features),
        "invalid_reason": None if features else (error_message or "no_prefill_activation_features"),
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chunk-aligned prefill telemetry through a sidecar llama.cpp server.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--host", default="root@vicuna-host")
    parser.add_argument("--base-url", default="http://127.0.0.1:28080")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--remote-em-v2-python", default="/root/vicuna/llama.cpp-runtime/portable/em_v2/python/src")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--activation-chunk-count", type=int, default=8)
    parser.add_argument("--request-prefix", default="activation-rag")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-reuse-existing-raw", action="store_true", help="Ignore existing raw activation manifests and recapture.")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = {
        "rows": rows,
        "base_url": args.base_url.rstrip("/"),
        "raw_root": args.raw_root,
        "remote_em_v2_python": args.remote_em_v2_python,
        "timeout": args.timeout,
        "activation_chunk_count": args.activation_chunk_count,
        "request_prefix": args.request_prefix,
        "progress_every": args.progress_every,
        "reuse_existing_raw": not args.no_reuse_existing_raw,
    }
    completed = run_remote_capture(args.host, args.remote_python, payload)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"ssh capture failed: {completed.returncode}")
    output_path.write_text(completed.stdout, encoding="utf-8")


def run_remote_capture(host: str, remote_python: str, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        local_payload = Path(handle.name)
        json.dump(payload, handle)
    remote_payload = f"/tmp/activation-rag-capture-payload-{local_payload.name}.json"
    try:
        copy = subprocess.run(
            ["scp", str(local_payload), f"{host}:{remote_payload}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if copy.returncode != 0:
            return copy
        command = (
            f"ACTIVATION_RAG_CAPTURE_PAYLOAD={shlex.quote(remote_payload)} "
            f"{shlex.quote(remote_python)} -c {shlex.quote(REMOTE_CAPTURE_CODE)}; "
            f"rm -f {shlex.quote(remote_payload)}"
        )
        return subprocess.run(
            ["ssh", host, command],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        local_payload.unlink(missing_ok=True)


REMOTE_CAPTURE_CODE = textwrap.dedent(
    r'''
    import json
    import os
    import re
    import sys
    import time
    import urllib.request
    from pathlib import Path

    payload_path = os.environ.get("ACTIVATION_RAG_CAPTURE_PAYLOAD")
    if payload_path:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    else:
        payload = json.loads(sys.stdin.read())
    sys.path.insert(0, payload["remote_em_v2_python"])
    from em_v2.activation_capture_features import summarize_activation_directory

    PREFILL_MARKERS = (":prefill_last:", ":post50_mean:", ":traj_prefill_mean:")
    ALLOWED_PREFILL_WINDOWS = ("prefill_last", "post50_mean", "trajectory")
    ALLOWED_RAW_SECTION_LABEL = "prompt"

    def safe_id(value):
        value = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value))
        return value[:180] or "request"

    def prefill_feature_subset(features):
        return {
            str(key): float(value)
            for key, value in features.items()
            if any(marker in str(key) for marker in PREFILL_MARKERS)
        }

    def requested_prompt_section_label(request):
        explicit = request.get("requested_prompt_section_label") or request.get("prompt_section_label")
        if explicit:
            return str(explicit)
        return "query" if str(request.get("document_id")) == "query" else "document_chunk"

    def build_output_row(request, *, features, request_id, activation_dir, response_usage, endpoint, error_message=None, raw_activation_reused=False, manifest_stats=None):
        completion_tokens = int(response_usage.get("completion_tokens") or 0)
        prompt_tokens = int(response_usage.get("prompt_tokens") or 0)
        total_tokens = int(response_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        generation_disabled = completion_tokens == 0
        prefill_only_extracted = bool(features)
        manifest_stats = manifest_stats or {}
        semantic_section_label = requested_prompt_section_label(request)
        provenance = {
            "capture_phase": "prefill",
            "generation_disabled": generation_disabled,
            "prefill_only_extracted": prefill_only_extracted,
            "source_generation_disabled": generation_disabled,
            "source_completion_tokens": completion_tokens,
            "source_prompt_tokens": prompt_tokens,
            "source_total_tokens": total_tokens,
            "requested_prompt_section_label": semantic_section_label,
            "raw_prompt_section_label": ALLOWED_RAW_SECTION_LABEL,
            "allowed_prefill_windows": list(ALLOWED_PREFILL_WINDOWS),
            "manifest_filter": manifest_stats,
            "raw_request_id": request_id,
            "raw_activation_dir": activation_dir,
            "endpoint": endpoint,
            "raw_activation_reused": bool(raw_activation_reused),
        }
        if error_message:
            provenance["capture_error"] = str(error_message)
        return {
            "schema_version": "activation_rag.sidecar_prefill_capture.row.v1",
            "chunk_id": request["chunk_id"],
            "document_id": request["document_id"],
            "capture_run_id": request["capture_run_id"],
            "provider_id": request["provider_id"],
            "model_id": request["model_id"],
            "site_id": request["site_id"],
            "layer_selection_policy": request["layer_selection_policy"],
            "prompt_template_id": request["prompt_template_id"],
            "prompt_template_hash": request["prompt_template_hash"],
            "normalization_policy": request["normalization_policy"],
            "prompt_section_label": semantic_section_label,
            "requested_prompt_section_label": semantic_section_label,
            "raw_prompt_section_label": ALLOWED_RAW_SECTION_LABEL,
            "token_start": 0,
            "token_end": int(request.get("token_count_estimate") or 0),
            "aggregation": "raw_capture_prefill_summary",
            "capture_phase": "prefill",
            "generation_disabled": generation_disabled,
            "prefill_only_extracted": prefill_only_extracted,
            "current_em_state": {},
            "neutral_baseline_state": {},
            "prior_current_state": {},
            "delta_vs_neutral": {},
            "delta_vs_current": {},
            "saturation": {},
            "residual_headroom": {},
            "sae_feature_values": features,
            "sae_delta_vs_neutral": features,
            "sae_delta_vs_current": features,
            "sae_feature_mask": {key: value != 0.0 for key, value in features.items()},
            "telemetry_valid": bool(features),
            "invalid_reason": None if features else (str(error_message) if error_message else "no_prefill_activation_features"),
            "provenance": provenance,
        }

    def post_chat(row, request_id):
        endpoint = "/v1/chat/completions"
        request_payload = {
            "model": "local",
            "messages": [{"role": "user", "content": row.get("prompt_text") or row.get("text") or ""}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 0,
            "x-vicuna-prefill-only": True,
            "stream": False,
        }
        request = urllib.request.Request(
            payload["base_url"] + endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Client-Request-Id": request_id},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=int(payload["timeout"])) as response:
            return endpoint, json.loads(response.read().decode("utf-8"))

    def wait_for_activation_dir(request_id):
        activation_dir = Path(payload["raw_root"]) / request_id
        manifest = activation_dir / "manifest.jsonl"
        deadline = time.time() + int(payload["timeout"])
        while time.time() < deadline:
            if manifest.exists():
                return activation_dir
            time.sleep(0.05)
        raise TimeoutError(f"raw activation manifest not found for {request_id}: {manifest}")

    def filtered_prefill_manifest_dir(activation_dir):
        manifest_path = activation_dir / "manifest.jsonl"
        binary_path = activation_dir / "activations.f16bin"
        if not manifest_path.exists() or not binary_path.exists():
            raise FileNotFoundError(f"missing activation artifacts under {activation_dir}")
        entries = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        allowed = []
        dropped_by_window = {}
        dropped_by_section = {}
        for entry in entries:
            window = str(entry.get("window") or "")
            section_label = str(entry.get("prompt_section_label") or "")
            if window not in ALLOWED_PREFILL_WINDOWS:
                dropped_by_window[window] = dropped_by_window.get(window, 0) + 1
                continue
            if section_label != ALLOWED_RAW_SECTION_LABEL:
                dropped_by_section[section_label] = dropped_by_section.get(section_label, 0) + 1
                continue
            allowed.append(entry)
        if not allowed:
            raise ValueError(f"no prompt-prefill manifest rows remained after filtering {activation_dir}")

        filtered_dir = activation_dir / ".prefill_prompt_only"
        filtered_dir.mkdir(exist_ok=True)
        filtered_manifest = filtered_dir / "manifest.jsonl"
        filtered_manifest.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in allowed),
            encoding="utf-8",
        )
        filtered_binary = filtered_dir / "activations.f16bin"
        if not filtered_binary.exists():
            try:
                filtered_binary.symlink_to(binary_path)
            except FileExistsError:
                pass
        stats = {
            "raw_manifest_rows": len(entries),
            "accepted_prompt_prefill_rows": len(allowed),
            "dropped_by_window": dropped_by_window,
            "dropped_by_section_label": dropped_by_section,
            "filtered_manifest_dir": str(filtered_dir),
        }
        return filtered_dir, stats

    def summarize_prefill_features(activation_dir):
        filtered_dir, stats = filtered_prefill_manifest_dir(activation_dir)
        features = prefill_feature_subset(
            summarize_activation_directory(
                filtered_dir,
                chunk_count=int(payload["activation_chunk_count"]),
            )
        )
        return features, stats

    for index, row in enumerate(payload["rows"], start=1):
        request_id = safe_id(f"{payload['request_prefix']}-{row['capture_run_id'][:12]}-{row['chunk_id'][:24]}")
        try:
            activation_dir = Path(payload["raw_root"]) / request_id
            if bool(payload.get("reuse_existing_raw")) and (activation_dir / "manifest.jsonl").exists():
                endpoint = "reuse://raw-activation"
                response = {"usage": {"prompt_tokens": int(row.get("token_count_estimate") or 0), "completion_tokens": 0}}
                features, manifest_stats = summarize_prefill_features(activation_dir)
                out = build_output_row(
                    row,
                    features=features,
                    request_id=request_id,
                    activation_dir=str(activation_dir),
                    response_usage=response.get("usage") or {},
                    endpoint=endpoint,
                    raw_activation_reused=True,
                    manifest_stats=manifest_stats,
                )
            else:
                endpoint, response = post_chat(row, request_id)
                activation_dir = wait_for_activation_dir(request_id)
                features, manifest_stats = summarize_prefill_features(activation_dir)
                out = build_output_row(
                    row,
                    features=features,
                    request_id=request_id,
                    activation_dir=str(activation_dir),
                    response_usage=response.get("usage") or {},
                    endpoint=endpoint,
                    manifest_stats=manifest_stats,
                )
        except Exception as exc:
            endpoint = "/v1/chat/completions"
            activation_dir = Path(payload["raw_root"]) / request_id
            out = build_output_row(
                row,
                features={},
                request_id=request_id,
                activation_dir=str(activation_dir),
                response_usage={},
                endpoint=endpoint,
                error_message=repr(exc),
            )
        print(json.dumps(out, sort_keys=True), flush=True)
        progress_every = int(payload.get("progress_every") or 0)
        if progress_every and index % progress_every == 0:
            print(f"captured {index}/{len(payload['rows'])}", file=sys.stderr, flush=True)
    '''
)


if __name__ == "__main__":
    main()

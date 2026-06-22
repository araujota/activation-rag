#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROW_SCHEMA = "activation_rag.qwen_sae_prefill_capture.row.v1"
DEFAULT_REMOTE_PYTHON = "/mnt/disk-3tb/rmt-tail-venv-cu128/bin/python"
DEFAULT_REMOTE_EM_V2 = "/mnt/disk-3tb/longmem-mechinterp-selector-release/vendor/em_v2_python/src"
DEFAULT_FEATURE_MANIFEST = "/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/selector_materialization/feature_manifest.json"
DEFAULT_MODEL_PATH = "/mnt/disk-3tb/models/qwen3-4b-hf"
DEFAULT_JOINT_CHECKPOINT = "/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/rmt/qwen3_rmt_joint_memory_latest.pt"
DEFAULT_SAE_CHECKPOINT = "/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/sae/topk_sae_latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chunk-aligned prefill telemetry with the longmem Qwen/RMT SAE encoder.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--host", default="root@vicuna-host")
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--remote-em-v2-python", default=DEFAULT_REMOTE_EM_V2)
    parser.add_argument("--feature-manifest", default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--joint-checkpoint", default=DEFAULT_JOINT_CHECKPOINT)
    parser.add_argument("--sae-checkpoint", default=DEFAULT_SAE_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-mode", default="qwen_no_memory_span_preview", choices=["qwen_no_memory_span_preview", "span_self_written_memory", "episode_prefix_memory"])
    parser.add_argument("--layer-index", type=int, default=7)
    parser.add_argument("--qwen-seq-tokens", type=int, default=128)
    parser.add_argument("--min-activation", type=float, default=0.0)
    parser.add_argument("--request-prefix", default="activation-rag-qwen-sae")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    requests = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = build_episode_rows(requests)
    payload = {
        "requests": requests,
        "episodes": episodes,
        "remote_em_v2_python": args.remote_em_v2_python,
        "feature_manifest": args.feature_manifest,
        "model_path": args.model_path,
        "joint_checkpoint": args.joint_checkpoint,
        "sae_checkpoint": args.sae_checkpoint,
        "device": args.device,
        "evidence_mode": args.evidence_mode,
        "layer_index": args.layer_index,
        "qwen_seq_tokens": args.qwen_seq_tokens,
        "min_activation": args.min_activation,
        "request_prefix": args.request_prefix,
    }
    completed = run_remote_capture(args.host, args.remote_python, payload)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"ssh qwen SAE capture failed: {completed.returncode}")
    output_path.write_text(completed.stdout, encoding="utf-8")


def build_episode_rows(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for request in requests:
        chunk_id = str(request["chunk_id"])
        text = str(request.get("prompt_text") or request.get("text") or "")
        episodes.append(
            {
                "episode_id": f"activation-rag-prefill-{chunk_id}",
                "request_id": str(request.get("capture_run_id") or chunk_id),
                "split": "activation_rag_prefill",
                "candidate_spans": [
                    {
                        "span_id": chunk_id,
                        "text_preview": text,
                        "text_hash": str(request.get("text_hash") or ""),
                    }
                ],
            }
        )
    return episodes


def adapt_sae_row(
    *,
    request: dict[str, Any],
    sae_row: dict[str, Any],
    feature_manifest_summary: dict[str, Any],
    encoder_summary: dict[str, Any],
) -> dict[str, Any]:
    values = _float_map(sae_row.get("sae_feature_values"))
    semantic_section_label = str(
        request.get("requested_prompt_section_label")
        or request.get("prompt_section_label")
        or ("query" if str(request.get("document_id")) == "query" else "document_chunk")
    )
    provenance = {
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "encoder": "longmem_qwen_rmt_sae",
        "evidence_mode": sae_row.get("evidence_mode") or encoder_summary.get("evidence_mode"),
        "feature_set_id": feature_manifest_summary.get("feature_set_id"),
        "feature_count": feature_manifest_summary.get("feature_count"),
        "layer_index": sae_row.get("layer_index"),
        "qwen_seq_tokens": sae_row.get("qwen_seq_tokens"),
        "sequence_tokens": sae_row.get("sequence_tokens"),
        "max_token_offset_by_feature": dict(sae_row.get("max_token_offset_by_feature") or {}),
        "encoder_summary": encoder_summary,
    }
    return {
        "schema_version": ROW_SCHEMA,
        "chunk_id": str(request["chunk_id"]),
        "document_id": str(request["document_id"]),
        "capture_run_id": str(request["capture_run_id"]),
        "provider_id": str(request.get("provider_id") or "qwen-rmt-sae-prefill"),
        "model_id": str(request.get("model_id") or "qwen3-4b-rmt-sae"),
        "site_id": str(request.get("site_id") or "l07_resid_pre"),
        "hook_name": "qwen.model.layers.7.resid_pre",
        "layer_index": int(sae_row.get("layer_index") or 7),
        "layer_selection_policy": str(request.get("layer_selection_policy") or "qwen3_rmt_l07_resid_pre_core245"),
        "prompt_template_id": str(request.get("prompt_template_id") or "raw_chunk_v1"),
        "prompt_template_hash": str(request.get("prompt_template_hash") or ""),
        "normalization_policy": str(request.get("normalization_policy") or "qwen_sae_checkpoint_mean_rms_topk64"),
        "prompt_section_label": semantic_section_label,
        "requested_prompt_section_label": semantic_section_label,
        "token_start": 0,
        "token_end": int(sae_row.get("sequence_tokens") or request.get("token_count_estimate") or 0),
        "aggregation": "selected_sae_max_over_prompt_tokens",
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "current_em_state": {},
        "neutral_baseline_state": {},
        "prior_current_state": {},
        "delta_vs_neutral": {},
        "delta_vs_current": {},
        "saturation": {},
        "residual_headroom": {},
        "sae_feature_values": values,
        "sae_delta_vs_neutral": dict(values),
        "sae_delta_vs_current": dict(values),
        "sae_feature_mask": {key: value != 0.0 for key, value in values.items()},
        "telemetry_valid": bool(values),
        "invalid_reason": None if values else "no_selected_sae_features",
        "provenance": provenance,
    }


def run_remote_capture(host: str, remote_python: str, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        local_payload = Path(handle.name)
        json.dump(payload, handle)
    remote_payload = f"/tmp/activation-rag-qwen-sae-payload-{local_payload.name}.json"
    try:
        copy = subprocess.run(["scp", str(local_payload), f"{host}:{remote_payload}"], text=True, capture_output=True, check=False)
        if copy.returncode != 0:
            return copy
        command = (
            f"ACTIVATION_RAG_QWEN_SAE_PAYLOAD={shlex.quote(remote_payload)} "
            f"{shlex.quote(remote_python)} -c {shlex.quote(REMOTE_CAPTURE_CODE)}; "
            f"rm -f {shlex.quote(remote_payload)}"
        )
        return subprocess.run(["ssh", host, command], text=True, capture_output=True, check=False)
    finally:
        local_payload.unlink(missing_ok=True)


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if raw is not None}


REMOTE_CAPTURE_CODE = r'''
import json
import os
import sys
import tempfile
from pathlib import Path

payload = json.loads(Path(os.environ["ACTIVATION_RAG_QWEN_SAE_PAYLOAD"]).read_text(encoding="utf-8"))
sys.path.insert(0, payload["remote_em_v2_python"])

from em_v2.encode_rmt_span_selector_sae_features import encode_span_sae_features
from em_v2.rmt_span_selector import read_json

def float_map(value):
    if not isinstance(value, dict):
        return {}
    return {str(key): float(raw) for key, raw in value.items() if raw is not None}

def requested_section(request):
    return str(request.get("requested_prompt_section_label") or request.get("prompt_section_label") or ("query" if str(request.get("document_id")) == "query" else "document_chunk"))

def adapt(request, sae_row, feature_manifest_summary, encoder_summary):
    values = float_map(sae_row.get("sae_feature_values"))
    section = requested_section(request)
    provenance = {
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "encoder": "longmem_qwen_rmt_sae",
        "evidence_mode": sae_row.get("evidence_mode") or encoder_summary.get("evidence_mode"),
        "feature_set_id": feature_manifest_summary.get("feature_set_id"),
        "feature_count": feature_manifest_summary.get("feature_count"),
        "layer_index": sae_row.get("layer_index"),
        "qwen_seq_tokens": sae_row.get("qwen_seq_tokens"),
        "sequence_tokens": sae_row.get("sequence_tokens"),
        "max_token_offset_by_feature": dict(sae_row.get("max_token_offset_by_feature") or {}),
        "encoder_summary": encoder_summary,
    }
    return {
        "schema_version": "activation_rag.qwen_sae_prefill_capture.row.v1",
        "chunk_id": str(request["chunk_id"]),
        "document_id": str(request["document_id"]),
        "capture_run_id": str(request["capture_run_id"]),
        "provider_id": str(request.get("provider_id") or "qwen-rmt-sae-prefill"),
        "model_id": str(request.get("model_id") or "qwen3-4b-rmt-sae"),
        "site_id": str(request.get("site_id") or "l07_resid_pre"),
        "hook_name": "qwen.model.layers.7.resid_pre",
        "layer_index": int(sae_row.get("layer_index") or 7),
        "layer_selection_policy": str(request.get("layer_selection_policy") or "qwen3_rmt_l07_resid_pre_core245"),
        "prompt_template_id": str(request.get("prompt_template_id") or "raw_chunk_v1"),
        "prompt_template_hash": str(request.get("prompt_template_hash") or ""),
        "normalization_policy": str(request.get("normalization_policy") or "qwen_sae_checkpoint_mean_rms_topk64"),
        "prompt_section_label": section,
        "requested_prompt_section_label": section,
        "token_start": 0,
        "token_end": int(sae_row.get("sequence_tokens") or request.get("token_count_estimate") or 0),
        "aggregation": "selected_sae_max_over_prompt_tokens",
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "current_em_state": {},
        "neutral_baseline_state": {},
        "prior_current_state": {},
        "delta_vs_neutral": {},
        "delta_vs_current": {},
        "saturation": {},
        "residual_headroom": {},
        "sae_feature_values": values,
        "sae_delta_vs_neutral": dict(values),
        "sae_delta_vs_current": dict(values),
        "sae_feature_mask": {key: value != 0.0 for key, value in values.items()},
        "telemetry_valid": bool(values),
        "invalid_reason": None if values else "no_selected_sae_features",
        "provenance": provenance,
    }

manifest = read_json(Path(payload["feature_manifest"]))
rows, summary = encode_span_sae_features(
    episodes=payload["episodes"],
    manifest=manifest,
    model_path=Path(payload["model_path"]),
    joint_checkpoint_path=Path(payload["joint_checkpoint"]),
    sae_checkpoint_path=Path(payload["sae_checkpoint"]),
    layer_index=int(payload["layer_index"]),
    qwen_seq_tokens=int(payload["qwen_seq_tokens"]),
    device=str(payload["device"]),
    evidence_mode=str(payload["evidence_mode"]),
    min_activation=float(payload["min_activation"]),
)
requests_by_chunk = {str(request["chunk_id"]): request for request in payload["requests"]}
feature_manifest_summary = {"feature_set_id": manifest.get("feature_set_id"), "feature_count": len(manifest.get("sae_feature_ids") or [])}
for sae_row in rows:
    chunk_id = str(sae_row.get("span_id") or "")
    request = requests_by_chunk.get(chunk_id)
    if request is None:
        continue
    print(json.dumps(adapt(request, sae_row, feature_manifest_summary, summary), sort_keys=True), flush=True)
'''


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--capture-execution-mode",
        choices=("full_forward", "early_stop_layer", "early_stop_layer_prefix_cache"),
        default="full_forward",
        help="Execution mode for prefill capture. Optimized modes are intended only for strict zero-token reranking telemetry.",
    )
    parser.add_argument(
        "--timing-summary-out",
        help="Optional path for a JSON timing summary emitted by the remote capture process.",
    )
    parser.add_argument(
        "--allow-experimental-prefix-cache",
        action="store_true",
        help="Allow early_stop_layer_prefix_cache despite known need for paired equivalence validation.",
    )
    parser.add_argument(
        "--optimized-batch-size",
        type=int,
        default=8,
        help="Batch size for early_stop_layer reranking telemetry only. Full-forward capture remains unbatched.",
    )
    parser.add_argument(
        "--allow-nonexact-batched-prefill",
        action="store_true",
        help="Permit early_stop_layer batch sizes greater than 1 even though batched Qwen execution changes SAE telemetry.",
    )
    args = parser.parse_args()
    if args.optimized_batch_size <= 0:
        raise SystemExit("--optimized-batch-size must be positive")
    if args.capture_execution_mode == "early_stop_layer_prefix_cache" and not args.allow_experimental_prefix_cache:
        raise SystemExit(
            "early_stop_layer_prefix_cache is experimental and must not be used for production reranking "
            "without paired equivalence validation; pass --allow-experimental-prefix-cache for diagnostic timing runs"
        )

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
        "capture_execution_mode": args.capture_execution_mode,
        "allow_experimental_prefix_cache": args.allow_experimental_prefix_cache,
        "optimized_batch_size": args.optimized_batch_size,
        "allow_nonexact_batched_prefill": args.allow_nonexact_batched_prefill,
    }
    completed = run_remote_capture(args.host, args.remote_python, payload)
    stderr, timing_summary = split_timing_summary(completed.stderr)
    if stderr:
        sys.stderr.write(stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"ssh qwen SAE capture failed: {completed.returncode}")
    output_path.write_text(completed.stdout, encoding="utf-8")
    if args.timing_summary_out:
        Path(args.timing_summary_out).write_text(json.dumps(timing_summary or {}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    if host in {"local", "localhost", "127.0.0.1", ""}:
        try:
            env = dict(**os.environ, ACTIVATION_RAG_QWEN_SAE_PAYLOAD=str(local_payload))
            return subprocess.run([remote_python, "-c", REMOTE_CAPTURE_CODE], text=True, capture_output=True, check=False, env=env)
        finally:
            local_payload.unlink(missing_ok=True)
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


def split_timing_summary(stderr: str) -> tuple[str, dict[str, Any] | None]:
    prefix = "ACTIVATION_RAG_QWEN_SAE_TIMING "
    remaining: list[str] = []
    summary: dict[str, Any] | None = None
    for line in stderr.splitlines():
        if line.startswith(prefix):
            try:
                parsed = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                remaining.append(line)
                continue
            if isinstance(parsed, dict):
                summary = parsed
            continue
        remaining.append(line)
    text = "\n".join(remaining)
    if text:
        text += "\n"
    return text, summary


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
from em_v2.run_sae_feature_causal_gates import JointMemoryQwenSaeRunner
from em_v2.rmt_span_selector import read_json

class _ActivationRagLayerCaptureComplete(Exception):
    pass

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

def iter_spans(episodes):
    for episode in episodes:
        for span in episode.get("candidate_spans") or []:
            if isinstance(span, dict) and span.get("span_id"):
                yield episode, span

def feature_ids(manifest):
    out = []
    for raw in manifest.get("sae_feature_ids") or []:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not out:
        raise ValueError("feature manifest has no integer SAE feature ids")
    return sorted(set(out))

def install_abort_hook(runner):
    if getattr(runner, "_activation_rag_abort_hook_installed", False):
        return
    def abort_after_existing_capture(_module, _inputs):
        if getattr(runner, "_activation_rag_stop_after_capture", False):
            raise _ActivationRagLayerCaptureComplete()
        return None
    runner.qwen.model.layers[runner.layer_index].register_forward_pre_hook(abort_after_existing_capture)
    runner._activation_rag_abort_hook_installed = True

def forward_until_capture(runner, *, input_ids, attention_mask=None, past_key_values=None, use_cache=False, squeeze_batch=True):
    torch = runner.torch
    runner.capture = None
    runner._activation_rag_stop_after_capture = True
    try:
        with torch.no_grad():
            runner.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
    except _ActivationRagLayerCaptureComplete:
        pass
    finally:
        runner._activation_rag_stop_after_capture = False
    if runner.capture is None:
        raise RuntimeError("layer capture hook did not fire before early stop")
    hidden = runner.capture.to(dtype=torch.float32)
    return hidden[0] if squeeze_batch else hidden

def summarize_hidden(runner, hidden, selected, selected_tensor, *, token_offset=0, min_activation=0.0):
    torch = runner.torch
    x = (hidden - runner.sae_mean) / runner.sae_rms
    with torch.no_grad():
        z, _values, _indices = runner.sae.encode(x)
    selected_z = z.index_select(dim=1, index=selected_tensor)
    max_values, token_indices = selected_z.max(dim=0)
    feature_values = {}
    token_offsets = {}
    for fid, value, token_index in zip(selected, max_values.detach().cpu().tolist(), token_indices.detach().cpu().tolist()):
        value = float(value)
        if value > min_activation:
            feature_values[str(fid)] = value
            token_offsets[str(fid)] = int(token_index) + int(token_offset)
    return feature_values, token_offsets

def summarize_hidden_batch(runner, hidden, attention_mask, selected, selected_tensor, *, min_activation=0.0):
    torch = runner.torch
    batch, seq, dim = hidden.shape
    x = (hidden.reshape(batch * seq, dim) - runner.sae_mean) / runner.sae_rms
    with torch.no_grad():
        z, _values, _indices = runner.sae.encode(x)
    selected_z = z.index_select(dim=1, index=selected_tensor).reshape(batch, seq, len(selected))
    valid = attention_mask.to(device=selected_z.device, dtype=torch.bool)
    selected_z = selected_z.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    max_values, token_indices = selected_z.max(dim=1)
    batch_values = []
    batch_offsets = []
    for row_values, row_indices in zip(max_values.detach().cpu().tolist(), token_indices.detach().cpu().tolist()):
        feature_values = {}
        token_offsets = {}
        for fid, value, token_index in zip(selected, row_values, row_indices):
            value = float(value)
            if value != float("-inf") and value > min_activation:
                feature_values[str(fid)] = value
                token_offsets[str(fid)] = int(token_index)
        batch_values.append(feature_values)
        batch_offsets.append(token_offsets)
    return batch_values, batch_offsets

def build_sae_row(*, episode, span, feature_values, token_offsets, evidence_mode, manifest, layer_index, qwen_seq_tokens, sequence_tokens):
    span_id = str(span.get("span_id") or "")
    return {
        "schema_version": "vicuna.rmt_span_selector.sae_activation_row.v1",
        "row_id": f"sae:{episode.get('episode_id')}:{span_id}",
        "split": episode.get("split"),
        "episode_id": episode.get("episode_id"),
        "request_id": episode.get("request_id"),
        "span_id": span_id,
        "candidate_text_hash": span.get("text_hash"),
        "evidence_mode": evidence_mode,
        "source_artifact": manifest.get("artifact_id"),
        "layer_index": layer_index,
        "qwen_seq_tokens": qwen_seq_tokens,
        "sequence_tokens": int(sequence_tokens),
        "sae_feature_values": feature_values,
        "sae_delta_vs_neutral": dict(feature_values),
        "sae_delta_vs_current": dict(feature_values),
        "max_token_offset_by_feature": token_offsets,
    }

def merge_feature_max(prefix_values, prefix_offsets, suffix_values, suffix_offsets):
    values = dict(prefix_values)
    offsets = dict(prefix_offsets)
    for key, value in suffix_values.items():
        if key not in values or float(value) > float(values[key]):
            values[key] = float(value)
            offsets[key] = int(suffix_offsets.get(key, 0))
    return values, offsets

def behavior_prompt_prefix(text):
    marker = "Candidate evidence:\n"
    idx = text.find(marker)
    if idx < 0:
        return None
    return text[: idx + len(marker)]

def prefix_token_length_from_offsets(encoded, prefix_char_len):
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    values = offsets[0].detach().cpu().tolist() if hasattr(offsets[0], "detach") else offsets[0]
    prefix_len = 0
    for index, pair in enumerate(values):
        start, end = int(pair[0]), int(pair[1])
        if start == 0 and end == 0 and index == 0:
            prefix_len = index + 1
            continue
        if end <= prefix_char_len:
            prefix_len = index + 1
            continue
        if start < prefix_char_len < end:
            return None
        break
    return prefix_len

def encode_span_sae_features_optimized(
    *,
    episodes,
    manifest,
    model_path,
    joint_checkpoint_path,
    sae_checkpoint_path,
    layer_index=7,
    qwen_seq_tokens=128,
    device="auto",
    evidence_mode="qwen_no_memory_span_preview",
    min_activation=0.0,
    max_rows=None,
    capture_execution_mode="early_stop_layer",
    optimized_batch_size=8,
    allow_nonexact_batched_prefill=False,
):
    import copy
    import time
    import torch
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        DynamicCache = None

    if evidence_mode != "qwen_no_memory_span_preview":
        raise ValueError("optimized zero-token reranking capture only supports qwen_no_memory_span_preview")
    selected = feature_ids(manifest)
    runner = JointMemoryQwenSaeRunner(
        model_path=model_path,
        joint_checkpoint_path=joint_checkpoint_path,
        sae_checkpoint_path=sae_checkpoint_path,
        layer_index=layer_index,
        qwen_seq_tokens=qwen_seq_tokens,
        device_name=device,
    )
    install_abort_hook(runner)
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=runner.device)
    runner.set_memory(None)
    runner.intervention = None
    rows = []
    started = time.time()
    prefix_cache = {}
    prefix_cache_hits = 0
    prefix_cache_misses = 0
    prefix_cache_fallbacks = 0
    requested_optimized_batch_size = max(1, int(optimized_batch_size))
    effective_optimized_batch_size = requested_optimized_batch_size if allow_nonexact_batched_prefill else 1
    if getattr(runner.tokenizer, "pad_token_id", None) is None and getattr(runner.tokenizer, "eos_token", None) is not None:
        runner.tokenizer.pad_token = runner.tokenizer.eos_token

    span_items = list(iter_spans(episodes))
    if max_rows is not None:
        span_items = span_items[: max(0, int(max_rows))]

    if capture_execution_mode == "early_stop_layer":
        prepared = []
        for item_index, (episode, span) in enumerate(span_items):
            text = str(span.get("text_preview") or "").strip()
            span_id = str(span.get("span_id") or "")
            if not text or not span_id:
                continue
            encoded = runner.encode_text(text)
            input_ids = encoded["input_ids"]
            prepared.append(
                {
                    "item_index": item_index,
                    "episode": episode,
                    "span": span,
                    "input_ids": input_ids,
                    "sequence_tokens": int(input_ids.shape[1]),
                }
            )
        by_length = {}
        for item in prepared:
            by_length.setdefault(int(item["sequence_tokens"]), []).append(item)
        rows_by_index = {}
        for sequence_tokens, same_length_items in by_length.items():
            for batch_start in range(0, len(same_length_items), effective_optimized_batch_size):
                batch_items = same_length_items[batch_start : batch_start + effective_optimized_batch_size]
                if not batch_items:
                    continue
                input_ids = torch.cat([item["input_ids"] for item in batch_items], dim=0).to(runner.device)
                attention_mask = torch.ones_like(input_ids, device=runner.device)
                hidden = forward_until_capture(
                    runner,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    squeeze_batch=False,
                )
                batch_values, batch_offsets = summarize_hidden_batch(
                    runner,
                    hidden,
                    attention_mask,
                    selected,
                    selected_tensor,
                    min_activation=min_activation,
                )
                for item, feature_values, token_offsets in zip(batch_items, batch_values, batch_offsets):
                    rows_by_index[int(item["item_index"])] = build_sae_row(
                        episode=item["episode"],
                        span=item["span"],
                        feature_values=feature_values,
                        token_offsets=token_offsets,
                        evidence_mode=evidence_mode,
                        manifest=manifest,
                        layer_index=layer_index,
                        qwen_seq_tokens=qwen_seq_tokens,
                        sequence_tokens=sequence_tokens,
                    )
                if batch_start == 0:
                    torch.cuda.empty_cache() if runner.device.type == "cuda" else None
        for item in prepared:
            row = rows_by_index.get(int(item["item_index"]))
            if row is not None:
                rows.append(row)
    else:
        for index, (episode, span) in enumerate(span_items):
            text = str(span.get("text_preview") or "").strip()
            if not text:
                continue
            encoded = runner.encode_text(text)
            full_input_ids = encoded["input_ids"].to(runner.device)
            feature_values = None
            token_offsets = None

            if capture_execution_mode == "early_stop_layer_prefix_cache" and DynamicCache is not None:
                prefix = behavior_prompt_prefix(text)
                if prefix:
                    prefix_len_from_offsets = prefix_token_length_from_offsets(encoded, len(prefix))
                    if prefix_len_from_offsets is not None and 0 < prefix_len_from_offsets < full_input_ids.shape[1]:
                        prefix_input_ids = full_input_ids[:, :prefix_len_from_offsets]
                        key = tuple(int(item) for item in prefix_input_ids[0].detach().cpu().tolist())
                        cached = prefix_cache.get(key)
                        if cached is None:
                            cache = DynamicCache()
                            prefix_hidden = forward_until_capture(
                                runner,
                                input_ids=prefix_input_ids,
                                attention_mask=torch.ones_like(prefix_input_ids, device=runner.device),
                                past_key_values=cache,
                                use_cache=True,
                            )
                            prefix_values, prefix_offsets = summarize_hidden(
                                runner,
                                prefix_hidden,
                                selected,
                                selected_tensor,
                                min_activation=min_activation,
                            )
                            cached = {
                                "cache": cache,
                                "prefix_len": int(prefix_input_ids.shape[1]),
                                "values": prefix_values,
                                "offsets": prefix_offsets,
                            }
                            prefix_cache[key] = cached
                            prefix_cache_misses += 1
                        else:
                            prefix_cache_hits += 1
                        prefix_len = int(cached["prefix_len"])
                        suffix_input_ids = full_input_ids[:, prefix_len:]
                        if suffix_input_ids.shape[1] > 0:
                            candidate_cache = copy.deepcopy(cached["cache"])
                            attention_mask = torch.ones((1, prefix_len + int(suffix_input_ids.shape[1])), dtype=torch.long, device=runner.device)
                            suffix_hidden = forward_until_capture(
                                runner,
                                input_ids=suffix_input_ids,
                                attention_mask=attention_mask,
                                past_key_values=candidate_cache,
                                use_cache=True,
                            )
                            suffix_values, suffix_offsets = summarize_hidden(
                                runner,
                                suffix_hidden,
                                selected,
                                selected_tensor,
                                token_offset=prefix_len,
                                min_activation=min_activation,
                            )
                            feature_values, token_offsets = merge_feature_max(
                                cached["values"],
                                cached["offsets"],
                                suffix_values,
                                suffix_offsets,
                            )

            if feature_values is None or token_offsets is None:
                if capture_execution_mode == "early_stop_layer_prefix_cache":
                    prefix_cache_fallbacks += 1
                hidden = forward_until_capture(
                    runner,
                    input_ids=full_input_ids,
                    attention_mask=torch.ones_like(full_input_ids, device=runner.device),
                    use_cache=False,
                )
                feature_values, token_offsets = summarize_hidden(
                    runner,
                    hidden,
                    selected,
                    selected_tensor,
                    min_activation=min_activation,
                )

            rows.append(
                build_sae_row(
                    episode=episode,
                    span=span,
                    feature_values=feature_values,
                    token_offsets=token_offsets,
                    evidence_mode=evidence_mode,
                    manifest=manifest,
                    layer_index=layer_index,
                    qwen_seq_tokens=qwen_seq_tokens,
                    sequence_tokens=int(full_input_ids.shape[1]),
                )
            )
            if index == 0 or index % 250 == 0:
                torch.cuda.empty_cache() if runner.device.type == "cuda" else None

    active_counts = [len(row["sae_feature_values"]) for row in rows]
    summary = {
        "schema_version": "vicuna.rmt_span_selector.sae_activation_summary.v1",
        "status": "completed",
        "row_count": len(rows),
        "matched_rows": sum(1 for row in rows if row["sae_feature_values"]),
        "unmatched_rows": sum(1 for row in rows if not row["sae_feature_values"]),
        "feature_set_id": manifest.get("feature_set_id"),
        "feature_count": len(selected),
        "evidence_mode": evidence_mode,
        "promotion_note": "optimized_strict_zero_token_reranking_prefill_only",
        "capture_execution_mode": capture_execution_mode,
        "requested_optimized_batch_size": requested_optimized_batch_size if capture_execution_mode == "early_stop_layer" else 1,
        "optimized_batch_size": effective_optimized_batch_size if capture_execution_mode == "early_stop_layer" else 1,
        "allow_nonexact_batched_prefill": bool(allow_nonexact_batched_prefill),
        "batching_exactness_policy": "force_effective_batch_1" if capture_execution_mode == "early_stop_layer" and not allow_nonexact_batched_prefill else "nonexact_batching_allowed",
        "prefix_cache_hits": prefix_cache_hits,
        "prefix_cache_misses": prefix_cache_misses,
        "prefix_cache_fallbacks": prefix_cache_fallbacks,
        "prefix_cache_entries": len(prefix_cache),
        "active_features_per_row_mean": (sum(active_counts) / len(active_counts)) if active_counts else 0.0,
        "duration_s": time.time() - started,
    }
    return rows, summary

manifest = read_json(Path(payload["feature_manifest"]))
capture_execution_mode = str(payload.get("capture_execution_mode") or "full_forward")
encoder = encode_span_sae_features if capture_execution_mode == "full_forward" else encode_span_sae_features_optimized
rows, summary = encoder(
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
    **(
        {
            "capture_execution_mode": capture_execution_mode,
            "optimized_batch_size": int(payload.get("optimized_batch_size") or 8),
            "allow_nonexact_batched_prefill": bool(payload.get("allow_nonexact_batched_prefill")),
        }
        if capture_execution_mode != "full_forward"
        else {}
    ),
)
requests_by_chunk = {str(request["chunk_id"]): request for request in payload["requests"]}
feature_manifest_summary = {"feature_set_id": manifest.get("feature_set_id"), "feature_count": len(manifest.get("sae_feature_ids") or [])}
for sae_row in rows:
    chunk_id = str(sae_row.get("span_id") or "")
    request = requests_by_chunk.get(chunk_id)
    if request is None:
        continue
    print(json.dumps(adapt(request, sae_row, feature_manifest_summary, summary), sort_keys=True), flush=True)
print("ACTIVATION_RAG_QWEN_SAE_TIMING " + json.dumps(summary, sort_keys=True), file=sys.stderr, flush=True)
'''


if __name__ == "__main__":
    main()

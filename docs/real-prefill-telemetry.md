# Real Prefill Telemetry Provider

`CommandPrefillTelemetryProvider` is the non-mock integration point for activation RAG experiments.
It writes chunk-aligned prefill capture requests to JSONL, invokes an external capture command, and reads selector-compatible rows back into `ActivationRecord` values.

The same provider is used for document ingestion and query-time activation capture.

## Command Contract

Pass a command with these placeholders:

```bash
--telemetry-command 'python3 /path/to/capture_prefill.py {input_jsonl} {output_jsonl} --manifest {manifest_json}'
```

The provider writes:

- `{input_jsonl}`: one `activation_rag.prefill_capture_request.v1` row per chunk.
- `{manifest_json}`: capture settings, provider/model/site IDs, prompt template hash, layer policy, and normalization policy.

The capture command must write `{output_jsonl}` with one row per input `chunk_id`.

Required behavior:

- capture phase is prefill only
- generation is disabled
- output contains selector-compatible baseline/current/delta fields
- output preserves `chunk_id`

Accepted selector field aliases include:

- `current_em_state`, `current_em`, or `absolute_em`
- `neutral_baseline_state`, `neutral_baseline_em`, or `baseline_em`
- `delta_vs_neutral`, `em_delta_vs_neutral`, `span_delta_vs_neutral`, or `delta`
- `delta_vs_current`, `em_delta_vs_current`, or `span_delta_vs_current`
- `sae_feature_values`, `sae_delta_vs_neutral`, `sae_delta_vs_current`, and `sae_feature_mask`

## Benchmark Guard

Non-fixture benchmark runs reject mock telemetry by default. Use `--telemetry-command` for real runs.

`--allow-mock-telemetry` exists only for harness smoke tests and should not be used for method comparisons.

Example:

```bash
PYTHONPATH=src python3 scripts/run_benchmark.py \
  --beir-dir data/benchmarks/beir/scifact/scifact \
  --beir-name beir-scifact \
  --split test \
  --top-k 10 \
  --candidate-k 100 \
  --telemetry-command 'python3 /path/to/sidecar_prefill_capture.py {input_jsonl} {output_jsonl} --manifest {manifest_json}' \
  --telemetry-model-id gemma4-e4b \
  --telemetry-site-id l16_resid_pre \
  --layer-selection-policy semantic_middle_late_resid_pre_l16 \
  --normalization-policy sidecar_centered_unit_norm_v1 \
  --out runs/benchmarks/beir-scifact-real-telemetry.json
```

The sidecar runtime currently writes capture artifacts/traces rather than exposing raw activations through a server API, so the command should wrap that existing capture/export path.

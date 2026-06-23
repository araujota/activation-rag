# behavior-latent-general-v0.1

This directory is the lightweight release bundle for the primary
query+candidate behavior-latent MLP reranker.

Files:

- `model.pt`: PyTorch checkpoint for the 248-input MLP.
- `training-metrics.json`: training run summary copied from the robust run.
- `expanded-text-reranker-comparison.json`: dense, behavior, Ettin, BGE, GTE,
  Qwen3 seq-cls, and Mixedbread comparison summary.
- `release-manifest.json`: checksums, model metadata, telemetry metadata, and
  primary result summary.

This bundle is suitable for a Hugging Face model repo after copying
`../../MODEL_CARD.md` to `README.md` in the model repo root.

The primary controlled release claim is LegalBench-RAG and R2MED reranking.
The same checkpoint also has a zero-shot CoIR coding-transfer result documented
in `../../MODEL_CARD.md` and
`../../docs/behavior-latent-reranker-experiment-report-20260618.md`.

Important: `model.pt` is not a text model. It requires transformed
query/candidate behavior telemetry as input. See `../../MODEL_CARD.md` and
`../../README.md` for the full usage path.

Telemetry capture for this artifact should use the truncated reranking path:

```bash
python scripts/capture_qwen_sae_prefill.py REQUESTS.jsonl ROWS.jsonl \
  --capture-execution-mode early_stop_layer \
  --optimized-batch-size 8
```

The layer-7 early-stop path preserves the Core245 signal while avoiding the
rest of Qwen prefill. The requested optimized batch size is `8`, but exact
production capture currently forces effective batch size `1` unless
`--allow-nonexact-batched-prefill` is explicitly passed; do not enable non-exact
batching for reported results without a paired equivalence audit.

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

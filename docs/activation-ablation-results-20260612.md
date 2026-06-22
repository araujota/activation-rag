# Activation Retrieval Ablation Results

Date: 2026-06-12

Inputs:

- Benchmark: BEIR SciFact test split.
- Telemetry cache: `runs/telemetry-cache/scifact-fixed-zero-20260611-140007`.
- Dense embedding model: `BAAI/bge-base-en-v1.5`, captured on `vicuna-host` GPU.
- Activation report: `runs/activation-ablation/scifact-dense-integration-20260612.json`.
- Raw manifest audit: `runs/activation-ablation/raw_manifest_audit_scifact_20260612.json`.

## Capture Hygiene

The cached `ActivationRecord` rows preserve prompt template, normalization policy, and zero-token provenance, but they do not persist prompt-section labels or section token spans.

The raw sidecar manifest audit found:

- 5,525 raw manifest files.
- 165,654 manifest rows.
- `prompt_section_label=prompt` for 155,808 rows.
- `prompt_section_label=assistant_response` for 9,846 rows.
- `window=prefill_last` for 82,188 rows.
- `window=post50_mean` for 82,188 rows.
- `window=decode_first` for 639 rows.
- `window=decode_last` for 639 rows.

Conclusion: corrected recapture is mandatory before making further benchmark claims. The raw capture artifacts still include assistant/decode-labeled rows, and the normalized cache does not retain enough section provenance to prove strict section alignment.

## Activation-Only Results

162 activation variants executed across pooling, site/feature selection, and geometry transforms.

Best activation-only variant:

- `site_emv2_p65_attn_out::rank`
- nDCG@10: 0.007540
- Recall@10: 0.010000
- MRR@10: 0.006667
- unique top-1 chunks: 185/300
- top hub count: 11/300

Baseline raw activation:

- `all_features::raw`
- nDCG@10: 0.006187
- Recall@10: 0.010000
- MRR@10: 0.005000
- unique top-1 chunks: 102/300
- top hub count: 26/300

Geometry observations:

- CSLS improved hubness for all features: unique top-1 rose from 102 to 156 and top hub share fell from 8.67% to 4.00%.
- CSLS did not materially improve retrieval: nDCG@10 rose only from 0.006187 to 0.006436.
- Mean-centering, z-score, all-but-the-top, and whitening did not rescue retrieval quality.
- Some transforms made hubness worse, especially all-feature z-score and whitening.

Pooling/site observations:

- Per-site rank features reduced hubness and produced the best activation-only score, but the absolute score remained near zero.
- `prefill_last` and `post50_mean` splits did not reveal a robust retrieval signal.
- Chunk-bin and scalar-moment separation did not identify a strong semantic retrieval representation.
- Rogue high-variance dimensions alone had zero retrieval value.

## Dense Integration

Dense-only BGE baseline in the ablation run:

- nDCG@10: 0.745126
- Recall@10: 0.865889
- MRR@10: 0.700329

Hard activation reranking remained destructive:

- Best hard rerank among inspected variants was still far below dense-only.
- `post50_mean_only::rank` hard rerank reached nDCG@10 0.067850.
- `all_features::raw` hard rerank matched the earlier failure: nDCG@10 0.062252.

Small dense-plus-activation blending showed a tiny diagnostic gain:

- Best blend: `all_features::raw` at lambda 0.03.
- nDCG@10: 0.748343.
- Recall@10: 0.869222.
- MRR@10: 0.703687.

This gain is not enough to promote the current activation representation. It is a reason to preserve score-blending as a future experiment after corrected recapture, not a reason to keep hard activation reranking.

## Interpretation

The current activation vectors are not simply noisy; they are weak as standalone semantic retrieval vectors. Hubness correction can make the nearest-neighbor distribution healthier, but healthier geometry did not translate into meaningful relevance ranking.

The most plausible remaining path is:

1. Fix capture hygiene first: strict section labels, section token spans, and rejection of assistant/decode windows for document/query prefill rows.
2. Stop over-pooling: preserve token/layer/site/window variants separately rather than only summary vectors.
3. Retest unsupervised geometry corrections after corrected recapture.
4. Treat activation as a conservative dense-score feature before treating it as a replacement retriever.
5. Move to supervised projection only with train/dev qrels and hard negatives.

No architecture should be selected from this pass alone because the capture audit invalidates the current cache as a clean representation of section-aligned document/query prefill activations.

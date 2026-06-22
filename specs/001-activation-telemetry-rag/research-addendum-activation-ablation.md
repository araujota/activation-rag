# Research Addendum: Activation Retrieval Ablation Pass

Date: 2026-06-12

## Trigger

The first real-prefill SciFact diagnostic run showed dense retrieval working normally while raw activation cosine and dense-first activation reranking failed badly:

- Dense BGE: nDCG@10 0.7451, Recall@10 0.8659, MRR@10 0.7003.
- Activation-only: nDCG@10 0.0062, Recall@10 0.0100, MRR@10 0.0050.
- Dense plus activation rerank: nDCG@10 0.0623, Recall@10 0.1274, MRR@10 0.0456.

Diagnostics showed saturated cosine geometry rather than ordinary retrieval noise: random document/document cosine was about 0.95, query/document top scores were about 0.97-0.99, and only 102 unique top-1 documents appeared for 300 queries.

## Research Hypothesis

The earlier selector primitive is not automatically a semantic retrieval embedding. It may still contain useful signal, but the current ingestion process over-pooled it, mixed section/template effects into the vector, and compared it with raw cosine in an anisotropic space.

The next pass is a research pass, not a product hardening pass. Its purpose is to identify whether any activation representation has retrieval value after correcting or ablating:

1. capture hygiene and canonicalization;
2. pooling granularity;
3. geometry and hubness correction;
4. layer/site/feature selection;
5. dense-retrieval integration.

## Ablation Matrix

### A1. Capture Hygiene

Checks:

- Prompt template IDs and hashes must be consistent across documents and queries.
- Records must preserve `generation_disabled=true` and prefill-only provenance.
- Raw sidecar manifests must expose prompt-section labels and token spans.
- A record captured for a document chunk must label the extracted section as document content, not assistant response.
- A record captured for a query must label the extracted section as query content, not assistant response.

Current-cache status:

- Template and zero-token provenance can be audited from cached rows.
- Section labels cannot be fully validated from cached rows because section boundaries were not persisted into `ActivationRecord`.
- Raw sidecar manifests must be audited while the remote raw activation directories still exist.

Required fix if audit fails:

- Add strict section labels to capture requests: `document_chunk`, `query`, and future `answer_span`.
- Store section ID, section label, token start/end, and prompt text hash in every activation row.
- Fail capture when the sidecar reports a section label outside the requested extraction section.

### A2. Pooling Granularity

Runnable post-processing variants from the current summary vectors:

- all summary features;
- `prefill_last` only;
- `post50_mean` only;
- per-site vectors for each selected EMV2 attention-out site;
- early/mid/late proxy site groups;
- chunk-bin features only;
- scalar moment features only.

Requires recapture:

- mean over all content tokens;
- last content token;
- final-N content token pooling;
- answer-span pooling;
- section-aligned multi-chunk prompt packing with token-level extraction.

### A3. Geometry and Hubness Correction

Runnable post-processing variants:

- raw cosine;
- document-mean-centered cosine;
- document z-score standardization;
- all-but-the-top removal with 1, 3, and 5 top principal components removed;
- whitening with dimensionality reduction;
- rank/Spearman-style feature ranking;
- CSLS-style local scaling over query/document similarity.

Required diagnostics for every promoted variant:

- unique top-1 chunk count;
- top hub share;
- nearest-neighbor in-degree distribution;
- query/document top-1 and top-10 score distributions;
- random document/document cosine distribution.

### A4. Layer, Site, and Feature Selection

Runnable post-processing variants:

- individual selected sites;
- early proxy: p15/p25/p35 sites;
- middle proxy: p45/p55/p65 sites;
- late proxy: p75/p85/p92 sites;
- chunk-bin features vs summary-stat features;
- low-variance feature drop;
- high-variance rogue-dimension drop.

Requires later supervised pass:

- qrel-ranked feature selection;
- learned linear projection;
- contrastive projection with hard negatives;
- answer-seeker target prediction.

### A5. Retrieval Integration

Runnable if dense embeddings/candidates are available:

- activation-only KNN;
- dense-only baseline;
- dense top-k followed by activation-only rerank;
- dense plus activation blend with a small lambda sweep;
- activation tie-breaker/margin gate.

Promotion rule:

- Activation reranking must not be promoted if it lowers dense nDCG@10 or recall@10 on the development split.
- Activation scores should first be blended conservatively, not used as a hard replacement for dense candidate ordering.

## Research Sources

- Hubness and local/global scaling: Dominik Schnitzer et al., "Local and Global Scaling Reduce Hubs in Space", JMLR 2012. https://jmlr.org/papers/v13/schnitzer12a.html
- Hubness reduction empirical comparison: Feldbauer et al., "A comprehensive empirical comparison of hubness reduction in high-dimensional spaces", 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/
- CSLS precedent: Conneau et al., "Word Translation Without Parallel Data", 2018. https://arxiv.org/abs/1710.04087
- All-but-the-Top: Jiaqi Mu and Pramod Viswanath, ICLR 2018. https://openreview.net/forum?id=HkuGJ3kCb
- Whitening sentence representations: Su et al., "Whitening Sentence Representations for Better Semantics and Faster Retrieval".
- Rogue dimensions: Timkey and van Schijndel, "All Bark and No Bite", EMNLP 2021.
- Transformer anisotropy: Godey et al., "Anisotropy Is Inherent to Self-Attention in Transformers", EACL 2024.
- Implementation precedent: CSLS-style hubness correction appears in open-source nearest-neighbor libraries such as `dobraczka/kiez`.

## Corrected Section-Prefill Run

Report: `runs/activation-ablation/scifact-section-prefill-dense-integration-20260612.json`

Input cache: `runs/telemetry-cache/scifact-section-prefill-20260612-section-prefill`

Status:

- Corrected recapture is present and usable: 5,224 valid document-chunk rows and 300 query rows.
- Prompt template is canonicalized as `rag_raw_chunk_prefill_v1_strict_zero_section_v2`.
- Normalization policy is `raw_summary_values_v2_prompt_prefill_filtered`.
- Cached semantic section labels are present: 5,224 `document_chunk` rows and 300 `query` rows.
- One SciFact chunk remains invalid from capture and is excluded from ablations.

Activation-only findings:

- Best activation-only variant: `site_emv2_p65_attn_out::rank`.
- Best activation-only metrics: MRR@10 0.0067, nDCG@10 0.0075, Recall@10 0.0100.
- Best activation-only hubness: 185 unique top-1 chunks for 300 queries, top hub share 0.0367.
- Raw all-feature cosine remains weak: nDCG@10 0.0062, Recall@10 0.0100.
- CSLS raw improves hubness over raw cosine but not task quality: nDCG@10 0.0064, 157 unique top-1 chunks, top hub share 0.0400.
- NICDM raw behaves similarly: nDCG@10 0.0058, 145 unique top-1 chunks, top hub share 0.0400.

Dense integration findings:

- Dense BGE baseline: MRR@10 0.7003, nDCG@10 0.7451, Recall@10 0.8659.
- Hard activation reranking still damages dense retrieval badly; the best corrected-cache activation rerank is `post50_mean_only::rank` at nDCG@10 0.0663.
- Small score blending is the only positive signal observed so far. The best blend is dense plus `all_features::raw`/equivalent scalar-moment raw activation at lambda 0.03: MRR@10 0.7037, nDCG@10 0.7484, Recall@10 0.8692.

Interpretation:

- Corrected capture hygiene fixed section labeling and prompt canonicalization, but it did not turn the selector-derived vector into a standalone semantic retrieval embedding.
- Hubness correction changes neighbor geometry and improves top-1 diversity, but activation-only relevance remains near zero on SciFact.
- The next causal claim should be narrow: current activation telemetry may contain a weak auxiliary score that can help dense retrieval when blended conservatively. It does not yet justify activation-only retrieval or hard activation reranking.
- Supervised projection/reranker work should start from dense candidate pools and hard negatives, not from full-index activation-only replacement.

## Research Output

The ablation pass MUST write an inspectable JSON report under `runs/activation-ablation/` containing:

- input cache path and benchmark dataset identity;
- all runnable variants and their metric summaries;
- all non-runnable variants with the reason they require recapture or supervised data;
- hubness diagnostics for each executed activation variant;
- dense integration results when dense embeddings are supplied;
- a short interpretation field limited to observations, not final architectural conclusions.

This addendum intentionally does not choose the next architecture. The next architecture should be synthesized only after the ablation report exists.

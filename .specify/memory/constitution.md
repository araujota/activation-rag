# Activation RAG Constitution

## Principles

1. **Stable Join Keys**
   Every representation of document content must share stable `document_id`, `chunk_id`, and span offsets. Dense embeddings, activation telemetry, rerank scores, and benchmark labels must be joinable without fuzzy matching.

2. **Provider Boundaries**
   Runtime-specific activation capture must live behind `TelemetryProvider`. Core ingestion and retrieval code must not depend on llama.cpp, TransformerLens, SAE Lens, or any specific pod runtime.

3. **Selector-Compatible Telemetry**
   Activation records must preserve the earlier selector primitive semantics: current state, neutral-baseline deltas, current-state deltas, saturation/headroom terms, signed mass, feature masks, provenance, and validity flags.

4. **Benchmark Isolation**
   Dense retrieval, dense-plus-rerank, activation-similarity, and answer-seeker experiments must be runnable as named strategies over the same frozen chunk store and query set.

5. **Inspectable Artifacts**
   All model/config choices, chunking settings, telemetry provider metadata, and index build manifests must be persisted as explicit artifacts.

## Constraints

- The first retrieval pass is single-hop only.
- Activation capture is prefill-only; generation is out of scope for document ingestion.
- The first implementation must run locally with a mock telemetry provider and dependency-light tests.
- Heavy optional providers may be added later without changing the core contracts.


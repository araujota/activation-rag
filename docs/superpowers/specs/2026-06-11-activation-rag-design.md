# Activation RAG Design

## Summary

Build a new Python repository that ingests documents once, assigns stable chunk identifiers, and writes two parallel representations for each chunk:

- dense text embeddings for conventional single-pass RAG
- prefill-only activation telemetry records that preserve selector-style baseline/current/delta features

The system must allow retrieval strategies to be compared over identical chunks: dense vector search, dense retrieval plus reranking, activation-space similarity, and later answer-seeker prediction.

## Architecture

The core package has four boundaries:

- `ingest`: load text documents and split them into stable chunk records.
- `retrieval`: embed chunks, search a vector index, and optionally rerank candidates.
- `telemetry`: define `TelemetryProvider` and selector-compatible activation records.
- `store`: persist chunks, dense vectors, activation records, and manifests.

The activation provider is intentionally abstract. The first implementation includes a deterministic mock provider for tests and offline pipeline development. A llama.cpp SAE/CAA provider can later implement the same interface using prefill-only runtime traces.

## Selector Primitive Translation

Each chunk is treated as a candidate span. The telemetry row keeps the same primitive categories used by the RMT span selector:

- `current_em_state`
- `neutral_baseline_state`
- `delta_vs_neutral`
- `delta_vs_current`
- `saturation`
- `residual_headroom`
- `sae_feature_values`
- `sae_delta_vs_neutral`
- `sae_delta_vs_current`
- `sae_feature_mask`
- `positive_mass`, `negative_mass`, `total_mass`, `signed_balance`
- provenance and validity fields

For document ingestion, memory-pressure fields remain optional extension fields because there may be no existing memory state.

## Data Flow

1. Documents are normalized into `DocumentRecord` values.
2. The chunker emits `ChunkRecord` values with stable IDs, offsets, token estimates, and text hashes.
3. The embedding model writes `EmbeddingRecord` values keyed by `chunk_id`.
4. The telemetry provider runs a prefill/activation pass over the same chunk text and writes `ActivationRecord` values keyed by `chunk_id`.
5. Retrieval strategies consume stored records and produce `RetrievalResult` values.

## Testing

Initial tests verify:

- chunk IDs are stable for identical document content and settings
- chunk records preserve offsets and hashes
- mock telemetry records include selector-compatible baseline/current/delta fields
- activation similarity can rank chunks without dense embeddings

## Approval

Approved direction: clean `TelemetryProvider` interface, preserving the earlier selector primitive semantics where they translate to document chunks.


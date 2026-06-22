# Research: Activation Telemetry RAG

## RAG Baseline

Current mainstream RAG systems separate document parsing/chunking, embeddings, vector stores, retrievers, and post-retrieval rankers. LlamaIndex `SentenceSplitter` uses paragraph and sentence boundaries before falling back to smaller splits and overlap. LangChain `RecursiveCharacterTextSplitter` similarly tries increasingly fine separators. Haystack's `SentenceTransformersDocumentEmbedder` demonstrates an explicit document-in/document-out embedding component with configurable normalization, batching, precision, and backend.

Implication: the baseline engine should keep chunking, embedding, indexing, and reranking as separate components and persist settings in manifests.

## Reranking

Sentence Transformers documents the standard retrieve-and-rerank pattern: a fast bi-encoder retrieves candidates, then a CrossEncoder scores query/document pairs more accurately. LlamaIndex's `SentenceTransformerRerank` follows this pattern by preserving candidate nodes, scoring query/content pairs, and sorting by cross-encoder score.

Implication: reranking should be optional and should consume the same `RetrievalResult` candidates rather than being mixed into vector search.

## Activation Capture

TransformerLens centers activation work around named hook points and an `ActivationCache` produced by `run_with_cache`. SAE Lens `ActivationsStore` captures hook activations from token batches, supports cached activation datasets, and includes multi-hook capture from a single forward pass.
NNsight exposes activation access during forward traces, and Hugging Face Transformers can return hidden states from forward calls when configured to do so.
The local sidecar runtime currently captures activation files/traces rather than serving raw activations from the server API.

Implication: `TelemetryProvider` should expose provider metadata, hook/site identity, token alignment, and aggregation policy. It should not require one specific activation library.
For the sidecar path, the first real provider should therefore be command/file backed: write chunk-aligned prefill capture requests, let the model runtime capture/export real rows, then normalize selector-compatible rows into `ActivationRecord`.

## Existing Local Selector Primitive

Prior local selector specs define rows with:

- `em_state`
- `em_delta_vs_neutral`
- `em_delta_vs_current`
- `em_saturation`
- `em_residual_headroom`
- `sae_feature_values`
- `sae_delta_vs_neutral`
- `sae_delta_vs_current`
- `sae_feature_mask`
- `sae_novelty`
- `sae_overlap_with_memory`
- provenance and validity fields

Sign-aware EM magnitude work adds aggregate `positive_mass`, `negative_mass`, `total_mass`, and `signed_balance` so equal unsigned norms can differ by constructive direction.

Implication: document chunks should be represented as candidate spans and activation records should retain this selector primitive vocabulary even when memory-specific fields are absent.

## Failure Modes To Treat As First-Class

### Dimensionality and Layer Selection

Full hidden-state capture across many transformer layers is too large for retrieval latency and will create brittle feature geometry. The provider contract must therefore capture a small, named set of hook sites with recorded rationale. Early layers are expected to emphasize token, syntax, and entity structure; middle-to-late layers are expected to better represent semantic intent, task structure, and answer formatting. This is a hypothesis to test, not a default to hard-code.

### Contextual Drift

The same chunk will not produce identical activations when processed alone, inside a canonical ingestion template, or inside a long live chat history. Position, token age, role formatting, and preceding context can all move hidden states. The index must record prompt wrappers and apply normalization/centering policies before comparing activation vectors across contexts.

### Hubness

Hubness is a known high-dimensional nearest-neighbor pathology: a few points become nearest neighbors for many queries. Activation-space retrieval must measure nearest-neighbor in-degree/k-occurrence and treat high hub concentration as a failed diagnostic gate. Learned mappings must include hard negatives rather than only easy random distractors.

### Data Scarcity

The answer-seeker objective needs high-quality query-to-evidence activation pairs. Weakly labeled chunks are useful for diagnostics, but training requires query, answer span or answer chunk, distractor chunks, capture template, and split provenance.

### Supervision Target Definition

"Answer activation" is ambiguous unless the training target is declared. The tractable target options are:

- teacher-forced activations at known answer-span token positions
- pooled activations of the answer-containing chunk
- contrastive answer-chunk versus distractor-chunk labels

The first answer-seeker experiment should likely start with the contrastive target because it directly matches retrieval evaluation and avoids pretending a pooled chunk vector is a precise answer-span vector.

### Prompt Canonicalization

Prefill under raw text plus separators is not equivalent to prefill under `Q: ... A:` or chat-role formatting. Every activation record must record template identity, separators, BOS/EOS handling, and role format. Index records captured under incompatible templates should not be mixed without an explicit normalization experiment.

## Decision

Use a dependency-light Python core with optional adapters:

- core tests use deterministic mock embeddings and mock telemetry
- dense production usage can plug in Sentence Transformers
- reranking can plug in Sentence Transformers CrossEncoder
- activation production usage can plug in sidecar/llama.cpp SAE/CAA through `CommandPrefillTelemetryProvider`, or future direct TransformerLens/NNsight/Hugging Face providers
- non-fixture benchmark runs should reject mock telemetry unless explicitly marked as harness-only smoke tests

## Full Selector-Compatible CAA/EM Restoration Audit

The current scaled RAG runs are not full selector-primitive runs. The Qwen/RMT/SAE Core245 capture wrapper populates `sae_feature_values`, `sae_delta_vs_neutral`, `sae_delta_vs_current`, and feature masks, but leaves `current_em_state`, `neutral_baseline_state`, `prior_current_state`, `delta_vs_neutral`, `delta_vs_current`, `saturation`, and `residual_headroom` empty. Those results must be described as SAE-only activation-aware reranking.

Existing mechanisms that can restore the missing fields:

- The longmem selector materialization path has strict EM row logic that computes current state, neutral-baseline deltas, current-state deltas, saturation/headroom, and signed mass fields.
- The production 8-head Qwen bundle on `vicuna-host` includes a learned feature bundle and `neutral-em-baseline-v1.json` for `constraint_imposed`, `final_answer_readiness`, `hair`, `plan_formation`, `repair_readiness`, `runtime_failure_pressure`, `stall_looping`, and `state_carryover`.
- Compact sidecar traces can contain `final_em_v2_features` and `final_em_v2_inputs`. Raw activation directories can also be summarized and rescored through the learned bundle when strict zero-token capture writes stable `manifest.jsonl` and `activations.f16bin`.

Rebuild decision:

- Do not launch a full recapture solely because the mechanisms exist. The existing positive claims are already meaningful as SAE-only results: decisive on SciFact and narrow but scaled-positive on LegalBench-RAG.
- Do run a bounded pilot if we want to make a full CAA/SAE selector-primitive claim. The pilot should merge Core245 SAE features with 8-head EM/CAA fields and derived scalars, then compare against the existing SAE-only Core245 direct-blend model, dense-only, and dense plus Ettin on SciFact and LegalBench-RAG dev/test candidate groups.
- Promote the rebuild only if the added EM/CAA fields create positive paired query-level movement on development splits and preserve locked heldout performance. If the pilot is flat or noisy, publish the current results as SAE-only activation reranking and list full selector-compatible CAA capture as future work.

## Benchmark Conditions

MS MARCO passage ranking uses a passage collection of about 8.8M passages and is typically evaluated with `MRR@10` on passage ranking dev/eval splits. Because qrels are sparse, success is about placing the judged relevant passage high, not exhaustive recall.

BEIR is a heterogeneous zero-shot IR benchmark and commonly reports `nDCG@10`; `Recall@100` is also used to understand whether relevant documents enter a retrievable candidate pool.

HotpotQA is a multi-hop QA dataset with supporting facts. It is not equivalent to a single-hop retrieval benchmark, but we can evaluate retrieval against supporting paragraphs/facts by measuring whether the supporting evidence appears in top-k. Because this project is explicitly single-hop for now, HotpotQA results must be reported as supporting-evidence retrieval diagnostics, not full QA task performance.

Implication: the benchmark harness must keep metrics per benchmark rather than collapse them into one score. It should also record when a single-hop method is being evaluated on a multi-hop benchmark.

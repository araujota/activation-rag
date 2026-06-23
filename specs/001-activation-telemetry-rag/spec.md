# Feature Specification: Activation Telemetry RAG

## User Need

Experimenters need an off-the-shelf document ingestion and RAG engine that can compare ordinary dense-vector retrieval against activation-derived document representations. The first version must ingest documents once, retrieve chunks in a single pass, and store prefill-only SAE/CAA-style telemetry for the same chunk boundaries so later benchmarks can test activation similarity and answer-seeker retrieval.

## Scope

- Build a new standalone repository for the experiment.
- Ingest plain text documents and split them into stable chunk records.
- Generate dense embedding records for conventional vector retrieval.
- Provide a single-pass retriever with optional reranking.
- Define a clean `TelemetryProvider` interface for prefill-only activation capture.
- Store selector-compatible activation records keyed to the same chunks as dense embeddings.
- Include a deterministic mock telemetry provider for tests and local development.
- Provide a non-mock prefill telemetry adapter for the existing sidecar/llama.cpp SAE/CAA capture path before making benchmark claims.
- Prepare contracts for future answer-seeker providers.

## Requirements

- The system MUST assign stable `document_id` and `chunk_id` values from source identity, normalized content, chunk settings, and chunk offsets.
- Chunking MUST support sentence/paragraph-aware splitting with bounded overlap and preserve source offsets.
- Dense embedding records MUST be keyed by `chunk_id`.
- Activation records MUST be keyed by `chunk_id` and generated from the exact chunk text used for embeddings.
- `TelemetryProvider.capture_prefill()` MUST be prefill-only and MUST NOT require text generation.
- Real benchmark runs MUST use non-mock prefill telemetry captured from the same runtime/model path for both document chunks and user queries.
- The real prefill telemetry adapter MUST support selector-compatible sidecar rows and MUST map them into `ActivationRecord` without losing baseline/current/delta fields.
- Runs whose activation rows leave `current_em_state`, `neutral_baseline_state`, `delta_vs_neutral`, `delta_vs_current`, `saturation`, and `residual_headroom` empty MUST be reported as SAE-only activation telemetry, not as full CAA/SAE selector telemetry.
- A rebuilt full selector-compatible ingestion path MUST be gated by a pilot that proves non-empty current/baseline/delta CAA fields are captured or reconstructed for both queries and candidate chunks under strict zero-token prefill.
- The benchmark CLI MUST refuse mock telemetry for non-fixture benchmark runs unless the caller explicitly opts into a harness-only smoke test.
- `TelemetryProvider` implementations MUST return selector-compatible fields:
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
  - `positive_mass`
  - `negative_mass`
  - `total_mass`
  - `signed_balance`
  - `telemetry_valid`
  - `invalid_reason`
  - provenance fields for model, provider, hook/site, token span, aggregation, and capture run
- The first activation similarity strategy MUST work without dense embeddings by comparing activation feature vectors.
- Conventional retrieval MUST support dense vector search and an optional reranker interface.
- The engine MUST expose activation-space KNN over prefill activations of the user query.
- The engine MUST expose dense-first retrieval followed by activation-similarity reranking as an alternative second-stage reranker.
- The engine MUST expose a comparison surface that returns dense, activation-KNN, and dense-plus-activation-reranked results for the same query.
- The benchmark harness MUST evaluate dense, activation-KNN, and dense-plus-activation-rerank under the same corpus, query set, qrels, `top_k`, and candidate pool settings.
- The benchmark harness MUST report benchmark-standard metrics where applicable: MS MARCO passage `MRR@10`, BEIR `nDCG@10` and `Recall@100`, and HotpotQA supporting-evidence retrieval `Recall@k`/`nDCG@k`.
- Full benchmark runs MUST persist run manifests recording dataset name, split, qrels source, corpus size, query count, approach settings, and timing.
- Full benchmark download commands MUST be explicit and resumable under `data/benchmarks/`; benchmark data MUST NOT be checked into git.
- Retrieval results MUST identify strategy name, chunk ID, score, and component scores.
- Index and capture manifests MUST record chunker settings, embedder identity, telemetry provider identity, schema versions, and artifact hashes when available.
- Tests MUST run without downloading models or requiring GPU access.
- Telemetry capture MUST require an explicit layer/site selection policy and MUST NOT concatenate arbitrary full-depth hidden states into retrieval vectors.
- Layer selection policy MUST distinguish early syntactic/entity sites from middle-to-late semantic/task-formatting sites and record the rationale in the capture manifest.
- Activation representations MUST be normalized to reduce prompt-position, token-age, and isolated-prefill context drift before they are compared across ingestion and live query contexts.
- Capture templates MUST be canonicalized. Document chunks, query chunks, and future answer-span captures MUST record the exact prompt wrapper, separators, BOS/EOS handling, and role formatting used during prefill.
- Activation similarity experiments MUST include hubness diagnostics, including nearest-neighbor in-degree or equivalent k-occurrence reporting, before promotion beyond diagnostic use.
- Learned answer-seeker experiments MUST avoid generic-vector collapse by using contrastive training with in-batch and hard negatives, such as InfoNCE-style objectives.
- Answer-seeker supervision MUST define the target representation explicitly before training. Supported target definitions are:
  - teacher-forced activations at the known answer span positions
  - pooled activations of the full answer-containing chunk
  - contrastive answer-chunk versus distractor-chunk labels
- Any answer-seeker training dataset MUST contain query-to-evidence pairs with known answer spans or known answer-containing chunks and sealed distractors.
- Data splits MUST prevent answer span leakage, document duplicate leakage, and prompt-template leakage across train, dev, and held-out benchmark evaluation.
- Full CAA/EM recapture MUST NOT be launched over full corpora until a small SciFact/LegalBench-RAG pilot shows positive dev movement over the existing SAE-only Core245 direct-blend artifact and no regression against dense plus Ettin on the same candidate groups.
- Query-plus-candidate behavior-latent pilots MUST use a canonical prompt containing the user query and one candidate evidence passage in the same prefill context, keyed by a stable pair id derived from dataset, query id, candidate chunk id, and prompt-template hash.
- Behavior-latent pilot reports MUST distinguish true final-input-token telemetry from available pair-prompt pooled telemetry. Runs using the current Qwen/RMT/SAE max-over-prefill-token wrapper MUST be named `query_candidate_pair_core245_max_prefill` and MUST NOT be described as final-token-only CAA/SAE behavior telemetry.
- Behavior-latent pilots MUST evaluate both a mixed general-purpose model trained from multiple benchmark train/dev subsets and benchmark-specific models trained only on each dataset's train/dev split.
- Behavior-latent pilot evaluation MUST compare against pure dense candidate order, the existing actpred blended reranker for the same candidate subset, and dense plus Ettin when the matching Ettin score cache exists.
- Final behavior-latent claims MUST compare against an expanded off-the-shelf text-reranker envelope, not only Ettin, when compatible text rerankers can score the same frozen candidate groups.
- Behavior-latent public-claim reports MUST run a deduplication and leakage audit covering query IDs, normalized query text, near-duplicate query text, positive/candidate query-doc pairs, repeated positive evidence text, and train/dev positive evidence appearing in heldout candidates.
- When repeated train/dev positive evidence appears in heldout test, behavior-latent reports MUST include a strict no-train/dev-positive-overlap heldout subset evaluation before using the result as a primary generalization claim.
- Coding retrieval spot checks MUST use frozen dense candidate groups from a retrieval-focused code benchmark, prepare the same canonical query-plus-candidate behavior prompt used by the trained behavior-latent artifact, and compare behavior-prefill reranking against Ettin over the exact same candidate slate.
- Behavior-latent feature attribution MUST identify which Core245 SAE feature IDs and semantic manifest groups most influence the trained MLP's support score, using at least two independent attribution methods before recommending a trimmed feature subset.
- Feature-trimming recommendations MUST be validated by reranking the same frozen candidate groups with dropped features replaced by train-normalizer means, reporting top-k feature subset curves against the full behavior teacher.
- Student-reranker distillation MUST train a wholly decoupled text reranker that consumes only query text and candidate text at inference. It MUST NOT receive activation telemetry, dense scores, dense ranks, appended-positive position/provenance, labels, or handcrafted lexical metadata as input features.
- Student-reranker reports MUST compare the text-only student against pure dense, dense plus Ettin, and the behavior-prefill teacher on the same frozen candidate slates, and MUST distinguish imitation success from retrieval success by reporting teacher-score correlation/KL alongside benchmark relevance metrics.
- Clean student-reranker evaluation MUST include a heldout domain/split not used for student training; APPS coding retrieval is the preferred first heldout because the behavior-prefill teacher showed a large zero-shot win there.
- Reranking-only behavior-prefill capture MAY use an opt-in layer-truncated execution path that aborts immediately after the configured SAE hook/site tensor is captured. This MUST be limited to strict zero-token telemetry capture and MUST NOT alter normal generation, answer production, or non-reranking capture defaults.
- Reranking-only behavior-prefill capture MAY reuse the shared canonical prompt-plus-query prefix across candidates for the same query, but only when tokenized prefix IDs exactly match the corresponding full-prompt prefix. Mismatches MUST fall back to non-cached early-stop capture.
- Any speedup claim for behavior-prefill reranking MUST report the baseline full-forward capture time, layer-truncated capture time, prefix-cache capture time, and a text-reranker baseline such as Ettin over the same frozen query/candidate slate.

## Non-Goals

- No multi-hop retrieval in the first milestone.
- No answer generation pipeline in the first milestone.
- No training of the answer-seeker artifact in the first milestone.
- No direct llama.cpp runtime dependency in the core package.
- No hidden tuning on held-out RAG benchmarks.

## Acceptance Criteria

- Spec Kit `spec.md`, `research.md`, `plan.md`, `data-model.md`, contracts, and `tasks.md` exist for the feature.
- A local test suite verifies stable chunk IDs, selector-compatible telemetry records, and activation-similarity ranking.
- The README explains the dense and activation retrieval paths.
- A quickstart command or snippet demonstrates ingesting sample documents, capturing mock telemetry, and querying by dense or activation strategy.
- A local comparison run demonstrates which chunks are surfaced by dense search, activation-KNN, and activation reranking.
- A fixture benchmark verifies that metric computation and approach comparison work before any full benchmark download is attempted.
- A sidecar/file/command-backed provider test verifies that non-mock prefill telemetry records can be ingested for document chunks and captured for query chunks through the same `TelemetryProvider` interface.
- The feature artifacts explicitly document dimensionality/layer selection, contextual drift, hubness, data scarcity, supervision target definition, and prompt canonicalization risks.

# Implementation Plan: Activation Telemetry RAG

## Architecture

Create a Python package named `activation_rag` with focused modules:

- `schema.py`: dataclasses for documents, chunks, embeddings, activation records, and retrieval results.
- `chunking.py`: paragraph/sentence-aware chunker with overlap and stable IDs.
- `embedding.py`: embedding provider protocol plus deterministic hash embedder.
- `telemetry.py`: `TelemetryProvider` protocol, mock selector-compatible provider, and a command/file-backed provider for real sidecar prefill telemetry.
- `retrieval.py`: in-memory cosine search, activation-similarity search, and explicit activation matching strategies for anisotropic telemetry spaces.
- `pipeline.py`: orchestration for ingesting documents, embedding chunks, capturing telemetry, and querying.
- `benchmarks.py`: dataset-neutral benchmark records, metrics, approach evaluation, and run summaries.
- `benchmark_download.py`: explicit dataset download/preparation entrypoints for MS MARCO, BEIR, and HotpotQA.

The first implementation avoided heavyweight dependencies in the core path. The anisotropic activation matching pass promotes NumPy into the package dependency set because whitening, PCA removal, and local-scaling score matrices are first-class retrieval operations rather than throwaway research-only transforms. The real telemetry path remains an adapter around sidecar/llama.cpp capture artifacts or a capture command, so the core RAG package does not own model startup.

## Data Flow

1. Caller creates `DocumentRecord` values from text.
2. `Chunker.split()` emits `ChunkRecord` values.
3. `EmbeddingProvider.embed_chunks()` emits `EmbeddingRecord` values.
4. `TelemetryProvider.capture_prefill()` emits `ActivationRecord` values from either deterministic test telemetry or real prefill capture rows.
5. `RagEngine` stores the records in memory for the first milestone.
6. Query methods return `RetrievalResult` values with strategy names and component scores.
7. Comparison methods return dense, activation-KNN, and dense-first activation-reranked result sets for the same query.
8. Benchmark methods evaluate those same approaches over a corpus/query/qrels dataset and write comparable metric summaries.

## Technical Choices

- Python 3.11+.
- Standard-library dataclasses for schema stability.
- No required GPU/model dependency for tests; non-fixture benchmark runs require a non-mock telemetry provider or an explicit harness-only override.
- Cosine similarity remains the default for both dense and activation vectors. CSLS-style local scaling, NICDM-style local scaling, whitening plus L2 normalization, top-PC removal, and per-site late fusion are explicit opt-in activation strategies.
- JSON-schema-style contracts under `specs/001-activation-telemetry-rag/contracts/`.
- Add provider metadata fields for hook/site selection, prompt template identity, normalization policy, and aggregation policy before integrating a real activation runtime.
- Keep activation vectors bounded to selected/pooled telemetry fields. Do not concatenate all layers or raw token-by-layer matrices into the retrieval index.
- Treat hubness reporting and hard-negative contrastive datasets as required gates for the later answer-seeker phase, not optional evaluation polish.
- Keep answer-seeker target mode explicit in every training manifest: `teacher_forced_answer_span`, `pooled_answer_chunk`, or `contrastive_answer_chunk`.

## Validation

- `python -m unittest discover -s tests`
- Test stable chunk IDs and offsets.
- Test mock telemetry fields and delta math.
- Test activation-similarity search ranks the intended chunk.
- Test dense retrieval remains independent from telemetry capture.
- Test activation-similarity reranking reorders an existing dense candidate pool without broadening the retrieval pool.
- Test every promoted activation matching strategy with small deterministic fixtures before using it in benchmarks.
- Test that the raw cosine activation default is unchanged unless a strategy config is supplied.
- Test comparison output includes dense, activation-KNN, and activation-reranked result groups.
- Test benchmark metrics on a fixture dataset where the expected `MRR@10`, `Recall@k`, and `nDCG@k` are known.
- Test fixture benchmark evaluation over the three approaches without downloading external data.
- Test command-backed real telemetry row adaptation, including selector-compatible baseline/current/delta fields.
- Reject mock telemetry for non-fixture benchmark runs unless `--allow-mock-telemetry` is set.
- Add future tests that reject mixed prompt-template activation indexes once persistent indexes are introduced.

## Phases

1. Create project metadata and package skeleton.
2. Write failing tests for chunking, telemetry, and retrieval.
3. Implement schema and chunking.
4. Implement mock embedding, telemetry, and retrieval.
5. Add quickstart and verify tests.
6. Add real-provider readiness gates for layer selection, normalization, prompt canonicalization, and hubness diagnostics.
7. Add answer-seeker dataset contracts and contrastive training once query-to-answer-span pairs exist.
8. Add activation-KNN comparison and activation-similarity reranking over dense candidates.
9. Add benchmark harness and fixture metrics.
10. Add real sidecar prefill telemetry adapter and benchmark guardrails.
11. Add dataset download adapters and long-running full-eval commands for MS MARCO, BEIR, and HotpotQA.
12. Run the activation retrieval research addendum as an ablation pass before changing the production retrieval architecture. The pass must separate post-processing variants that can use the existing SciFact cache from variants that require corrected recapture.
13. Promote no activation-based reranking path unless it survives hubness diagnostics and improves or preserves dense retrieval metrics on a development split.
14. Rerun the ablation harness on corrected section-prefill telemetry before making causal claims about activation-space retrieval.
15. Prepare supervised activation reranker/projection training data only after corrected-cache diagnostics show which activation transformations produce stable signal.
16. Abandon the single universal activation reranker as the primary experimental artifact until it generalizes. Train dataset-specific learned blended rerankers for SciFact, NFCorpus, and FiQA over frozen dense candidate pools.
17. Treat blending as a supervised reranking objective, not a post-hoc reporting choice. The learned blended reranker must consume dense score/rank features plus answer-bearing activation-prediction scores and optimize query-group ranking on train/dev splits.
18. Preserve faithful heldout evaluation: train only on each dataset's train split, select hyperparameters/checkpoints only on dev, and run test once per selected artifact against pure dense and dense+Ettin baselines with paired query-level audits.
19. Use an interpretable constrained blend first, then promote heavier learning-to-rank models only if the constrained trainer shows stable dev/test signal. Initial acceptable forms are a fixed-alpha grid and a pairwise logistic linear scorer over per-query normalized dense and activation features; later optional forms include LambdaMART-style rankers when the dependency/runtime environment supports them.
20. Run a TechQA-RAG-Eval transfer benchmark as a documentation-retrieval test. Convert answerable TechQA rows into chunk-level candidate groups over the unique linked IBM Technotes, label answer-bearing chunks by answer-span overlap with a lexical fallback, capture the same Qwen/RMT/SAE Core245 prefill telemetry for query and candidate chunks, and compare pure dense, dense+Ettin, and dense+SciFact-direct-blend-actpred under the same frozen dense candidate pools.
21. Prepare the next validation suite around domain-specific expert prose rather than generic transfer: legal retrieval via MLEB/LegalBench-RAG, biomedical retrieval via R2MED, and coding retrieval/reranking via CoREB. Each vertical should get its own adapter, leakage controls, and train/dev/test path before training a vertical-specific activation-aware hyper-reranker.
22. Train vertical-specific direct-blend answer-activation rerankers for legal, medical, and coding benchmarks instead of expecting one universal reranker to transfer. When a benchmark only publishes a test split, create deterministic query-level train/dev/test splits and label them as internal adaptation splits, not benchmark-standard heldout splits.
23. Use native retrieval/candidate structure where possible: MLEB legal RAG passage IDs remain atomic retrieval units, R2MED uses query/corpus/qrels retrieval components with BGE dense candidates, and CoREB uses its published top-ranked reranking candidate lists with reciprocal-rank dense proxy features unless a fresh dense cache is explicitly built.
24. Scale the vertical training regime with larger same-shaped supervised evidence retrieval sources before interpreting failures as negative evidence about activation telemetry. Legal should add LegalBench-RAG exact span evidence windows, medical should pool all eight R2MED retrieval tasks, and code should move from saturated CoREB reranking lists toward CoIR/MTEB retrieval tasks where dense candidates can actually miss.
25. Add cross-fitted score generation and stronger regularization to the direct-blend answer-activation trainer. Out-of-fold train scores should be generated by training fold models on query-disjoint folds and scoring held-out train queries, while the final reported model is still selected only on the dev split and evaluated once on the heldout test split.
26. Diagnose vertical failures with per-query candidate movement audits. For each serious run, persist examples where activation-aware reranking improves or harms nDCG@10, including dense top candidates, model top candidates, labels, scores, and snippets. Use these audits to separate near-topic overpromotion, missing positives, dense pool failures, and true activation-signal failures.
27. Restore full selector-compatible CAA/EM fields only through a gated pilot. The existing Qwen/RMT/SAE Core245 capture path is valid as an SAE-only artifact, but it bypasses the 8-head current/baseline/delta machinery. Before rebuilding ingestion, implement or reuse a strict zero-token path that emits either compact `final_em_v2_features` plus `final_em_v2_inputs` or raw activation captures that can be rescored against the production learned bundle and neutral baseline. Run this first on SciFact and LegalBench-RAG candidate unions; promote to full recapture only if the added CAA/EM fields improve dev performance over the current SAE-only model and preserve locked baselines.
28. Add a query-plus-candidate behavior-latent pilot before any further isolated actpred retraining. The pilot builds stable pair-prompt IDs over `dataset/query/candidate/template`, captures the available Qwen/RMT/SAE Core245 pair-prompt prefill telemetry, trains a direct support reranker over pair telemetry plus dense features, and evaluates mixed-general and dataset-specific models on SciFact, LegalBench-RAG, and R2MED against dense, existing actpred blends, and Ettin where available.
29. Treat the first behavior-latent pilot as a proxy for the desired final-input-token behavior objective unless the telemetry provider is extended. The report must state whether the capture used `query_candidate_pair_core245_max_prefill` or a true final-token CAA/SAE row.
30. Before making public behavior-latent claims, run split-level deduplication and leakage audits for exact/near-duplicate query text, exact positive/candidate pairs, repeated positive evidence text, and train/dev positives appearing in heldout candidate pools. If positive evidence recurs across splits, evaluate an additional strict heldout subset that removes duplicate query text and train/dev-positive evidence overlap.
31. Run a coding retrieval transfer spot check on CoIR/CoSQA using already-prepared BGE dense candidate groups. Prepare behavior-pair prompts for the heldout split, capture or reuse strict zero-token Qwen/RMT/SAE Core245 pair telemetry, score with the released general behavior-latent MLP, and compare against dense-only plus dense candidates reranked by Ettin over the identical candidate IDs.
32. Extend the coding transfer spot check into a fuller CoIR suite without document-prefill ingestion. For each selected CoIR task, build or reuse BGE dense candidate groups, prepare query+candidate behavior prompts, capture strict zero-token pair telemetry only, score the released behavior-latent checkpoint, run Ettin over the same candidate IDs, and report dense, dense+Ettin, and dense+behavior metrics with paired audits.
33. Run behavior-latent MLP feature attribution against the published robust checkpoint. Combine first-layer normalized weight magnitude, gradient-times-input attribution, permutation importance, and top-k keep/drop curves over the same frozen candidate groups. Aggregate raw Core245 feature IDs back to longmem manifest labels and categories before recommending any future trimmed telemetry subset.
34. Train a text-only student reranker from behavior-prefill teacher scores. The student must be a conventional query/candidate cross-encoder and may only consume query text plus candidate text at inference, with no dense metadata, ranks, labels, provenance, handcrafted lexical features, or activation telemetry. Optimize teacher-score distillation with pointwise score matching plus group-wise listwise and margin losses; evaluate against dense, Ettin, and the behavior teacher on a clean heldout domain, with APPS coding retrieval as the first target.
35. Optimize behavior-prefill reranking latency without changing the signal. Add an explicit strict-zero-token execution mode that stops the Qwen/RMT/SAE forward pass immediately after the configured layer-7 SAE hook captures `resid_pre`, preserving the tensor used by the trained Core245 behavior artifact while avoiding layers 7+. Add a second opt-in mode that caches the token-exact canonical `Query ... Candidate evidence:` prefix per query and reuses its KV/cache state across candidate suffixes when tokenization permits exact prefix matching. Benchmark full-forward capture, layer-truncated capture, prefix-cache capture, and Ettin over the same frozen candidate slate before using the optimization in reports.

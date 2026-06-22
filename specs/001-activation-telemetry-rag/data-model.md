# Data Model: Activation Telemetry RAG

## Document Record

- `document_id`
- `source_uri`
- `title`
- `text`
- `metadata`
- `text_hash`

## Chunk Record

- `schema_version`: `activation_rag.chunk.v1`
- `chunk_id`
- `document_id`
- `ordinal`
- `text`
- `text_hash`
- `char_start`
- `char_end`
- `token_count_estimate`
- `chunker`
- `chunk_size`
- `chunk_overlap`

## Embedding Record

- `schema_version`: `activation_rag.embedding.v1`
- `chunk_id`
- `model_id`
- `vector`
- `dimension`
- `normalized`

## Activation Record

- `schema_version`: `activation_rag.activation_record.v1`
- `chunk_id`
- `document_id`
- `capture_run_id`
- `provider_id`
- `model_id`
- `model_hash`
- `tokenizer_hash`
- `site_id`
- `hook_name`
- `layer_index`
- `layer_selection_policy`
- `prompt_template_id`
- `prompt_template_hash`
- `normalization_policy`
- `token_start`
- `token_end`
- `aggregation`
- `current_em_state`
- `neutral_baseline_state`
- `prior_current_state`
- `delta_vs_neutral`
- `delta_vs_current`
- `saturation`
- `residual_headroom`
- `positive_mass`
- `negative_mass`
- `total_mass`
- `signed_balance`
- `sae_feature_values`
- `sae_delta_vs_neutral`
- `sae_delta_vs_current`
- `sae_feature_mask`
- `sae_novelty`
- `sae_overlap_with_memory`
- `telemetry_valid`
- `invalid_reason`
- `provenance`

## Prefill Capture Request

- `schema_version`: `activation_rag.prefill_capture_request.v1`
- `capture_run_id`
- `chunk_id`
- `document_id`
- `ordinal`
- `text`
- `text_hash`
- `token_count_estimate`
- `prompt_text`
- `provider_id`
- `model_id`
- `site_id`
- `layer_selection_policy`
- `prompt_template_id`
- `prompt_template_hash`
- `normalization_policy`
- `capture_phase`: `prefill`
- `generation_disabled`: `true`

## Retrieval Result

- `chunk_id`
- `strategy`
- `score`
- `component_scores`
- `rank`

## Search Comparison

- `query`
- `dense_results`
- `activation_results`
- `activation_reranked_results`
- `dense_activation_overlap`
- `dense_rerank_overlap`
- `notes`

## Manifest

- `schema_version`
- `artifact_id`
- `created_at`
- `chunker_settings`
- `embedding_provider`
- `telemetry_provider`
- `layer_selection_policy`
- `prompt_canonicalization`
- `normalization_policy`
- `hubness_report`
- `record_counts`
- `source_hashes`

## Answer-Seeker Training Row

- `schema_version`: `activation_rag.answer_seeker.row.v1`
- `query_id`
- `query_text_hash`
- `query_activation_id`
- `positive_chunk_id`
- `positive_answer_span_start`
- `positive_answer_span_end`
- `target_mode`: `teacher_forced_answer_span`, `pooled_answer_chunk`, or `contrastive_answer_chunk`
- `positive_activation_id`
- `hard_negative_chunk_ids`
- `hard_negative_activation_ids`
- `prompt_template_id`
- `normalization_policy`
- `split`: `train`, `dev`, or `heldout_test`
- `dataset_source`
- `leakage_group_id`

## Benchmark Dataset

- `name`
- `split`
- `corpus`: mapping of benchmark document ids to text
- `queries`: mapping of query ids to text
- `qrels`: mapping of query ids to relevant document ids and grades
- `metric_profile`: `msmarco_passage`, `beir`, `hotpotqa_supporting_evidence`, or `custom`

## Benchmark Run Summary

- `schema_version`: `activation_rag.benchmark_run.v1`
- `dataset_name`
- `split`
- `query_count`
- `corpus_count`
- `approaches`
- `metrics_by_approach`
- `candidate_k`
- `top_k`
- `started_at`
- `finished_at`
- `duration_seconds`
- `notes`

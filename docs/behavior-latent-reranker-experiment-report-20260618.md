# Query+Candidate Behavior-Latent Reranking Experiment Report

Date: 2026-06-18

Status: Final interim report after duplicate-evidence, leakage, expanded text-reranker baseline due diligence, and a coding-retrieval transfer spot check, updated 2026-06-21.

## Executive Summary

We tested whether internal activation telemetry from a language model can improve retrieval-augmented generation (RAG) evidence selection. The decisive result was not achieved by treating activations as ordinary embedding vectors. It was achieved by using model telemetry from a query-plus-candidate evidence prompt and training a small reranker to recognize when a candidate passage induces an internal "this evidence supports the query" behavior.

The strongest finding is that a learned query+candidate behavior-latent reranker using Qwen layer-7 Core245 SAE prefill telemetry from the exact evidence-support prompt decisively improves dense-candidate evidence retrieval in expert domains. After expanding the text-reranker baselines beyond Ettin to include BGE v2-m3, GTE ModernBERT, Qwen3-0.6B seq-cls, and Mixedbread base-v2, the behavior-latent reranker still beats the best observed text reranker by large paired-significant margins on LegalBench-RAG and R2MED. SciFact remains directionally encouraging in the full split, but strict duplicate-evidence filtering weakens the SciFact headline claim.

A fuller retrieval-focused CoIR coding suite is also highly informative and may be the most strategically important transfer result outside legal/medical RAG. Without document-prefill ingestion and without training a coding-specific artifact, the released general behavior-latent reranker beat Ettin on CoSQA and APPS over identical frozen dense candidate groups. APPS was especially large: behavior reached `0.9339` nDCG@10 versus Ettin `0.6418` and dense `0.0572`, p approximately `0.0001` versus Ettin. The suite is mixed, because behavior loses to Ettin on StackOverflow-QA and CodeTrans-Contest, but the 5-task macro nDCG still favors behavior (`0.6861`) over Ettin (`0.6676`), and the query-weighted nDCG favors behavior more strongly (`0.8354` versus `0.7125`). This is not yet a universal coding-vertical claim, but it is a potentially important signal because APPS and CoSQA are close to the retrieval bottleneck faced by coding agents: retrieving code that actually solves, implements, or operationally supports a task from a slate of plausible distractors.

This is a promising result because it points to a specific and interpretable use of activation telemetry: not as a replacement embedding space, but as a high-precision evidence-support adjudicator over a small set of plausible dense candidates.

## Background: What Problem Are We Solving?

RAG systems answer questions by retrieving passages from a document collection and giving those passages to a language model as context. The quality of the final answer depends heavily on whether the retrieval system surfaces the right evidence.

A standard RAG retrieval pipeline usually works like this:

1. Split documents into chunks.
2. Embed each chunk into a vector.
3. Embed the user's query into the same vector space.
4. Retrieve chunks whose vectors are close to the query vector.
5. Optionally rerank those chunks with a stronger text reranker.
6. Pass the top chunks to an answer model.

This works well for many broad semantic search tasks, but it can fail in high-stakes expert domains. Legal, biomedical, scientific, and compliance documents often contain many passages that are topically similar but differ in legally or medically critical details. Dense vector search can surface near-topic distractors. Text rerankers improve this, but they still operate externally over text rather than inspecting the internal behavior induced inside a model by the query and candidate evidence.

Our central question was:

Can activation telemetry from a model give us a better signal about which candidate passage actually supports answering the query?

## What Is Activation Telemetry?

A transformer language model processes text through many layers. At each layer, it creates internal hidden states. Sparse autoencoders (SAEs) can decompose those hidden states into more interpretable features. In this project, the telemetry used in the winning method is a selected set of 245 SAE features, called Core245, captured from Qwen layer 7 at the `resid_pre` site.

The relevant telemetry row records which selected SAE features activate, and by how much, when the model reads a prompt. This lets us ask not only "what text is similar to what query?", but "what internal behavior does this query/evidence pair induce?"

The current winning run uses:

- Model: `qwen3-4b-rmt-sae`
- Site: `qwen.model.layers.7.resid_pre`
- Feature set: `core245_corrected_longmem_query_conditioned_train_dev`
- Feature count: 245
- Capture phase: zero-token prefill
- Aggregation: max over prompt tokens
- Prompt representation: `query_candidate_pair_core245_max_prefill`

Important limitation: this is SAE-only Core245 telemetry. It is not yet full final-token CAA/SAE selector telemetry. The current rows do not populate the full current/baseline/headroom EM/CAA fields.

## Initial Hypotheses And What Failed

The project began with a natural hypothesis: maybe document chunks can be represented by activation vectors in the same way they are represented by embedding vectors. If so, we might retrieve by activation-space similarity.

We tested several variants:

- activation-only nearest-neighbor search,
- activation cosine similarity,
- hubness-aware matching,
- whitening and top-PC removal,
- CSLS/NICDM-style corrections,
- per-site late fusion,
- selected Core245 feature subsets,
- answer-bearing activation prediction.

These methods were mostly weak or inconsistent. The reason became clearer over time: isolated query activations and isolated document-chunk activations do not necessarily live in a geometry where "the answer" is close to "the question." Activation space is anisotropic, context-sensitive, and behavior-dependent. Raw similarity in that space was the wrong abstraction.

The successful shift was to stop asking:

> Is this document chunk close to the query in activation space?

and instead ask:

> When the model sees this query and this candidate evidence together, does its internal telemetry look like evidence support rather than near-topic distraction?

## Winning Method: Query+Candidate Behavior-Latent Reranking

For each candidate passage, we construct a canonical prompt:

```text
Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:
```

We then run a zero-output-token prefill pass through the telemetry model and capture Core245 SAE activations over this full prompt. No answer is generated. The system only observes the model's internal response to the query/evidence pair.

Each query/candidate pair is represented as:

```text
245 SAE feature values
+ dense_score
+ dense_z
+ dense_rank_reciprocal
= 248 input features
```

The SAE values are transformed with `log1p_l2`:

1. Apply signed `log1p` compression to reduce extreme feature magnitudes.
2. L2-normalize the 245-dimensional SAE vector.
3. Append the three dense-retrieval metadata features.
4. Normalize the full feature vector with train-set mean and standard deviation.

The model is a small MLP:

```text
Linear(248 -> 128)
ReLU
Dropout(0.15)
Linear(128 -> 128)
ReLU
Dropout(0.15)
Linear(128 -> 1)
```

The output is a learned support score for the candidate.

Training uses a listwise query-group objective. Within each query group, positives are answer-supporting candidates and negatives are dense hard negatives. The model learns to rank positives above negatives. The final score during training can blend dense score and behavior score:

```text
final_score = (1 - alpha) * z_dense_score + alpha * z_behavior_support_score
```

The robust runs swept alpha over:

```text
0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0
```

All selected robust models chose `alpha = 1.0`, meaning the learned behavior score dominated the final ranking. However, the model still had dense metadata features available as inputs, so the correct description is:

> a learned query+candidate behavior telemetry reranker over dense candidates, with dense retrieval metadata available as features, but no final dense-score blend in the selected artifacts.

## Experimental Setup

We evaluated on frozen dense-candidate groups so that methods were compared over the same candidate pool.

Datasets:

- SciFact: scientific claim/evidence retrieval.
- LegalBench-RAG: legal evidence/snippet retrieval.
- R2MED: biomedical and medical retrieval tasks.
- CoIR bounded coding transfer suite: retrieval-focused natural-language-to-code and problem-to-code evidence retrieval.

Candidate pool:

- Dense retrieval produced candidate groups.
- Robust behavior runs used 16 candidates per query.
- Dense, Ettin, actpred, and behavior-latent systems were evaluated over the same frozen groups.

Telemetry capture:

- 141,592 query/candidate pair-prompt telemetry rows.
- Zero invalid rows.
- All rows labeled `query_candidate_behavior_prompt`.
- Mean active Core245 features per row: 61.27.
- Minimum active features: 18.
- Maximum active features: 88.

Baselines:

- Dense: original dense candidate ordering.
- Text rerankers: `cross-encoder/ettin-reranker-150m-v1`, `BAAI/bge-reranker-v2-m3`, `Alibaba-NLP/gte-reranker-modernbert-base`, `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, and `mixedbread-ai/mxbai-rerank-base-v2`.
- Actpred: earlier answer-bearing activation-prediction reranker.
- Behavior-latent: the current query+candidate telemetry reranker.

Coding transfer spot check:

- Source groups: `runs/supervised/verticals/coir-cosqa/test-groups.k100.bge.appendpos.jsonl`
- Prepared behavior-pair groups: `runs/coding-retrieval-behavior-20260621/coir-cosqa/test-groups.behavior-pair.jsonl`
- Scope: 500 heldout CoIR/CoSQA queries, 16 candidates per query, 8,000 total query/candidate pairs.
- Candidate protocol: frozen BGE dense slate with appended positives, then identical candidate IDs reranked by Ettin and by the released general behavior-latent checkpoint.
- Telemetry: strict zero-token Qwen/RMT/SAE Core245 pair-prompt prefill, `query_candidate_pair_core245_max_prefill`.
- Behavior checkpoint: `release/behavior-latent-general-v0.1/model.pt`.
- Ettin baseline: `cross-encoder/ettin-reranker-150m-v1`, max length 512.

Fuller coding suite:

- Tasks: CoSQA, CodeTrans-DL, CodeTrans-Contest, StackOverflow-QA, and APPS.
- Total scope: 6,660 heldout queries and 106,560 query/candidate pairs across the five tasks.
- Dense candidates: BGE base dense candidates, `k=100`, with qrel positives appended when missing.
- Rerank candidate count: 16 per query for both Ettin and behavior-prefill.
- Artifact summary: `runs/coding-retrieval-behavior-suite-20260621/summary.json`

Metrics:

- MRR@10: how high the first relevant result appears.
- nDCG@10: ranking quality in the top 10, rewarding relevant evidence near the top.
- Recall@10: whether relevant evidence appears anywhere in the top 10.

Significance testing:

- Paired per-query randomization tests.
- The minimum p-value with 10,000 randomization iterations is approximately `0.0001`.
- A p-value of `0.0001` means the observed effect was more extreme than all sampled randomized sign flips.

## Results

### nDCG@10 Summary

| Dataset / model | Dense | Ettin | Actpred | Behavior-latent |
| --- | ---: | ---: | ---: | ---: |
| SciFact specific | 0.7682 | 0.8161 | 0.8514 | 0.8381 |
| SciFact general | 0.7682 | 0.8161 | 0.8514 | 0.8016 |
| R2MED specific | 0.1732 | 0.4586 | 0.1762 | 0.8414 |
| R2MED general | 0.1732 | 0.4586 | 0.1762 | 0.8616 |
| LegalBench-RAG specific | 0.2733 | 0.5023 | 0.2884 | 0.7292 |
| LegalBench-RAG general | 0.2733 | 0.5023 | 0.2884 | 0.7510 |

### Expanded Text-Reranker Baseline Check

Because Ettin is a strong text reranker but not the only reasonable modern baseline, we reran the same frozen dense-candidate groups with additional off-the-shelf rerankers:

- `BAAI/bge-reranker-v2-m3`
- `Alibaba-NLP/gte-reranker-modernbert-base`
- `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- `mixedbread-ai/mxbai-rerank-base-v2`
- `cross-encoder/ettin-reranker-150m-v1`, rescored through the same explicit artifact path

All were evaluated as rerankers over the same dense candidates. Artifacts are under:

- `runs/text-reranker-expanded-20260621/`
- `runs/text-reranker-expanded-20260621/expanded-text-reranker-comparison.json`

Full-split nDCG@10:

| Dataset | Dense | Behavior-general | Ettin | BGE v2-m3 | GTE ModernBERT | Qwen3 0.6B seq-cls | Mixedbread base-v2 | Best text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.7682 | 0.8016 | 0.8163 | 0.8042 | 0.8134 | 0.3762 | 0.7938 | Ettin, narrowly over GTE |
| LegalBench-RAG | 0.2733 | 0.7510 | 0.5023 | 0.5187 | 0.4854 | 0.3075 | 0.4454 | BGE v2-m3 |
| R2MED | 0.1732 | 0.8616 | 0.4586 | 0.3764 | 0.4630 | 0.4120 | 0.3789 | GTE ModernBERT |

Strict no-train/dev-positive-overlap nDCG@10:

| Dataset | Dense | Behavior-general | Ettin | BGE v2-m3 | GTE ModernBERT | Qwen3 0.6B seq-cls | Mixedbread base-v2 | Best text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.7626 | 0.7960 | 0.8098 | 0.7867 | 0.8477 | 0.4595 | 0.8363 | GTE ModernBERT |
| LegalBench-RAG | 0.2655 | 0.7437 | 0.5055 | 0.5178 | 0.4894 | 0.3027 | 0.4544 | BGE v2-m3 |
| R2MED | 0.1862 | 0.8527 | 0.4711 | 0.3863 | 0.4752 | 0.4042 | 0.3872 | GTE ModernBERT |

Interpretation:

- Ettin remains a very strong baseline, but it is not uniformly the best text reranker in this local evaluation. BGE v2-m3 is strongest on LegalBench-RAG, and GTE ModernBERT is strongest on R2MED and strict SciFact.
- The primary LegalBench-RAG and R2MED claim survives the stronger text-reranker envelope. On LegalBench-RAG full split, behavior-general beats the best expanded text reranker, BGE v2-m3, by `+0.2323` nDCG@10, p approximately `0.0001`. On strict LegalBench-RAG, behavior beats BGE by `+0.2259`, p approximately `0.0001`.
- On R2MED full split, behavior-general beats the best expanded text reranker, GTE ModernBERT, by `+0.3986` nDCG@10, p approximately `0.0001`. On strict R2MED, behavior beats GTE by `+0.3775`, p approximately `0.0001`.
- SciFact remains mixed. On the full split, Ettin narrowly leads behavior by `+0.0147` nDCG@10, not significant. On the strict subset, GTE ModernBERT leads behavior by `+0.0517`, also not significant against behavior, but it does beat Ettin by `+0.0379`, p = `0.0250`. This reinforces that SciFact should remain supportive rather than primary.

### Coding Retrieval Transfer: CoIR

We then ran retrieval-focused coding evaluations before publishing the artifact. This is important because coding assistants and agentic software-engineering systems often fail or succeed based on whether they retrieve the right snippet, function, API usage, or repository-local implementation pattern. The retrieval setting is also structurally similar to the domains where behavior-latent reranking has looked strongest: many candidates are lexically and semantically near-topic, but only one actually supports the requested action.

Benchmark choice:

- CoIR was used because it is explicitly a code information retrieval benchmark and follows a BEIR/MTEB-style retrieval structure.
- CORE-Bench is a promising newer benchmark for coding agents, but no prepared local candidate groups exist in this workspace yet.
- The bounded suite covers five CoIR tasks available through the local adapter and should be read as a serious transfer signal, not yet as a complete coding benchmark suite.

Initial CoSQA protocol:

- 500 heldout CoIR/CoSQA queries.
- 16 candidates per query.
- 8,000 total query/candidate pairs.
- Same frozen dense candidate IDs for dense, Ettin, and behavior-prefill reranking.
- Behavior arm used the released general checkpoint trained on the prior behavior-latent corpora, with no coding-specific retraining.
- Behavior telemetry used the same canonical query+candidate evidence-support prompt and strict zero-output-token prefill path.

CoSQA results:

| System | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense BGE candidates | 0.2515 | 0.3258 | 0.5680 |
| Dense + Ettin | 0.3484 | 0.4447 | 0.7620 |
| Dense + behavior-prefill reranker | 0.4549 | 0.5277 | 0.7700 |

Paired deltas:

| Comparison | MRR@10 delta | nDCG@10 delta | Recall@10 delta | Significance |
| --- | ---: | ---: | ---: | --- |
| Behavior minus dense | +0.2035 | +0.2020 | +0.2020 | p approximately 0.0001 for all three metrics |
| Ettin minus dense | +0.0970 | +0.1189 | +0.1940 | p approximately 0.0001 for all three metrics |
| Behavior minus Ettin | +0.1065 | +0.0830 | +0.0080 | MRR p approximately 0.0001; nDCG p approximately 0.0003; recall not significant, p = 0.8318 |

Artifacts:

- Capture cache: `runs/coding-retrieval-behavior-20260621/telemetry-cache/`
- Behavior metrics/scores: `runs/coding-retrieval-behavior-20260621/coir-cosqa/behavior-general-metrics.json`, `behavior-general-scores.jsonl`
- Ettin metrics/scores: `runs/coding-retrieval-behavior-20260621/coir-cosqa/ettin-metrics.json`, `ettin-scores.jsonl`
- Paired comparison: `runs/coding-retrieval-behavior-20260621/coir-cosqa/comparison.json`
- Changed-query audit: `runs/coding-retrieval-behavior-20260621/coir-cosqa/audit-behavior-vs-ettin.json`

Interpretation:

- This was a major encouraging transfer result. The behavior-latent model was not trained specifically for coding, yet it beat Ettin decisively on rank-quality metrics over the same candidate slate.
- The gain appears to be ranking-quality rather than candidate-pool recall. Recall@10 is nearly tied with Ettin, but behavior places positives much higher when it succeeds.
- The changed-query audit shows plausible qualitative behavior: the model can promote exact function-intent support from deep dense ranks when the top dense/Ettin candidates are near-duplicate lexical distractors.
- The failure mode is also concrete: when Ettin already places an exact positive first, behavior sometimes overpromotes nearby code variants. This keeps the result in the "very promising transfer evidence" category until we run a broader coding suite, strict duplicate controls, and ideally coding-specific training.

Fuller five-task CoIR suite:

| Task | Queries | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 | Behavior minus Ettin nDCG@10 | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoSQA | 500 | 0.3258 | 0.4447 | 0.5277 | +0.0830 | 0.0003 |
| CodeTrans-DL | 180 | 0.2123 | 0.3587 | 0.3337 | -0.0251 | 0.2927 |
| CodeTrans-Contest | 221 | 0.9095 | 0.9770 | 0.8664 | -0.1106 | 0.0001 |
| StackOverflow-QA | 1,994 | 0.7993 | 0.9156 | 0.7685 | -0.1471 | 0.0001 |
| APPS | 3,765 | 0.0572 | 0.6418 | 0.9339 | +0.2921 | 0.0001 |

Aggregate CoIR suite:

| Aggregate | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 |
| --- | ---: | ---: | ---: |
| Macro average over 5 tasks | 0.4608 | 0.6676 | 0.6861 |
| Query-weighted average over 6,660 queries | 0.3320 | 0.7125 | 0.8354 |

Suite interpretation:

- The coding result is not universal, but it is stronger than a one-off spot check. Behavior-prefill wins decisively on APPS and CoSQA, loses decisively on CodeTrans-Contest and StackOverflow-QA, and is statistically tied/slightly behind Ettin on CodeTrans-DL nDCG while slightly ahead on MRR.
- APPS is the standout: behavior reaches `0.9339` nDCG@10 against Ettin `0.6418` and dense `0.0572`. That is a large paired-significant win on the biggest task in the bounded suite.
- StackOverflow-QA and CodeTrans-Contest show the failure mode. When dense/Ettin already perform strongly, or when qrels reward exact known items among many plausible near-equivalent snippets, behavior can overpromote unlabeled alternatives.
- The right next coding step is a coding-specific behavior model plus duplicate/equivalence-aware analysis, not a claim that the current general behavior reranker beats Ettin on every coding retrieval task.

#### Agentic Coding Implications

The APPS and CoSQA results matter more for coding agents than the aggregate CoIR number alone. They are closer to the actual retrieval questions an agent faces during software work:

- APPS asks whether a candidate code solution satisfies a rich programming problem.
- CoSQA asks whether a candidate function implements a natural-language intent.

Those are both operational support judgments. A coding agent rarely needs a generic "similar text" result; it needs code that can be used to solve the task in front of it. This is precisely the form of judgment the behavior-latent prompt was trained to expose:

```text
Query:
{query}

Candidate evidence:
{candidate}

Task:
Decide whether the candidate evidence directly supports answering the query.
```

The zero-shot nature of the APPS and CoSQA wins is therefore unusually encouraging. The released general model was trained on non-coding behavior-latent evidence-support data, yet it transferred to task-to-code and intent-to-function retrieval. That suggests the captured telemetry may be closer to a general "does this candidate support accomplishing the requested task?" signal than to a narrow domain classifier.

This could matter for agentic coding systems in several concrete places:

- selecting relevant repository functions for a requested change,
- choosing implementation examples from a codebase or documentation corpus,
- retrieving solution patterns for algorithmic subtasks,
- deciding which candidate files/snippets deserve expensive reasoning,
- reducing context pollution from plausible but non-supporting code.

The production implication is a cascade:

1. Use cheap repository search, dense retrieval, lexical search, dependency graphs, call graphs, and symbol indexes to assemble a broad candidate slate.
2. Use inexpensive rerankers to reduce the slate.
3. Apply behavior-prefill reranking only to the final hard candidates where near-topic false positives are costly.

The important caveat is that these are reranking results. APPS used qrel-positive appending when dense retrieval missed the positive, so the result proves that behavior-prefill can identify the right solution when it is in the candidate slate. It does not prove that behavior-prefill alone can retrieve the solution from the full corpus. For coding agents, that is still a meaningful result because production systems are already cascaded: the hardest problem is often not finding any related code, but choosing which related code is actually useful.

The negative CoIR tasks sharpen the claim rather than invalidate it:

- CodeTrans-Contest is mostly exact translation-pair matching, and dense/Ettin are already near-saturated.
- StackOverflow-QA is accepted-answer text reranking, where Ettin is a better fit.

So the emerging claim is not "behavior-prefill beats text rerankers on all coding retrieval." The stronger and more useful claim is:

> Behavior-prefill reranking appears to transfer zero-shot to task-to-code support detection, producing a very large APPS win and a meaningful CoSQA win, which is exactly the retrieval shape that could improve coding agents once the right candidate code is present in the slate.

## Deduplication And Leakage Audit

After the initial robust runs, we expanded the leakage auditor from query-id and query-document pair checks to also cover normalized text duplicates and near-duplicates. The enhanced audit now checks:

- exact query-id overlap,
- exact normalized query-text overlap,
- near-duplicate query text using word-shingle Jaccard,
- exact positive `(query, doc)` pair overlap,
- exact candidate `(query, doc)` pair overlap,
- repeated positive evidence text,
- train/dev positive evidence appearing as a heldout candidate.

The clean result: no exact query-id overlap and no exact positive or candidate `(query, doc)` pair overlap were found across train/dev/test for SciFact, LegalBench-RAG, or R2MED.

The important caveat: evidence text recurs across splits. This is common in retrieval benchmarks built over a shared corpus, but it can still inflate a learned reranker if train/dev positives teach passage-level signatures that reappear as heldout positives or candidates.

Strict heldout subsets were therefore created by dropping heldout test queries when either the query text duplicated train/dev query text or any positive evidence doc/text had appeared as train/dev positive evidence.

| Dataset | Strict test kept | Drop rate | Dense | Ettin | Actpred | Behavior-specific | Behavior-general | Strict interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 103 / 291 | 64.6% | 0.7626 | 0.8096 | 0.7517 | 0.7123 | 0.7960 | SciFact weakens materially versus the dense/Ettin headline, but general behavior still beats actpred by `+0.0443` nDCG@10. That behavior-minus-actpred margin is not paired-significant on the 103-query strict subset (`p=0.2452`), but it is a clean numerical win over the earlier activation-prediction method. |
| LegalBench-RAG | 525 / 689 | 23.8% | 0.2655 | 0.5055 | 0.2722 | 0.7177 | 0.7437 | The LegalBench-RAG result survives strongly. General behavior beats Ettin by `+0.2382` nDCG@10, p approximately `0.0001`. |
| R2MED | 116 / 131 | 11.5% | 0.1862 | 0.4711 | 0.1879 | 0.8308 | 0.8527 | The R2MED result survives strongly. General behavior beats Ettin by `+0.3816` nDCG@10, p approximately `0.0001`. |

Audit artifacts:

- `runs/behavior-latent-robust-20260618/scifact/leakage-dedup-audit.json`
- `runs/behavior-latent-robust-20260618/legalbenchrag/leakage-dedup-audit.json`
- `runs/behavior-latent-robust-20260618/r2med-all/leakage-dedup-audit.json`

Strict subset artifacts:

- `runs/behavior-latent-robust-20260618/scifact/test-groups.strict-no-train-dev-positive-overlap.jsonl`
- `runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.strict-no-train-dev-positive-overlap.jsonl`
- `runs/behavior-latent-robust-20260618/r2med-all/test-groups.strict-no-train-dev-positive-overlap.jsonl`

This audit changes the narrative. The strongest controlled claim is now LegalBench-RAG and R2MED, where the gains survive a strict no-train/dev-positive-overlap heldout subset. SciFact should be reported as full-split positive and historically useful for method discovery, with one important strict-subset result preserved: the general behavior model still beats actpred by `+0.0443` nDCG@10, though it does not beat Ettin and the actpred margin is not paired-significant.

### SciFact

SciFact was the domain where earlier actpred work was strongest. The behavior-latent method still performs strongly:

- Behavior-specific beats dense by `+0.0699` nDCG@10, p = `0.0001`.
- Behavior-specific is numerically above Ettin by `+0.0220` nDCG@10, but this is not paired-significant.
- Actpred is numerically above behavior-specific by `+0.0134` nDCG@10, also not paired-significant.

Interpretation: SciFact remains highly favorable to the earlier answer-bearing activation-prediction method on the full split, and behavior-latent reranking is competitive there. Under the stricter no-train/dev-positive-overlap subset, the SciFact behavior result is no longer decisive against dense or Ettin, but the general behavior model does beat actpred by `+0.0443` nDCG@10.

### R2MED

R2MED is a decisive win.

Dense retrieval is weak on this frozen candidate setup:

- Dense nDCG@10: `0.1732`

Ettin is a strong text reranker:

- Ettin nDCG@10: `0.4586`
- Ettin improves over dense by `+0.2854`, p = `0.0001`

Behavior-latent reranking is much stronger:

- R2MED-specific behavior nDCG@10: `0.8414`
- R2MED-general behavior nDCG@10: `0.8616`
- General behavior beats dense by `+0.6884`, p = `0.0001`
- General behavior beats Ettin by `+0.4031`, p = `0.0001`

The changed-query audit supports the aggregate metric. For general behavior versus dense, nDCG@10 improved on 115 queries, harmed 16, and was unchanged on 0. For Ettin versus behavior, Ettin was worse on 100 queries, better on 24, and unchanged on 7.

Interpretation: the behavior-latent signal is not a small complement to dense retrieval here. It is identifying evidence support in a way that both dense retrieval and the text reranker miss.

### LegalBench-RAG

LegalBench-RAG is also a decisive win.

Baselines:

- Dense nDCG@10: `0.2733`
- Ettin nDCG@10: `0.5023`
- Actpred nDCG@10: `0.2884`

Behavior results:

- Legal-specific behavior nDCG@10: `0.7292`
- General behavior nDCG@10: `0.7510`
- General behavior beats dense by `+0.4777`, p = `0.0001`
- General behavior beats Ettin by `+0.2487`, p = `0.0001`
- General behavior beats actpred by `+0.4626`, p = `0.0001`

Changed-query audit:

- Behavior versus dense: 478 improved, 144 harmed, 67 unchanged.
- Behavior versus Ettin: Ettin was better on 205, worse on 408, unchanged on 76.

Interpretation: this is the cleanest result because Ettin is a strong relevant baseline and behavior-latent still wins by a large margin.

## What The Results Mean

The key result is not merely "activation telemetry helps." Earlier activation similarity methods did not reliably help. The stronger claim is:

> Activation telemetry becomes highly useful when captured from the model's response to the exact query/candidate evidence-support decision.

That distinction matters. The winning method does not ask the model to generate an answer. It also does not ask whether a query vector is close to a document vector. Instead, it probes the model's internal state after it has read both the question and a candidate evidence passage under a canonical support-judgment prompt.

This suggests that the useful signal is behavioral rather than geometric. The telemetry captures whether the model internally treats the candidate as answer-supporting evidence, not whether the text is merely topically similar.

The results are especially encouraging in domains with high semantic density and high costs for near-miss evidence:

- legal passages that discuss similar clauses but do not answer the precise question,
- biomedical passages that share terminology but differ in diagnosis or treatment relevance,
- scientific evidence where claim support depends on exact experimental findings.
- coding retrieval where many snippets share identifiers, syntax, and comments, but only one function or implementation detail actually supports the developer's next action.

## Why The Result Is Optimistic But Still Bounded

Reasons for optimism:

- Large effect sizes on LegalBench-RAG and R2MED.
- Strong paired significance.
- Strong changed-query movement, not just a few outlier wins.
- Wins over the expanded text-reranker envelope on LegalBench-RAG and R2MED, not only over dense retrieval or Ettin.
- A zero-shot five-task CoIR coding retrieval suite has a positive aggregate nDCG result, with decisive APPS and CoSQA wins over Ettin on the same frozen candidate IDs.
- The APPS and CoSQA wins are especially relevant to coding agents because they test task-to-code and intent-to-function support, not generic text similarity.
- General behavior model outperforms dataset-specific models on LegalBench-RAG and R2MED, suggesting possible cross-domain behavior signal.
- The LegalBench-RAG and R2MED wins survive strict duplicate-evidence filtering.

Reasons for caution:

- The current telemetry is SAE-only Core245 max-over-prefill-token telemetry, not final-token full CAA/SAE telemetry.
- The method is evaluated over frozen dense candidate pools, so it is a reranking result rather than end-to-end corpus retrieval.
- The general model outperforming dataset-specific models triggered leakage and duplication audits; those audits found no exact query-id or query-doc pair leakage, but they did find repeated evidence text across splits.
- The strict duplicate-evidence audit weakens SciFact substantially; SciFact should not be the primary controlled claim.
- Behavior reranking is computationally expensive because it requires query+candidate prefill passes.
- Additional external datasets would still strengthen publication-grade generalization claims, but the current report now includes the due-diligence duplicate-evidence control for the completed benchmark set.
- The CoIR coding result is mixed by task. It should motivate a serious coding-specific training and validation pass rather than be oversold as a universal coding reranker claim.
- The APPS result used appended positives in the rerank slate. It is evidence for powerful final-stage evidence selection, not standalone first-stage retrieval.

The right interpretation is that we have a decisive and promising reranking result in high-structure expert evidence retrieval, bounded by the duplicate-evidence audit and by the fact that this is a reranking method rather than a finished universal retrieval system.

## Feature Attribution And Student Distillation

After the main benchmark runs, we ran two follow-up diagnostics:

1. Which Core245 SAE features actually drive the behavior-prefill MLP?
2. Can a cheaper no-activation student reranker imitate the behavior-prefill reranker?

### Feature Attribution

The attribution utility loaded the released general behavior-latent checkpoint and reconstructed the SciFact, LegalBench-RAG, and R2MED robust heldout examples. It measured feature contribution with four lenses:

- first-layer input weight magnitude,
- gradient-times-input attribution,
- permutation importance by nDCG@10 drop,
- keep-top-k SAE subset curves.

The key finding is mixed but useful. The MLP relies heavily on dense retrieval metadata, but Core245 still adds real lift.

| Arm | nDCG@10 |
| --- | ---: |
| Raw dense candidate order | 0.3911 |
| Behavior MLP with dense metadata but all SAE features ablated to train means | 0.7164 |
| Behavior MLP with all Core245 SAE features | 0.7773 |

So the released artifact is not a pure activation-only reranker. Dense score and rank metadata provide a strong base inside the MLP. But the activation features still add `+0.0610` nDCG@10 over that dense-metadata-only ablation and are part of the full result.

Feature trimming is plausible but not extreme:

| SAE Features Kept | nDCG@10 |
| ---: | ---: |
| 16 | 0.7399 |
| 32 | 0.7381 |
| 64 | 0.7555 |
| 96 | 0.7629 |
| 128 | 0.7683 |
| 192 | 0.7740 |
| 245 | 0.7773 |

Interpretation: a future retrain could probably target a 96- to 128-feature Core subset, but the current checkpoint does not support trimming down to only a tiny handful of features without losing meaningful ranking quality.

The most influential labeled SAE features were not narrow legal or medical features. They were mostly general answer-binding, task/instruction, relation/discourse, and structured quantity/code features. Top examples included:

| Feature ID | Label | Category |
| --- | --- | --- |
| 9449 | variable assignment values | quantity_math_code |
| 767 | language learning context tokens | relation_discourse |
| 16426 | math problem answer reference | task_instruction / relation_discourse / quantity_math_code |
| 14424 | question answer variables | task_instruction |
| 9373 | problem node answer subtraction | task_instruction |
| 19715 | variable definition in problem context | task_instruction / quantity_math_code |

This supports the behavioral interpretation: the model seems to use telemetry related to answer binding, task framing, discourse structure, and support-like relations, rather than a single obvious domain keyword feature.

### No-Activation Student Reranker

We then trained a student neural reranker to imitate the behavior-prefill teacher without using activation telemetry at inference. The student sees only cheap features:

- dense score,
- dense z-score,
- dense reciprocal/log rank,
- query and candidate length,
- unigram and bigram overlap,
- exact query substring indicators,
- numeric overlap.

The cleanest student was trained with pure teacher distillation:

- listwise KL to match the teacher's per-query candidate distribution,
- margin-MSE to match teacher score differences within each query,
- no relevance-label regularization,
- checkpoint selected by dev teacher KL rather than dev relevance.

Combined heldout results:

| Model | nDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense candidate order | 0.3911 | n/a | n/a |
| Behavior-prefill teacher | 0.7773 | 0.7617 | 0.9096 |
| No-activation student | 0.7848 | 0.7708 | 0.9052 |

The student's `+0.0075` nDCG@10 over the teacher was not significant (`p=0.2681`), so this should be interpreted as parity, not a student win. Against Ettin, however, the student remained strongly ahead on the combined benchmark set: `+0.2054` nDCG@10, p approximately `0.0001`.

Per-dataset nDCG@10:

| Dataset | Teacher | Student |
| --- | ---: | ---: |
| SciFact | 0.8016 | 0.7905 |
| LegalBench-RAG | 0.7510 | 0.7659 |
| R2MED | 0.8616 | 0.8716 |

This result changes the cost story. It suggests that the behavior-prefill reranker may be compressible into a much cheaper student for the same benchmark family and frozen candidate setup. It does not prove that the original activation telemetry was unnecessary. The student learned from telemetry-generated teacher scores; the telemetry system produced the soft ranking policy.

The production implication is important:

- use behavior-prefill telemetry as a high-quality teacher and adjudicator,
- distill its judgments into cheaper students for ordinary traffic,
- reserve runtime telemetry for high-risk, hard, out-of-distribution, or audit-required queries,
- periodically refresh the student from newly scored telemetry examples.

The next validation question is whether this no-activation student transfers to new domains and coding-retrieval tasks as well as the telemetry teacher. Until that is tested, the student is best viewed as a cost-amortization path, not a replacement for the research claim.

Important correction: this artifact is now classified as a tabular surrogate, not the intended "student reranker." It consumes dense score/rank and handcrafted lexical features. That makes it useful as a cost diagnostic, but it is not equivalent to a normal reranker such as Ettin. In candidate groups with appended positives, dense-rank features can become protocol artifacts. The proper student-reranker test is a text-only cross-encoder that consumes only query text and candidate text.

### Text-Only Student Reranker

We corrected the protocol and trained a conventional text-only cross-encoder student:

- backbone: `cross-encoder/ettin-reranker-17m-v1`,
- inputs: query text and candidate text only,
- forbidden inputs: activation telemetry, dense score, dense rank, labels, candidate position, provenance, and handcrafted lexical features,
- teacher: behavior-prefill reranker scores,
- loss: pointwise score MSE, listwise KL, and Margin-MSE over within-query score differences.

Two APPS evaluations were run.

First, we trained the text-only student on non-coding behavior-prefill teacher scores from SciFact, LegalBench-RAG, and R2MED, then evaluated zero-shot on the full APPS heldout slate. This is the cleanest test of whether a conventional reranker can inherit the behavior-prefill APPS advantage without seeing APPS data.

| Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense | 0.0481 | 0.0572 | 0.0866 |
| Ettin | 0.5666 | 0.6418 | 0.8834 |
| Text-only student, non-coding train | 0.2454 | 0.3460 | 0.6818 |
| Behavior-prefill teacher | 0.9236 | 0.9339 | 0.9676 |

This failed as a zero-shot replacement. The student beat dense but was far below Ettin and far below the behavior-prefill teacher.

Second, we asked whether the behavior-prefill APPS policy is distillable within distribution. We split APPS query groups into 2,636 train, 376 dev, and 753 heldout test queries. The text-only student trained on APPS teacher scores and was evaluated once on the APPS test split.

| Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense | 0.0448 | 0.0534 | 0.0810 |
| Ettin | 0.5623 | 0.6370 | 0.8765 |
| Text-only student, APPS-specific | 0.8348 | 0.8695 | 0.9774 |
| Behavior-prefill teacher | 0.9292 | 0.9384 | 0.9681 |

The APPS-specific student decisively beat Ettin: `+0.2326` nDCG@10, p approximately `0.0001`. It still trailed the behavior-prefill teacher by `-0.0689` nDCG@10, also significant.

Interpretation:

- The behavior-prefill APPS signal is not automatically captured by a conventional text-only reranker trained on non-coding data.
- But behavior-prefill can generate domain-specific teacher scores that train a much cheaper text-only reranker which beats Ettin decisively on heldout APPS.
- The production path is therefore not "replace telemetry with a universal student." It is "use telemetry as a high-quality teacher/adjudicator, then distill domain-specific students where enough teacher-scored data exists."

## Cost And Production Implications

The current method is expensive because it requires prefill telemetry for every reranked query/candidate pair. For a query with 16 candidates, the naive version performs 16 zero-token prefill passes:

```text
per query cost ~= K * prefill(Query + Candidate)
```

The original document-only prefill ingestion is not required for this winning method. Offline ingestion only needs to chunk, embed, and index documents. The behavior-latent reranker must see the query and candidate together, because the signal is interaction-dependent.

This means the method is not a drop-in replacement for cheap dense retrieval. It is a premium reranking stage.

## Ideal Production System

The best production version should be built as a high-precision evidence adjudication tier.

### Offline

1. Ingest documents.
2. Split into stable chunks.
3. Build dense embeddings.
4. Build lexical indexes.
5. Optionally build late-interaction indexes.
6. Store text, metadata, corpus version, and chunk hashes.

No doc-only activation prefill is required for the winning path.

### Online

1. Receive query.
2. Retrieve a broad candidate pool with dense plus lexical search.
3. Apply cheap domain filters and metadata constraints.
4. Use a cheap neural reranker, such as Ettin or a domain cross-encoder, to reduce candidates.
5. Use an uncertainty/risk gate to decide whether behavior telemetry is needed.
6. Run behavior-latent telemetry only on the top 8 to 20 hard candidates.
7. Select final evidence.
8. Generate an answer with citations.
9. Store a replayable audit trail.

### Cost Mitigation

The production system should use:

- Cascading retrieval to reduce candidate count before telemetry.
- Confidence gating so easy queries skip behavior scoring.
- Prefix caching only after paired equivalence tests prove that cached execution reproduces the uncached layer-7 SAE vector. Our first implementation produced cache hits but changed the feature vector, so it is not production-safe yet.
- A dedicated prefill-only scoring service with no decode path.
- Continuous batching and paged KV cache for high GPU utilization.
- A truncated telemetry model that stops at layer 7 rather than running the full model.
- Fused SAE extraction and compiled inference kernels.
- Pair-score caching keyed by query hash, candidate chunk hash, prompt template, model version, feature manifest, and corpus version.
- Student distillation so a smaller reranker handles ordinary traffic while the expensive behavior scorer handles hard/high-risk cases and labels new training data.

Modern serving stacks already support several relevant primitives. vLLM is built around PagedAttention, continuous batching, chunked prefill, prefix caching, and CUDA/HIP graph execution; SGLang's RadixAttention keeps reusable KV cache entries in a radix tree; and Hugging Face's cache guidance covers static KV caches that can make compiled inference practical. The caution for this project is that telemetry is not ordinary answer generation: any serving optimization must reproduce the exact hidden-state/SAE feature contract, not merely produce equivalent next-token logits. References: [vLLM docs](https://docs.vllm.ai/), [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/), and [Hugging Face KV cache docs](https://huggingface.co/docs/transformers/en/kv_cache).

### Maximum-Speed Production Design

The fastest credible production design is not "run the current Python script faster." It is a dedicated resident telemetry service with a strict equivalence gate.

The service should keep the Qwen/RMT/SAE stack resident on GPU and expose one endpoint: score canonical query+candidate prompts with zero output tokens. The request should include query ID, candidate IDs, prompt-template hash, model/feature-manifest versions, corpus version, and risk tier. The service should return the compact Core245 feature vector plus the MLP score, not raw hidden states.

The serving path should be:

1. Dense/lexical retrieval produces a broad candidate pool.
2. Ettin or a cheaper domain reranker reduces the slate to the hardest candidates.
3. A risk/uncertainty gate decides whether behavior telemetry is worth the cost.
4. The telemetry service scores only the remaining hard candidates.
5. Pair scores are cached under a versioned key: normalized query hash, candidate chunk hash, prompt-template hash, model checkpoint, SAE manifest, and corpus version.
6. Accepted telemetry scores are logged as distillation data for future cheap students.

The runtime should prioritize these optimizations in order:

1. **Layer truncation:** stop immediately after layer 7. This is already validated and exactly preserves the signal.
2. **Resident workers:** avoid per-request model startup. Our benchmark wrapper's wall time is dominated by loading; production should price the in-process timing.
3. **Candidate count reduction:** reduce from 100 dense candidates to 8-20 hard candidates before telemetry whenever policy permits.
4. **Compiled/fused layer-0-through-layer-7 path:** export or hand-build the exact prefix of Qwen needed for `l07_resid_pre`, then fuse SAE normalization/top-k/max-pooling so the service never materializes or serializes unnecessary tensors.
5. **Batch-invariant batching:** implement batching only if an equivalence suite proves that batched execution yields the same Core245 feature vector as single-prompt execution. Our naive HF batched run changed the selected SAE vector, including duplicate-prompt batches, so default production must not use non-exact batching.
6. **Exact prefix reuse:** implement KV/prefix reuse only if it reproduces the uncached layer-7 activations. Our first query-prefix cache changed feature values, so this is future work, not a current optimization.
7. **Multi-GPU data parallelism:** if exact batching remains hard, replicate resident telemetry workers across GPUs and shard candidates by query. This is less elegant than batching but preserves single-prompt semantics.
8. **Student cascade:** train domain-specific text-only students from telemetry teacher scores so most traffic avoids telemetry entirely. Keep telemetry for high-stakes/hard cases and for generating fresh training labels.

Approximate cost envelope from the current Vicuna measurements:

- Full Qwen forward over 100 candidates: about `2.7s` in-process.
- Valid layer-7 early-stop over 100 candidates: about `0.8s` in-process.
- Naive true batch-8 looked faster in a microbenchmark, but failed feature-equivalence checks and is not acceptable as a production default.
- With a resident service, candidate gating to 8-20 candidates is more valuable than unsafe batching: it moves behavior telemetry from roughly `0.8s/100 candidates` to roughly `60-160ms` of layer-7 prefill compute at the same measured throughput, before service overhead.

The engineering target is therefore:

```text
dense/lexical retrieval -> cheap reranker -> risk gate -> exact layer-7 resident telemetry on 8-20 candidates -> final evidence
```

Under that design, behavior-prefill reranking is not a universal high-QPS reranker. It is a high-precision adjudication tier for cases where the extra 100-300ms can buy materially better evidence selection.

### Where It Matters

This system is most valuable when wrong evidence is expensive:

- legal research and contract review,
- clinical or biomedical evidence retrieval,
- compliance and policy interpretation,
- scientific literature review,
- financial/regulatory diligence,
- engineering incident investigation,
- agentic coding and repository/tool/documentation retrieval,
- safety-critical documentation search.

It is probably not worth the cost for generic FAQ search, casual chat memory, or low-value high-QPS consumer retrieval. In those settings, dense retrieval plus a cheap reranker is usually the right tradeoff.

## Bottom Line

The experiment supports a strong, specific claim:

> Query+candidate behavior-latent telemetry turns activation capture from a weak retrieval geometry into a powerful evidence-support signal, producing large paired-significant wins over dense retrieval and the strongest observed off-the-shelf text-reranker envelope on LegalBench-RAG and R2MED that survive strict duplicate-evidence due diligence, with SciFact remaining a supportive but not primary controlled result.

The new coding-transfer suite strengthens the forward-looking claim:

> The same released general behavior-latent reranker also transfers zero-shot to a bounded five-task CoIR coding retrieval suite, beating Ettin in aggregate nDCG and producing a very large paired-significant APPS win, suggesting that query+candidate activation telemetry may be valuable for agentic coding systems where exact evidence selection among near-duplicate code snippets is a core bottleneck, while still requiring task-specific training and duplicate-aware evaluation before broad coding claims.

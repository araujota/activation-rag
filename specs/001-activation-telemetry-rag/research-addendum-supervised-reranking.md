# Research Addendum: Supervised Activation-Aware Reranking

Date: 2026-06-12

## Trigger

Corrected prefill capture and anisotropy-aware activation similarity did not produce a usable activation-only retrieval space. The only positive signal observed was a small dense-score blend, and it was not statistically decisive on SciFact. The next hypothesis is that activation telemetry is useful as supervised evidence inside a dense-candidate reranker, not as unsupervised KNN.

## Baselines

Every supervised activation-aware result must be compared against:

- dense-only candidate ranking from the same embedding model and same candidate pool;
- a normal text reranker over the same candidates, initially a cross-encoder reranker from Sentence Transformers;
- dense plus the best unsupervised activation blend as a weak non-supervised reference.

The target is not "better than dense by a tiny amount"; it is a decisive win over the normal text reranker on paired query-level metrics.

## Training Unit

Training examples are query groups, not independent pairs:

- one query activation record;
- top-N dense candidate chunks;
- qrel positive candidates;
- explicit qrel negative candidates when a benchmark provides zero-grade judgments;
- mined hard negatives from dense candidates when explicit negatives are unavailable;
- feature provenance for every candidate.

SciFact BEIR qrels in the local copy contain only positive labels. Mined dense negatives must therefore be labeled as `dense_hard_negative` with a false-negative policy, not as ground-truth negatives.

## Loss Functions

First training pass:

- pairwise logistic ranking loss over positive-vs-negative candidates from the same query group;
- L2 regularization;
- query-level feature standardization or train-split feature standardization only;
- early stopping on dev nDCG@10.

Second pass if the pairwise model shows signal:

- listwise softmax/ListNet-style loss over each dense candidate group;
- optional LambdaRank-style weighting by nDCG delta;
- hard-negative depth sampling so training does not overfit only the easiest top candidates.

Fallback if supervised reranking fails:

- train a nonlinear MLP to predict an answer-bearing activation representation from query activation, but only after the supervised reranker fails against the text-reranker baseline under the same dense candidates.

## Feature Families

Activation-aware candidate features should include:

- dense score and dense rank features;
- raw activation cosine;
- activation L2 distance and absolute-difference summary statistics;
- activation elementwise-product summary statistics;
- per-site late-fusion cosine features for SAE feature names shaped like `act:{site}:{window}:{stat}`;
- optional CSLS/NICDM-derived candidate scores when full query and candidate matrices are available.

The model must record the exact feature list and normalization statistics with every run.

## Scaled Vertical Training Regime

The first legal, medical, and code vertical runs were too small or too structurally saturated to explain the SciFact anomaly. The next regime treats each vertical as a pooled set of same-shaped query-to-evidence-passage retrieval problems, while preserving source labels for per-dataset reporting.

Candidate larger sources:

- Legal: LegalBench-RAG, because it provides exact character-level gold snippets over contracts and privacy policies. The adapter should convert snippets into span-centered evidence windows and keep the exact snippet span in provenance.
- Medical: all eight R2MED retrieval tasks, because the benchmark is already BEIR-like with query, corpus, and qrels files and covers biology, bioinformatics, medical science, exams, diagnosis, treatment, clinical evidence, and long clinical cases.
- Code: CoIR/MTEB code retrieval tasks, because CoIR uses BEIR/MTEB-compatible retrieval schemas over code/text pairs and should avoid the saturated CoREB reranking condition where the positive was already rank 1.

Calibration and regularization changes:

- generate out-of-fold train scores for activation-aware direct-blend models by training fold models on query-disjoint train folds and scoring the held-out fold;
- select the final artifact only on the dev split, then evaluate the locked test split once;
- add dropout, input noise, label smoothing, gradient clipping, weight decay, and early stopping so direct-blend training cannot memorize small vertical train sets as easily;
- keep alpha selection explicit and compare fixed-alpha grids against learned calibration only when out-of-fold scores exist.

Candidate movement diagnosis is now mandatory for serious vertical runs. The failed first legal and medical rerankers mostly promoted near-topic but non-answer-bearing passages over dense positives. That points toward calibration/false-negative pressure and insufficient supervised diversity, not toward random activation noise alone.

## STREAM And Sparse-Latent RAG Guidance

Added: 2026-06-15

Two related papers matter for the next decision point:

- Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention (`https://arxiv.org/abs/2510.19875`). This is not a retrieval reranker, but it is relevant to our capture design. Stream uses sparse tracing to preserve salient attention pathways in long contexts and shows needle retrieval paths can survive aggressive attention pruning. If our chunk-level prefill telemetry keeps failing, this argues for moving away from pooled per-chunk vectors and toward prompt-packed, section-aligned attention/flow diagnostics over multi-document contexts.
- Sparse Latents Steer Retrieval-Augmented Generation (`https://aclanthology.org/2025.acl-long.228/`). This is directly parallel to our SAE/CAA direction. It identifies SAE latents correlated with RAG behaviors such as context-following versus memory-based responses and refusal versus non-refusal, using separation in latent activation frequency between target and baseline behavior groups. The key lesson is that the useful supervision target may not be "which chunk is semantically closest?" but "which internal behavior mode does this query/context pair induce?"

Implications for this project:

- Our current actpred reranker still treats SAE telemetry as a query-to-answer representation matching problem. Sparse-Latent RAG suggests a different target: learn behavior latents that predict context reliance, refusal/non-refusal, and near-topic distraction, then use those as reranker/control features.
- The final input token position matters. Sparse-Latent RAG focuses on residual-stream SAE activations at the position before the first answer token. Our isolated chunk prefill capture is cheaper, but it may miss the decision boundary that appears only when query and candidate evidence are jointly presented.
- Middle-layer latent selection deserves renewed attention. Their findings point to middle layers for context-following/refusal decisions, while our core245 layer-7 capture may be too early or too selector-specific for RAG behavior control.
- If the scaled LegalBench/R2MED rerankers fail, the next serious ablation should not be another raw similarity objective. It should build query+candidate prompt pairs, capture final-input-token SAE activations, label whether the candidate supports context-following answer generation versus near-topic distraction, and test whether those behavior latents rerank dense candidates better than the answer-representation predictor.

## Scaled LegalBench-RAG And R2MED Cross-Fit Results

Run date: 2026-06-15

Artifacts:

- LegalBench-RAG groups: `runs/supervised/verticals/legalbenchrag/*-groups.k100.bge.appendpos.jsonl`
- LegalBench-RAG telemetry cache: `runs/telemetry-cache/legalbenchrag-qwen-core245-20260615`
- LegalBench-RAG cross-fit run: `runs/supervised/verticals/crossfit/legalbenchrag-scaled`
- R2MED pooled groups: `runs/supervised/verticals/r2med-all/*-groups.k100.appendpos.jsonl`
- R2MED telemetry cache: `runs/telemetry-cache/r2med-all-qwen-core245-20260615`
- R2MED cross-fit run: `runs/supervised/verticals/crossfit/r2med-all-scaled`

Data:

| Vertical | Train Groups | Dev Groups | Test Groups | Telemetry Records |
| --- | ---: | ---: | ---: | ---: |
| LegalBench-RAG | 5,511 | 689 | 689 | 16,821 |
| R2MED pooled | 614 | 131 | 131 | 48,070 |

LegalBench-RAG cross-fit behavior:

| Fold | Selected Alpha | Best Epoch | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4 | 4 | 0.2268 | 0.2332 | 0.3413 | 0.3424 |
| 1 | 0.4 | 4 | 0.2379 | 0.2491 | 0.3665 | 0.3753 |
| 2 | 0.3 | 4 | 0.2460 | 0.2600 | 0.3851 | 0.3898 |
| 3 | 0.3 | 5 | 0.2445 | 0.2594 | 0.3775 | 0.3942 |
| 4 | 0.4 | 3 | 0.2322 | 0.2402 | 0.3597 | 0.3727 |

LegalBench-RAG final locked split:

| Split | Selected Alpha | Dense MRR@10 | Model MRR@10 | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.4 | 0.2318 | 0.2522 | 0.2493 | 0.2710 | 0.3917 | 0.4061 |
| test | 0.4 | 0.2456 | 0.2565 | 0.2728 | 0.2791 | 0.4184 | 0.4290 |

LegalBench-RAG locked test comparison against dense plus Ettin:

Artifacts:

- Candidate groups: `runs/supervised/verticals/legalbenchrag/test-groups.k100.bge.appendpos.jsonl`
- Actpred scores: `runs/supervised/verticals/crossfit/legalbenchrag-scaled/final/test-scores.jsonl`
- Ettin scores: `runs/supervised/verticals/ettin-baselines/legalbenchrag/test-ettin-reranker-150m-v1.scores.jsonl`
- Ettin metrics: `runs/supervised/verticals/ettin-baselines/legalbenchrag/test-ettin-reranker-150m-v1.metrics.json`
- Paired audits: `runs/supervised/verticals/ettin-baselines/legalbenchrag/audit-dense-vs-ettin.json`, `audit-dense-vs-actpred.json`, and `audit-actpred-vs-ettin.json`

| Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Pure dense candidates | 0.2456 | 0.2728 | 0.4184 |
| Dense + LegalBench-RAG actpred direct blend | 0.2565 | 0.2791 | 0.4290 |
| Dense + Ettin reranker | 0.3699 | 0.3905 | 0.5277 |

Paired randomization audits over 689 test queries:

| Comparison | Delta MRR@10 | Delta nDCG@10 | Delta Recall@10 | nDCG p-value | Changed-query summary |
| --- | ---: | ---: | ---: | ---: | --- |
| Actpred minus dense | +0.0110 | +0.0063 | +0.0106 | 0.3431 | 148 improved / 135 harmed / 406 unchanged |
| Ettin minus dense | +0.1244 | +0.1177 | +0.1094 | 0.0001 | 275 improved / 100 harmed / 314 unchanged |
| Ettin minus actpred | +0.1134 | +0.1114 | +0.0988 | 0.0001 | 278 improved / 111 harmed / 300 unchanged |

R2MED pooled cross-fit behavior:

| Fold | Selected Alpha | Best Epoch | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4 | 40 | 0.1595 | 0.1607 | 0.1660 | 0.1660 |
| 1 | 0.1 | 6 | 0.1281 | 0.1324 | 0.1683 | 0.1724 |
| 2 | 0.1 | 3 | 0.1621 | 0.1386 | 0.2113 | 0.1817 |
| 3 | 0.2 | 4 | 0.1324 | 0.1301 | 0.1611 | 0.1687 |
| 4 | 0.1 | 3 | 0.1581 | 0.1506 | 0.1957 | 0.1747 |

R2MED pooled final locked split:

| Split | Selected Alpha | Dense MRR@10 | Model MRR@10 | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.4 | 0.2644 | 0.2919 | 0.1995 | 0.2127 | 0.2580 | 0.2709 |
| test | 0.4 | 0.2146 | 0.2044 | 0.1661 | 0.1532 | 0.2062 | 0.1929 |

Candidate movement diagnosis:

- LegalBench-RAG test changed 283 of 689 queries: 148 improved, 135 harmed, and 406 unchanged. Mean nDCG@10 delta was `+0.0063`. The improvement set includes cases where answer-bearing clauses moved from just outside top-10 to rank 1. The harmed set is dominated by legally adjacent boilerplate, especially clauses that share the same contract topic but not the exact queried obligation.
- R2MED test changed 38 of 131 queries: 18 improved, 20 harmed, and 93 unchanged. Mean nDCG@10 delta was `-0.0129`. The model can promote answer-bearing medical passages in some cases, but it also overpromotes near-topic clinical/scientific text and does not recover enough positives to generalize.

Interpretation:

- The LegalBench-RAG result is the first scaled run where every out-of-fold train fold moves positive and the final test split also improves dense MRR, nDCG, and recall. However, the dense-plus-Ettin comparison materially changes the claim: actpred is a narrow, statistically non-significant improvement over dense on this split, while Ettin is a large and statistically decisive improvement over both dense and actpred.
- R2MED remains negative on locked test despite a positive dev split. This argues against a universal activation-aware reranker and supports the dataset-specific/domain-specific hyper-reranker framing.
- The failure mode is not pure noise. Both verticals show meaningful candidate movement. The current model is learning a near-topic/answer-bearing direction that is helpful for exact legal snippet retrieval and harmful or undercalibrated for pooled medical retrieval.
- LegalBench-RAG no longer supports a strong claim that the current activation reranker beats a strong text reranker. It remains useful evidence that SAE-only actpred can move legal snippets in the right direction over dense retrieval, but the publishable competitive reranking claim is still concentrated in SciFact.
- The next implementation step should complete any remaining dense+Ettin comparisons before making broader vertical claims. If publishing now, frame LegalBench-RAG as a scaled diagnostic/narrow dense lift, not as a win over text reranking.

## Deep Semantic And Representation Diagnostics

Run date: 2026-06-17

Artifacts:

- Diagnostic script: `scripts/deep_activation_reranker_diagnostics.py`
- Unit tests: `tests/test_deep_activation_reranker_diagnostics.py`
- Diagnostic report: `runs/supervised/verticals/deep-activation-reranker-diagnostics-20260617.json`
- External protocol anchor: BEIR's reranking example evaluates a cross-encoder over fixed top-k retrieval results before computing standard IR metrics (`https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py`).
- External modeling anchor: Sparse-Latent RAG (`https://aclanthology.org/2025.acl-long.228/`) uses SAE latents to characterize RAG behavior modes, while Stream (`https://arxiv.org/abs/2510.19875`) motivates inspecting long-context information flow rather than only pooled chunk vectors.

The diagnostic compares SciFact, LegalBench-RAG, R2MED pooled, and CoIR/CosQA on:

- first-stage candidate hardness;
- within-query positive-vs-negative score separation for dense, actpred, and Ettin where available;
- dense-to-reranker first-positive rank movement;
- source/span overlap between positives and dense negatives;
- query and passage length distributions;
- representative improved/harmed examples with snippets.

Cross-dataset summary:

| Dataset | Queries | Dense positive@1 | Dense positive@10 | Actpred nDCG delta | Actpred pairwise AUC | Ettin nDCG delta | Ettin pairwise AUC | Train nDCG delta | Dev nDCG delta | Test nDCG delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 300 | 0.610 | 0.877 | +0.0779 | 0.962 | +0.0186 | 0.962 | +0.2200 | +0.0650 | +0.0805 |
| LegalBench-RAG | 689 | 0.151 | 0.479 | +0.0063 | 0.588 | +0.1177 | 0.785 | +0.0924 | +0.0217 | +0.0063 |
| R2MED pooled | 131 | 0.145 | 0.366 | -0.0129 | 0.358 | n/a | n/a | +0.3461 | +0.0132 | -0.0129 |
| CoIR/CosQA | 500 | 0.136 | 0.568 | -0.0021 | 0.786 | n/a | n/a | +0.0654 | +0.0090 | -0.0021 |

Key findings:

- SciFact is not merely a lucky nDCG artifact. The activation predictor separates positives from negatives inside each query group almost as well as Ettin: mean within-query pairwise AUC `0.962` for actpred and `0.962` for Ettin. It lifts true biomedical evidence abstracts from ranks like 13, 14, 20, and 29 to rank 1 in multiple high-delta examples.
- LegalBench-RAG fails against Ettin because the actpred representation has weak positive/negative separation, not because the candidate pool is impossible. Actpred AUC is only `0.588`, while Ettin AUC is `0.785` on the same groups. The semantic examples show actpred often promotes adjacent contract boilerplate or nearby clause text, while Ettin better recognizes exact legal obligations, exceptions, and dates.
- R2MED is the clearest model-target failure. Actpred is inverted on heldout positives: mean pairwise AUC `0.358`, positive score mean below negative score mean, and positive z-mean below negative z-mean. The model learned something on train/dev but it is not a stable medical-evidence direction.
- CoIR/CosQA has a different failure. Actpred pairwise AUC is reasonably high (`0.786`), close to dense (`0.777`), but final nDCG is slightly negative. Candidate movement examples show many benchmark-specific duplicate/near-duplicate code snippets: the reranker can promote a semantically equivalent function that is unlabeled, which is harmful under exact qrel IDs.
- Core245 feature-label pruning did not solve the issue. The feature diagnostic had zero stable useful individual features for LegalBench-RAG and CoIR, and only one for R2MED. Subset32 ablations preserved a tiny legal lift, reduced harm on R2MED/CoIR, but did not produce positive heldout wins. The problem is therefore not just "too many noisy Core245 labels."

Root-cause hypothesis after this pass:

The SciFact win appears to come from an activation-space direction that is genuinely aligned with scientific claim evidence at the abstract level. The same isolated query-to-answer-chunk representation target does not capture exact legal span selection, medical answer support, or code duplicate identity. In those domains, relevance depends on query-candidate interaction details that are erased by separate prefill pooling:

- Legal: exact clause boundaries and obligation wording matter more than broad semantic clause type.
- R2MED: long clinical/user questions and multi-task medical pools induce unstable or inverted answer-bearing directions.
- CoIR: semantic equivalence and benchmark qrel identity diverge because many snippets are duplicates or near-duplicates.

Training-regime diagnosis:

- The current MLP can strongly memorize or fit train groups. R2MED train nDCG delta was `+0.3461` while heldout test was `-0.0129`; LegalBench-RAG train was `+0.0924` while test was `+0.0063`; SciFact train was `+0.2200` and still test-positive.
- SciFact is the only setting where train/dev/test movement all remain strongly positive and score separation remains high on heldout queries.
- Cross-fitting reduces in-sample leakage, but it does not fix a mismatched supervision target. The current target asks "what answer-bearing chunk activation should this query map to?" while the failed examples require "does this query-candidate pair support the answer under this exact benchmark label?"

Implications:

- Do not publish a broad activation-reranker generalization claim from these runs.
- The strongest defensible result remains: SAE-only Core245 activation-aware direct-blend reranking decisively beats dense and Ettin on SciFact.
- LegalBench-RAG is a weak positive diagnostic over dense but a clear loss to Ettin; report it only as evidence that the activation signal can move legal candidates, not as competitive legal reranking.
- The next credible research direction is not another isolated-chunk actpred retrain. It is a query+candidate behavior-latent objective: prefill a canonical `Q + candidate evidence` prompt, capture final-input-token SAE/CAA telemetry, and train labels for answer support versus near-topic distraction. This aligns better with Sparse-Latent RAG's behavior-mode framing and with the examples where actpred confuses adjacent/equivalent text.

## Query+Candidate Behavior-Latent Pilot

Run date: 2026-06-17

Artifacts:

- Preparation script: `scripts/prepare_behavior_latent_reranker_pilot.py`
- Cache materialization script: `scripts/materialize_behavior_telemetry_cache.py`
- Training script: `scripts/train_behavior_latent_reranker.py`
- Comparison script: `scripts/compare_behavior_latent_pilot.py`
- Unit tests: `tests/test_behavior_latent_reranker_pilot.py`
- Run root: `runs/behavior-latent-pilot-20260617/`

Design:

- Prompt template:

```text
Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:
```

- Stable pair IDs are keyed by dataset, query id, original candidate chunk id, and prompt-template hash.
- The support model is a pointwise MLP trained with a query-group softmax/listwise ranking objective over dense-candidate groups.
- The selected score is a dev-chosen blend:

```text
final_score = (1 - alpha) * z_dense_score + alpha * z_behavior_support_score
```

- The pilot used 160 train, 60 dev, and 100 test queries per dataset, with 8 candidates per query. Candidate positives were appended where needed, so the pilot measures reranking quality over bounded candidate groups rather than end-to-end first-stage retrieval.
- This run is `query_candidate_pair_core245_max_prefill`. It uses the available Qwen/RMT/SAE Core245 selected-feature capture over the full pair prompt, pooled by max over prefill tokens. It is not true final-input-token SAE/CAA behavior telemetry.

Telemetry capture:

| Dataset | Train Queries | Dev Queries | Test Queries | Pair Requests |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 160 | 60 | 100 | 2,560 |
| LegalBench-RAG | 160 | 60 | 100 | 2,566 |
| R2MED pooled | 160 | 60 | 100 | 2,580 |
| Total | 480 | 180 | 300 | 7,706 |

The merged capture completed with `7,706` valid rows in `217.98` seconds. Mean active Core245 features per pair prompt was `58.69`. All rows used the `query_candidate_behavior_prompt` section label and were materialized to `runs/behavior-latent-pilot-20260617/telemetry-cache`.

Training summary:

| Model | Train Scope | Dev Scope | Test Scope | Selected Alpha | Best Epoch | Test Dense nDCG@10 | Test Behavior nDCG@10 | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| General behavior | SciFact + LegalBench-RAG + R2MED | mixed | mixed | 0.5 | 30 | 0.6168 | 0.6888 | +0.0720 |
| SciFact-specific behavior | SciFact | SciFact | SciFact | 0.4 | 9 | 0.8238 | 0.8271 | +0.0033 |
| LegalBench-RAG-specific behavior | LegalBench-RAG | LegalBench-RAG | LegalBench-RAG | 0.5 | 33 | 0.4809 | 0.5643 | +0.0834 |
| R2MED-specific behavior | R2MED | R2MED | R2MED | 0.5 | 22 | 0.5456 | 0.7132 | +0.1675 |

Per-dataset comparisons:

| Dataset / Behavior Model | Dense nDCG@10 | Existing Actpred nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 | Behavior vs Dense | Behavior vs Actpred | Behavior vs Ettin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact / general | 0.8238 | 0.8838 | 0.8776 | 0.8081 | -0.0157, p=0.4509 | -0.0757, p=0.0018 | -0.0695, p=0.0058 |
| SciFact / specific | 0.8238 | 0.8838 | 0.8776 | 0.8271 | +0.0033, p=0.8243 | -0.0567, p=0.0140 | -0.0505, p=0.0359 |
| LegalBench-RAG / general | 0.4809 | 0.5040 | 0.6087 | 0.5503 | +0.0694, p=0.0035 | +0.0463, p=0.0581 | -0.0584, p=0.0204 |
| LegalBench-RAG / specific | 0.4809 | 0.5040 | 0.6087 | 0.5643 | +0.0834, p=0.0001 | +0.0603, p=0.0145 | -0.0445, p=0.0976 |
| R2MED / general | 0.5456 | 0.5464 | n/a | 0.7079 | +0.1623, p=0.0001 | +0.1616, p=0.0001 | n/a |
| R2MED / specific | 0.5456 | 0.5464 | n/a | 0.7132 | +0.1675, p=0.0001 | +0.1668, p=0.0001 | n/a |

Interpretation:

- This objective is worth pursuing further. It directly targets the failure mode identified by the deep diagnostics: isolated query-to-answer-chunk activation prediction erases query-candidate interaction details, while pair-prompt telemetry can learn "answer support versus near-topic distraction."
- The pilot does not improve SciFact over the existing answer-bearing activation predictor or Ettin. SciFact remains the domain where isolated actpred is strongest; replacing it with pair-prompt behavior telemetry is not justified by this pilot.
- LegalBench-RAG becomes materially stronger under the behavior objective. The dataset-specific model beats dense by `+0.0834` nDCG@10 and beats existing actpred by `+0.0603` nDCG@10; Ettin remains numerically ahead, but its `+0.0445` nDCG@10 advantage over behavior was not significant on this 100-query pilot.
- R2MED is the clearest directional win. Existing actpred was flat against dense, while pair-prompt behavior reranking improved nDCG@10 by `+0.1675` over dense and `+0.1668` over actpred on the pilot subset.
- The general mixed model transfers reasonably to LegalBench-RAG and R2MED, but dataset-specific training is still slightly better. That suggests a shared pair-prompt answer-support signal exists, with domain calibration still useful.

Limitations and next gate:

- These are pilot reranking results over small, positive-appended, 8-candidate groups. They are not full benchmark-standard retrieval results.
- The capture is max-pooled Core245 telemetry over the whole pair prompt. It is not the final-input-token behavior telemetry recommended by Sparse-Latent RAG-style reasoning and by our own failure analysis.
- The next serious implementation should add true final-input-token SAE/CAA capture for the exact same Q+candidate support prompt, rerun the pilot at larger candidate depth, and then repeat locked full-split comparisons against dense, dense+Ettin, existing actpred, and pair-prompt behavior.

## Behavior-Latent Robust Training Plan And Transform Sanity Check

Run date: 2026-06-18

Research/implementation anchors:

- BEIR reranking examples use the standard two-stage pattern: retrieve a candidate set, rerank top-k candidates, then evaluate nDCG/MAP/Recall over qrels (`https://github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py`).
- Sparse Latents Steer Retrieval-Augmented Generation motivates behavior-latent targets rather than raw chunk-similarity targets for RAG behavior (`https://aclanthology.org/2025.acl-long.228/`).
- Hard-negative training can be harmed by ambiguous or false negatives; bounded dense-hard-negative slates, label smoothing, and heldout-only reporting are therefore mandatory for this objective (`https://arxiv.org/html/2604.11092v1`).
- Recent dense-retrieval/reranking work continues to support listwise distillation or listwise reranking objectives when the evaluation target is ranking quality over a candidate group (`https://arxiv.org/html/2505.19274v1`).
- GitHub implementation reference: `ielab/llm-rankers` pointwise reranking uses a query-document interaction prompt as the scoring unit (`https://github.com/ielab/llm-rankers/blob/b36517bf17a3956dc56c4c967f972d02390b1cdd/llmrankers/pointwise.py`).

Design changes made before the larger run:

- Added `scripts/run_resumable_behavior_capture.py`, a batch capture wrapper that skips already materialized cache rows and writes per-batch artifacts. This is required because the robust request set is large enough that SSH failures cannot force a full restart.
- Extended `scripts/train_behavior_latent_reranker.py` with activation feature transforms: `raw`, `log1p`, `log1p_l2`, and `binary`.
- Extended the trainer to record every alpha in the dev sweep, not just the selected alpha. This preserves behavior-only diagnostics when `alpha=1.0` is in the grid.
- Added gradient clipping so larger full-split runs cannot be derailed by sparse-feature outlier gradients.

Prepared full-split request set:

| Dataset | Train Groups | Dev Groups | Test Groups | Candidates / Query | Pair Requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 676 | 117 | 291 | 16 | 17,344 |
| LegalBench-RAG | 5,511 | 689 | 689 | 16 | 110,228 |
| R2MED pooled | 614 | 131 | 131 | 16 | 14,020 |
| Total | 6,801 | 937 | 1,111 | 16 | 141,592 |

Artifacts:

- Robust request root: `runs/behavior-latent-robust-20260618/`
- Merged capture requests: `runs/behavior-latent-robust-20260618/all-capture-requests.jsonl`
- Planned resumable capture: 35 batches of 4,096 requests into `runs/behavior-latent-robust-20260618/telemetry-cache`

Blocker:

- The robust full-split capture could not start because `vicuna-host` was unreachable over Tailscale/SSH. Diagnostics showed `ssh: connect to host vicuna-host port 22: Operation timed out`, `ping` had 100% packet loss, and `tailscale status` reported `vicuna-host` as offline, last seen roughly five hours earlier.
- No robust full-split telemetry rows were captured yet. The prepared request set and resumable wrapper are ready to run once vicuna is reachable again.

Pilot-cache transform ablation:

While blocked on vicuna reachability, the existing 8-candidate pilot cache was used to test the stronger trainer with `activation_transform=log1p_l2`, hidden size 128, stronger regularization, and alpha grid `{0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}`.

| Dataset | Selected Alpha | Dense nDCG@10 | Raw Behavior nDCG@10 | log1p/L2 Behavior nDCG@10 | Actpred nDCG@10 | Ettin nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.3 | 0.8238 | 0.8271 | 0.8298 | 0.8838 | 0.8776 |
| LegalBench-RAG | 0.7 | 0.4809 | 0.5643 | 0.7850 | 0.5040 | 0.6087 |
| R2MED pooled | 0.7 | 0.5456 | 0.7132 | 0.9091 | 0.5464 | n/a |

Paired pilot-cache comparison:

| Dataset | Comparison | Delta nDCG@10 | p-value |
| --- | --- | ---: | ---: |
| SciFact | log1p/L2 behavior minus dense | +0.0060 | 0.5180 |
| SciFact | actpred minus log1p/L2 behavior | +0.0540 | 0.0071 |
| SciFact | Ettin minus log1p/L2 behavior | +0.0477 | 0.0262 |
| LegalBench-RAG | log1p/L2 behavior minus dense | +0.3041 | 0.0001 |
| LegalBench-RAG | log1p/L2 behavior minus actpred | +0.2810 | 0.0001 |
| LegalBench-RAG | log1p/L2 behavior minus Ettin | +0.1762 | 0.0004 |
| R2MED pooled | log1p/L2 behavior minus dense | +0.3635 | 0.0001 |
| R2MED pooled | log1p/L2 behavior minus actpred | +0.3627 | 0.0001 |

Interpretation:

- The sparse activation geometry matters. Raw max-pooled Core245 magnitudes were leaving substantial signal on the table; log1p compression plus per-row L2 normalization is a much better default for the behavior-prompt objective.
- The higher selected alpha values on LegalBench-RAG and R2MED mean the behavior signal is no longer merely a tiny dense-score perturbation on the pilot cache. On R2MED, alpha `1.0` also performed extremely well, but alpha `0.7` was selected on dev and slightly better on test.
- SciFact still does not overtake the existing answer-bearing actpred reranker under the behavior-prompt objective. It improves slightly over dense, but actpred and Ettin remain significantly ahead on the pilot subset. Full-split log1p/L2 behavior capture is still worth running, but SciFact overtaking actpred is not yet supported by evidence.
- The LegalBench/R2MED pilot gains are large enough that the robust full-split capture should proceed as soon as vicuna is reachable. If the gains survive the larger 16-candidate/full-split run, the query+candidate behavior-latent method becomes the strongest direction in the project.

## Scaled CoIR Code Retrieval Result

Run date: 2026-06-16

Artifacts:

- CoIR CosQA groups: `runs/supervised/verticals/coir-cosqa/*-groups.k100.bge.appendpos.jsonl`
- CoIR CosQA dense cache: `runs/supervised/verticals/coir-cosqa/coir-cosqa-bge-base-dense-cache.npz`
- CoIR CosQA telemetry cache: `runs/telemetry-cache/coir-cosqa-qwen-core245-20260616`
- CoIR CosQA cross-fit run: `runs/supervised/verticals/crossfit/coir-cosqa-scaled`

Data:

- train groups: 19,604
- dev groups: 500
- test groups: 500
- telemetry records captured: 41,206
- train positives in dense top-100 before appended positives: 9,020 of 19,604
- dev/test positives in candidate pool after appended positives: 500 of 500 each

CoIR CosQA cross-fit behavior:

| Fold | Selected Alpha | Best Epoch | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2 | 6 | 0.3162 | 0.3131 | 0.5648 | 0.5626 |
| 1 | 0.1 | 6 | 0.3176 | 0.3221 | 0.5578 | 0.5612 |
| 2 | 0.3 | 3 | 0.3122 | 0.3132 | 0.5556 | 0.5578 |
| 3 | 0.3 | 4 | 0.3158 | 0.3036 | 0.5649 | 0.5455 |
| 4 | 0.2 | 4 | 0.3235 | 0.3227 | 0.5711 | 0.5750 |

CoIR CosQA final locked split:

| Split | Selected Alpha | Dense MRR@10 | Model MRR@10 | Dense nDCG@10 | Model nDCG@10 | Dense Recall@10 | Model Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.3 | 0.2681 | 0.2745 | 0.3422 | 0.3512 | 0.5840 | 0.6000 |
| test | 0.3 | 0.2515 | 0.2495 | 0.3258 | 0.3237 | 0.5680 | 0.5660 |

Candidate movement diagnosis:

- CoIR CosQA test changed 135 of 500 queries: 63 improved, 72 harmed, and 365 unchanged. Mean nDCG@10 delta was `-0.0021`.
- The improvement set includes plausible answer-bearing rescues where dense buried the positive around ranks 5-30 and the model moved it to rank 1, such as tokenization, comma-separated-list formatting, file cleanup, and HTML stripping examples.
- The harmed set is dominated by duplicate or near-duplicate code snippets. Dense often puts the labeled duplicate at rank 1, while the activation model prefers another function with nearly identical text but a different document id. Under exact qrels, those movements are counted as errors even when the surfaced code is semantically equivalent.

Interpretation:

- CoIR does not give us the desired code-domain win. Dev improves, but locked test is slightly negative and OOF behavior is mixed.
- The result is diagnostically useful: activation-aware reranking appears to recover some functional intent, but it is weak at preserving exact benchmark identity when many snippets are textual or functional duplicates.
- This strengthens the case that the current target is not a universal answer-bearing representation. For code, the next credible version would need duplicate-aware positives or behavior-pair supervision; simply scaling this direct-blend objective is unlikely to produce a decisive CoIR win.
- LegalBench-RAG remains the only scaled vertical with clean OOF-positive and locked-test-positive behavior under this objective.

## Core245 Named-Feature And CAA Shape Audit

Run date: 2026-06-16

Artifacts:

- Feature diagnostics: `runs/supervised/verticals/core245-feature-signal-diagnostics-20260616.json`
- Generated subset manifests: `runs/supervised/verticals/core245-feature-subsets-20260616/`
- Subset ablations: `runs/supervised/verticals/subset-ablation-20260616/`

Shape finding:

- The longmem feature manifest has 245 labeled SAE features plus a 299-field selector feature order. The extra selector fields include span/query metadata, memory pressure, EM/CAA delta summaries, SAE aggregate mass, causal-weighted aggregate mass, and category-level aggregate fields.
- The current RAG prefill caches populate `sae_feature_values`, `sae_delta_vs_neutral`, and `sae_delta_vs_current`, but `current_em_state`, `neutral_baseline_state`, `prior_current_state`, `delta_vs_neutral`, `delta_vs_current`, `saturation`, and `residual_headroom` are empty dictionaries. Therefore the scaled direct-blend rerankers were SAE-only Core245 rerankers, not full selector-primitive CAA/SAE rerankers.
- This diverges from the original selector primitive. If the next phase claims CAA/SAE rather than SAE-only, the capture wrapper must restore true current-vs-baseline CAA fields or reconstruct them from the original sidecar trace source.

Named-feature diagnostic:

- The diagnostic computes within-query positive-vs-negative AUC for each named SAE feature using four simple primitives: document value, query-document product, negative absolute difference, and shared minimum. It is explanatory, not a model-selection result.
- LegalBench-RAG had weak single-feature stability. Only four features cleared a loose train/dev/test `+0.01` margin: `11961` ("sum of first k local values on processor p"), `6317` ("contextual token within phrases"), `15368` ("lambda expression focus"), and `20108` ("drug-related context tokens"). Several legal-sounding labels were actually stably harmful in this candidate pool, including `3249` ("rule language segment"), `654` ("condition-related tokens"), and `11615` ("document-related terms"). Relation/discourse category mass was positive on train and test but not stable on dev.
- R2MED had stronger-looking dev/test category signal from event/action, quantity/math/code, and task/instruction mass, but many specific features flipped sign. Stable positive examples under the loose threshold included `19067` ("verb followed by preposition"), `9958` ("action verbs in instructions"), `13063` ("task-focused text in problem-solving context"), `6896` ("organization mentions in text"), and `1607` ("list item start tokens"). Stable harmful examples included `4637` ("pain-related token contexts"), `7310` ("size-related text patterns"), `4485` ("contextual instruction phrases"), and `1270` ("json key-value pairs in code").
- CoIR/CosQA showed a small code/math/task subset but remained dominated by duplicate-label ambiguity. Weak stable positives included `14424` ("question answer variables"), `1607` ("list item start tokens"), `7310` ("size-related text patterns"), `4632` ("math expression delimiter"), `19917` ("scale rating context"), `7369` ("coin-related probability terms"), and `7231` ("delta in algorithm context"). The only loose stable harmful feature was `15772` ("math problem context with and").
- High-mass noisy features appeared in all three verticals. `19708` ("noun followed by adjective") fired on effectively all positives and negatives, making it conical background mass rather than a ranking signal. Other frequent noisy dimensions included generic segment-initial/focus-word, chemical-structure, JSON/code, and variable-definition latents depending on the corpus.

Subset ablations:

| Dataset | Subset | Alpha | Dense nDCG@10 | Subset Model nDCG@10 | Full Core245 Model nDCG@10 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LegalBench-RAG | 32 features | 0.1 | 0.2728 | 0.2735 | 0.2791 | Pruning preserved only a tiny positive delta and removed most of the legal gain. |
| R2MED pooled | 32 features | 0.2 | 0.1661 | 0.1584 | 0.1532 | Pruning reduced harm versus full Core245 but still failed against dense. |
| CoIR/CosQA | 32 features | 0.1 | 0.3258 | 0.3252 | 0.3237 | Pruning reduced harm versus full Core245 but stayed slightly below dense. |

Interpretation:

- Full Core245 is noisy, but the legal win does not come from a small obvious list of individually stable latents. The legal model appears to use weak, distributed interactions across many SAE dimensions. Removing most dimensions makes the model safer but also strips the effect.
- R2MED and CoIR benefit from pruning in the sense that they are less bad than the full-Core245 direct-blend runs, but neither becomes positive on heldout test. Their dev improvements still overstate heldout behavior.
- A better next model is not simply "top useful labels only." It should use structured regularization over groups: category gates, sparsity penalties, per-feature dropout, and learned non-negative or low-rank feature masks, while preserving the broader latent field needed for legal interactions.
- The fact that the EM/CAA fields are empty is material. Before another serious "same selector primitive" claim, we need a capture path that populates current/baseline/delta/headroom fields or we should rename the current branch as SAE-only activation reranking.

## CAA/EM Restoration Audit And Rebuild Gate

Run date: 2026-06-16

Finding:

- The mechanisms for current/baseline/delta CAA capture already exist in the longmem selector stack, but the current activation-RAG Qwen/RMT/SAE capture path does not call them. It directly encodes selected layer-7 RMT SAE features and writes empty EM/CAA maps into the activation records.
- The reusable production assets are present on `vicuna-host`: the 8-head learned bundle, the neutral EM baseline, compact sidecar traces with `final_em_v2_features`/`final_em_v2_inputs`, and raw-activation summarization code that can rescore raw captures against the learned bundle.
- The currently observed SciFact and LegalBench-RAG results should therefore be claimed as SAE-only Core245 activation-aware reranking. They are not evidence that the full selector primitive improves RAG yet.

Pilot design:

| Arm | Features | Purpose |
| --- | --- | --- |
| SAE-only Core245 | Existing `sae_feature_values` and SAE deltas | Reproduce current result. |
| CAA8-only | 8-head current EM, neutral baseline, delta vs neutral/current, saturation/headroom, mass scalars | Test whether behavior-state fields carry independent ranking signal. |
| SAE + CAA8 | Core245 plus 8-head fields | Test whether full selector shape improves the current artifact. |
| CAA8 aggregates | L2 norms, positive/negative mass, signed balance, per-head query-doc products/differences | Test low-dimensional robust derived features. |
| Matched controls | Shuffled or query-mismatched CAA8 fields | Check that any gain is not split leakage or calibration noise. |

Promotion gate:

- Start with SciFact and LegalBench-RAG candidate unions only.
- Require strict zero-token prefill for both queries and candidate chunks.
- Require non-empty current/baseline/delta fields and prompt-section provenance in every promoted row.
- Require positive dev movement over the current SAE-only direct-blend model and no degradation against dense plus Ettin before full-corpus recapture.
- If the pilot fails, publish the current result as SAE-only and describe full CAA/EM restoration as future work rather than spending the next cycle on full recapture.

## Gated CAA/EM Pilot Result

Run date: 2026-06-17

Artifacts:

- Pilot report: `runs/caa-em-pilot-20260617/pilot-report.json`
- SciFact CAA cache: `runs/caa-em-pilot-20260617/scifact/caa-cache/`
- SciFact CAA diagnostic: `runs/caa-em-pilot-20260617/scifact/caa-cache-diagnostic.json`
- SciFact arm runs: `runs/caa-em-pilot-20260617/scifact/arms/`
- LegalBench-RAG request plan: `runs/caa-em-pilot-20260617/legalbenchrag/request-summary.sample1000each.json`

Capture result:

- Full SciFact candidate union was captured under strict zero-token prefill: 6,258 requests, 6,253 new rows plus 5 smoke rows, zero failures.
- Strict zero was verified via response usage: `completion_tokens=0` and empty assistant content.
- The LegalBench-RAG pilot plan included 1,000 sampled train groups plus full dev/test, 11,495 requests total. Capture was stopped after 3,394 rows once the SciFact and remote compact-trace diagnostics showed the CAA source was constant under this capture mode.

Critical diagnostic:

- Every SciFact row had identical `current_em_state` and identical `delta_vs_neutral` across all eight CAA dimensions.
- The diagnostic reported `nonconstant_key_count=0` for current EM and neutral deltas over 6,258 rows.
- Remote compact trace inspection showed request IDs were correct, so this was not stale trace reuse. The underlying `final_em_v2_inputs` were also constant under strict zero-token isolated prefill: 98 common keys and zero nonzero-variance keys in the inspected request set.
- Interpretation: the production compact 8-head CAA bundle is not content-sensitive in this strict zero-token isolated-prefill configuration. It appears tied to decode/trajectory inputs or defaults that are not populated by this capture path.

SciFact pilot metrics:

| Arm | Dev nDCG@10 | Test nDCG@10 | Test MRR@10 | Test Recall@10 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense baseline | 0.8009 | 0.7682 | 0.7220 | 0.8927 | Shared dense candidate baseline. |
| CAA8-only | 0.8009 | 0.7682 | 0.7220 | 0.8927 | Collapsed to dense-only because CAA was constant. |
| CAA8 aggregates | 0.8009 | 0.7682 | 0.7220 | 0.8927 | Same collapse. |
| CAA8 shuffled control | 0.8009 | 0.7682 | 0.7220 | 0.8927 | Correctly no fake gain. |
| SAE-only Core245 same-seed reproduction | 0.8655 | 0.8407 | 0.8099 | 0.9245 | Positive artifact remains. |
| SAE+CAA8 | 0.8644 | 0.8388 | 0.8037 | 0.9332 | Slightly below SAE-only nDCG/MRR despite higher recall. |

Decision:

- Do not promote full recapture around the current compact CAA/EM path.
- Do not claim the full selector primitive improved RAG. The current publishable result remains SAE-only Core245 activation-aware reranking: decisive on SciFact and narrow on LegalBench-RAG.
- Future CAA work must first produce nonzero-variance, content-sensitive strict-prefill CAA inputs. The likely options are a prefill-trained/rescored CAA bundle, raw activation capture scored by prefill-compatible feature terms, or a query+candidate final-input-token behavior-latent objective rather than isolated chunk compact traces.

## Negative Policy

Hard negatives:

- prefer dense top-N candidates that are not positive for the query;
- exclude any candidate whose document is a positive qrel for that query;
- when possible, exclude candidates that are positives for near-duplicate query text;
- keep provenance fields that distinguish explicit qrel negatives from mined negatives.

False-negative risk is expected on SciFact because unjudged does not mean irrelevant. This is acceptable for a first pass only if the manifest names the policy and the evaluation remains on held-out queries.

## Validation

Required run outputs:

- train/dev/test query counts;
- candidate count distribution;
- positive-in-candidate-pool recall;
- explicit-negative count and mined-negative count;
- dense-only, unsupervised blend, activation-aware supervised, and text-reranker metrics;
- paired query-level significance for nDCG@10, MRR@10, and Recall@10;
- changed-query audit showing improved and harmed cases.

Promotion rule:

- supervised activation-aware reranking must beat the normal text reranker, not merely dense-only;
- it must preserve or improve Recall@10;
- it must show a paired-query improvement that is not explained by a handful of lucky query movements.

## First SciFact Supervised Run

Run date: 2026-06-12

Artifacts:

- train/dev/test candidate groups: `runs/supervised/scifact-train-reranker-groups.k100.train.jsonl`, `runs/supervised/scifact-train-reranker-groups.k100.dev.jsonl`, `runs/supervised/scifact-test-reranker-groups.k100.jsonl`
- activation MLP metrics: `runs/supervised/scifact-activation-mlp-reranker.metrics.json`
- text reranker metrics: `runs/supervised/scifact-text-reranker-baseline.metrics.json`

Data:

- train groups: 688
- dev groups: 121
- test groups: 300
- candidate pool: top-100 BGE-base dense candidates per query
- positives in candidate pool before train/dev split: 793 of 809 SciFact train queries
- positives in test candidate pool: 291 of 300 SciFact test queries
- negative policy: non-positive dense candidates are mined hard negatives with `unjudged_assumed_negative` trust, not true negatives

Activation-aware MLP reranker:

- features: dense score/rank/z-score, aggregate activation distance/product features, and per-site activation cosine late-fusion features
- hidden dimension: 128
- loss: listwise softmax
- device: CUDA on vicuna host

Metrics:

| Split | Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: |
| train | dense | 0.7388 | 0.7745 | 0.8845 |
| train | activation MLP | 0.8707 | 0.8926 | 0.9605 |
| dev | dense | 0.7540 | 0.7744 | 0.8691 |
| dev | activation MLP | 0.7501 | 0.7681 | 0.8691 |
| test | dense | 0.7003 | 0.7451 | 0.8659 |
| test | activation MLP | 0.7099 | 0.7489 | 0.8684 |
| test | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.6640 | 0.7025 | 0.8222 |

Interpretation:

- The activation MLP strongly overfits the training split and does not beat dense-only on dev.
- The small test lift over dense-only is not sufficient to promote the method because the dev result is negative.
- The MS MARCO MiniLM cross-encoder baseline is weak on this SciFact setup and likely domain-mismatched; beating it is not enough to claim a decisive win over normal reranking.
- The next supervised run needs a stronger text baseline, preferably a modern reranker over the same candidate groups, before promotion or fallback to answer-span activation prediction.

## Stronger Text Baselines And Dev-Selected Activation Sweep

Run date: 2026-06-12

Additional text reranker artifacts:

- BGE reranker test metrics: `runs/supervised/scifact-bge-reranker-v2-m3-baseline.metrics.json`
- Ettin reranker test metrics: `runs/supervised/scifact-ettin-reranker-150m-v1-baseline.metrics.json`
- Ettin reranker dev metrics: `runs/supervised/scifact-dev-ettin-reranker-150m-v1-baseline.metrics.json`

Activation sweep artifacts:

- dense-only control: `runs/supervised/scifact-activation-mlp-dense-only-listwise-h32.metrics.json`
- activation-only aggregate: `runs/supervised/scifact-activation-mlp-activation-agg-only-listwise-h64.metrics.json`
- activation-only aggregate plus per-site: `runs/supervised/scifact-activation-mlp-activation-all-only-listwise-h64.metrics.json`
- dense plus aggregate activation: `runs/supervised/scifact-activation-mlp-dense-plus-agg-listwise-h64.metrics.json`
- dense plus per-site activation: `runs/supervised/scifact-activation-mlp-dense-plus-sites-listwise-h64.metrics.json`
- dense plus all activation features: `runs/supervised/scifact-activation-mlp-dense-plus-all-listwise-h64.metrics.json`
- dense plus all activation features, wider MLP: `runs/supervised/scifact-activation-mlp-dense-plus-all-listwise-h128.metrics.json`
- dense plus all activation features, pairwise loss: `runs/supervised/scifact-activation-mlp-dense-plus-all-pairwise-h128.metrics.json`

Trainer correction:

- The MLP trainer now evaluates dev after each epoch, selects by dev `nDCG@10`, restores that checkpoint, and records `best_epoch`, `best_dev_score`, and `selection_metric`.
- This replaces the previous last-epoch reporting, which was too vulnerable to train overfit.

Text reranker metrics:

| Split | Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: |
| dev | dense | 0.7540 | 0.7744 | 0.8691 |
| dev | `cross-encoder/ettin-reranker-150m-v1` | 0.7345 | 0.7620 | 0.8609 |
| test | dense | 0.7003 | 0.7451 | 0.8659 |
| test | `BAAI/bge-reranker-v2-m3` | 0.7075 | 0.7430 | 0.8486 |
| test | `cross-encoder/ettin-reranker-150m-v1` | 0.7288 | 0.7641 | 0.8587 |

Dev-selected activation sweep:

| Variant | Best Epoch | Dev MRR@10 | Dev nDCG@10 | Dev Recall@10 | Test MRR@10 | Test nDCG@10 | Test Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense-only listwise h32 | 1 | 0.7540 | 0.7744 | 0.8691 | 0.7003 | 0.7451 | 0.8659 |
| activation aggregate-only listwise h64 | 78 | 0.0796 | 0.0954 | 0.1612 | 0.0469 | 0.0662 | 0.1390 |
| activation all-only listwise h64 | 80 | 0.1053 | 0.1282 | 0.2204 | 0.0674 | 0.0864 | 0.1651 |
| dense plus aggregate activation listwise h64 | 9 | 0.7620 | 0.7822 | 0.8774 | 0.7011 | 0.7463 | 0.8692 |
| dense plus per-site activation listwise h64 | 1 | 0.7541 | 0.7738 | 0.8691 | 0.7000 | 0.7455 | 0.8676 |
| dense plus all activation listwise h64 | 7 | 0.7597 | 0.7813 | 0.8774 | 0.7044 | 0.7483 | 0.8712 |
| dense plus all activation listwise h128 | 3 | 0.7588 | 0.7789 | 0.8733 | 0.7029 | 0.7474 | 0.8686 |
| dense plus all activation pairwise h128 | 9 | 0.7588 | 0.7796 | 0.8691 | 0.7027 | 0.7454 | 0.8642 |

Interpretation:

- Activation-only reranking remains non-viable.
- Aggregate activation features add useful supervised signal when anchored by dense features. The best dev-selected activation variant is dense plus aggregate activation, with dev nDCG@10 `0.7822` versus dense `0.7744`.
- Per-site activation features do not help in this sweep; adding them slightly reduces dev performance relative to dense plus aggregate activation.
- The strongest text reranker tested so far is Ettin on test: nDCG@10 `0.7641`, MRR@10 `0.7288`. It beats the activation reranker on test nDCG/MRR but reduces Recall@10 versus dense.
- Ettin does not beat dense on the dev split, so the current split has validation/test disagreement. This argues against making a promotion decision from SciFact alone.
- The next research step should not be answer-bearing activation prediction yet. The supervised reranker has a real dev signal in aggregate activation features; the immediate next step is to test this dense-plus-aggregate activation reranker on additional BEIR datasets and add paired query-level significance plus changed-query audits.

## Paired SciFact Audit And Cross-BEIR Repeat Attempt

Run date: 2026-06-12

Audit artifacts:

- dense versus activation dense-plus-aggregate: `runs/supervised/scifact-audit-dense-vs-activation-dense-plus-agg.json`
- dense versus Ettin: `runs/supervised/scifact-audit-dense-vs-ettin.json`
- activation dense-plus-aggregate versus Ettin: `runs/supervised/scifact-audit-activation-dense-plus-agg-vs-ettin.json`

SciFact paired randomization results, 10,000 iterations over 300 test queries:

| Comparison | Delta MRR@10 | p | Delta nDCG@10 | p | Delta Recall@10 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| activation dense-plus-aggregate minus dense | +0.0008 | 0.6256 | +0.0012 | 0.4721 | +0.0033 | 1.0000 |
| Ettin minus dense | +0.0285 | 0.0879 | +0.0189 | 0.2104 | -0.0072 | 0.7060 |
| Ettin minus activation dense-plus-aggregate | +0.0277 | 0.1009 | +0.0178 | 0.2392 | -0.0106 | 0.5699 |

Changed-query audit:

- Activation dense-plus-aggregate versus dense changes nDCG@10 on only 20 of 300 queries: 10 improved, 10 harmed, 280 unchanged. This explains the tiny mean lift and makes the result non-decisive.
- Ettin versus dense changes nDCG@10 on 113 of 300 queries: 62 improved, 51 harmed, 187 unchanged. The mean MRR/nDCG lift is materially larger than the activation reranker, but the p-values still do not cross 0.05 on SciFact and Recall@10 drops.
- Ettin versus activation dense-plus-aggregate changes nDCG@10 on 111 of 300 queries: 61 improved, 50 harmed, 189 unchanged. On this split, the stronger text reranker still carries the larger reranking effect.

Interpretation:

- The best activation-aware reranker is not significantly ahead of dense on SciFact and does not show broad query-level movement.
- The stronger text reranker is ahead by a non-trivial mean margin on MRR/nDCG, but SciFact alone is still underpowered or noisy enough that the paired tests do not make it decisive.
- No promotion decision should be made from SciFact alone. The next valid step remains additional BEIR repeats with the same audit harness.

NFCorpus repeat status:

- Downloaded BEIR NFCorpus and built a BGE-base dense embedding cache at `runs/supervised/nfcorpus-test-bge-base-dense-cache.npz`.
- Added durable BEIR telemetry cache capture in `scripts/capture_beir_telemetry_cache.py` so document and query prefill activations can resume in batches.
- Added invalid-telemetry cache guards: rows with `telemetry_valid=false`, an `invalid_reason`, or empty `sae_feature_values` are ignored as cache hits, are not written as valid cache rows, and cannot enter supervised group preparation.
- The first NFCorpus telemetry attempt exposed an operational raw-root mismatch: the server wrote manifests under `/tmp/activation-rag-20260612-section-prefill/activations`, while the probe waited under a probe-specific raw root. The bad invalid cache row was flushed.
- The corrected NFCorpus capture was relaunched against the active server raw root, but SSH/SCP to `vicuna-host` timed out before the first batch completed. Local NFCorpus telemetry cache count remained zero. Cross-BEIR activation comparison is therefore blocked on vicuna reachability, not on benchmark code.

## Semantic Feature Catalog And Counterfactual Sweep

Run date: 2026-06-12

Trigger:

- The earlier activation reranker reused the same selector primitive shape that had worked for durable span selection, but the telemetry object contains more fields than the prior aggregate feature family.
- The hypothesis was that labeled SAE/CAA feature subsets from the longmem/vicuna selector work may carry more semantic and causal signal than unlabeled site/window/stat aggregates.
- The diagnostic requirement was to compare semantic groups against counterfactual feature groups before treating any gain as a causal semantic effect.

Implementation artifacts:

- Feature catalog adapter: `src/activation_rag/feature_catalog.py`
- Catalog-aware group preparation: `scripts/prepare_supervised_reranker_groups.py --feature-catalog`
- Semantic/counterfactual sweep runner: `scripts/run_semantic_activation_feature_sweep.py`
- Current proxy catalog: `configs/activation_feature_catalog.section_prefill_proxy.json`
- Generated candidate groups:
  - `runs/supervised/scifact-train-reranker-groups.k100.semantic-proxy.train.jsonl`
  - `runs/supervised/scifact-train-reranker-groups.k100.semantic-proxy.dev.jsonl`
  - `runs/supervised/scifact-test-reranker-groups.k100.semantic-proxy.jsonl`
- Sweep manifest: `runs/supervised/semantic-proxy-sweep.manifest.json`
- Paired audits:
  - `runs/supervised/scifact-audit-dense-vs-semantic-proxy-categories.json`
  - `runs/supervised/scifact-audit-dense-vs-semantic-proxy-counterfactual.json`

Local longmem/vicuna label status:

- Local longmem selector code exists at `/Users/tyleraraujo/longmem-mechinterp-selector`.
- The selector manifest schema `vicuna.rmt_span_selector.feature_manifest.v1` is supported by the adapter. It maps labeled rows into `sae.feature.{feature_id}` entries with label, category, validation status, causal effect, and feature order metadata.
- The portable artifact manifest references `artifacts/local/selector_materialization/feature_manifest.json`, but that materialized artifact is not present locally.
- SSH to `vicuna-host` timed out during this pass, so the true vicuna/longmem labeled SAE subset could not be fetched.
- The corrected SciFact telemetry cache currently exposes `act:{site}:{window}:{stat}` summary features, not `sae.feature.*` latent IDs. Therefore the run below uses a section-prefill proxy catalog rather than the true labeled causal SAE subset.

Telemetry-shape finding:

- Real sidecar rows have populated `sae_feature_values`, `sae_delta_vs_neutral`, and `sae_delta_vs_current`.
- The currently useful features are section-prefill summaries such as `act:emv2_p65_attn_out:prefill_last:chunk_03`.
- `current_em_state`, `delta_vs_neutral`, `delta_vs_current`, `saturation`, and `residual_headroom` are empty in the inspected real sidecar cache rows.
- This means the implemented catalog path is ready for true SAE-labeled features, but the current SciFact cache cannot validate the longmem causal subset until capture/export stores matching `sae.feature.*` values or provides a deterministic mapping from summary features back to labeled SAE features.

SciFact proxy sweep:

| Variant | Features | Best Epoch | Dev nDCG@10 | Dev MRR@10 | Dev Recall@10 | Test nDCG@10 | Test MRR@10 | Test Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense plus categories | 50 | 12 | 0.7442 | 0.7018 | 0.8829 | 0.7504 | 0.7081 | 0.8678 |
| dense plus counterfactual | 26 | 10 | 0.7413 | 0.7018 | 0.8760 | 0.7510 | 0.7105 | 0.8619 |
| dense plus all semantic | 134 | 10 | 0.7367 | 0.6936 | 0.8788 | 0.7495 | 0.7088 | 0.8629 |
| dense plus causal confidence | 14 | 19 | 0.7339 | 0.6957 | 0.8609 | 0.7463 | 0.7025 | 0.8642 |
| dense plus polarity | 14 | 19 | 0.7322 | 0.6940 | 0.8650 | 0.7447 | 0.7023 | 0.8609 |
| dense plus semantic labels | 62 | 1 | 0.7321 | 0.6931 | 0.8733 | 0.7426 | 0.6989 | 0.8669 |
| dense only | 2 | 1 | 0.7289 | 0.6902 | 0.8567 | 0.7451 | 0.7003 | 0.8659 |
| semantic labels only | 60 | 16 | 0.1238 | 0.1011 | 0.2066 | 0.0988 | 0.0731 | 0.1836 |
| categories only | 48 | 20 | 0.1181 | 0.0918 | 0.2252 | 0.1144 | 0.0798 | 0.2357 |
| counterfactual only | 24 | 16 | 0.0999 | 0.0803 | 0.1736 | 0.0683 | 0.0468 | 0.1411 |
| polarity only | 12 | 13 | 0.0932 | 0.0707 | 0.1798 | 0.0692 | 0.0484 | 0.1463 |
| causal confidence only | 12 | 16 | 0.0758 | 0.0569 | 0.1488 | 0.0537 | 0.0360 | 0.1184 |

Paired SciFact proxy audits against dense, 10,000 randomization iterations over 300 test queries:

| Variant | Delta MRR@10 | p | Delta nDCG@10 | p | Delta Recall@10 | p | nDCG changed queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense plus categories | +0.0078 | 0.4198 | +0.0053 | 0.5022 | +0.0019 | 0.8589 | 36 improved, 36 harmed, 228 unchanged |
| dense plus counterfactual | +0.0102 | 0.2920 | +0.0059 | 0.4618 | -0.0040 | 0.6880 | 33 improved, 29 harmed, 238 unchanged |

Interpretation:

- The proxy semantic categories modestly improve mean SciFact test nDCG/MRR over dense, but the effect is not statistically significant and the improved/harmed query count is balanced.
- The counterfactual group performs about as well as the semantic category proxy and slightly better on test nDCG/MRR, while hurting Recall@10. This prevents a causal claim that the current proxy labels are identifying semantic activation evidence.
- Activation-only semantic features remain non-viable. The usable signal appears only when anchored by dense candidates.
- The result does not clear the kill criteria for "semantic activation reranking beats text reranking." It is a useful implementation and diagnostic harness, not yet a promotion result.
- The next decisive experiment requires the true labeled longmem/vicuna SAE subset to be materialized and the telemetry cache to store matching `sae.feature.*` values or a lossless mapping from the sidecar summaries to labeled SAE features.

## Vicuna Longmem Core245 Manifest And Capture Permutations

Run date: 2026-06-13

Artifacts copied/inspected:

- Longmem feature manifest: `external/vicuna-longmem/feature_manifest.json`
- Longmem method note: `external/vicuna-longmem/method.md`
- Longmem artifact note: `external/vicuna-longmem/artifacts.md`
- Capture permutation plan: `configs/activation_capture_permutations.longmem_core245.json`
- Qwen/RMT/SAE capture wrapper: `scripts/capture_qwen_sae_prefill.py`

Actual manifest shape:

- Schema: `vicuna.rmt_span_selector.feature_manifest.v1`
- Feature set: `core245_corrected_longmem_query_conditioned_train_dev`
- Selected SAE features: 245
- Validation status: all 245 features have `causal_support`
- Feature-order size: 299, including selector metadata, EM scalar slots, SAE scalar slots, `saeagg.*` category aggregates, and `sae.feature.{id}` raw selected feature channels
- Category distribution:
  - `uncategorized_content`: 87
  - `relation_discourse`: 60
  - `quantity_math_code`: 52
  - `task_instruction`: 43
  - `entity_domain`: 30
  - `event_action`: 13
  - `state_affect`: 7
- Causal-effect distribution:
  - min: `0.0205`
  - median: `0.0887`
  - mean: `0.2594`
  - p75: `0.2549`
  - p90: `0.5715`
  - max: `3.9726`

Actual selector-row shape:

- Runtime selector rows store SAE values as raw string IDs such as `"5439"` and `"18172"`, not only as `sae.feature.5439`.
- The feature catalog adapter was fixed to match both raw selector IDs and `sae.feature.{id}` aliases.
- Longmem selector rows contain sparse nonnegative SAE prominences in:
  - `sae_feature_values`
  - `sae_delta_vs_neutral`
  - `sae_delta_vs_current`
  - `sae_residual_headroom`
  - `sae_novelty`
- Smoke selector rows showed all 245 selected IDs somewhere across 246 rows, with about 10-56 active selected IDs per row.
- Some features are nearly ubiquitous in the smoke set, for example `11488`, `12533`, `18172`, and `19708`; these are likely hub-like unless document-frequency filtering or normalization controls them.

Model compatibility finding:

- The longmem core245 SAE checkpoint is for `qwen3-4b-hf` / RMT layer `l07_resid_pre`, with SAE checkpoint schema `vicuna.em_sae.checkpoint.v1`, `d_model=2560`, `d_sae=20480`, and `target_l0=64`.
- The previous activation-rag corrected SciFact cache was produced by the DeepSeek sidecar summary path and exposes `act:{site}:{window}:{stat}` summary features.
- Those DeepSeek section summaries are not compatible with the longmem Qwen/RMT SAE manifest. Loading the real manifest into the old cache would silently test the wrong representation.
- Therefore the next valid capture path is not a better catalog over the old cache; it is a new Qwen/RMT/SAE prefill capture that emits the 245 selected raw SAE IDs.

Operational status:

- `vicuna-host` is reachable over Tailscale again.
- Port `28080`, used by the prior DeepSeek activation capture, is not running.
- Port `8080` is a provider-backed DeepSeek relay and cannot produce local activations.
- The Qwen/RMT/SAE release paths are present:
  - model: `/mnt/disk-3tb/models/qwen3-4b-hf`
  - joint checkpoint: `/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/rmt/qwen3_rmt_joint_memory_latest.pt`
  - SAE checkpoint: `/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/sae/topk_sae_latest.pt`
  - feature manifest: `/mnt/disk-3tb/longmem-mechinterp-selector-release/artifacts/local/selector_materialization/feature_manifest.json`
- CUDA was not available after the host came back: `/dev/nvidia*` was absent, `nvidia-smi` failed, and `modprobe nvidia` reported no module for kernel `6.17.0-35-generic`.
- A tiny Qwen/RMT/SAE smoke could not run because torch reported `No CUDA GPUs are available`. Full recapture should wait until the NVIDIA module/driver is restored.
- Later on 2026-06-13, the host was restored with a working NVIDIA driver/CUDA path. The Qwen/RMT/SAE two-row smoke and SciFact recapture proceeded through that repaired runtime.

Capture permutations to try, in order:

1. `qwen_l07_core245_raw_max`: primary capture. Store raw selected SAE ID values using max activation over prompt tokens. This matches the longmem selector primitive most directly.
2. `qwen_l07_core245_log1p_l2`: post-capture transform. Apply `log1p` and per-row L2 normalization to reduce domination by high-mass always-on features.
3. `qwen_l07_core245_causal_weighted_category`: post-capture transform. Use category active count, total mass, max value, and causal-effect weighted mass. This tests semantic/causal grouping without over-wide raw IDs.
4. `qwen_l07_core245_high_effect_topk`: post-capture transform. Restrict raw IDs to the highest causal-effect features and sweep top-K.
5. `qwen_l07_core245_df_filtered`: post-capture transform. Drop document-frequency hubs and near-constant features before reranking.
6. `qwen_l07_core245_counterfactual_matched`: required control. Random groups matched to semantic category size must lose to semantic groups before we call a semantic result real.
7. `deepseek_section_prefill_summary_proxy`: retain only as a negative control/proxy baseline, not as a causal SAE-label experiment.

Design decision:

- Do not proceed to answer-bearing activation prediction yet.
- First run the Qwen/RMT/SAE selected-feature capture and the above supervised reranker ablations over the same frozen dense candidates.
- Treat any semantic/category improvement as non-causal unless it beats matched counterfactual groups and survives paired per-query significance.
- Treat raw selected SAE IDs as a sparse feature representation, not as a high-dimensional cosine search space. Prefer supervised reranking and controlled transforms over activation-only KNN.

## Qwen/RMT/SAE Core245 Permutation Sweep

Run date: 2026-06-13

Capture artifacts:

- Qwen/RMT/SAE telemetry cache: `runs/telemetry-cache/scifact-qwen-core245-20260613-215705`
- Initial train/query capture log: `runs/qwen-core245-capture-20260613-215705/stdout.log`
- Corrected BEIR namespace document recapture log: `runs/qwen-core245-capture-beir-scifact-docs-20260613-220040/stdout.log`
- Permutation sweep manifest: `runs/supervised/core245-permutation-sweep-20260613.manifest.json`
- Variant outputs: `runs/supervised/core245-permutation-sweep-20260613/{variant}/`

Capture correction:

- The first Qwen/RMT/SAE document capture used the dataset namespace `scifact`, while the frozen dense candidate groups use `beir-scifact`.
- Query chunk IDs were stable, but document candidate chunk IDs did not align. This would have made the ablation outputs empty or non-comparable.
- Documents were recaptured with `dataset_name=beir-scifact`; after correction, train/dev/test group preparation found telemetry for all dense candidates in the frozen candidate groups.

Sweep design:

- Candidate pool: same frozen top-100 BGE-base dense groups used by the previous SciFact supervised reranker runs.
- Loss: listwise softmax.
- Hidden dimension: 128.
- Epochs: 25.
- Best checkpoint: selected by dev nDCG@10.
- Training location: local CPU over materialized JSONL feature groups. Capture used the repaired vicuna CUDA path, but these compact supervised MLP runs were not moved back to vicuna.
- Shared anisotropic-space candidate features: raw cosine plus CSLS, NICDM, top-PC removal, whitening plus L2, and z-scored cosine diagnostics computed over each variant representation.

Dense test baseline for all rows: MRR@10 `0.7003`, nDCG@10 `0.7451`, Recall@10 `0.8659`.

| Variant | Features | Dev nDCG@10 | Test MRR@10 | Test nDCG@10 | Test Recall@10 | Test nDCG delta vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen_l07_core245_causal_weighted_category` | 57 | 0.7860 | 0.7246 | 0.7576 | 0.8568 | +0.0124 |
| `qwen_l07_core245_counterfactual_matched` | 57 | 0.7811 | 0.7160 | 0.7575 | 0.8719 | +0.0124 |
| `qwen_l07_core245_high_effect_topk` | 76 | 0.7616 | 0.6953 | 0.7397 | 0.8639 | -0.0055 |
| `qwen_l07_core245_raw_max` | 257 | 0.7718 | 0.7017 | 0.7337 | 0.8232 | -0.0114 |
| `qwen_l07_core245_df_filtered` | 254 | 0.7691 | 0.6978 | 0.7325 | 0.8230 | -0.0126 |
| `qwen_l07_core245_log1p_l2` | 257 | 0.7765 | 0.6848 | 0.7220 | 0.8209 | -0.0231 |

Paired audit artifacts:

- Dense versus causal category: `runs/supervised/core245-permutation-sweep-20260613/audit-category-vs-dense.json`
- Dense versus matched counterfactual: `runs/supervised/core245-permutation-sweep-20260613/audit-counterfactual-vs-dense.json`
- Matched counterfactual versus causal category: `runs/supervised/core245-permutation-sweep-20260613/audit-category-vs-counterfactual.json`

Paired SciFact audit over 300 test queries, 10,000 randomization iterations:

| Comparison | Delta MRR@10 | p | Delta nDCG@10 | p | Delta Recall@10 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal category minus dense | +0.0243 | 0.0977 | +0.0124 | 0.3292 | -0.0091 | 0.5155 |
| matched counterfactual minus dense | +0.0157 | 0.1143 | +0.0124 | 0.1432 | +0.0060 | 0.6804 |
| causal category minus matched counterfactual | +0.0087 | 0.5570 | +0.0000 | 0.9984 | -0.0151 | 0.1495 |

Interpretation:

- The result is not negligible: causal-category and matched-counterfactual variants both improve mean SciFact test nDCG over dense by about `+0.0124`, and causal-category improves MRR by about `+0.0243`.
- The semantic causal interpretation fails the required control. Matched counterfactual groups are effectively tied on nDCG and have better Recall@10 than the category aggregates.
- Raw selected SAE IDs, log1p/L2, high-effect top-K, and document-frequency filtering do not preserve useful reranking signal in this setup; several harm Recall@10 substantially.
- The useful signal is likely coming from compact learned transformations and shared anisotropic matcher diagnostics, not demonstrably from the human-readable category labels.
- This does not yet trigger the answer-bearing activation predictor fallback under the original rule, because not all activation permutations are negligible. However, before spending more effort on semantic labeling, the next diagnostic should isolate matcher-only features versus category/counterfactual aggregate features. If the matched control remains tied or the gain is mostly matcher-only, the research path should escalate to supervised answer-bearing representation prediction.

## Core245 Hubness And Feature-Source Isolation Diagnostics

Run date: 2026-06-14

Implementation artifacts:

- Diagnostic runner: `scripts/run_core245_hubness_diagnostics.py`
- Unit tests: `tests/test_run_core245_hubness_diagnostics.py`
- Diagnostic report: `runs/supervised/core245-hubness-diagnostics-20260614/report.json`
- Diagnostic model/metric artifacts: `runs/supervised/core245-hubness-diagnostics-20260614/{diagnostic}/`
- Paired audits:
  - `runs/supervised/core245-hubness-diagnostics-20260614/audit-dense-vs-matcher-only.json`
  - `runs/supervised/core245-hubness-diagnostics-20260614/audit-dense-vs-category-plus-matcher.json`
  - `runs/supervised/core245-hubness-diagnostics-20260614/audit-dense-vs-counterfactual-only.json`
  - `runs/supervised/core245-hubness-diagnostics-20260614/audit-category-plus-matcher-vs-counterfactual-only.json`

Diagnostic setup:

- Reused the already materialized Qwen/RMT/SAE Core245 train/dev/test groups.
- Trained feature-filtered MLP rerankers with the same listwise objective, hidden size 128, 25 epochs, seed 13, and dev nDCG@10 checkpoint selection.
- Evaluated activation-only ranking by each matcher feature directly over the same frozen dense top-100 candidate groups.
- Measured hubness by top-1 and top-10 neighbor occurrence skew across 300 SciFact test queries.

Filtered supervised reranker results:

| Diagnostic | Features | Dev nDCG@10 | Test MRR@10 | Test nDCG@10 | Test Recall@10 | Test nDCG delta vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense only | 2 | 0.7744 | 0.7003 | 0.7451 | 0.8659 | +0.0000 |
| matcher only | 8 | 0.7781 | 0.7024 | 0.7461 | 0.8642 | +0.0010 |
| category aggregates only | 51 | 0.7871 | 0.7085 | 0.7437 | 0.8428 | -0.0014 |
| category aggregates plus matcher | 57 | 0.7860 | 0.7246 | 0.7576 | 0.8568 | +0.0124 |
| counterfactual aggregates only | 51 | 0.8069 | 0.7200 | 0.7577 | 0.8652 | +0.0126 |
| counterfactual aggregates plus matcher | 57 | 0.7811 | 0.7160 | 0.7575 | 0.8719 | +0.0124 |

Paired randomization audits over 300 test queries:

| Comparison | Delta MRR@10 | p | Delta nDCG@10 | p | Delta Recall@10 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matcher only minus dense | +0.0021 | 0.5029 | +0.0010 | 0.7564 | -0.0017 | 1.0000 |
| category plus matcher minus dense | +0.0243 | 0.0977 | +0.0124 | 0.3292 | -0.0091 | 0.5155 |
| counterfactual only minus dense | +0.0196 | 0.0520 | +0.0126 | 0.1466 | -0.0007 | 1.0000 |
| counterfactual only minus category plus matcher | -0.0047 | 0.7399 | +0.0001 | 0.9911 | +0.0084 | 0.4742 |

Activation-only matcher ranking and hubness:

| Ranking score | MRR@10 | nDCG@10 | Recall@10 | Top-1 unique | Top-1 max count | Top-1 Gini | Top-10 unique | Top-10 max count | Top-10 Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense rank | 0.7003 | 0.7451 | 0.8659 | 263 | 4 | 0.1101 | 1965 | 7 | 0.2515 |
| raw activation cosine | 0.0658 | 0.0851 | 0.1646 | 177 | 9 | 0.3181 | 938 | 25 | 0.4573 |
| CSLS | 0.0749 | 0.0992 | 0.1879 | 197 | 9 | 0.2713 | 1097 | 22 | 0.4450 |
| NICDM | 0.0756 | 0.1010 | 0.1901 | 201 | 8 | 0.2568 | 1217 | 20 | 0.4259 |
| top-PC removed | 0.0773 | 0.1071 | 0.2115 | 203 | 6 | 0.2669 | 1491 | 13 | 0.3760 |
| whitening plus L2 | 0.0710 | 0.1018 | 0.2025 | 213 | 6 | 0.2406 | 1544 | 14 | 0.3585 |

Interpretation:

- Hubness correction did work in the narrow geometric sense. CSLS, NICDM, top-PC removal, and whitening reduce neighbor occurrence skew versus raw activation cosine, especially in top-10 hubness.
- That correction does not make activation-only retrieval useful. The best activation-only corrected score is top-PC removal at nDCG@10 `0.1071`, still far below dense nDCG@10 `0.7451`.
- Matcher-only supervised features barely improve dense: nDCG delta `+0.0010`, paired p `0.7564`. The shared anisotropic matcher diagnostics are not carrying the earlier `+0.012` lift.
- Semantic category aggregates alone do not generalize on test and damage Recall@10. Adding matcher features recovers the previous category result, but it still does not beat the matched control.
- Matched counterfactual aggregates alone reproduce the full gain and preserve Recall@10 better than category aggregates. This is the strongest evidence so far that the lift is a random low-dimensional aggregation/projection regularization effect, not a validated semantic SAE-label effect.
- The causal semantic reranking hypothesis should not be promoted from these results. The reasonable next research move is to either test deliberate random projection/aggregation baselines to characterize this regularization effect, or proceed to the answer-bearing activation representation predictor. The original semantic-label path has not cleared its control.

## Answer-Bearing Activation Representation Searcher

Run date: 2026-06-14

Implementation artifacts:

- Trainer/evaluator: `scripts/train_activation_representation_searcher.py`
- Unit tests: `tests/test_activation_representation_searcher.py`
- Clean run directory: `runs/supervised/activation-representation-searcher-20260614-v2/`
- Full-set paired audits:
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-full-dense-vs-raw-blend.json`
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-full-ettin-vs-raw-blend.json`
- Matched 291-query subset audits:
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-dense-vs-raw-model-only.json`
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-dense-vs-raw-blend.json`
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-dense-vs-log1p-blend.json`
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-ettin-vs-raw-blend.json`
  - `runs/supervised/activation-representation-searcher-20260614-v2/audit-counterfactual-only-vs-raw-blend.json`

Training setup:

- Task: learn a mapping from query prefill activation representation to the activation representation of answer-bearing candidate chunks.
- Objective: soft InfoNCE/listwise cross-entropy over the frozen dense top-100 candidate group, with positive qrel chunks as targets and dense unjudged candidates as hard negatives.
- Auxiliary term: small MSE term toward the mean positive answer-chunk representation.
- Model: MLP `input_dim -> 128 -> 128 -> input_dim`.
- Candidate pool: same frozen SciFact dense top-100 groups.
- Checkpoint selection: dev model-only nDCG@10.
- Reranker mode: dev-selected dense/model blend alpha over z-scored dense scores and predicted activation-vector similarity.
- Query coverage: 676 train, 117 dev, and 291 test groups had query telemetry, candidate telemetry, at least one positive candidate, and at least one negative candidate. Full 300-query audits treat the 9 uncovered test queries as dense-order ties because the predictor cannot improve queries whose answer-bearing candidate has no usable activation target.

Matched 291-query subset results:

| Representation | Pure model nDCG@10 | Pure model Recall@10 | Blend alpha | Blend MRR@10 | Blend nDCG@10 | Blend Recall@10 | Dense nDCG@10 | Blend nDCG delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw Core245 | 0.4609 | 0.5215 | 0.3 | 0.8027 | 0.8362 | 0.9281 | 0.7682 | +0.0681 |
| log1p/L2 Core245 | 0.4566 | 0.5295 | 0.4 | 0.7961 | 0.8313 | 0.9321 | 0.7682 | +0.0632 |
| counterfactual aggregate | 0.2556 | 0.3494 | 0.3 | 0.7578 | 0.7974 | 0.9016 | 0.7682 | +0.0292 |
| category aggregate | 0.1931 | 0.2773 | 0.1 | 0.7229 | 0.7688 | 0.8932 | 0.7682 | +0.0006 |

Full 300-query fallback audit:

| Comparison | Candidate MRR@10 | Candidate nDCG@10 | Candidate Recall@10 | Delta MRR@10 | p | Delta nDCG@10 | p | Delta Recall@10 | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw Core245 predictor blend minus dense | 0.7786 | 0.8112 | 0.9002 | +0.0783 | 0.0001 | +0.0660 | 0.0001 | +0.0343 | 0.0077 |
| raw Core245 predictor blend minus Ettin text reranker | 0.7786 | 0.8112 | 0.9002 | +0.0498 | 0.0090 | +0.0471 | 0.0042 | +0.0416 | 0.0126 |

Interpretation:

- Primary activation search is still not solved. Pure predicted-vector ranking is much better than raw activation KNN, but raw Core245 model-only nDCG@10 `0.4609` remains far below dense nDCG@10 `0.7682` on the matched subset.
- The answer-representation predictor is a major reranking win when blended with dense scores. The raw Core245 predictor blend beats dense, the previous counterfactual aggregate reranker, and the Ettin text reranker on this SciFact setup with paired significance.
- The best primitive is now raw Core245 answer-representation prediction, not category/counterfactual aggregation. The log1p/L2 predictor is close, but raw has the best nDCG and MRR in the selected blend.
- This result is strong enough to continue the activation-telemetry line, but only as a dense-candidate reranking/search-assist method for now. It should be repeated on at least one additional BEIR dataset before treating it as a general retrieval result.
- The next diagnostic should verify that the gain is not caused by query/document leakage, duplicate document IDs, or candidate-pool artifacts. After that, promote this path over the semantic-label reranker and run cross-BEIR repeats.

## Cross-BEIR Answer-Predictor Repeats

Run date: 2026-06-14

Implementation artifacts:

- Dense candidate builder: `scripts/prepare_dense_candidate_groups.py`
- Targeted group telemetry capture: `scripts/capture_group_telemetry_cache.py`
- Remote text reranker runner: `scripts/rerank_remote_sentence_transformers.py`
- Dense/reranker blend sweep: `scripts/run_score_blend_sweep.py`
- Leakage/memorization audit: `scripts/audit_group_leakage.py`
- Unit tests:
  - `tests/test_prepare_dense_candidate_groups.py`
  - `tests/test_score_blend_sweep.py`
  - `tests/test_audit_group_leakage.py`

Protocol:

- Retrieval candidate pool: BGE-base dense top-100 candidates.
- Activation primitive: corrected Qwen/RMT/SAE Core245 prefill telemetry, raw selected SAE max-over-prompt representation.
- Predictor: same MLP and loss as the SciFact answer-bearing activation representation searcher.
- Selection: dev-selected model checkpoint and dev-selected dense/act_pred blend alpha.
- Baselines: pure dense candidate order, dense+Ettin (`cross-encoder/ettin-reranker-150m-v1`) blend sweep, and pure/weighted act_pred.
- Significance: paired randomization over per-query MRR@10, nDCG@10, and Recall@10 on the full test split.

NFCorpus artifacts:

- Groups:
  - `runs/supervised/cross-beir/nfcorpus/nfcorpus-train-reranker-groups.k100.jsonl`
  - `runs/supervised/cross-beir/nfcorpus/nfcorpus-dev-reranker-groups.k100.jsonl`
  - `runs/supervised/cross-beir/nfcorpus/nfcorpus-test-reranker-groups.k100.jsonl`
- Telemetry cache: `runs/telemetry-cache/nfcorpus-qwen-core245-20260614`
- Capture log: `runs/supervised/cross-beir/nfcorpus/capture-qwen-core245.log`
- Predictor run: `runs/supervised/cross-beir/nfcorpus/activation-representation-searcher-raw/`
- Leakage audit: `runs/supervised/cross-beir/nfcorpus/leakage-audit.json`
- Ettin scores/sweep:
  - `runs/supervised/cross-beir/nfcorpus/nfcorpus-ettin-reranker-150m.scores.jsonl`
  - `runs/supervised/cross-beir/nfcorpus/nfcorpus-ettin-reranker-150m.blend-sweep.json`
- Paired audits:
  - `runs/supervised/cross-beir/nfcorpus/audit-full-dense-vs-actpred-raw-blend.json`
  - `runs/supervised/cross-beir/nfcorpus/audit-full-dense-vs-ettin-blend.json`
  - `runs/supervised/cross-beir/nfcorpus/audit-full-ettin-blend-vs-actpred-raw-blend.json`

NFCorpus setup and leakage controls:

- Dense pool coverage: train `2267/2590`, dev `285/324`, test `279/323` queries had positives in the top-100 pool.
- Telemetry capture: `6871` unique query/candidate chunks, one batch, no cache misses after capture.
- Predictor coverage: train `2267`, dev `285`, test `279` positive-covered groups.
- Leakage: train/test query ID overlap `0`; train/test exact normalized query-text overlap `4`; train/test positive query-doc pair overlap `0`.
- Same-corpus document overlap is high by design: train/test positive-document overlap `3115` docs and candidate-document overlap `3480` docs. Because positive query-doc pairs do not repeat, this is corpus reuse rather than exact supervised-pair leakage.

NFCorpus full-test results:

| System | Blend alpha | MRR@10 | nDCG@10 | Recall@10 | Delta nDCG vs dense | nDCG p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.0 | 0.5670 | 0.4446 | 0.3778 | 0.0000 | - |
| dense + act_pred | 0.3 | 0.5627 | 0.4544 | 0.3945 | +0.0098 | 0.0187 |
| dense + Ettin | 0.6 | 0.6182 | 0.4849 | 0.4157 | +0.0404 | 0.0001 |

NFCorpus interpretation:

- Activation prediction improves nDCG and Recall versus dense, but loses a little MRR.
- The gain is statistically visible for nDCG and Recall, but it is much smaller than the Ettin reranker.
- Dense+Ettin beats dense+act_pred by nDCG `+0.0305` with p `0.0003`; this does not reproduce the SciFact activation win.

FiQA artifacts:

- Groups:
  - `runs/supervised/cross-beir/fiqa/fiqa-train-reranker-groups.k100.jsonl`
  - `runs/supervised/cross-beir/fiqa/fiqa-dev-reranker-groups.k100.jsonl`
  - `runs/supervised/cross-beir/fiqa/fiqa-test-reranker-groups.k100.jsonl`
- Telemetry cache: `runs/telemetry-cache/fiqa-qwen-core245-20260614`
- Capture log: `runs/supervised/cross-beir/fiqa/capture-qwen-core245.log`
- Predictor run: `runs/supervised/cross-beir/fiqa/activation-representation-searcher-raw/`
- Leakage audit: `runs/supervised/cross-beir/fiqa/leakage-audit.json`
- Ettin scores/sweep:
  - `runs/supervised/cross-beir/fiqa/fiqa-ettin-reranker-150m.scores.jsonl`
  - `runs/supervised/cross-beir/fiqa/fiqa-ettin-reranker-150m.blend-sweep.json`
- Paired audits:
  - `runs/supervised/cross-beir/fiqa/audit-full-dense-vs-actpred-raw-blend.json`
  - `runs/supervised/cross-beir/fiqa/audit-full-dense-vs-ettin-blend.json`
  - `runs/supervised/cross-beir/fiqa/audit-full-ettin-blend-vs-actpred-raw-blend.json`

FiQA setup and leakage controls:

- Dense pool coverage: train `4854/5500`, dev `430/500`, test `573/648` queries had positives in the top-100 pool.
- Telemetry capture: `48757` unique query/candidate chunks, 7 batches, no cache misses after capture.
- Predictor coverage: train `4854`, dev `430`, test `573` positive-covered groups.
- Leakage: train/test query ID overlap `0`; train/test exact normalized query-text overlap `0`; train/test positive-document overlap `0`; train/test positive query-doc pair overlap `0`.
- Candidate-document overlap is still high (`23295` train/test docs) because every split retrieves from the same FiQA corpus.

FiQA full-test results:

| System | Blend alpha | MRR@10 | nDCG@10 | Recall@10 | Delta nDCG vs dense | nDCG p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.0 | 0.4755 | 0.4402 | 0.5399 | 0.0000 | - |
| dense + act_pred | 0.2 | 0.4809 | 0.4419 | 0.5366 | +0.0017 | 0.5453 |
| dense + Ettin | 0.8 | 0.5770 | 0.5501 | 0.6513 | +0.1098 | 0.0001 |

FiQA interpretation:

- The activation predictor gives only a tiny, non-significant MRR/nDCG movement over dense and slightly reduces Recall@10.
- Dense+Ettin dominates both dense and act_pred by a large, paired-significant margin.
- Pure act_pred remains weak: on the positive-covered test subset, pure model nDCG@10 is `0.0845` versus dense nDCG@10 `0.4978`.

Cross-BEIR synthesis:

- The SciFact result does not currently generalize. On two additional BEIR datasets, activation-prediction reranking is at best a small dense-complement signal and loses decisively to the text reranker baseline.
- The best activation primitive remains raw Core245 answer-representation prediction, but only because it won SciFact and produced a small NFCorpus nDCG/Recall lift. It is not a reliable replacement for dense retrieval or a strong general reranker yet.
- Leakage controls do not explain the FiQA failure: FiQA has zero repeated exact query text, positive docs, or positive query-doc pairs across train/test. NFCorpus has repeated documents and four repeated normalized query texts, but no repeated positive query-doc pairs.
- The failure mode now looks less like simple leakage and more like dataset/domain specificity or insufficient supervision/model capacity for translating query activations into answer-bearing activation geometry across heterogeneous corpora.
- Next research options:
  - improve the predictor objective and architecture, with regularization and calibration against a strong text-reranker distillation target;
  - train across multiple BEIR datasets instead of per-dataset in-domain runs;
  - run a retrieval-cost analysis, because every candidate reranked by act_pred requires stored document prefill telemetry;
  - demote activation telemetry from "candidate reranker replacement" to "diagnostic/auxiliary feature" until it beats dense+Ettin on at least one non-SciFact dataset.

## Research Sources

- Sentence Transformers CrossEncoder training overview: https://sbert.net/docs/cross_encoder/training_overview.html
- CrossEncoder usage for retrieve-and-rerank: https://sbert.net/docs/cross_encoder/usage/usage.html
- Hard-negative training for dense retrieval: https://dl.acm.org/doi/10.1145/3404835.3462880
- Negative sampling and false-negative risk survey: https://aclanthology.org/2026.findings-eacl.157.pdf
- IR paired significance tests: https://ciir-publications.cs.umass.edu/getpdf.php?id=744
- Causal feature explanation for SAEs: https://arxiv.org/html/2509.00749v1
- Sparse feature circuits: https://openreview.net/forum?id=I4e82CIDxv
- Neural feature selection for learning-to-rank: https://machinelearning.apple.com/research/neural-feature-selection
- Local and global scaling for hubness reduction: https://jmlr.org/papers/v13/schnitzer12a.html
- Hubness empirical comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC7327987/
- All-but-the-Top anisotropy postprocessing: https://openreview.net/forum?id=HkuGJ3kCb
- Dense Passage Retrieval with in-batch negatives: https://aclanthology.org/2020.emnlp-main.550/
- Sentence Transformers multiple negatives ranking loss: https://github.com/huggingface/sentence-transformers/blob/ae1acc3fb2aa2004577b297eb4a915ce7a03316a/sentence_transformers/cross_encoder/losses/multiple_negatives_ranking.py

## Meta-BEIR Answer-Predictor Training Design

Run date: 2026-06-15

Goal:

- Train a more general answer-bearing activation representation predictor by pooling train-only supervision across available BEIR datasets, while preserving an untouched heldout validation/test environment for later execution.
- Do not execute heldout validation in this phase. This phase may report training loss, train-split diagnostics, geometry diagnostics, and artifact integrity checks only.

Research basis:

- BEIR is explicitly heterogeneous and intended to expose out-of-domain generalization failures. A row-random train/test split is therefore insufficient; valid evidence needs dataset-level heldout evaluation.
- Reranker training guidance emphasizes hard negatives, because easy negatives do not teach the model the distinction between plausible and answer-bearing evidence.
- Activation representations are high-dimensional and anisotropic. Prior ablations showed raw cosine was hub-dominated; therefore the predictor should not train directly on raw cosine geometry without centering and dominant-direction controls.
- Hubness controls should be both preprocessing-level and loss-level: mean/centroid subtraction, feature scaling, optional top-PC removal or whitening, L2 scoring, in-batch contrastive alignment, and prediction-uniformity regularization.

Chosen training design:

- Training sources for this first run:
  - SciFact train split: `runs/supervised/scifact-train-reranker-groups.k100.train.jsonl`
  - NFCorpus train split: `runs/supervised/cross-beir/nfcorpus/nfcorpus-train-reranker-groups.k100.jsonl`
  - FiQA train split: `runs/supervised/cross-beir/fiqa/fiqa-train-reranker-groups.k100.jsonl`
- Activation sources:
  - SciFact: `runs/telemetry-cache/scifact-qwen-core245-20260613-215705`
  - NFCorpus: `runs/telemetry-cache/nfcorpus-qwen-core245-20260614`
  - FiQA: `runs/telemetry-cache/fiqa-qwen-core245-20260614`
- Locked non-executed validation environment:
  - SciFact test, NFCorpus dev/test, FiQA dev/test, plus future BEIR datasets after telemetry capture.
  - These are recorded in the manifest but not scored by the training script.

Geometry preprocessing:

- Start with raw Core245 SAE max-over-prompt vectors.
- Fit geometry on train-only query and candidate activations.
- Subtract the train activation centroid.
- Divide by train feature standard deviation.
- Remove the top principal components fitted on train-only vectors.
- L2-normalize all vectors at scoring time.
- Persist the fitted centroid, feature scale, PCA directions, singular values, and explained variance so validation can use exactly the same transform later.

MLP shape:

- Input/output dimension: Core245 feature count.
- Model: residual MLP with LayerNorm, GELU, dropout, and output dimension equal to activation representation dimension.
- Shape: `d -> hidden -> hidden -> d`, with optional residual projection from normalized query vector to output vector.
- Rationale: the mapping is a representation translation problem, not scalar relevance scoring. The model should predict a vector that can be compared against candidate answer representations, while the residual path preserves query activation information if the answer representation lies near a transformed query manifold.

Losses:

- Per-query listwise InfoNCE over frozen dense top-100 candidates, with qrel-positive candidates as the target distribution and dense hard negatives as in-group negatives.
- In-batch positive-centroid contrastive loss: each query prediction should prefer its own positive centroid over other examples' positive centroids. This discourages collapse to a generic answer-like hub.
- Positive-centroid cosine/MSE alignment loss: a small auxiliary term stabilizes vector prediction.
- Hard-negative margin loss: penalize the hardest negative if it scores too close to the weakest positive.
- Prediction uniformity loss across each batch to discourage conic collapse.
- Dataset-balanced example weighting so FiQA does not dominate simply because it has more train examples.

Promotion gate for later validation:

- Training succeeds only if the artifact is reproducible, train loss decreases, hubness diagnostics do not show severe prediction collapse, and the validation manifest remains locked/unexecuted.
- Later validation must compare pure dense, dense+Ettin, dense+act_pred, and blend sweeps on dataset-heldout splits. Row-level heldout is only a debugging tool, not publishable evidence.

Training implementation:

- Meta-trainer: `scripts/train_meta_activation_representation_searcher.py`
- Unit tests: `tests/test_meta_activation_representation_searcher.py`
- Run directory: `runs/supervised/meta-beir-actpred-20260615/`
- Model: `runs/supervised/meta-beir-actpred-20260615/model.pt`
- Metrics/training diagnostics: `runs/supervised/meta-beir-actpred-20260615/metrics.json`
- Locked, non-executed validation manifest: `runs/supervised/meta-beir-actpred-20260615/locked-validation-manifest.json`
- Stdout log: `runs/supervised/meta-beir-actpred-20260615/stdout.log`

Important scope note:

- This run used all BEIR datasets that currently have complete dense groups and real Qwen/RMT/SAE Core245 telemetry in the repo: SciFact, NFCorpus, and FiQA.
- It did not yet train on every BEIR dataset in the public suite. Extending to the full suite requires dense groups and prefill telemetry capture for the additional train-side datasets before they can be added without changing the protocol.
- No heldout validation or final validation run was executed.

Train-side capture continuation:

- Objective: add more train-side real prefill telemetry before any heldout validation run, using qrel-positive query/chunk capture groups as the immediate training-ingestion path.
- Quora audit: the local BEIR Quora archive has `dev` and `test` qrels but no `train.tsv`, so it was not used for train-side supervision.
- HotpotQA train qrels: `85,000` query groups, `170,001` positive qrel rows, `101,307` unique positive docs, `101,308` unique positive chunks.
- FEVER train qrels: `109,810` query groups, `163,028` positive qrel rows, `12,549` unique positive docs, `13,419` unique positive chunks.
- FEVER capture completed with real Qwen/RMT/SAE Core245 prefill telemetry: `115,711` query/chunk rows, cache `runs/telemetry-cache/fever-train-qrel-positive-qwen-core245-20260615`, verification `uncached_count=0`.
- HotpotQA capture completed with real Qwen/RMT/SAE Core245 prefill telemetry: `186,302` query/chunk rows, cache `runs/telemetry-cache/hotpotqa-train-qrel-positive-qwen-core245-20260615`, verification `uncached_count=0`.
- Group files: `runs/supervised/train-side-capture/fever/fever-train-qrel-positive-groups.jsonl` and `runs/supervised/train-side-capture/hotpotqa/hotpotqa-train-qrel-positive-groups.jsonl`.
- Logs: `runs/supervised/train-side-capture/fever/capture-qwen-core245.log`, `runs/supervised/train-side-capture/hotpotqa/capture-qwen-core245.log`, and matching `prepare-qrel-positive.log` files.
- Full dense top-100 mining for FEVER/HotpotQA/MS MARCO remains a separate storage/indexing task. Local disk is too tight for naive full-corpus dense materialization over five-million-document corpora; the correct next implementation should use remote or streaming ANN storage on `/mnt/disk-3tb`.
- The qrel-positive captures provide positives and queries for answer-representation training with in-batch negatives, centroid contrast, or later negative mining. They are not a drop-in replacement for frozen dense candidate groups with explicit dense hard negatives.

Training configuration:

| Setting | Value |
| --- | --- |
| Representation | raw Core245 SAE max-over-prompt |
| Geometry policy | train-only z-score, centroid subtraction, top-PC removal, L2 scoring |
| Top PCs removed | 3 |
| Model | residual LayerNorm MLP `245 -> 384 -> 384 -> 245` |
| Dropout | 0.10 |
| Epochs | 40 |
| Batch size | 64 |
| LR / weight decay | `3e-4` / `1e-4` |
| Temperature | 0.07 |
| Loss weights | listwise `1.0`, centroid `0.10`, in-batch `0.25`, margin `0.10`, uniformity `0.01` |

Training coverage:

| Dataset | Train group rows | Usable train examples |
| --- | ---: | ---: |
| SciFact | 688 | 676 |
| NFCorpus | 2590 | 2267 |
| FiQA | 5500 | 4854 |
| Total | 8778 | 7797 |

Training outcome:

| Epoch | Total loss | Listwise | In-batch | Centroid | Margin | Uniformity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.9164 | 4.6058 | 3.9523 | 0.9446 | 2.4012 | -1.2123 |
| 10 | 3.6701 | 2.8461 | 2.5632 | 0.7740 | 1.3891 | -3.3139 |
| 20 | 3.1332 | 2.4791 | 1.9980 | 0.7282 | 1.1699 | -3.5279 |
| 30 | 2.8424 | 2.2859 | 1.6669 | 0.7018 | 1.0562 | -3.6063 |
| 40 | 2.6362 | 2.1423 | 1.4612 | 0.6821 | 0.9681 | -3.6459 |

Train-only hubness/cone diagnostics:

- Sample size: `512` train examples.
- Mean pairwise cosine among normalized predictions: `0.0240`.
- Pairwise cosine std: `0.1490`.
- Unique nearest positive centroids: `394/512`.
- Max nearest-centroid count: `34/512`.
- Top nearest-centroid counts: `[34, 11, 8, 7, 4, 3, 3, 3, 3, 3]`.

Interpretation before heldout validation:

- The optimization behaved correctly: listwise, in-batch, centroid, and margin losses all decreased materially.
- The prediction space does not show obvious full conic collapse: mean pairwise prediction cosine is near zero, and nearest-centroid usage is spread over most of the sample.
- There is still a moderate hub in the train-only nearest-centroid diagnostic (`34/512`), so later validation should include CSLS/NICDM or calibrated dense/act_pred blending rather than relying on raw predicted-vector cosine alone.
- This training artifact is suitable for the next phase: locked heldout validation against dense and dense+Ettin. It is not evidence of retrieval improvement until that validation is run.

Full train-side MLP run after FEVER/HotpotQA capture:

- Run directory: `runs/supervised/meta-beir-actpred-full-train-20260615/`
- Remote training host: `vicuna-host`, CUDA environment `/mnt/disk-3tb/rmt-tail-venv-cu128`, GPU `NVIDIA GeForce RTX 5080`.
- Model: `runs/supervised/meta-beir-actpred-full-train-20260615/model.pt`
- Metrics: `runs/supervised/meta-beir-actpred-full-train-20260615/metrics.json`
- Locked, non-executed validation manifest: `runs/supervised/meta-beir-actpred-full-train-20260615/locked-validation-manifest.json`
- Stdout log: `runs/supervised/meta-beir-actpred-full-train-20260615/stdout.log`
- Training mode: `--allow-positive-only-groups` was explicitly enabled so qrel-positive-only FEVER/HotpotQA groups could contribute via centroid alignment and in-batch positive-centroid contrast. Default dense-negative training behavior remains unchanged.
- No heldout validation or final validation run was executed.

Full train-side coverage:

| Dataset | Usable train examples | Group type |
| --- | ---: | --- |
| SciFact | 676 | dense top-100 with qrel positives and dense negatives |
| NFCorpus | 2,267 | dense top-100 with qrel positives and dense negatives |
| FiQA | 4,854 | dense top-100 with qrel positives and dense negatives |
| FEVER | 109,810 | qrel-positive-only query/chunk groups |
| HotpotQA | 85,000 | qrel-positive-only query/chunk groups |
| Total | 202,607 | mixed dense-negative and qrel-positive-only |

Full train-side configuration:

| Setting | Value |
| --- | --- |
| Representation | raw Core245 SAE max-over-prompt |
| Geometry policy | train-only z-score, centroid subtraction, top-PC removal, L2 scoring |
| Top PCs removed | 3 |
| Model | residual LayerNorm MLP `245 -> 384 -> 384 -> 245` |
| Dropout | 0.10 |
| Epochs | 40 |
| Batch size | 256 |
| LR / weight decay | `3e-4` / `1e-4` |
| Temperature | 0.07 |
| Loss weights | listwise `1.0`, centroid `0.10`, in-batch `0.25`, margin `0.10`, uniformity `0.01` |
| Best epoch by train loss | 40 |

Full train-side training outcome:

| Epoch | Total loss | Listwise | In-batch | Centroid | Margin | Uniformity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.2788 | 2.8893 | 4.7613 | 0.9345 | 1.1826 | -1.2537 |
| 10 | 3.0045 | 1.8362 | 4.2050 | 0.8045 | 0.7322 | -3.6579 |
| 20 | 2.6696 | 1.5663 | 4.0311 | 0.7651 | 0.5661 | -3.7536 |
| 30 | 2.5103 | 1.4419 | 3.9279 | 0.7449 | 0.4970 | -3.7714 |
| 40 | 2.4200 | 1.3752 | 3.8563 | 0.7311 | 0.4544 | -3.7816 |

Full train-side train-only hubness/cone diagnostics:

- Sample size: `2048` train examples.
- Mean pairwise cosine among normalized predictions: `0.0137`.
- Pairwise cosine std: `0.1382`.
- Unique nearest positive centroids: `975/2048`.
- Max nearest-centroid count: `30/2048`.
- Top nearest-centroid counts: `[30, 21, 20, 17, 17, 16, 16, 13, 13, 11]`.

Interpretation before heldout validation:

- Optimization behaved cleanly at full train-side scale: total, listwise, centroid, in-batch, and margin losses all decreased across the 40 epochs.
- The train-only prediction geometry does not show obvious conic collapse. Mean pairwise prediction cosine is close to zero, and the largest nearest-centroid hub accounts for only `30/2048` sampled predictions.
- This is still not retrieval evidence. The next valid evidence step is locked heldout validation against pure dense and dense+Ettin baselines using the persisted geometry/model exactly as trained.

Locked heldout comparative benchmark for the full-train artifact:

- Run directory: `runs/supervised/full-train-comparative-20260615/`
- Evaluator: `scripts/evaluate_meta_activation_representation_searcher.py`
- Artifact evaluated: `runs/supervised/meta-beir-actpred-full-train-20260615/model.pt`
- Compared systems:
  - pure dense candidates;
  - dense candidates plus Ettin reranking using existing cached scores;
  - dense candidates plus full-train answer-bearing activation-prediction reranking;
  - pure answer-bearing activation-prediction retrieval over the available activation index.
- Scope caveat: pure activation-prediction retrieval ranks the union of captured heldout candidate chunks for each dataset, not an uncaptured full BEIR corpus.

Heldout test metrics:

| Dataset | System | MRR@10 | nDCG@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: |
| SciFact test | pure dense candidates | 0.7003 | 0.7451 | 0.8659 |
| SciFact test | dense candidates + Ettin | 0.7288 | 0.7641 | 0.8587 |
| SciFact test | dense candidates + full-train actpred | 0.4181 | 0.4446 | 0.5336 |
| SciFact test | pure actpred available index | 0.3410 | 0.3439 | 0.3647 |
| NFCorpus test | pure dense candidates | 0.5670 | 0.4446 | 0.3778 |
| NFCorpus test | dense candidates + Ettin | 0.5964 | 0.4547 | 0.3918 |
| NFCorpus test | dense candidates + full-train actpred | 0.2178 | 0.1601 | 0.1497 |
| NFCorpus test | pure actpred available index | 0.0368 | 0.0234 | 0.0221 |
| FiQA test | pure dense candidates | 0.4755 | 0.4402 | 0.5399 |
| FiQA test | dense candidates + Ettin | 0.5605 | 0.5365 | 0.6430 |
| FiQA test | dense candidates + full-train actpred | 0.0703 | 0.0731 | 0.1344 |
| FiQA test | pure actpred available index | 0.0020 | 0.0020 | 0.0043 |

Paired test summary:

| Dataset | Comparison | Delta nDCG@10 | p |
| --- | --- | ---: | ---: |
| SciFact test | full-train actpred minus dense | -0.3006 | 0.0001 |
| SciFact test | Ettin minus dense | +0.0189 | 0.2104 |
| SciFact test | full-train actpred minus Ettin | -0.3195 | 0.0001 |
| NFCorpus test | full-train actpred minus dense | -0.2845 | 0.0001 |
| NFCorpus test | Ettin minus dense | +0.0101 | 0.3667 |
| NFCorpus test | full-train actpred minus Ettin | -0.2946 | 0.0001 |
| FiQA test | full-train actpred minus dense | -0.3672 | 0.0001 |
| FiQA test | Ettin minus dense | +0.0963 | 0.0001 |
| FiQA test | full-train actpred minus Ettin | -0.4634 | 0.0001 |

Interpretation:

- The pure activation-prediction score is not viable as a standalone reranker. It loses decisively to pure dense candidates and to Ettin on every heldout test set.
- The pure-score failure is not subtle or merely nonsignificant. The actpred-only reranker harms nDCG@10 by roughly `-0.28` to `-0.37` versus dense, with paired randomization p near the minimum possible under 10k iterations.
- Pure activation-prediction retrieval over the available activation index is much worse than dense on all three datasets. This is especially severe on FiQA, where nDCG@10 is approximately `0.002`.
- This does not by itself reverse the earlier smaller SciFact-specific activation-predictor finding, because that finding used a dense/activation score blend rather than pure activation reranking. The comparable blended diagnostic is below.

Post-hoc diagnostic: dense/activation blend with the full-train artifact:

| Dataset | Best alpha | MRR@10 | nDCG@10 | Recall@10 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact test | 0.40 | 0.7711 | 0.8059 | 0.8996 | beats dense and Ettin |
| NFCorpus test | 0.25 | 0.5672 | 0.4475 | 0.3871 | tiny lift over dense, below Ettin |
| FiQA test | 0.00 | 0.4755 | 0.4402 | 0.5399 | best setting ignores actpred |

SciFact blended paired audit:

| Comparison | Delta nDCG@10 | p | nDCG changed queries |
| --- | ---: | ---: | --- |
| full-train actpred blend minus dense | +0.0608 | 0.0001 | 57 improved / 32 harmed / 211 unchanged |
| full-train actpred blend minus Ettin | +0.0419 | 0.0075 | 63 improved / 43 harmed / 194 unchanged |

Fixed and validation-selected blend arms:

The test-selected sweep above is useful diagnostically but optimistic because it selects alpha on the same test queries. Two stricter blend arms were therefore run:

- fixed `70/30` dense/activation blend, matching the earlier SciFact-style protocol (`alpha=0.30`);
- dev-selected alpha, chosen on each dataset dev split and then applied once to test.

Fixed `70/30` test results:

| Dataset | Alpha | MRR@10 | nDCG@10 | Recall@10 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact test | 0.30 | 0.7631 | 0.7998 | 0.8996 | beats dense and Ettin |
| NFCorpus test | 0.30 | 0.5583 | 0.4429 | 0.3837 | below dense nDCG and below Ettin |
| FiQA test | 0.30 | 0.4526 | 0.4208 | 0.5214 | below dense and Ettin |

Fixed `70/30` paired audit:

| Dataset | Comparison | Delta nDCG@10 | p | nDCG changed queries |
| --- | --- | ---: | ---: | --- |
| SciFact test | fixed blend minus dense | +0.0547 | 0.0001 | 55 improved / 26 harmed / 219 unchanged |
| SciFact test | fixed blend minus Ettin | +0.0357 | 0.0163 | 62 improved / 41 harmed / 197 unchanged |
| NFCorpus test | fixed blend minus dense | -0.0017 | 0.7005 | 81 improved / 90 harmed / 152 unchanged |
| NFCorpus test | fixed blend minus Ettin | -0.0118 | 0.3005 | 105 improved / 111 harmed / 107 unchanged |
| FiQA test | fixed blend minus dense | -0.0194 | 0.0001 | 92 improved / 167 harmed / 389 unchanged |
| FiQA test | fixed blend minus Ettin | -0.1156 | 0.0001 | 110 improved / 283 harmed / 255 unchanged |

Dev-selected alphas:

| Dataset | Selected alpha | Dev nDCG@10 | Test MRR@10 | Test nDCG@10 | Test Recall@10 | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.35 | 0.7936 | 0.7709 | 0.8058 | 0.9029 | beats dense and Ettin |
| NFCorpus | 0.25 | 0.4193 | 0.5672 | 0.4475 | 0.3871 | tiny dense lift, below Ettin |
| FiQA | 0.20 | 0.4650 | 0.4653 | 0.4311 | 0.5312 | below dense and Ettin |

Dev-selected paired audit:

| Dataset | Comparison | Delta nDCG@10 | p | nDCG changed queries |
| --- | --- | ---: | ---: | --- |
| SciFact test | dev-selected blend minus dense | +0.0606 | 0.0001 | 56 improved / 31 harmed / 213 unchanged |
| SciFact test | dev-selected blend minus Ettin | +0.0417 | 0.0056 | 65 improved / 40 harmed / 195 unchanged |
| NFCorpus test | dev-selected blend minus dense | +0.0029 | 0.4341 | 87 improved / 79 harmed / 157 unchanged |
| NFCorpus test | dev-selected blend minus Ettin | -0.0072 | 0.5207 | 108 improved / 108 harmed / 107 unchanged |
| FiQA test | dev-selected blend minus dense | -0.0091 | 0.0030 | 85 improved / 128 harmed / 435 unchanged |
| FiQA test | dev-selected blend minus Ettin | -0.1053 | 0.0001 | 112 improved / 281 harmed / 255 unchanged |

Corrected interpretation:

- The benchmark distinguishes three different systems that should not be conflated: pure actpred reranking, pure actpred retrieval, and dense/actpred blending.
- The earlier SciFact win is reproducible in kind for the full-train artifact when the same dense/activation blending protocol is used. It is weaker than the old smaller SciFact-specific artifact on its matched subset, but it still beats Ettin on the full SciFact test candidate groups.
- Cross-dataset generalization remains weak. NFCorpus gets only a very small blended gain over dense and still trails Ettin; FiQA is harmed by both fixed and dev-selected activation blending on test.
- The full-train MLP was trained as a standalone activation-prediction scorer, not under blend-calibrated/listwise conditions that include the dense score. The blend win is therefore real as a composition, but the current artifact is not yet trained for the objective that actually performs best.
- The likely training issue is not global vector collapse. It is objective/data mismatch: FEVER and HotpotQA dominate the added train-side data as qrel-positive-only groups, so most new supervision teaches query-to-positive-centroid alignment without dense hard negatives or candidate-list calibration. That can preserve a useful weak SciFact signal for blending while making pure activation ranking poor and cross-domain calibration unreliable.
- Next research should compare the old smaller artifact and the full-train artifact under the same blended protocol, then add real dense hard negatives for FEVER/HotpotQA before another full retrain. Positive-only expansion is not enough for a standalone activation reranker.

## Dataset-Specific Learned Blended Rerankers

Objective:

- Abandon the single universal activation reranker as the primary artifact.
- Train/evaluate dataset-specific blended rerankers for SciFact, NFCorpus, and FiQA.
- Treat the activation score as a learned/calibrated reranking feature over dense candidates, not as a standalone retrieval score.
- Preserve heldout discipline: train or score models on train, select blend weights on dev, and evaluate once on test.

Artifacts:

- Learned blend trainer: `scripts/train_learned_blended_reranker.py`
- Dataset-specific checkpoint evaluator: `scripts/evaluate_activation_representation_searcher.py`
- Run directory: `runs/supervised/dataset-blended-rerankers-20260615/`
- Input feature: raw answer-bearing activation-prediction score from the dataset-specific Core245 answer-representation searcher, blended with per-query z-scored dense score.

Two blend regimes were launched:

1. Constrained alpha selection: choose a scalar dense/activation alpha on dev, then apply that fixed alpha to test.
2. Pairwise learned-linear blend: optimize a query-group pairwise logistic objective over `z_dense`, `z_actpred`, reciprocal dense rank, `z_dense*z_actpred`, and `z_actpred-z_dense`, with dense-initialized epoch-0 checkpoint selection on dev.

The learned-linear regime exposed a calibration failure. Because the actpred scorer is trained on the same train split, train actpred scores are much cleaner than dev/test scores. Pairwise training overfits this train-only cleanliness and often learns destructive activation-heavy weights. Adding a dense epoch-0 checkpoint prevents destructive promotion, but then SciFact and NFCorpus select dense-only. FiQA learns a small positive linear blend on covered queries, but the full-test comparable alpha artifact remains the cleaner benchmark object.

Constrained alpha results, selected on dev and applied to heldout test:

| Dataset | Selected alpha | MRR@10 | nDCG@10 | Recall@10 | Dense nDCG@10 | Raw Ettin nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact test | 0.30 | 0.7786 | 0.8112 | 0.9002 | 0.7451 | 0.7641 |
| NFCorpus test | 0.25 | 0.5692 | 0.4561 | 0.3953 | 0.4446 | 0.4547 |
| FiQA test | 0.20 | 0.4809 | 0.4419 | 0.5366 | 0.4402 | 0.5365 |

Paired audits for constrained alpha:

| Dataset | Comparison | Delta MRR@10 | Delta nDCG@10 | Delta Recall@10 | p nDCG@10 | nDCG changed queries |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SciFact test | alpha blend minus dense | +0.0783 | +0.0660 | +0.0343 | 0.0001 | 62 improved / 29 harmed / 209 unchanged |
| SciFact test | alpha blend minus raw Ettin | +0.0498 | +0.0471 | +0.0416 | 0.0042 | 64 improved / 45 harmed / 191 unchanged |
| NFCorpus test | alpha blend minus dense | +0.0022 | +0.0115 | +0.0174 | 0.0017 | 96 improved / 64 harmed / 163 unchanged |
| NFCorpus test | alpha blend minus raw Ettin | -0.0272 | +0.0013 | +0.0035 | 0.9019 | 110 improved / 104 harmed / 109 unchanged |
| FiQA test | alpha blend minus dense | +0.0054 | +0.0017 | -0.0033 | 0.5453 | 90 improved / 92 harmed / 466 unchanged |
| FiQA test | alpha blend minus raw Ettin | -0.0796 | -0.0946 | -0.1064 | 0.0001 | 123 improved / 265 harmed / 260 unchanged |

Interpretation:

- SciFact remains the strong result. Dataset-specific activation prediction plus dev-selected dense/activation blending decisively beats both dense and raw Ettin on heldout test.
- NFCorpus now shows a real but modest activation contribution over dense: nDCG and Recall improve with paired significance versus dense. It is effectively tied with raw Ettin on nDCG but loses on MRR, and it remains below the earlier stronger dense+Ettin blend-best reference.
- FiQA still does not clear the bar. The activation blend has a tiny, nonsignificant nDCG lift over dense and loses decisively to Ettin.
- The publishable claim should be scoped as task-sensitive activation telemetry: useful for claim/evidence retrieval and modestly useful for broad biomedical IR, not a general reranker.
- The next training improvement should not be a naive pairwise blend over in-sample actpred train scores. It should use cross-fitted/out-of-fold actpred train scores, direct blend-aware training of the activation predictor, or a LambdaMART-style ranker once the feature leakage is controlled.

## Direct-Blend Activation-Aware Training

Objective:

- Move beyond post-hoc alpha blending by training the answer-representation predictor against the final dense-plus-activation ranking score.
- Avoid the stacked-feature overfitting observed in the pairwise learned-linear blend.
- Use a listwise query-group loss over `final_score = (1-alpha) * z_dense + alpha * z_activation_prediction`, with alpha selected on dev.

Method note:

- Cross-fitted/out-of-fold scores are the correct next step for stacked/meta-feature training because each training prediction must come from a model that did not train on that row. Scikit-learn documents this as cross-validated estimates where each sample belongs to one test fold.
- Direct blend-aware training attacks a different issue: the activation predictor is optimized for the ranking score we actually deploy, rather than for pure activation cosine followed by post-hoc blending.
- Future tree-based LTR work should use a grouped ranking objective such as XGBoost `rank:ndcg`/LambdaMART, which is designed to optimize ranking metrics over query groups.

Artifacts:

- Trainer: `scripts/train_direct_blend_activation_searcher.py`
- Run directory: `runs/supervised/direct-blend-rerankers-20260615/`
- Full-test score files with dense fallback for uncovered activation rows:
  - `runs/supervised/direct-blend-rerankers-20260615/scifact/test-scores.full-fallback.jsonl`
  - `runs/supervised/direct-blend-rerankers-20260615/nfcorpus/test-scores.full-fallback.jsonl`
  - `runs/supervised/direct-blend-rerankers-20260615/fiqa/test-scores.full-fallback.jsonl`

Training status:

- SciFact completed the full direct-blend grid `alpha in {0.1, 0.2, 0.3, 0.4}`, 60 epochs, selected `alpha=0.4`, epoch `27`.
- NFCorpus completed the same full grid, selected `alpha=0.1`, epoch `13`.
- FiQA full grid was too slow with the unbatched CPU trainer and was interrupted. A bounded same-objective probe ran with `alpha=0.2`, 5 epochs. This is sufficient as a diagnostic but should not be treated as the final optimized FiQA direct-blend artifact.

Full-test comparable results use dense fallback for query/candidate rows without activation coverage:

| Dataset | Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: |
| SciFact test | dense | 0.7003 | 0.7451 | 0.8659 |
| SciFact test | raw Ettin | 0.7288 | 0.7641 | 0.8587 |
| SciFact test | current constrained alpha | 0.7786 | 0.8112 | 0.9002 |
| SciFact test | direct blend | 0.7951 | 0.8232 | 0.8968 |
| NFCorpus test | dense | 0.5670 | 0.4446 | 0.3778 |
| NFCorpus test | raw Ettin | 0.5964 | 0.4547 | 0.3918 |
| NFCorpus test | current constrained alpha | 0.5692 | 0.4561 | 0.3953 |
| NFCorpus test | direct blend | 0.5730 | 0.4449 | 0.3748 |
| FiQA test | dense | 0.4755 | 0.4402 | 0.5399 |
| FiQA test | raw Ettin | 0.5605 | 0.5365 | 0.6430 |
| FiQA test | current constrained alpha | 0.4809 | 0.4419 | 0.5366 |
| FiQA test | direct blend probe | 0.4718 | 0.4383 | 0.5399 |

Paired audit deltas for direct blend:

| Dataset | Comparison | Delta MRR@10 | Delta nDCG@10 | Delta Recall@10 | p nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | direct minus dense | +0.0947 | +0.0781 | +0.0309 | 0.0001 |
| SciFact | direct minus raw Ettin | +0.0662 | +0.0592 | +0.0381 | 0.0013 |
| SciFact | direct minus current alpha | +0.0165 | +0.0121 | -0.0034 | 0.2337 |
| NFCorpus | direct minus dense | +0.0060 | +0.0003 | -0.0030 | 0.8928 |
| NFCorpus | direct minus raw Ettin | -0.0234 | -0.0098 | -0.0170 | 0.3806 |
| NFCorpus | direct minus current alpha | +0.0038 | -0.0112 | -0.0204 | 0.0047 |
| FiQA | direct probe minus dense | -0.0038 | -0.0020 | +0.0000 | 0.5325 |
| FiQA | direct probe minus raw Ettin | -0.0888 | -0.0982 | -0.1031 | 0.0001 |
| FiQA | direct probe minus current alpha | -0.0091 | -0.0036 | +0.0033 | 0.2883 |

Interpretation:

- Direct blend materially strengthens the SciFact result. It improves over dense and raw Ettin with paired significance, and it improves mean nDCG/MRR over the current constrained-alpha activation blend. The direct-minus-current-alpha improvement is not paired-significant on nDCG, but the absolute heldout score is the best observed SciFact activation result in this line.
- Direct blend does not improve NFCorpus. It slightly improves MRR over dense but loses nDCG/Recall versus the current constrained-alpha activation blend, and it remains below Ettin on the main metrics.
- Direct blend does not rescue FiQA. Even the bounded direct objective probe is slightly below dense/current-alpha and far below Ettin.
- The next serious engineering task is to batch/vectorize the direct-blend trainer before running larger grids, especially FiQA. The next serious modeling task is cross-fitted actpred scores or grouped LambdaMART over dense plus activation features, because direct neural training only produced a clear win on SciFact.

## TechQA-RAG-Eval Documentation Retrieval Transfer

Purpose: test whether the SciFact-trained direct-blend answer-bearing activation predictor transfers from scientific claim evidence to technical documentation retrieval.

Dataset construction:

- Source: `nvidia/TechQA-RAG-Eval`.
- Usable retrieval rows: answerable rows only. The package has 610 answerable questions with one linked IBM Technote context each, and 300 unanswerable rows with no context.
- Corpus: unique linked Technote contexts from the answerable rows.
- Labels: chunk-level positives, selected by normalized answer-span overlap where possible, then answer-line substring matching, then a lexical-overlap fallback within the linked Technote.
- Candidate pool: global dense retrieval over all TechQA chunks, with the positive answer-bearing chunk forced into the candidate group if dense top-k misses it so rerankers are evaluated on comparable groups.

Arms:

- pure dense candidates
- dense candidates plus Ettin reranking
- dense candidates plus the SciFact-trained direct-blend activation-prediction reranker, using the SciFact-selected blend alpha from `runs/supervised/direct-blend-rerankers-20260615/scifact/model.pt`

Interpretation guardrail: this is a transfer benchmark over a derived chunk-level retrieval task, not the original TechQA blind leaderboard. A positive result would indicate domain transfer into technical documentation retrieval. A negative result would not falsify the SciFact result, but would strongly motivate training a TechQA-specific actpred MLP before claiming documentation-domain utility.

Observed run:

- Prepared corpus: 496 unique IBM Technotes, 924 chunks.
- Query set: 610 answerable questions.
- Positive labels: 601 by answer-span overlap, 5 by answer-line substring match, 4 by lexical-overlap fallback.
- Candidate pool: top-100 dense candidates per query, with all 610 queries containing a positive answer-bearing chunk.
- Prefill telemetry: 1,505 unique query/candidate rows captured through Qwen/RMT/SAE Core245 zero-token prefill into `runs/telemetry-cache/techqa-qwen-core245-20260615`.
- Ettin scoring required `max_length=512`; the unconstrained remote cross-encoder path hit CUDA OOM on long TechQA technical chunks.

Results:

| Method | MRR@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense BGE-base candidates | 0.8041 | 0.8315 | 0.9189 |
| Dense + Ettin reranker | 0.7931 | 0.8227 | 0.9164 |
| Dense + SciFact direct-blend actpred | 0.7791 | 0.8076 | 0.9008 |
| Pure SciFact actpred score | 0.0315 | 0.0497 | 0.1115 |

Paired audits:

| Comparison | Delta MRR@10 | Delta nDCG@10 | Delta Recall@10 | p nDCG@10 | nDCG changed queries |
| --- | ---: | ---: | ---: | ---: | --- |
| Ettin minus dense | -0.0110 | -0.0088 | -0.0025 | 0.4306 | 82 improved / 96 harmed / 432 unchanged |
| SciFact direct-blend minus dense | -0.0250 | -0.0240 | -0.0180 | 0.0002 | 35 improved / 74 harmed / 501 unchanged |
| SciFact direct-blend minus Ettin | -0.0140 | -0.0151 | -0.0156 | 0.2111 | 90 improved / 96 harmed / 424 unchanged |

Interpretation:

- The SciFact-trained activation predictor does not transfer to TechQA technical support documentation. The degradation versus dense is paired-significant.
- Dense retrieval is unusually strong on this derived TechQA task, likely because many questions and Technote titles/errors share exact product names, error codes, CVEs, and support-prose terminology.
- Ettin also fails to improve over dense under this formulation. That makes TechQA a useful hard negative for the "generic reranker helps" assumption, not just for activation transfer.
- This supports the dataset-specific hyper-reranker direction. If TechQA remains important, the next valid experiment is a TechQA-specific actpred/direct-blend MLP trained on heldout-safe TechQA splits, not reuse of the SciFact checkpoint.

## Vertical Benchmark Queue

The better external validation target is a family of expert-prose retrieval benchmarks rather than generic BEIR transfer. Candidate verticals:

- Legal: MLEB and LegalBench-RAG. MLEB is a broad legal embedding benchmark, while LegalBench-RAG focuses on precise legal snippet retrieval rather than whole-document retrieval.
- Biomedical: R2MED. R2MED is a high-resolution medical IR benchmark with 876 queries across multiple retrieval tasks, medical scenarios, and body systems.
- Coding: CoREB. CoREB is a contamination-limited code retrieval and reranking benchmark with graded relevance judgments across text-to-code, code-to-text, and code-to-code tasks.

These are a better match for the emerging claim: activation-aware methods may be useful as domain-specific hyper-rerankers for expert, structured prose where answer-bearing evidence is syntactically tight and high-stakes. They should be prepared as separate vertical adapters with train/dev/test discipline, not pooled into another "one reranker to rule them all" artifact.

## Vertical-Specific Direct-Blend Rerankers

Run date: 2026-06-15

Decision:

- Stop treating the SciFact direct-blend activation reranker as a universal artifact.
- Train separate direct-blend answer-activation rerankers for legal, biomedical, and coding verticals.
- Keep the same core protocol as the SciFact direct-blend result: train-only supervision, dev-selected alpha/checkpoint, and heldout evaluation against candidate-order/dense and dense+Ettin where that baseline is meaningful.

Dataset access findings:

- Legal: `isaacus/mleb-legal-rag-bench` exposes MTEB-style `queries`, `corpus`, and qrel `default` configs. It is the MLEB legal RAG task and is a more direct first legal adapter than attempting all MLEB tasks at once.
- Biomedical: R2MED exposes MTEB-compatible retrieval datasets such as `mteb/R2MEDMedQADiagRetrieval`, each with `queries`, `corpus`, and `qrels` configs. This supports normal dense-candidate mining, but full all-task R2MED capture can be large, so begin with one diagnostic subset and only escalate after telemetry throughput is confirmed.
- Coding: CoREB exposes MTEB-compatible reranking datasets such as `mteb/coreb-t2c-reranking`, `mteb/coreb-c2t-reranking`, and `mteb/coreb-c2c-reranking` with `queries`, `corpus`, qrel `default`, and `top_ranked` configs. These are ideal for first vertical training because candidate sets are already fixed.

Split policy:

- Several of these HF datasets publish only a `test` split. For model training, create deterministic query-level internal train/dev/test splits and report them as adaptation splits rather than official benchmark-standard splits.
- Do not compare these internal split results to leaderboard numbers. Use them only to answer whether activation-aware reranking can learn a vertical-specific improvement over the candidate-order/dense baseline and the available text reranker baseline.

Candidate-pool policy:

- MLEB legal RAG and R2MED MedQA-Diag both had weak BGE positive-in-pool coverage at bounded candidate depths. Legal improved from 49/100 at k=50 to 85/100 at k=500; R2MED MedQA-Diag reached only 58/118 at k=100.
- For supervised vertical training, use `--append-qrel-positives` candidate pools: keep the dense top-k hard negatives and append missing qrel positives with their true dense cosine scores. This is not benchmark-standard first-stage retrieval, but it is the right training pool for learning whether activation prediction can lift answer-bearing expert passages that dense retrieval scored too low.
- Report any later benchmark result from appended-positive pools separately from true dense-candidate retrieval results.

Observed vertical direct-blend run:

- Group preparation:
  - MLEB legal RAG: 100 query groups, k=100 dense candidates plus appended missing positives, train/dev/test = 60/20/20.
  - R2MED MedQA-Diag: 118 query groups, k=100 dense candidates plus appended missing positives, train/dev/test = 82/18/18.
  - CoREB combined reranking: 2,354 query groups from t2c/c2t/c2c native reranking candidate lists, train/dev/test = 1,646/354/354.
- Telemetry capture:
  - MLEB legal RAG: 3,247 unique query/candidate rows captured.
  - R2MED MedQA-Diag: 7,529 unique query/candidate rows captured.
  - CoREB combined: 4,961 unique query/candidate rows captured.
- Training: all three direct-blend answer-activation rerankers ran on vicuna CUDA using raw Core245 vectors, hidden dimension 128, dev-selected alpha from `{0.1, 0.2, 0.3, 0.4}`.

Results:

| Dataset | Selected alpha | Dev dense nDCG@10 | Dev model nDCG@10 | Test dense nDCG@10 | Test model nDCG@10 | Test conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLEB legal RAG | 0.4 | 0.2931 | 0.3837 | 0.2491 | 0.1544 | Dev lift did not generalize; clear heldout degradation. |
| R2MED MedQA-Diag | 0.4 | 0.1926 | 0.2013 | 0.0687 | 0.0487 | Tiny dev lift with heldout degradation. |
| CoREB combined reranking | 0.1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Saturated candidate order; not diagnostic. |

Interpretation:

- These first vertical-specific direct-blend runs do not replicate the SciFact win.
- The appended-positive legal and R2MED pools are useful for supervised training diagnostics, but they are small enough that the current MLP overfits train/dev and fails heldout.
- CoREB's published top-ranked reranking order, as loaded here, places positives at rank 1 for every prepared group. This makes it unusable for testing activation reranking unless we construct harder candidate pools or use a different CoREB split/configuration where the first-stage order is not already perfect.
- The next modeling step should not be simply "train more epochs." It should be either cross-fitted calibration on larger vertical train pools, harder candidate generation for code, or a lower-capacity/strongly regularized vertical model selected by more stable folds.

## Robust Query+Candidate Behavior-Latent Reranking

Run date: 2026-06-18

Purpose: replace query-only or candidate-only activation similarity with a canonical query-plus-candidate support prompt, then classify whether the candidate behaviorally supports the query from the pair-prompt Core245 max-prefill telemetry. This is still not the future ideal final-token SAE/CAA selector primitive; it is a stronger Core245 pair-prompt proxy designed to test whether activation telemetry becomes useful when captured on the actual query/candidate decision.

Capture and hygiene:

- Prepared full-split dense-hard-negative candidate groups for SciFact, LegalBench-RAG, and pooled R2MED with 16 candidates per query.
- Captured 141,592 query/candidate pair-prompt telemetry rows through the resume-safe cache.
- Validation found 141,592 valid rows, zero invalid rows, all rows labeled `query_candidate_behavior_prompt`, mean active Core245 features 61.27, minimum active 18, maximum active 88.
- The merged request JSONL was deleted after cache completion to relieve local scratch pressure. The durable telemetry cache and all train/dev/test group files remain.
- Disk finding during the run: the 3TB vicuna data volume was not exhausted. `/mnt/disk-3tb` had about 1.8T free; the pressure was local/root scratch and generated JSONL artifacts.

Training protocol:

- Artifacts: SciFact-specific, LegalBench-RAG-specific, R2MED-specific, and mixed general model.
- Feature transform: `log1p_l2`.
- Model: 245-input behavior vector, 128 hidden units, dropout 0.15, label smoothing 0.03, weight decay 0.001, gradient clipping 1.0.
- Optimization: group/listwise binary support objective, train-only fitting, dev-selected alpha over `{0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0}`, heldout test reporting.
- All completed models selected `alpha=1.0`, meaning the learned behavior telemetry score dominated the final ranking under this protocol.

Heldout test results:

| Dataset/model | Dense nDCG@10 | Ettin nDCG@10 | Actpred nDCG@10 | Behavior nDCG@10 | Main paired result |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact-specific | 0.7682 | 0.8161 | 0.8514 | 0.8381 | Behavior beats dense, p=0.0001; numerically above Ettin but not paired-significant; below actpred by a non-significant 0.0134 nDCG. |
| SciFact general | 0.7682 | 0.8161 | 0.8514 | 0.8016 | General behavior is a weaker transfer model; dense delta is not significant at 0.05. |
| R2MED-specific | 0.1732 | 0.4586 | 0.1762 | 0.8414 | Behavior beats dense, actpred, and Ettin, all p=0.0001. |
| R2MED general | 0.1732 | 0.4586 | 0.1762 | 0.8616 | General behavior is strongest on R2MED; beats dense, actpred, and Ettin, all p=0.0001. |
| LegalBench-RAG-specific | 0.2733 | 0.5023 | 0.2884 | 0.7292 | Behavior beats dense, actpred, and Ettin, all p=0.0001. |
| LegalBench-RAG general | 0.2733 | 0.5023 | 0.2884 | 0.7510 | General behavior is strongest on LegalBench-RAG; beats dense, actpred, and Ettin, all p=0.0001. |

Interpretation:

- The query+candidate behavior objective is a qualitatively different result from the earlier answer-bearing activation-prediction reranker. It does not merely compare separate query and document representations; it captures the model's behavior on the candidate-evidence decision prompt.
- The large LegalBench-RAG and R2MED lifts suggest the failure of prior actpred transfer was not simply "activation telemetry has no ranking signal." The signal appears much stronger when telemetry is captured on the exact query/candidate support decision rather than on isolated chunks or predicted answer-bearing chunk representations.
- R2MED now has a dense plus Ettin comparison point over the same frozen robust test candidate groups. Ettin strongly improves over dense (`+0.2854` nDCG@10, p=0.0001), but the general behavior model still beats Ettin by `+0.4031` nDCG@10 with paired p=0.0001.
- SciFact remains strong but no longer uniquely dominant. The behavior-latent objective comes close to the best actpred SciFact result, while materially exceeding Ettin only numerically on this split.
- The general model outperforming dataset-specific models on LegalBench-RAG and R2MED is encouraging but also a leakage/control warning: the next validation pass should audit source/query identity separation, duplicate passages, and candidate construction artifacts before making a public claim.
- The result justifies pursuing a true final-token SAE/CAA implementation for this query+candidate prompt. The current Core245 max-prefill proxy is strong enough to warrant replacing the proxy with the actual selector-compatible telemetry shape before publication-grade claims.

## Behavior-Latent Deduplication And Leakage Audit

Run date: 2026-06-21

Purpose: address the leakage/control warning from the robust behavior-latent results by auditing exact identifiers, exact normalized text, near-duplicate query text, repeated evidence text, and cross-split positive-evidence reuse. This audit is stricter than the earlier query-id and positive-pair checks.

Implementation updates:

- `scripts/audit_group_leakage.py` now records candidate and positive text hashes in addition to query/doc/pair IDs.
- The audit reports exact candidate-text overlap, exact positive-text overlap, train/dev positive text appearing in heldout candidates, near-duplicate query text, and near-duplicate train/dev positive text appearing in heldout candidates.
- Near duplicates use canonicalized word-shingle Jaccard with configurable `--near-duplicate-threshold`, `--shingle-size`, and `--max-shingle-postings`.
- `tests/test_audit_group_leakage.py` now covers exact candidate/positive text overlap and near-duplicate query detection.

Audit artifacts:

- SciFact: `runs/behavior-latent-robust-20260618/scifact/leakage-dedup-audit.json`
- LegalBench-RAG: `runs/behavior-latent-robust-20260618/legalbenchrag/leakage-dedup-audit.json`
- R2MED: `runs/behavior-latent-robust-20260618/r2med-all/leakage-dedup-audit.json`

Core audit finding:

- No exact query-id overlap was found across train/dev/test for SciFact, LegalBench-RAG, or R2MED.
- No exact positive `(query_id, doc_id)` pair overlap was found across train/dev/test.
- No exact candidate `(query_id, doc_id)` pair overlap was found across train/dev/test.
- Repeated evidence text does appear across splits, especially in SciFact and LegalBench-RAG. This is corpus reuse rather than exact supervised-pair leakage, but it can still make a learned reranker look stronger if it memorizes answer-bearing passages or passage signatures.

Train-vs-test audit details:

| Dataset | Test queries | Query text overlap | Near-duplicate query text | Positive text overlap | Candidate text overlap | Train-positive text in test candidates | Near-duplicate train-positive text in test candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 291 | 2 | 2 | 161 | 2208 | 329 | 988 |
| LegalBench-RAG | 689 | 6 | 6 | 190 | 3765 | 2859 | 10591 |
| R2MED | 131 | 0 | 0 | 19 | 489 | 57 | 65 |

Strict heldout construction:

Heldout test groups were filtered by dropping any query whose normalized query text duplicated train/dev query text, or whose positive evidence doc/text appeared as train/dev positive evidence. This is intentionally conservative and can remove legitimate benchmark queries over a shared corpus.

Strict heldout artifacts:

- SciFact: `runs/behavior-latent-robust-20260618/scifact/test-groups.strict-no-train-dev-positive-overlap.jsonl`
- LegalBench-RAG: `runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.strict-no-train-dev-positive-overlap.jsonl`
- R2MED: `runs/behavior-latent-robust-20260618/r2med-all/test-groups.strict-no-train-dev-positive-overlap.jsonl`

Strict subset sizes:

| Dataset | Source test queries | Strict kept | Dropped | Drop rate |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 291 | 103 | 188 | 64.6% |
| LegalBench-RAG | 689 | 525 | 164 | 23.8% |
| R2MED | 131 | 116 | 15 | 11.5% |

Strict heldout nDCG@10 results:

| Dataset/model | Dense | Ettin | Actpred | Behavior-specific | Behavior-general | Main strict result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.7626 | 0.8096 | 0.7517 | 0.7123 | 0.7960 | SciFact weakens materially as a dense/Ettin headline. General behavior is above dense by `+0.0334` but not significant (`p=0.3661`), below Ettin by `-0.0136` but not significant (`p=0.7284`), and above actpred by `+0.0443` but not significant (`p=0.2452`). Specific behavior falls below dense and Ettin. |
| LegalBench-RAG | 0.2655 | 0.5055 | 0.2722 | 0.7177 | 0.7437 | LegalBench-RAG survives. General behavior beats dense by `+0.4782`, p approximately `0.0001`, and beats Ettin by `+0.2382`, p approximately `0.0001`. |
| R2MED | 0.1862 | 0.4711 | 0.1879 | 0.8308 | 0.8527 | R2MED survives. General behavior beats dense by `+0.6665`, p approximately `0.0001`, and beats Ettin by `+0.3816`, p approximately `0.0001`. |

Interpretation update:

- The strict audit materially weakens the SciFact dense/Ettin headline. SciFact remains useful as a discovery and full-split positive result, and the general behavior model still beats actpred by a clean `+0.0443` nDCG@10 on the strict subset, but SciFact is not the cleanest controlled claim after duplicate-evidence filtering because the actpred margin is not paired-significant and Ettin remains numerically higher.
- LegalBench-RAG and R2MED remain decisive even after strict filtering. These are now the primary controlled evidence for the behavior-latent reranker.
- The broad claim should be revised from "decisive wins on SciFact, LegalBench-RAG, and R2MED" to "decisive strict-control wins on LegalBench-RAG and R2MED, with full-split SciFact support and a weakened SciFact strict subset."
- Repeated evidence text is not the same as exact pair leakage. The system did not see the same query/evidence pair in train and test. Still, strict no-train/dev-positive-overlap reporting is required for public claims because behavior telemetry can plausibly learn passage-level support signatures.

## Expanded Off-The-Shelf Text Reranker Baselines

Run date: 2026-06-21

Purpose: test whether the behavior-latent result is merely beating Ettin specifically, or whether it survives a broader off-the-shelf text-reranker envelope over the same frozen dense candidate groups.

Models scored:

- `cross-encoder/ettin-reranker-150m-v1`
- `BAAI/bge-reranker-v2-m3`
- `Alibaba-NLP/gte-reranker-modernbert-base`
- `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- `mixedbread-ai/mxbai-rerank-base-v2`

Artifact root:

- `runs/text-reranker-expanded-20260621/`
- Summary: `runs/text-reranker-expanded-20260621/expanded-text-reranker-comparison.json`

Full-split nDCG@10:

| Dataset | Dense | Behavior-general | Ettin | BGE v2-m3 | GTE ModernBERT | Qwen3 0.6B seq-cls | Mixedbread base-v2 | Best text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.7682 | 0.8016 | 0.8163 | 0.8042 | 0.8134 | 0.3762 | 0.7938 | Ettin |
| LegalBench-RAG | 0.2733 | 0.7510 | 0.5023 | 0.5187 | 0.4854 | 0.3075 | 0.4454 | BGE v2-m3 |
| R2MED | 0.1732 | 0.8616 | 0.4586 | 0.3764 | 0.4630 | 0.4120 | 0.3789 | GTE ModernBERT |

Strict no-train/dev-positive-overlap nDCG@10:

| Dataset | Dense | Behavior-general | Ettin | BGE v2-m3 | GTE ModernBERT | Qwen3 0.6B seq-cls | Mixedbread base-v2 | Best text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.7626 | 0.7960 | 0.8098 | 0.7867 | 0.8477 | 0.4595 | 0.8363 | GTE ModernBERT |
| LegalBench-RAG | 0.2655 | 0.7437 | 0.5055 | 0.5178 | 0.4894 | 0.3027 | 0.4544 | BGE v2-m3 |
| R2MED | 0.1862 | 0.8527 | 0.4711 | 0.3863 | 0.4752 | 0.4042 | 0.3872 | GTE ModernBERT |

Paired findings:

- LegalBench-RAG full: behavior-general beats best text, BGE v2-m3, by `+0.2323` nDCG@10, p approximately `0.0001`.
- LegalBench-RAG strict: behavior-general beats BGE v2-m3 by `+0.2259` nDCG@10, p approximately `0.0001`.
- R2MED full: behavior-general beats best text, GTE ModernBERT, by `+0.3986` nDCG@10, p approximately `0.0001`.
- R2MED strict: behavior-general beats GTE ModernBERT by `+0.3775` nDCG@10, p approximately `0.0001`.
- SciFact full: Ettin beats behavior-general by `+0.0147` nDCG@10, p = `0.4835`.
- SciFact strict: GTE ModernBERT beats behavior-general by `+0.0517` nDCG@10, p = `0.1733`, and beats Ettin by `+0.0379`, p = `0.0250`.

Interpretation:

- The expanded text-reranker pass strengthens the LegalBench-RAG and R2MED claim because behavior no longer only beats Ettin; it beats the best observed off-the-shelf text reranker in the model set.
- The expanded text-reranker pass weakens any SciFact headline, especially on the strict subset. SciFact remains supportive historically and methodologically, but not a primary controlled result.
- The converted Qwen3 sequence-classifier checkpoint is not competitive in this protocol. That should not be interpreted as a full Qwen3 reranker-family failure because the original Qwen3 reranker uses a generative yes/no scoring protocol that we did not run in this pass.

## Coding Retrieval Transfer Spot Check

Run date: 2026-06-21

Purpose: run a retrieval-focused coding benchmark before publishing, using the trained general query-plus-candidate behavior-latent reranker as a zero-shot transfer artifact and comparing it against Ettin over the exact same frozen dense candidate groups.

Benchmark choice:

- CoIR/CoSQA was used because CoIR is explicitly a code information retrieval benchmark and uses a BEIR/MTEB-style retrieval structure. CORE-Bench is a promising newer agentic-code retrieval benchmark, but no prepared candidate groups exist in this workspace yet.
- Local candidate source: `runs/supervised/verticals/coir-cosqa/test-groups.k100.bge.appendpos.jsonl`
- Prepared behavior-pair groups: `runs/coding-retrieval-behavior-20260621/coir-cosqa/test-groups.behavior-pair.jsonl`
- Scope: 500 heldout queries, 8,000 total query-candidate pairs, 16 candidates per query.
- Behavior telemetry: strict zero-token Qwen/RMT/SAE Core245 pair-prompt prefill, `query_candidate_pair_core245_max_prefill`.
- Behavior model: `release/behavior-latent-general-v0.1/model.pt`, `log1p_l2` activation transform, selected alpha `1.0`.
- Ettin baseline: `cross-encoder/ettin-reranker-150m-v1`, max length 512, same candidate IDs.

Artifacts:

- Capture cache: `runs/coding-retrieval-behavior-20260621/telemetry-cache/`
- Behavior metrics/scores: `runs/coding-retrieval-behavior-20260621/coir-cosqa/behavior-general-metrics.json`, `behavior-general-scores.jsonl`
- Ettin metrics/scores: `runs/coding-retrieval-behavior-20260621/coir-cosqa/ettin-metrics.json`, `ettin-scores.jsonl`
- Paired comparison: `runs/coding-retrieval-behavior-20260621/coir-cosqa/comparison.json`
- Changed-query audit: `runs/coding-retrieval-behavior-20260621/coir-cosqa/audit-behavior-vs-ettin.json`

Heldout CoIR/CoSQA results:

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

Changed-query interpretation:

- Behavior improved rank quality enough to beat Ettin decisively on MRR@10 and nDCG@10, while Recall@10 was effectively tied.
- The audit shows a plausible mechanism: behavior often promotes exact support functions from deep dense ranks when many near-duplicate lexical distractors occupy the top dense positions.
- Failure cases remain concrete: when Ettin already places an exact positive first, behavior can overpromote near-topic code variants and lose the positive from the top 10. This keeps the run diagnostic rather than a final coding-vertical claim.

## Fuller CoIR Coding Retrieval Suite

Run date: 2026-06-21

Purpose: extend the CoIR/CoSQA spot check into a bounded multi-task coding retrieval suite while preserving the same fair protocol: no document-prefill ingestion, frozen BGE dense candidate groups, Ettin reranking over the exact same candidate IDs, and behavior-prefill reranking through the canonical query+candidate adapter plus the released general behavior-latent MLP.

Tasks:

- CoSQA
- CodeTrans-DL
- CodeTrans-Contest
- StackOverflow-QA
- APPS

Protocol:

- Dense candidates: BGE base dense candidates, `k=100`, with qrel positives appended when missing.
- Rerank candidate count: 16 per query for both Ettin and behavior-prefill.
- Behavior telemetry: strict zero-token Qwen/RMT/SAE Core245 pair-prompt prefill.
- Behavior scorer: `release/behavior-latent-general-v0.1/model.pt`.
- Ettin scorer: `cross-encoder/ettin-reranker-150m-v1`, max length 512.
- CoSQA uses the earlier completed artifact root `runs/coding-retrieval-behavior-20260621/coir-cosqa/`; the four additional tasks use `runs/coding-retrieval-behavior-suite-20260621/`.

Suite artifact:

- `runs/coding-retrieval-behavior-suite-20260621/summary.json`

Per-task results:

| Task | Queries | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 | Behavior minus Ettin nDCG@10 | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CoSQA | 500 | 0.3258 | 0.4447 | 0.5277 | +0.0830 | 0.0003 |
| CodeTrans-DL | 180 | 0.2123 | 0.3587 | 0.3337 | -0.0251 | 0.2927 |
| CodeTrans-Contest | 221 | 0.9095 | 0.9770 | 0.8664 | -0.1106 | 0.0001 |
| StackOverflow-QA | 1,994 | 0.7993 | 0.9156 | 0.7685 | -0.1471 | 0.0001 |
| APPS | 3,765 | 0.0572 | 0.6418 | 0.9339 | +0.2921 | 0.0001 |

Aggregate results:

| Aggregate | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 |
| --- | ---: | ---: | ---: |
| Macro average over 5 tasks | 0.4608 | 0.6676 | 0.6861 |
| Query-weighted average over 6,660 queries | 0.3320 | 0.7125 | 0.8354 |

Recall:

- Macro Recall@10: dense `0.5971`, Ettin `0.8889`, behavior `0.8555`.
- Query-weighted Recall@10: dense `0.4039`, Ettin `0.9084`, behavior `0.9222`.

Interpretation:

- The fuller suite is not uniformly positive, but it is still consequential. Behavior-prefill wins decisively on APPS and CoSQA, loses decisively on CodeTrans-Contest and StackOverflow-QA, and is statistically tied/slightly behind Ettin on CodeTrans-DL nDCG while slightly ahead on MRR.
- APPS is the largest and most dramatic win: behavior nDCG@10 `0.9339` versus Ettin `0.6418` and dense `0.0572`, p approximately `0.0001` versus Ettin. This drives the query-weighted suite result.
- StackOverflow-QA and CodeTrans-Contest reveal a clear failure mode: when dense/Ettin already perform strongly or when benchmark labels reward exact known items among many near-equivalent snippets, behavior can overpromote plausible but unlabeled alternatives.
- The correct public framing is now stronger and more nuanced than the single-task spot check: behavior-prefill appears highly promising for some coding retrieval workloads, especially APPS-style problem-to-solution retrieval, but it is not a universal coding reranker over all CoIR tasks.
- A coding-specific behavior model and duplicate/equivalence-aware qrel treatment are the next serious steps before making a broad agentic-coding retrieval claim.

Agentic coding implications:

- APPS and CoSQA are the most relevant tasks for coding agents in this bounded suite because they ask task-to-code and intent-to-function support questions.
- The zero-shot APPS result is unusually positive: the current general behavior-latent artifact was not trained on coding data, but it still strongly recognized candidate code that solves a programming problem when the solution was present in the rerank slate.
- This suggests the current query+candidate prefill telemetry may encode a broader "candidate supports accomplishing the task" behavior rather than only domain-specific legal/medical evidence support.
- The result is most applicable to cascaded coding-agent retrieval: cheap repository/dense/lexical/symbol search gets plausible snippets into a slate, then behavior-prefill reranking selects the snippets that actually support the task.
- The main qualification is that APPS uses appended positives when dense misses the target, so this is evidence for final-stage reranking/evidence selection, not standalone full-corpus retrieval.
- The negative StackOverflow-QA and CodeTrans-Contest results are diagnostic rather than fatal. They indicate that the current artifact is not a generic accepted-answer reranker or exact translation-pair matcher. Its strength appears to be operational support detection.

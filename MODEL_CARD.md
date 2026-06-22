---
license: mit
library_name: pytorch
pipeline_tag: text-ranking
language:
  - en
tags:
  - retrieval
  - reranking
  - rag
  - activation-telemetry
  - sparse-autoencoder
  - legal
  - biomedical
  - scientific-literature
  - code-retrieval
datasets:
  - SciFact
  - LegalBench-RAG
  - R2MED
  - CoIR
metrics:
  - ndcg
  - mrr
  - recall
base_model:
  - qwen3-4b-rmt-sae
model-index:
  - name: behavior-latent-general-v0.1
    results:
      - task:
          type: text-ranking
          name: Dense-candidate evidence reranking
        dataset:
          type: LegalBench-RAG
          name: LegalBench-RAG strict no-train/dev-positive-overlap
        metrics:
          - type: ndcg_at_10
            value: 0.7436818148405017
            name: nDCG@10
          - type: mrr_at_10
            value: 0.7208450491307634
            name: MRR@10
          - type: recall_at_10
            value: 0.8993227513227513
            name: Recall@10
      - task:
          type: text-ranking
          name: Dense-candidate evidence reranking
        dataset:
          type: R2MED
          name: R2MED strict no-train/dev-positive-overlap
        metrics:
          - type: ndcg_at_10
            value: 0.8526802845107041
            name: nDCG@10
          - type: mrr_at_10
            value: 0.8866584564860427
            name: MRR@10
          - type: recall_at_10
            value: 0.9076981246218996
            name: Recall@10
      - task:
          type: text-ranking
          name: Zero-shot dense-candidate code reranking
        dataset:
          type: CoIR
          name: CoIR APPS bounded rerank slate
        metrics:
          - type: ndcg_at_10
            value: 0.9339
            name: nDCG@10
          - type: mrr_at_10
            value: 0.9410
            name: MRR@10
      - task:
          type: text-ranking
          name: Zero-shot dense-candidate code reranking
        dataset:
          type: CoIR
          name: CoIR CoSQA bounded rerank slate
        metrics:
          - type: ndcg_at_10
            value: 0.5277
            name: nDCG@10
---

# Behavior-Latent General Reranker v0.1

This model is a small PyTorch MLP that reranks dense-retrieved evidence
candidates using query-plus-candidate activation telemetry from a language
model. It is not a standalone text reranker. It requires telemetry captured from
a canonical evidence-support prompt.

## Model Summary

The model scores whether a candidate passage appears to directly support
answering a query, based on the internal sparse-autoencoder features activated
when Qwen reads the query and candidate together.

Release artifact:

- `release/behavior-latent-general-v0.1/model.pt`

Source checkpoint:

- `runs/behavior-latent-robust-20260618/general-log1p-l2/model.pt`

SHA-256:

- `2de017e5387697f6156786f43a1bfc4a97392941cb515c59e1edd49e632f2f36`

## Intended Use

Use this model as a premium second-stage or third-stage reranker for expert
evidence retrieval:

- legal evidence/snippet retrieval,
- biomedical and medical evidence retrieval,
- scientific literature evidence selection,
- code and API evidence selection when the right candidate is already in the
  rerank slate,
- compliance and policy evidence search,
- high-stakes RAG systems where near-topic distractors are costly.

The intended pipeline is:

1. Retrieve a dense or dense+lexical candidate slate.
2. Build a query/candidate support prompt for each candidate.
3. Run zero-token prefill through the telemetry model.
4. Extract Qwen layer-7 Core245 SAE activations.
5. Feed the transformed telemetry plus dense metadata into this MLP.
6. Sort candidates by the MLP support score.

## Out-Of-Scope Use

Do not use this model as:

- a first-stage retriever,
- a text-only reranker,
- an answer generator,
- a safety classifier,
- a general semantic similarity model,
- a medical/legal decision system without downstream expert validation.

The model only ranks candidates already surfaced by another retrieval system.

## Input Format

The MLP consumes a 248-dimensional numeric vector:

- 245 Core245 SAE feature values from Qwen layer 7 at `resid_pre`,
- dense retrieval score,
- per-query dense z-score,
- reciprocal dense rank.

The telemetry prompt representation is:

- `query_candidate_pair_core245_max_prefill`

The activation transform is:

- `log1p_l2`

The feature set is:

- `core245_corrected_longmem_query_conditioned_train_dev`

The checkpoint includes:

- `feature_ids`
- `input_dim`
- `hidden_dim`
- `normalizer`
- `state_dict`
- `selected_alpha`
- `selected_epoch`
- `prompt_representation`
- `activation_transform`

## Prompt Template

```text
Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:
```

Telemetry capture must be zero-output-token prefill. The system should not
generate an answer during scoring.

## Architecture

```text
Linear(248 -> 128)
ReLU
Dropout(0.15 during training)
Linear(128 -> 128)
ReLU
Dropout(0.15 during training)
Linear(128 -> 1)
```

Saved checkpoint fields:

- `input_dim`: 248
- `hidden_dim`: 128
- `selected_alpha`: 1.0
- `selected_epoch`: 6
- `activation_transform`: `log1p_l2`
- `prompt_representation`: `query_candidate_pair_core245_max_prefill`

At inference time, use dropout probability 0.0 by calling `model.eval()`.

## Training Procedure

Training data consists of dense candidate groups from:

- SciFact,
- LegalBench-RAG,
- pooled R2MED.

Each training group contains one query, answer-supporting positive candidates,
and dense hard negatives. The model is trained with a query-group listwise
objective. Within each group, labels define a smoothed target distribution over
positive candidates, and the model learns to rank positives above near-topic
distractors.

Training settings:

- hidden dimension: 128
- optimizer: AdamW
- learning rate: `5e-4`
- weight decay: `1e-3`
- temperature: `0.25`
- dropout: `0.15`
- label smoothing: `0.03`
- gradient clipping: `1.0`
- early stopping patience: `8`
- alpha grid: `0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0`
- selected alpha: `1.0`

`selected_alpha = 1.0` means the learned behavior score controls final ranking
under the selected checkpoint. Dense metadata is still part of the input vector.

## Evaluation

Evaluation uses frozen dense candidate groups. All rerankers score the same
candidates. Metrics are nDCG@10, MRR@10, and Recall@10.

### Full Test Split

| Dataset | Dense nDCG@10 | Best text nDCG@10 | Behavior nDCG@10 | Main result |
| --- | ---: | ---: | ---: | --- |
| SciFact | 0.7682 | 0.8163 | 0.8016 | Text reranker slightly ahead. |
| LegalBench-RAG | 0.2733 | 0.5187 | 0.7510 | Behavior beats best text by +0.2323, p approx 0.0001. |
| R2MED | 0.1732 | 0.4630 | 0.8616 | Behavior beats best text by +0.3986, p approx 0.0001. |

### Strict No-Train/Dev-Positive-Overlap Split

| Dataset | Dense nDCG@10 | Best text nDCG@10 | Behavior nDCG@10 | Main result |
| --- | ---: | ---: | ---: | --- |
| SciFact | 0.7626 | 0.8477 | 0.7960 | Best text ahead; SciFact is supportive, not primary. |
| LegalBench-RAG | 0.2655 | 0.5178 | 0.7437 | Behavior beats best text by +0.2259, p approx 0.0001. |
| R2MED | 0.1862 | 0.4752 | 0.8527 | Behavior beats best text by +0.3775, p approx 0.0001. |

Best text reranker envelope includes:

- `cross-encoder/ettin-reranker-150m-v1`
- `BAAI/bge-reranker-v2-m3`
- `Alibaba-NLP/gte-reranker-modernbert-base`
- `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`
- `mixedbread-ai/mxbai-rerank-base-v2`

### Zero-Shot Coding Retrieval Transfer

The same released checkpoint was also evaluated without coding-specific
training on a bounded five-task CoIR coding retrieval suite. Each arm reranked
the same frozen dense candidate IDs with 16 candidates per query.

| CoIR task | Queries | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 | Behavior minus Ettin |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoSQA | 500 | 0.3258 | 0.4447 | 0.5277 | +0.0830 |
| CodeTrans-DL | 180 | 0.2123 | 0.3587 | 0.3337 | -0.0251 |
| CodeTrans-Contest | 221 | 0.9095 | 0.9770 | 0.8664 | -0.1106 |
| StackOverflow-QA | 1,994 | 0.7993 | 0.9156 | 0.7685 | -0.1471 |
| APPS | 3,765 | 0.0572 | 0.6418 | 0.9339 | +0.2921 |

Aggregate nDCG@10 across the five-task suite:

| Aggregate | Dense | Ettin | Behavior |
| --- | ---: | ---: | ---: |
| Macro average | 0.4608 | 0.6676 | 0.6861 |
| Query-weighted | 0.3320 | 0.7125 | 0.8354 |

This is a transfer result, not a coding-trained claim. APPS and CoSQA are the
most encouraging tasks because they resemble coding-agent retrieval bottlenecks:
selecting code that supports a requested task from plausible near-topic
distractors. APPS used qrel-positive appending when dense retrieval missed the
positive, so it demonstrates strong final-stage evidence selection once the
right code is in the candidate slate; it does not demonstrate standalone
first-stage code retrieval.

## Leakage And Deduplication Controls

The audit checks:

- query ID overlap,
- normalized query-text overlap,
- near-duplicate query text,
- exact positive `(query, doc)` pair overlap,
- exact candidate `(query, doc)` pair overlap,
- repeated positive evidence text,
- train/dev positive evidence appearing as heldout candidates.

No exact query-id overlap and no exact positive/candidate `(query, doc)` pair
overlap were found across SciFact, LegalBench-RAG, or R2MED.

Repeated evidence text does appear across splits, so strict heldout subsets were
created by dropping test queries with duplicate train/dev query text or positive
evidence that appeared as train/dev positive evidence. LegalBench-RAG and R2MED
remain decisive under this stricter control.

## How To Load

```python
import torch
from torch import nn

payload = torch.load("model.pt", map_location="cpu")

model = nn.Sequential(
    nn.Linear(payload["input_dim"], payload["hidden_dim"]),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(payload["hidden_dim"], payload["hidden_dim"]),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(payload["hidden_dim"], 1),
)
model.load_state_dict(payload["state_dict"])
model.eval()
```

The input vector must already contain transformed activation telemetry and dense
metadata. Raw text is not accepted by this model.

## Limitations

- Requires the specific Qwen/RMT/SAE Core245 telemetry provider.
- Requires query/candidate prefill scoring at inference time.
- Current telemetry is max-over-prefill-token SAE-only telemetry, not full
  final-token CAA/SAE selector telemetry.
- Evaluated as a reranker over frozen dense candidate pools, not as end-to-end
  corpus retrieval.
- Strongest evidence is LegalBench-RAG and R2MED. SciFact is mixed after strict
  duplicate-evidence controls and expanded text baselines.
- Zero-shot coding transfer is mixed by task. APPS and CoSQA are strong and
  strategically important for coding-agent retrieval, while StackOverflow-QA and
  CodeTrans-Contest favor Ettin.
- The model may not transfer to casual FAQ retrieval, broad web search, or
  domains without expert-prose evidence structure.

## Ethical And Operational Considerations

This model is intended to improve evidence selection, not to make final legal,
medical, or scientific judgments. In high-stakes settings it should be paired
with:

- citation-preserving answer generation,
- replayable audit logs,
- human review,
- corpus versioning,
- uncertainty/risk gating,
- monitoring for domain drift.

Because the model depends on telemetry from a larger language model, production
systems should document:

- telemetry model version,
- SAE feature manifest,
- prompt template hash,
- dense retriever version,
- candidate-generation settings,
- model checkpoint checksum.

## Citation

If publishing this artifact, cite the repository report and the model-card
framework:

- Experiment report:
  `docs/behavior-latent-reranker-experiment-report-20260618.md`
- Mitchell et al., "Model Cards for Model Reporting":
  https://arxiv.org/abs/1810.03993
- Hugging Face model-card documentation:
  https://huggingface.co/docs/hub/en/model-cards

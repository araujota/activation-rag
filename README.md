# Activation RAG

Activation RAG is a research repository for testing whether language-model
activation telemetry can improve retrieval-augmented generation evidence
selection.

The strongest result in this repo is not activation-vector nearest-neighbor
search. That failed or stayed weak. The strongest result is a query-plus-
candidate behavior-latent reranker:

1. Retrieve candidate passages with a normal dense retriever.
2. For each query/candidate pair, build a canonical evidence-support prompt.
3. Run a zero-output-token prefill pass through the telemetry model.
4. Capture selected sparse-autoencoder activations from Qwen layer 7.
5. Score the candidate with a small MLP trained to detect whether the pair
   induces an internal evidence-support behavior.

This turns activation capture into a high-precision reranking signal over a
small candidate set. On the completed benchmark suite, the general behavior
reranker decisively beats dense retrieval and the best observed off-the-shelf
text reranker envelope on LegalBench-RAG and R2MED, including strict
duplicate-evidence controls.

## Current Primary Result

Primary checkpoint:

- `release/behavior-latent-general-v0.1/model.pt`
- Model card: `MODEL_CARD.md`
- Full experiment report: `docs/behavior-latent-reranker-experiment-report-20260618.md`
- Expanded text-reranker comparison:
  `release/behavior-latent-general-v0.1/expanded-text-reranker-comparison.json`

Strict no-train/dev-positive-overlap nDCG@10:

| Dataset | Dense | Best text reranker | Behavior-general | Delta vs best text |
| --- | ---: | ---: | ---: | ---: |
| LegalBench-RAG | 0.2655 | 0.5178 | 0.7437 | +0.2259 |
| R2MED | 0.1862 | 0.4752 | 0.8527 | +0.3775 |
| SciFact | 0.7626 | 0.8477 | 0.7960 | -0.0517 |

SciFact remains useful as a discovery and supporting result, but it is not the
primary controlled claim after strict duplicate-evidence filtering and the
expanded text-reranker baseline pass.

## Zero-Shot Coding Transfer

The released general checkpoint was also run without coding-specific retraining
on a bounded five-task CoIR coding retrieval suite. All arms reranked the same
frozen dense candidate IDs with 16 candidates per query.

| Task | Queries | Dense nDCG@10 | Ettin nDCG@10 | Behavior nDCG@10 | Behavior - Ettin |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoSQA | 500 | 0.3258 | 0.4447 | 0.5277 | +0.0830 |
| CodeTrans-DL | 180 | 0.2123 | 0.3587 | 0.3337 | -0.0251 |
| CodeTrans-Contest | 221 | 0.9095 | 0.9770 | 0.8664 | -0.1106 |
| StackOverflow-QA | 1,994 | 0.7993 | 0.9156 | 0.7685 | -0.1471 |
| APPS | 3,765 | 0.0572 | 0.6418 | 0.9339 | +0.2921 |

Across the five-task suite, macro nDCG@10 is `0.6861` for behavior versus
`0.6676` for Ettin, and query-weighted nDCG@10 is `0.8354` for behavior versus
`0.7125` for Ettin. The result is mixed, but APPS and CoSQA are notable because
they resemble agentic coding retrieval: selecting code that actually supports a
requested task from plausible distractors. APPS used qrel-positive appending
when dense retrieval missed the positive, so this is evidence for final-stage
selection once the right code is present, not for standalone first-stage code
retrieval.

## What This Repository Contains

- `src/activation_rag/`: core chunking, telemetry, retrieval, benchmark, and
  ranking utilities.
- `scripts/`: benchmark preparation, telemetry capture, reranker training,
  leakage audits, text-reranker baselines, and comparison utilities.
- `specs/001-activation-telemetry-rag/`: Spec Kit artifacts tracking the
  research plan, requirements, tasks, and addenda.
- `docs/`: experiment reports and telemetry notes.
- `examples/`: small local demos that do not require the full telemetry stack.
- `release/behavior-latent-general-v0.1/`: lightweight release bundle for the
  primary MLP checkpoint and evaluation summaries.

Large generated data is intentionally ignored by git:

- `data/`
- `runs/`
- `artifacts/`

The telemetry caches are large and environment-specific. Publish the small
release bundle and scripts, not the raw cache directory.

## Install

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[supervised,embeddings,benchmarks]'
```

For CPU-only local tests, the base package is enough:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests
```

For GPU telemetry capture and text-reranker scoring, use the vicuna host or an
equivalent CUDA machine with:

- PyTorch
- sentence-transformers
- transformers
- access to the Qwen/RMT/SAE telemetry capture command

## Quick Local Demo

The basic dense/activation-RAG toy demo does not require model downloads:

```bash
PYTHONPATH=src python examples/compare_activation_search.py
```

It prints:

- dense retrieval results,
- activation-KNN results,
- dense-first activation reranking results.

This demo is intentionally small. It validates package mechanics, not the
behavior-latent result.

## Core Prompt

The behavior-latent reranker scores query/candidate pairs, not isolated
document chunks. The canonical prompt shape is:

```text
Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:
```

The telemetry model runs a strict zero-output-token prefill pass over this
prompt. No answer is generated.

## Input Features

The released MLP consumes 248 scalar features:

- 245 Qwen layer-7 Core245 SAE feature values captured from the query/candidate
  support prompt,
- dense retrieval score,
- per-query dense z-score,
- reciprocal dense rank.

Activation transform:

1. Apply signed `log1p` compression to the 245 SAE values.
2. L2-normalize the transformed 245-dimensional activation vector.
3. Append the three dense metadata features.
4. Apply the checkpoint's train-set mean/scale normalizer over all 248 inputs.

The selected release checkpoint uses `selected_alpha = 1.0`, so final ranking is
driven by the learned behavior score. Dense metadata is still available as input
features.

## Reproducing The Primary Benchmark

The completed run artifacts live under `runs/behavior-latent-robust-20260618/`
on the research machine. To reproduce from prepared groups and a telemetry
cache, use the same structure.

### 1. Prepare Candidate Groups

Candidate groups are JSONL files with one query per row and a fixed dense
candidate slate:

```text
runs/behavior-latent-robust-20260618/scifact/*-groups.behavior-pair.jsonl
runs/behavior-latent-robust-20260618/legalbenchrag/*-groups.behavior-pair.jsonl
runs/behavior-latent-robust-20260618/r2med-all/*-groups.behavior-pair.jsonl
```

Each group row contains:

- `query_id`
- `query_text`
- dense-ranked `candidates`
- candidate `text`
- candidate `label`
- stable `behavior_chunk_id` for telemetry matching

### 2. Capture Pair-Prompt Telemetry

The telemetry capture path is intentionally command-backed so the repo is not
tightly coupled to a specific model server:

```bash
PYTHONPATH=src python scripts/capture_group_telemetry_cache.py \
  --groups runs/behavior-latent-robust-20260618/scifact/train-groups.behavior-pair.jsonl \
  --groups runs/behavior-latent-robust-20260618/scifact/dev-groups.behavior-pair.jsonl \
  --groups runs/behavior-latent-robust-20260618/scifact/test-groups.behavior-pair.jsonl \
  --telemetry-command 'python scripts/capture_qwen_sae_prefill.py {input} {output} --capture-execution-mode early_stop_layer --optimized-batch-size 8' \
  --cache-dir runs/behavior-latent-robust-20260618/telemetry-cache \
  --provider-id qwen-rmt-sae-prefill \
  --model-id qwen3-4b-rmt-sae \
  --site-id qwen.model.layers.7.resid_pre \
  --prompt-template-id query_candidate_pair_support_v1
```

The exact telemetry command depends on the vicuna/longmem sidecar environment.
The important contract is:

- input JSONL contains prompt/chunk records,
- output JSONL contains selector-compatible telemetry rows,
- rows include `sae_feature_values`,
- capture is prefill-only with zero generated tokens,
- reranking telemetry uses `--capture-execution-mode early_stop_layer` so Qwen
  stops immediately after the layer-7 SAE site is captured,
- rows are keyed by stable chunk IDs.

### 3. Train The General Behavior Reranker

```bash
PYTHONPATH=src python scripts/train_behavior_latent_reranker.py \
  --train-groups runs/behavior-latent-robust-20260618/scifact/train-groups.behavior-pair.jsonl \
  --train-groups runs/behavior-latent-robust-20260618/legalbenchrag/train-groups.behavior-pair.jsonl \
  --train-groups runs/behavior-latent-robust-20260618/r2med-all/train-groups.behavior-pair.jsonl \
  --dev-groups runs/behavior-latent-robust-20260618/scifact/dev-groups.behavior-pair.jsonl \
  --dev-groups runs/behavior-latent-robust-20260618/legalbenchrag/dev-groups.behavior-pair.jsonl \
  --dev-groups runs/behavior-latent-robust-20260618/r2med-all/dev-groups.behavior-pair.jsonl \
  --test-groups runs/behavior-latent-robust-20260618/scifact/test-groups.behavior-pair.jsonl \
  --test-groups runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.behavior-pair.jsonl \
  --test-groups runs/behavior-latent-robust-20260618/r2med-all/test-groups.behavior-pair.jsonl \
  --telemetry-cache-dir runs/behavior-latent-robust-20260618/telemetry-cache \
  --feature-manifest configs/activation_capture_permutations.longmem_core245.json \
  --model-out runs/behavior-latent-robust-20260618/general-log1p-l2/model.pt \
  --metrics-out runs/behavior-latent-robust-20260618/general-log1p-l2/metrics.json \
  --scores-out runs/behavior-latent-robust-20260618/general-log1p-l2/test-scores.jsonl \
  --dataset-name general-log1p-l2-robust \
  --hidden-dim 128 \
  --epochs 40 \
  --learning-rate 5e-4 \
  --weight-decay 1e-3 \
  --temperature 0.25 \
  --alpha-grid 0,0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0 \
  --dropout 0.15 \
  --label-smoothing 0.03 \
  --activation-transform log1p_l2 \
  --grad-clip 1.0 \
  --early-stopping-patience 8 \
  --device cuda
```

### 4. Compare Against Dense, Actpred, And Text Rerankers

```bash
PYTHONPATH=src python scripts/compare_behavior_latent_pilot.py \
  --groups runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.behavior-pair.jsonl \
  --behavior-scores runs/behavior-latent-robust-20260618/general-log1p-l2/test-scores.jsonl \
  --out runs/behavior-latent-robust-20260618/legalbenchrag/general-comparison.json
```

Run expanded text-reranker baselines with:

```bash
PYTHONPATH=src python scripts/rerank_remote_sentence_transformers.py \
  runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.behavior-pair.jsonl \
  runs/text-reranker-expanded-20260621/bge-v2-m3/legalbenchrag/scores.jsonl \
  --host root@vicuna-host \
  --remote-python /root/vicuna/venvs/activation-rag-embeddings/bin/python \
  --model BAAI/bge-reranker-v2-m3 \
  --batch-size 64 \
  --max-length 512 \
  --trust-remote-code
```

Evaluate those scores with:

```bash
PYTHONPATH=src python scripts/run_text_reranker_baseline.py \
  --groups runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.behavior-pair.jsonl \
  --scores-jsonl runs/text-reranker-expanded-20260621/bge-v2-m3/legalbenchrag/scores.jsonl \
  --out runs/text-reranker-expanded-20260621/bge-v2-m3/legalbenchrag/metrics.json
```

### 5. Run Leakage And Duplicate-Evidence Audits

```bash
PYTHONPATH=src python scripts/audit_group_leakage.py \
  --train-groups runs/behavior-latent-robust-20260618/legalbenchrag/train-groups.behavior-pair.jsonl \
  --dev-groups runs/behavior-latent-robust-20260618/legalbenchrag/dev-groups.behavior-pair.jsonl \
  --test-groups runs/behavior-latent-robust-20260618/legalbenchrag/test-groups.behavior-pair.jsonl \
  --out runs/behavior-latent-robust-20260618/legalbenchrag/leakage-dedup-audit.json \
  --near-duplicate-threshold 0.85 \
  --shingle-size 5
```

Strict subsets used in the report drop heldout test queries if their normalized
query text duplicates train/dev query text, or if any positive evidence doc/text
appeared as train/dev positive evidence.

## Scoring A New Candidate Pair

At inference time, do not feed raw text directly into `model.pt`. The MLP
expects the 248-dimensional feature vector described above. Production scoring
requires:

1. Retrieve candidate chunks with dense and/or lexical search.
2. For each query/candidate pair, build the canonical support prompt.
3. Capture Qwen layer-7 Core245 SAE telemetry with zero generated tokens.
4. Transform the 245 SAE values with `log1p_l2`.
5. Append dense score, dense z-score, and reciprocal dense rank.
6. Apply the checkpoint normalizer.
7. Run the MLP and sort candidates by support score.

Minimal PyTorch loader:

```python
import torch
from torch import nn

payload = torch.load("release/behavior-latent-general-v0.1/model.pt", map_location="cpu")

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

# x must already be the transformed 248-dimensional feature vector.
x = torch.tensor(feature_vector, dtype=torch.float32)
mean = torch.tensor(payload["normalizer"]["mean"], dtype=torch.float32)
scale = torch.tensor(payload["normalizer"]["scale"], dtype=torch.float32)
score = model(((x - mean) / scale).unsqueeze(0)).item()
```

## Limitations

- This is a reranker, not a first-stage retriever.
- It requires query/candidate prefill telemetry at serving time.
- The current artifact is SAE-only Core245 max-over-prefill telemetry, not full
  final-token CAA/SAE selector telemetry.
- It is strongest on LegalBench-RAG and R2MED. SciFact is supportive but not the
  primary strict-control claim.
- The model is not a general text reranker; text-only inference is unsupported.
- Raw telemetry caches may contain benchmark text and should be handled as
  dataset artifacts, not casually published.

## Production Shape

The intended production system is a premium evidence-adjudication tier:

1. Dense plus lexical retrieval produces a broad pool.
2. A cheap text reranker reduces the slate.
3. A risk/uncertainty gate decides whether behavior telemetry is warranted.
4. The behavior-latent reranker scores only the hard top candidates.
5. The answer model receives the final evidence with a replayable audit trail.

Cost mitigation should use a resident prefill-only service, validated layer-7
early-stop telemetry capture, pair-score caching, candidate-count gating, and
eventual distillation into cheaper student rankers. Prefix caching and true
batched prefill must pass activation-vector equivalence checks before production
use; the current exact path records requested batch size `8` but forces effective
batch size `1` unless non-exact batching is explicitly allowed.

## Documentation Standards

The model card follows the Hugging Face model-card convention, including YAML
metadata in README-style front matter, and the broader Model Cards for Model
Reporting guidance: intended use, factors, metrics, training data, evaluation
data, ethical considerations, and limitations.

References:

- Hugging Face model-card docs: https://huggingface.co/docs/hub/en/model-cards
- Mitchell et al., Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993

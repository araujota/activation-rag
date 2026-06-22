# LongMem Mech-Interp Selector Release

This repository packages the best current artifact from our LongMemEval applied mechanistic-interpretability experiments.

The core claim is simple: activation-derived memory signals can be matched back to the text spans that caused them, and those structured signals can train a selector that improves long-memory question answering over summary carryover and first/last carryover baselines.

## What This Contains

The release system has five parts:

1. **Selector**: a query/span cross-encoder that chooses the top four memory spans likely to contain answer evidence. Internally this artifact was called **T539**.
2. **Selected-memory builder**: expands each selected span to a sentence envelope, so the reader sees raw surrounding text rather than feature labels or generated summaries.
3. **Reader adapter**: a layer 0-7 Qwen3-4B answer-consumer adapter trained to answer from selected raw memory text without packet annotations. Internally this was **T488**.
4. **Output policy**: a deterministic post-generation policy that turns empty answers and unsupported abstention-row numbers into explicit no-information answers, with one narrow optional numeric repair path. Internally this was **T613/T614**.
5. **Reproduction runtime**: the original Python EM v2 evaluator is vendored under `vendor/em_v2_python` so the benchmark path can be rerun.

The release does **not** include the Qwen3-4B base model. You provide that separately.

## Reported Result

Externally judged LongMemEval results:

| Split | Released policy | Baseline comparison |
| --- | ---: | ---: |
| Dev98 | T613/T614 `17/98` | raw summary `10/98`, first-last `10/98` |
| Foldout101 | T613/T618 `19/101` | T569 no-packet `14/101`, T576 repair reader `12/101` |

The foldout paired comparison against the T569 no-packet reader was `5` gains and `0` losses. The switch audit is bundled in `benchmark_reports/`.

## Quickstart

Clone the code/docs repo:

```bash
git clone https://github.com/araujota/longmem-mechinterp-selector.git
cd longmem-mechinterp-selector
```

Create an environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

Run the CPU smoke test:

```bash
make smoke
make test
make summarize-results
```

The smoke test does not require Qwen3-4B. It verifies selected-memory construction and the audited T613 policy behavior on a synthetic stale-number fixture.

## Download Artifacts

The artifact bundle is hosted separately as a private Hugging Face model repo:

`https://huggingface.co/araujota/longmem-mechinterp-selector-artifacts`

After you have access to that repo, download the release artifacts into this checkout:

```bash
pip install -e '.[download]'
huggingface-cli login
python scripts/download_artifacts.py --local-dir .
python scripts/verify_artifacts.py --manifest artifacts/manifest.lock.json
```

You can also avoid `huggingface-cli login` by setting `HF_TOKEN` in the environment or passing `--token` to the download script.

## Full System Run

Install the ML dependencies in a CUDA-capable environment:

```bash
pip install -e '.[ml,test]'
```

Copy and edit the config:

```bash
cp configs/artifacts.example.yaml configs/artifacts.local.yaml
```

Set:

- `base_model_path` to your Qwen3-4B Hugging Face model directory.
- `llama_cpp_runtime_path` to a llama.cpp checkout or runtime directory.
- artifact paths to the bundled `artifacts/local/...` files or wherever you unpacked the artifact bundle.

Run a one-episode full smoke:

```bash
make full-smoke
```

Run the full dev reproduction by removing `--limit 1` from `scripts/run_full_t613_eval.py` usage or invoking:

```bash
python scripts/run_full_t613_eval.py \
  --config configs/artifacts.local.yaml \
  --output-dir outputs/dev98_t569_reader
```

The full evaluator uses the LongMemEval-compatible contract:

- reader mode: `con`
- thinking mode: Qwen default
- max new tokens: `800`
- candidate exposure: full history
- selected memory: T539 top-4 sentence-envelope raw text
- reader arm: `answer_consumer_no_packet`
- no generated summaries
- no model-facing SAE labels

## Artifact Layout

The full artifact bundle should contain:

```text
artifacts/local/
  t539_selector/cross_encoder_model/
  t488_reader/qwen3_consumer_adapter_latest.pt
  t576_reader/qwen3_consumer_adapter_latest.pt
  rmt/qwen3_rmt_joint_memory_latest.pt
  sae/topk_sae_latest.pt
  selector_materialization/feature_manifest.json
  selector_materialization/dev_cross_encoder_top4_action_predictions.jsonl
  longmem_inputs/dev_episodes.jsonl
  longmem_inputs/dev_selector_rows.jsonl
```

The Qwen3-4B base model remains external.

For hosting guidance, see `docs/artifact_hosting.md`. The recommended public shape is GitHub for this code/docs repo and Hugging Face Hub for the large artifact bundle.

## llama.cpp Patch

No C++ llama.cpp patch is required for the released reproduction path. See `patches/README.md`.

## Important Limitations

- The release is a research artifact, not a general memory product.
- The selector and reader were validated on LongMemEval-style memory QA.
- T576 is included only for exact reproduction of the frozen T613 policy. It is not a general repair recommendation.
- Held-out foldout rows must not be used for tuning thresholds or retraining.
- Redistribution of raw LongMemEval-derived rows may require checking the benchmark license before public release.

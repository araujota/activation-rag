# Publishing Checklist

This repo contains both lightweight release artifacts and large generated
research artifacts. Publish them separately.

## Publish To GitHub

Recommended GitHub contents:

- source code under `src/`
- scripts under `scripts/`
- tests under `tests/`
- configs under `configs/`
- docs under `docs/`
- Spec Kit artifacts under `specs/`
- lightweight release bundle under `release/behavior-latent-general-v0.1/`
- `README.md`
- `MODEL_CARD.md`
- `PUBLISHING.md`
- `pyproject.toml`
- `.gitignore`

Do not publish raw caches by default:

- `runs/`
- `data/`
- `artifacts/`
- full telemetry cache JSON directories
- raw benchmark corpora

Those may contain benchmark text, generated intermediate outputs, and
environment-specific runtime paths.

## GitHub Commands

Check status:

```bash
git status --short
```

Create a normal commit manually after review:

```bash
git add README.md MODEL_CARD.md PUBLISHING.md pyproject.toml src scripts tests specs docs configs release .gitignore
git commit -m "Prepare Activation RAG release materials"
```

Create a GitHub repository with the GitHub CLI:

```bash
gh repo create activation-rag --private --source . --remote origin
git push -u origin main
```

Switch to public only after checking:

- no private credentials,
- no raw benchmark data,
- no raw telemetry cache dump,
- no private hostnames or tokens in committed files,
- model-card limitations and intended-use sections are present.

## Publish Model To Hugging Face

The Hugging Face model repository should contain:

- `README.md`, copied from `MODEL_CARD.md`
- `model.pt`
- `training-metrics.json`
- `expanded-text-reranker-comparison.json`
- optional `release-manifest.json`

Suggested local assembly:

```bash
mkdir -p /tmp/behavior-latent-general-v0.1-hf
cp MODEL_CARD.md /tmp/behavior-latent-general-v0.1-hf/README.md
cp release/behavior-latent-general-v0.1/model.pt /tmp/behavior-latent-general-v0.1-hf/model.pt
cp release/behavior-latent-general-v0.1/training-metrics.json /tmp/behavior-latent-general-v0.1-hf/training-metrics.json
cp release/behavior-latent-general-v0.1/expanded-text-reranker-comparison.json /tmp/behavior-latent-general-v0.1-hf/expanded-text-reranker-comparison.json
cp release/behavior-latent-general-v0.1/release-manifest.json /tmp/behavior-latent-general-v0.1-hf/release-manifest.json
```

Upload with `huggingface_hub`:

```bash
python -m pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create behavior-latent-general-v0.1 --type model --private
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/tmp/behavior-latent-general-v0.1-hf",
    repo_id="YOUR_ORG_OR_USER/behavior-latent-general-v0.1",
    repo_type="model",
)
PY
```

Keep it private until:

- the model card renders correctly,
- the YAML metadata is valid,
- checksums match the release manifest,
- the README clearly states that raw text inference is unsupported,
- usage instructions explain the telemetry prerequisite.

## Release Manifest Checks

Verify local release files:

```bash
sha256sum release/behavior-latent-general-v0.1/model.pt \
  release/behavior-latent-general-v0.1/training-metrics.json \
  release/behavior-latent-general-v0.1/expanded-text-reranker-comparison.json
```

Expected primary checkpoint checksum:

```text
2de017e5387697f6156786f43a1bfc4a97392941cb515c59e1edd49e632f2f36  model.pt
```

## Required Caveats For Any Public Claim

Use this bounded claim:

> Query+candidate behavior-latent telemetry turns activation capture from a weak
> retrieval geometry into a powerful evidence-support signal, producing large
> paired-significant wins over dense retrieval and the strongest observed
> off-the-shelf text-reranker envelope on LegalBench-RAG and R2MED that survive
> strict duplicate-evidence due diligence, with SciFact remaining supportive but
> not primary.

Do not claim:

- universal retrieval improvement,
- text-only reranking,
- activation-only first-stage retrieval,
- final-token CAA/SAE telemetry,
- medical or legal correctness,
- leaderboard-standard external benchmark status.


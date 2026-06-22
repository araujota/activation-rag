#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


BEIR_DATASETS = {
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
}

MSMARCO_NOTES = {
    "loader": "ir_datasets",
    "dataset": "msmarco-passage/dev/small",
    "headline_metric": "MRR@10",
    "notes": [
        "MS MARCO passage corpus is about 8.8M passages.",
        "Use ir_datasets or official MS MARCO downloads for full standard runs.",
        "This project does not vendor the corpus into git.",
    ],
}

HOTPOTQA_NOTES = {
    "loader": "huggingface_datasets",
    "dataset": "hotpotqa/hotpot_qa",
    "config": "fullwiki",
    "headline_metric": "supporting-evidence Recall@k / nDCG@k",
    "notes": [
        "HotpotQA is multi-hop QA; this harness treats it as supporting-evidence retrieval.",
        "Fullwiki preparation requires mapping supporting facts to retrievable paragraphs.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or register RAG benchmark datasets.")
    parser.add_argument("--dataset", required=True, help="beir:<name>, beir:all, msmarco-passage, or hotpotqa-fullwiki")
    parser.add_argument("--out", default="data/benchmarks", help="Output directory")
    parser.add_argument("--download", action="store_true", help="Actually download supported direct-download datasets")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset.startswith("beir:"):
        name = args.dataset.split(":", 1)[1]
        names = sorted(BEIR_DATASETS) if name == "all" else [name]
        for dataset_name in names:
            if dataset_name not in BEIR_DATASETS:
                raise SystemExit(f"Unknown BEIR dataset: {dataset_name}")
            prepare_beir(dataset_name, out_dir, args.download)
        return

    if args.dataset == "msmarco-passage":
        write_manifest(out_dir / "msmarco-passage" / "manifest.json", MSMARCO_NOTES)
        print("Wrote MS MARCO manifest. Install/use ir_datasets for full corpus download.")
        return

    if args.dataset == "hotpotqa-fullwiki":
        write_manifest(out_dir / "hotpotqa-fullwiki" / "manifest.json", HOTPOTQA_NOTES)
        print("Wrote HotpotQA manifest. Use Hugging Face datasets for fullwiki preparation.")
        return

    raise SystemExit(f"Unsupported dataset: {args.dataset}")


def prepare_beir(dataset_name: str, out_dir: Path, should_download: bool) -> None:
    dataset_dir = out_dir / "beir" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    manifest = {
        "loader": "beir_zip",
        "dataset": dataset_name,
        "url": url,
        "headline_metric": "nDCG@10",
        "secondary_metric": "Recall@100",
    }
    write_manifest(dataset_dir / "manifest.json", manifest)
    if not should_download:
        print(f"Wrote BEIR manifest for {dataset_name}; pass --download to fetch {url}")
        return

    archive = dataset_dir / f"{dataset_name}.zip"
    if not archive.exists():
        print(f"Downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dataset_dir)
    print(f"Prepared BEIR dataset at {dataset_dir}")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(str(tmp), str(path))


if __name__ == "__main__":
    main()


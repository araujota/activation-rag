#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch embed JSONL text rows with Sentence Transformers.")
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = SentenceTransformer(args.model, device=args.device)
    vectors = model.encode(
        [row["text"] for row in rows],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, vector in zip(rows, vectors, strict=True):
            handle.write(json.dumps({"id": row["id"], "vector": [float(value) for value in vector]}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

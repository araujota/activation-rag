import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord, stable_hash
from scripts.prepare_supervised_reranker_groups import prepare_reranker_groups


class PrepareSupervisedRerankerGroupsTests(unittest.TestCase):
    def test_prepare_groups_adds_dense_candidates_activation_features_and_negative_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            cache = root / "cache"
            beir.mkdir()
            (beir / "qrels").mkdir()
            cache.mkdir()
            corpus = {
                "doc-pos": "positive answer evidence",
                "doc-neg": "near miss hard negative",
            }
            query = "answer evidence"
            (beir / "corpus.jsonl").write_text(
                "\n".join(
                    json.dumps({"_id": doc_id, "title": doc_id, "text": text})
                    for doc_id, text in corpus.items()
                )
                + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": query}) + "\n",
                encoding="utf-8",
            )
            (beir / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\tdoc-pos\t1\n",
                encoding="utf-8",
            )
            dataset = load_beir_dataset(beir, name="toy", split="test")
            chunks = Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)).split(
                [
                    DocumentRecord.from_text(f"benchmark://toy/{doc_id}", doc_id, text, {"benchmark_doc_id": doc_id})
                    for doc_id, text in dataset.corpus.items()
                ]
            )
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            query_chunk_id = stable_hash(f"query\n{query}", 32)
            for chunk in chunks:
                values = {"act:site_a:prefill_last:mean": 1.0 if "positive" in chunk.text else -1.0}
                _write_cache(cache, chunk.chunk_id, chunk.document_id, values)
            _write_cache(cache, query_chunk_id, "query", {"act:site_a:prefill_last:mean": 1.0})
            dense_cache = root / "dense.npz"
            np.savez_compressed(
                dense_cache,
                chunk_ids=np.array(chunk_ids),
                query_ids=np.array(["q1"]),
                doc_vectors=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64),
                query_vectors=np.array([[1.0, 0.0]], dtype=np.float64),
            )

            groups = prepare_reranker_groups(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_cache_dir=cache,
                dense_embedding_cache=dense_cache,
                candidate_k=2,
            )

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["query_id"], "q1")
        self.assertEqual(group["positive_in_candidate_pool"], True)
        self.assertEqual(len(group["candidates"]), 2)
        positive = next(candidate for candidate in group["candidates"] if candidate["label"] == 1)
        negative = next(candidate for candidate in group["candidates"] if candidate["label"] == 0)
        self.assertIn("activation_cosine", positive["features"])
        self.assertIn("activation_site:site_a", positive["features"])
        self.assertEqual(negative["negative_source"], "dense_hard_negative")
        self.assertEqual(negative["negative_trust"], "unjudged_assumed_negative")

    def test_prepare_groups_adds_catalog_semantic_activation_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            cache = root / "cache"
            beir.mkdir()
            (beir / "qrels").mkdir()
            cache.mkdir()
            corpus = {
                "doc-pos": "positive answer evidence",
                "doc-neg": "near miss hard negative",
            }
            query = "answer evidence"
            (beir / "corpus.jsonl").write_text(
                "\n".join(
                    json.dumps({"_id": doc_id, "title": doc_id, "text": text})
                    for doc_id, text in corpus.items()
                )
                + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": query}) + "\n", encoding="utf-8")
            (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tdoc-pos\t1\n", encoding="utf-8")
            dataset = load_beir_dataset(beir, name="toy", split="test")
            chunks = Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)).split(
                [
                    DocumentRecord.from_text(f"benchmark://toy/{doc_id}", doc_id, text, {"benchmark_doc_id": doc_id})
                    for doc_id, text in dataset.corpus.items()
                ]
            )
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            query_chunk_id = stable_hash(f"query\n{query}", 32)
            for chunk in chunks:
                _write_cache(
                    cache,
                    chunk.chunk_id,
                    chunk.document_id,
                    {
                        "act:site_a:prefill_last:mean": 1.0 if "positive" in chunk.text else -1.0,
                        "act:site_b:prefill_last:mean": 0.25,
                    },
                )
            _write_cache(
                cache,
                query_chunk_id,
                "query",
                {
                    "act:site_a:prefill_last:mean": 1.0,
                    "act:site_b:prefill_last:mean": 0.25,
                },
            )
            dense_cache = root / "dense.npz"
            np.savez_compressed(
                dense_cache,
                chunk_ids=np.array(chunk_ids),
                query_ids=np.array(["q1"]),
                doc_vectors=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64),
                query_vectors=np.array([[1.0, 0.0]], dtype=np.float64),
            )
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "schema_version": "activation_rag.activation_feature_catalog.v1",
                        "features": [
                            {
                                "feature_pattern": "act:site_a:*:mean",
                                "semantic_label": "answer_evidence",
                                "categories": ["evidence"],
                                "causal_confidence": "high",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            groups = prepare_reranker_groups(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_cache_dir=cache,
                dense_embedding_cache=dense_cache,
                candidate_k=2,
                feature_catalog_path=catalog,
            )

        positive = next(candidate for candidate in groups[0]["candidates"] if candidate["label"] == 1)
        negative = next(candidate for candidate in groups[0]["candidates"] if candidate["label"] == 0)
        self.assertEqual(groups[0]["activation_feature_catalog"]["path"], str(catalog))
        self.assertAlmostEqual(positive["features"]["activation_semantic:semantic_label:answer_evidence:cosine"], 1.0)
        self.assertAlmostEqual(negative["features"]["activation_semantic:semantic_label:answer_evidence:cosine"], -1.0)
        self.assertIn("activation_semantic:category:evidence:abs_diff_mean", positive["features"])


def _write_cache(cache: Path, chunk_id: str, document_id: str, features: dict[str, float]) -> None:
    (cache / f"{chunk_id}.json").write_text(
        json.dumps(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "telemetry_valid": True,
                "sae_feature_values": features,
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

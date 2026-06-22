import json
import tempfile
import unittest
from pathlib import Path

from activation_rag.schema import stable_hash
from scripts.prepare_activation_reranker_training import prepare_training_rows


class PrepareActivationRerankerTrainingTests(unittest.TestCase):
    def test_prepare_training_rows_joins_qrels_to_cached_query_and_chunk_activations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            cache = root / "cache"
            beir.mkdir()
            (beir / "qrels").mkdir()
            cache.mkdir()
            (beir / "corpus.jsonl").write_text(
                json.dumps({"_id": "doc-pos", "title": "Positive", "text": "answer evidence"}) + "\n"
                + json.dumps({"_id": "doc-neg", "title": "Negative", "text": "distractor evidence"}) + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "where is the answer"}) + "\n",
                encoding="utf-8",
            )
            (beir / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\tdoc-pos\t1\n",
                encoding="utf-8",
            )
            query_chunk_id = stable_hash("query\nwhere is the answer", 32)
            cached_ids = {query_chunk_id}
            rows = prepare_training_rows(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_cache_dir=cache,
                available_chunk_ids=cached_ids,
                fallback_hard_negative_count=1,
            )
            self.assertEqual(rows, [])

            for chunk_id in _chunk_ids_for_docs(beir):
                cached_ids.add(chunk_id)
                (cache / f"{chunk_id}.json").write_text(
                    json.dumps(_valid_cache_row(chunk_id)) + "\n",
                    encoding="utf-8",
                )
            (cache / f"{query_chunk_id}.json").write_text(
                json.dumps(_valid_cache_row(query_chunk_id)) + "\n",
                encoding="utf-8",
            )

            rows = prepare_training_rows(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_cache_dir=cache,
                fallback_hard_negative_count=1,
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["query_id"], "q1")
        self.assertEqual(row["query_activation_chunk_id"], query_chunk_id)
        self.assertEqual(row["target_mode"], "contrastive_answer_chunk")
        self.assertEqual(row["qrel_positive_doc_ids"], ["doc-pos"])
        self.assertEqual(len(row["positive_chunk_ids"]), 1)
        self.assertEqual(len(row["hard_negative_chunk_ids"]), 1)
        self.assertNotEqual(row["positive_chunk_ids"], row["hard_negative_chunk_ids"])


def _valid_cache_row(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "telemetry_valid": True,
        "sae_feature_values": {"act:test:prefill_last:mean": 1.0},
    }


def _chunk_ids_for_docs(beir_dir: Path) -> list[str]:
    from activation_rag.benchmarks import load_beir_dataset
    from activation_rag.chunking import Chunker, ChunkerSettings
    from activation_rag.schema import DocumentRecord

    dataset = load_beir_dataset(beir_dir, name="toy", split="test")
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://toy/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in dataset.corpus.items()
    ]
    return [chunk.chunk_id for chunk in Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)).split(documents)]


if __name__ == "__main__":
    unittest.main()

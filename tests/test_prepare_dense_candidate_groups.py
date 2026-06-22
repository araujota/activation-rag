import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_dense_candidate_groups import prepare_dense_candidate_groups


class PrepareDenseCandidateGroupsTests(unittest.TestCase):
    def test_prepares_dense_groups_without_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            (beir / "qrels").mkdir(parents=True)
            _write_jsonl(beir / "corpus.jsonl", [{"_id": "d1", "title": "", "text": "alpha"}, {"_id": "d2", "title": "", "text": "beta"}])
            _write_jsonl(beir / "queries.jsonl", [{"_id": "q1", "text": "alpha"}])
            (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
            # Match the deterministic one-chunk-per-document order.
            from activation_rag.chunking import Chunker, ChunkerSettings
            from activation_rag.schema import DocumentRecord

            docs = [DocumentRecord.from_text(source_uri="benchmark://fixture/d1", title="d1", text="alpha", metadata={"benchmark_doc_id": "d1"}),
                    DocumentRecord.from_text(source_uri="benchmark://fixture/d2", title="d2", text="beta", metadata={"benchmark_doc_id": "d2"})]
            chunks = Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)).split(docs)
            cache = root / "dense.npz"
            np.savez_compressed(cache, chunk_ids=np.array([chunk.chunk_id for chunk in chunks]), query_ids=np.array(["q1"]), doc_vectors=np.array([[1.0, 0.0], [0.0, 1.0]]), query_vectors=np.array([[1.0, 0.0]]))

            groups = prepare_dense_candidate_groups(beir_dir=beir, dataset_name="fixture", split="test", dense_embedding_cache=cache, candidate_k=2)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["candidates"][0]["doc_id"], "d1")
        self.assertEqual(groups[0]["candidates"][0]["label"], 1)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

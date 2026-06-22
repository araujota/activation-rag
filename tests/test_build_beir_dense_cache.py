import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from activation_rag.embedding import HashEmbeddingProvider
from scripts.build_beir_dense_cache import build_dense_cache


class BuildBeirDenseCacheTests(unittest.TestCase):
    def test_build_dense_cache_writes_qid_aligned_query_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            beir.mkdir()
            (beir / "qrels").mkdir()
            (beir / "corpus.jsonl").write_text(
                json.dumps({"_id": "d1", "title": "D1", "text": "alpha"}) + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "alpha query"}) + "\n",
                encoding="utf-8",
            )
            (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
            out = root / "dense.npz"

            build_dense_cache(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                out=out,
                embedder=HashEmbeddingProvider(dimension=8),
            )

            cached = np.load(out, allow_pickle=False)

        self.assertEqual(list(cached["query_ids"]), ["q1"])
        self.assertEqual(cached["query_vectors"].shape, (1, 8))
        self.assertEqual(cached["doc_vectors"].shape[1], 8)


if __name__ == "__main__":
    unittest.main()

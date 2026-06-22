from __future__ import annotations

import unittest

from activation_rag.embedding import HashEmbeddingProvider
from scripts.prepare_coir_groups import prepare_coir_groups


class PrepareCoirGroupsTests(unittest.TestCase):
    def test_prepare_coir_groups_preserves_native_splits_and_appends_missing_positives(self):
        bundle = {
            "dataset_name": "coir-unit",
            "queries": {
                "q-train": "parse json in python",
                "q-dev": "sort list javascript",
                "q-test": "read file go",
            },
            "corpus": {
                "d-train-positive": "json.loads parses JSON strings in Python.",
                "d-dev-positive": "Array sort compares JavaScript values.",
                "d-test-positive": "os.ReadFile reads a file in Go.",
                "d-distractor": "Unrelated database migration notes.",
            },
            "qrels_by_split": {
                "train": {"q-train": {"d-train-positive": 1.0}},
                "dev": {"q-dev": {"d-dev-positive": 1.0}},
                "test": {"q-test": {"d-test-positive": 1.0}},
            },
        }

        train_rows, dev_rows, test_rows = prepare_coir_groups(
            bundle,
            candidate_k=1,
            embedder=HashEmbeddingProvider(dimension=8),
            append_qrel_positives=True,
        )

        self.assertEqual([row["split"] for row in train_rows], ["train"])
        self.assertEqual([row["split"] for row in dev_rows], ["dev"])
        self.assertEqual([row["split"] for row in test_rows], ["test"])
        self.assertEqual(train_rows[0]["split_policy"], "coir_native_qrels_split")
        self.assertIn("d-train-positive", {candidate["doc_id"] for candidate in train_rows[0]["candidates"]})
        self.assertTrue(train_rows[0]["positive_in_candidate_pool"])

    def test_prepare_coir_groups_supports_train_limit(self):
        bundle = {
            "dataset_name": "coir-unit",
            "queries": {f"q{i}": f"query {i}" for i in range(4)},
            "corpus": {f"d{i}": f"document {i}" for i in range(4)},
            "qrels_by_split": {
                "train": {f"q{i}": {f"d{i}": 1.0} for i in range(4)},
                "dev": {"q0": {"d0": 1.0}},
                "test": {"q1": {"d1": 1.0}},
            },
        }

        train_rows, _, _ = prepare_coir_groups(
            bundle,
            candidate_k=2,
            embedder=HashEmbeddingProvider(dimension=8),
            train_limit=2,
        )

        self.assertEqual(len(train_rows), 2)


if __name__ == "__main__":
    unittest.main()

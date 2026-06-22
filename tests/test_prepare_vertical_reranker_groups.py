from __future__ import annotations

import unittest

from activation_rag.embedding import HashEmbeddingProvider
from scripts.prepare_vertical_reranker_groups import (
    RerankingComponents,
    RetrievalComponents,
    prepare_reranking_groups,
    prepare_retrieval_groups,
    split_groups,
)


class PrepareVerticalRerankerGroupsTests(unittest.TestCase):
    def test_prepare_retrieval_groups_labels_dense_candidates(self) -> None:
        components = RetrievalComponents(
            dataset_name="fixture-legal",
            queries={"q1": "jury impartiality", "q2": "contract privacy notice"},
            corpus={
                "d1": "jury judge impartiality juror",
                "d2": "contract privacy disclosure notice",
                "d3": "unrelated tax procedure",
            },
            qrels={"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        )
        groups = prepare_retrieval_groups(
            components,
            candidate_k=2,
            embedder=HashEmbeddingProvider(dimension=64),
        )

        self.assertEqual(2, len(groups))
        by_query = {group["query_id"]: group for group in groups}
        self.assertTrue(by_query["q1"]["positive_in_candidate_pool"])
        self.assertIn("embedding_cosine", by_query["q1"]["dense_score_source"])
        positive_candidates = [candidate for candidate in by_query["q1"]["candidates"] if candidate["label"] > 0]
        self.assertEqual(["d1"], [candidate["doc_id"] for candidate in positive_candidates])
        self.assertIn("query_activation_chunk_id", by_query["q1"])

    def test_prepare_retrieval_groups_can_append_missing_positives(self) -> None:
        components = RetrievalComponents(
            dataset_name="fixture-med",
            queries={"q1": "alpha"},
            corpus={"d1": "alpha", "d2": "alpha alpha", "d3": "zzzz rare positive"},
            qrels={"q1": {"d3": 1.0}},
        )
        groups = prepare_retrieval_groups(
            components,
            candidate_k=1,
            embedder=HashEmbeddingProvider(dimension=32),
            append_qrel_positives=True,
        )

        self.assertTrue(groups[0]["positive_in_candidate_pool"])
        self.assertIn("d3", [candidate["doc_id"] for candidate in groups[0]["candidates"]])
        self.assertGreater(len(groups[0]["candidates"]), 1)

    def test_prepare_reranking_groups_preserves_top_ranked_order(self) -> None:
        components = RerankingComponents(
            dataset_name="fixture-coreb",
            queries={"q1": "find equivalent code"},
            corpus={"d1": "correct code", "d2": "wrong code", "d3": "also wrong"},
            qrels={"q1": {"d1": 1.0, "d2": 0.0}},
            top_ranked={"q1": ["d2", "d1", "d3"]},
        )
        groups = prepare_reranking_groups(components, candidate_k=3)

        self.assertEqual(1, len(groups))
        candidates = groups[0]["candidates"]
        self.assertEqual(["d2", "d1", "d3"], [candidate["doc_id"] for candidate in candidates])
        self.assertEqual([1, 2, 3], [candidate["dense_rank"] for candidate in candidates])
        self.assertEqual([0, 1, 0], [candidate["label"] for candidate in candidates])
        self.assertEqual("top_ranked_reciprocal_rank", groups[0]["dense_score_source"])

    def test_split_groups_is_disjoint_and_marks_split(self) -> None:
        groups = [
            {
                "query_id": f"q{i}",
                "split": "unsplit",
                "candidates": [{"label": 1}],
                "positive_in_candidate_pool": True,
            }
            for i in range(20)
        ]

        train_rows, dev_rows, test_rows = split_groups(groups, dev_fraction=0.2, test_fraction=0.2, seed="fixture")

        self.assertEqual(20, len(train_rows) + len(dev_rows) + len(test_rows))
        self.assertEqual({"train"}, {row["split"] for row in train_rows})
        self.assertEqual({"dev"}, {row["split"] for row in dev_rows})
        self.assertEqual({"test"}, {row["split"] for row in test_rows})
        self.assertFalse({row["query_id"] for row in train_rows} & {row["query_id"] for row in dev_rows})
        self.assertFalse({row["query_id"] for row in train_rows} & {row["query_id"] for row in test_rows})


if __name__ == "__main__":
    unittest.main()

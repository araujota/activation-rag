import unittest

from activation_rag.supervised_reranking import (
    evaluate_group_rankings,
    train_pairwise_linear_ranker,
)


class SupervisedRerankingTests(unittest.TestCase):
    def test_pairwise_linear_ranker_learns_positive_candidate_direction(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "p1", "doc_id": "d1", "label": 1, "features": {"dense_score": 0.5, "activation_match": 2.0}},
                    {"chunk_id": "n1", "doc_id": "d2", "label": 0, "features": {"dense_score": 0.9, "activation_match": -1.0}},
                ],
            },
            {
                "query_id": "q2",
                "candidates": [
                    {"chunk_id": "p2", "doc_id": "d3", "label": 1, "features": {"dense_score": 0.4, "activation_match": 1.5}},
                    {"chunk_id": "n2", "doc_id": "d4", "label": 0, "features": {"dense_score": 0.8, "activation_match": -0.5}},
                ],
            },
        ]

        model = train_pairwise_linear_ranker(
            groups,
            feature_names=("dense_score", "activation_match"),
            epochs=200,
            learning_rate=0.2,
            l2=0.0,
        )

        self.assertGreater(model.weights["activation_match"], 0.0)
        scores = model.score_group(groups[0])
        self.assertGreater(scores["p1"], scores["n1"])

    def test_evaluate_group_rankings_computes_dense_and_model_metrics(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "n1", "doc_id": "d2", "label": 0, "dense_rank": 1, "features": {"f": 0.0}},
                    {"chunk_id": "p1", "doc_id": "d1", "label": 1, "dense_rank": 2, "features": {"f": 1.0}},
                ],
            }
        ]
        model = train_pairwise_linear_ranker(
            groups,
            feature_names=("f",),
            epochs=100,
            learning_rate=0.1,
            l2=0.0,
        )

        metrics = evaluate_group_rankings(groups, model=model, top_k=2)

        self.assertLess(metrics["dense"]["mrr@2"], metrics["model"]["mrr@2"])
        self.assertEqual(metrics["model"]["recall@2"], 1.0)


if __name__ == "__main__":
    unittest.main()

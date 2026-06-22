import unittest

from scripts.run_text_reranker_baseline import evaluate_with_scores


class TextRerankerBaselineTests(unittest.TestCase):
    def test_evaluate_with_scores_reranks_same_candidate_groups(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "n1", "doc_id": "d2", "label": 0, "dense_rank": 1},
                    {"chunk_id": "p1", "doc_id": "d1", "label": 1, "dense_rank": 2},
                ],
            }
        ]
        scores = {("q1", "n1"): 0.1, ("q1", "p1"): 0.9}

        metrics = evaluate_with_scores(groups, scores=scores, top_k=2)

        self.assertLess(metrics["dense"]["mrr@2"], metrics["text_reranker"]["mrr@2"])


if __name__ == "__main__":
    unittest.main()

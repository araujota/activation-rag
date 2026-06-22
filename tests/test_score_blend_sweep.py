import unittest

import numpy as np

from scripts.run_score_blend_sweep import _blend_scores


class ScoreBlendSweepTests(unittest.TestCase):
    def test_blends_dense_and_reranker_scores_per_query(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "a", "dense_score": 2.0},
                    {"chunk_id": "b", "dense_score": 1.0},
                ],
            }
        ]
        scores = {("q1", "a"): 0.0, ("q1", "b"): 10.0}

        dense_only = _blend_scores(groups, scores, alpha=0.0)
        rerank_only = _blend_scores(groups, scores, alpha=1.0)

        self.assertGreater(dense_only[("q1", "a")], dense_only[("q1", "b")])
        self.assertGreater(rerank_only[("q1", "b")], rerank_only[("q1", "a")])


if __name__ == "__main__":
    unittest.main()

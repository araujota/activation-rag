import unittest

from scripts.run_core245_hubness_diagnostics import _occurrence_stats, _score_diagnostics


class RunCore245HubnessDiagnosticsTests(unittest.TestCase):
    def test_occurrence_stats_exposes_hub_skew(self):
        stats = _occurrence_stats(["a", "a", "a", "b"])

        self.assertEqual(stats["max_occurrences"], 3)
        self.assertEqual(stats["unique_items"], 2)
        self.assertAlmostEqual(stats["max_fraction"], 0.75)
        self.assertGreater(stats["gini"], 0.0)

    def test_score_diagnostics_evaluates_metric_and_hubness(self):
        groups = [
            {
                "query_id": "q1",
                "query_activation_chunk_id": "query",
                "candidates": [
                    {"chunk_id": "p", "doc_id": "p", "label": 1, "dense_rank": 2, "features": {"score": 0.9}},
                    {"chunk_id": "n", "doc_id": "n", "label": 0, "dense_rank": 1, "features": {"score": 0.1}},
                ],
            },
            {
                "query_id": "q2",
                "query_activation_chunk_id": "query",
                "candidates": [
                    {"chunk_id": "hub", "doc_id": "hub", "label": 0, "dense_rank": 1, "features": {"score": 1.0}},
                    {"chunk_id": "p2", "doc_id": "p2", "label": 1, "dense_rank": 2, "features": {"score": 0.8}},
                ],
            },
        ]

        diagnostics = _score_diagnostics(
            groups,
            score_name="score",
            scorer=lambda candidate: float(candidate["features"]["score"]),
            top_k=1,
        )

        self.assertEqual(diagnostics["metrics"]["mrr@1"], 0.5)
        self.assertEqual(diagnostics["top1_hubness"]["total_occurrences"], 2)
        self.assertIn("top1_hubness", diagnostics)


if __name__ == "__main__":
    unittest.main()

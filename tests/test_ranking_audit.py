import unittest

from activation_rag.ranking_audit import compare_ranked_systems, paired_randomization_test


class RankingAuditTests(unittest.TestCase):
    def test_paired_randomization_detects_consistent_improvement(self):
        baseline = [0.0, 0.0, 0.0, 0.0]
        candidate = [1.0, 1.0, 1.0, 1.0]

        result = paired_randomization_test(baseline, candidate, iterations=1000, seed=7)

        self.assertEqual(result["query_count"], 4)
        self.assertEqual(result["mean_delta"], 1.0)
        self.assertLess(result["p_value"], 0.2)

    def test_compare_ranked_systems_reports_metric_deltas_and_changed_queries(self):
        groups = [
            _group("q-helped", "positive helped", "negative helped"),
            _group("q-harmed", "positive harmed", "negative harmed"),
            _group("q-same", "positive same", "negative same"),
        ]
        baseline_scores = {
            ("q-helped", "q-helped-neg"): 2.0,
            ("q-helped", "q-helped-pos"): 1.0,
            ("q-harmed", "q-harmed-pos"): 2.0,
            ("q-harmed", "q-harmed-neg"): 1.0,
            ("q-same", "q-same-pos"): 2.0,
            ("q-same", "q-same-neg"): 1.0,
        }
        candidate_scores = {
            ("q-helped", "q-helped-pos"): 2.0,
            ("q-helped", "q-helped-neg"): 1.0,
            ("q-harmed", "q-harmed-neg"): 2.0,
            ("q-harmed", "q-harmed-pos"): 1.0,
            ("q-same", "q-same-pos"): 2.0,
            ("q-same", "q-same-neg"): 1.0,
        }

        audit = compare_ranked_systems(
            groups,
            baseline_name="baseline",
            candidate_name="candidate",
            baseline_scores=baseline_scores,
            candidate_scores=candidate_scores,
            top_k=1,
            randomization_iterations=200,
            changed_query_limit=2,
            seed=11,
        )

        self.assertEqual(audit["query_count"], 3)
        self.assertIn("ndcg@1", audit["paired_significance"])
        self.assertEqual(audit["changed_query_summary"]["ndcg@1"]["improved_query_count"], 1)
        self.assertEqual(audit["changed_query_summary"]["ndcg@1"]["harmed_query_count"], 1)
        self.assertEqual(audit["changed_query_summary"]["ndcg@1"]["unchanged_query_count"], 1)
        self.assertEqual(audit["changed_queries"]["helped"][0]["query_id"], "q-helped")
        self.assertEqual(audit["changed_queries"]["harmed"][0]["query_id"], "q-harmed")
        self.assertEqual(audit["changed_queries"]["helped"][0]["candidate_top"][0]["chunk_id"], "q-helped-pos")
        self.assertEqual(audit["changed_queries"]["harmed"][0]["baseline_top"][0]["chunk_id"], "q-harmed-pos")


def _group(query_id: str, positive_text: str, negative_text: str) -> dict:
    return {
        "query_id": query_id,
        "query_text": f"query {query_id}",
        "candidates": [
            {
                "chunk_id": f"{query_id}-pos",
                "doc_id": f"{query_id}-pos-doc",
                "label": 1,
                "dense_rank": 1,
                "text": positive_text,
            },
            {
                "chunk_id": f"{query_id}-neg",
                "doc_id": f"{query_id}-neg-doc",
                "label": 0,
                "dense_rank": 2,
                "text": negative_text,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
